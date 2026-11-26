<#
.SYNOPSIS
  Reconcile a PresentMon CSV against a VIctory diagnostics log (present-to-present ground truth).

.DESCRIPTION
  Takes one PresentMon capture CSV plus one victory diag log (a run with
  VICTORY_GPU_TIMING=1: Meter::frame fps lines, costs::log GPU-cost reports)
  and prints:

  - app fps (victory Meter::frame median) vs present fps (PresentMon
    (N-1)/span over the aligned window)
  - frame-time p50/p95/p99 for both (victory ms/frame samples vs
    PresentMon MsBetweenPresents samples)
  - wall-busy-gap reconciliation: victory wall median (costs::log) vs
    PresentMon MsBetweenPresents p50, with summed-busy / gap / noted /
    residual context so unstamped work (preludes + present-copy list,
    docs/gpu-profiling-methodology.md section 1.3) is visible instead of
    hiding inside the delta
  - a mismatch flag when the two disagree beyond the documented timed-mode
    distortion band (12-40 ms, methodology section 1.2): |diff| <= 12 ms is
    agreement, 12-40 ms is inside the distortion band (timed waits/readback
    plausibly explain it), > 40 ms is a MISMATCH.

  Time alignment: the victory log has no per-line timestamps, so alignment is
  by run window. Capture PresentMon across the whole victory run (see
  tools/presentmon/README.md), then optionally pass -StartTime/-EndTime
  (PresentMon TimeInSeconds bounds) and/or -SkipFirst/-SkipLast (drop
  load-transient presents) to compare the same steady window both sides saw.

  PresentMon CSV schema varies by release; this script needs MsBetweenPresents
  plus a time column (TimeInSeconds preferred, QpcTime+QpcFrequency fallback).
  Column lookup is case-insensitive. Optional columns used when present:
  Application / ProcessID (for -App filtering), MsBetweenDisplayChange,
  MsUntilDisplayed.

.EXAMPLE
  ./compare-presentmon.ps1 -PresentMonCsv run.csv -DiagLog victory.log

.EXAMPLE
  ./compare-presentmon.ps1 -PresentMonCsv run.csv -DiagLog victory.log `
    -App victory -SkipFirst 30 -SkipLast 10 -Out recon.txt

.EXAMPLE
  ./compare-presentmon.ps1 -SelfTest
#>
param(
  [string]$PresentMonCsv = "",
  [string]$DiagLog = "",
  [string]$App = "",
  [double]$StartTime = [double]::NaN,
  [double]$EndTime = [double]::NaN,
  [int]$SkipFirst = 0,
  [int]$SkipLast = 0,
  [string]$Out = "",
  [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
# Invariant output: CSV/log decimals are dots, and [double] casts parse
# invariantly; force the display culture too so columns line up anywhere.
[System.Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [cultureinfo]::InvariantCulture

function Get-Median([double[]]$v) {
  if ($null -eq $v -or $v.Count -eq 0) { return [double]::NaN }
  $s = $v | Sort-Object
  $n = $s.Count
  if ($n % 2 -eq 1) { return $s[[int]($n / 2)] }
  return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

function Get-Percentile([double[]]$v, [double]$p) {
  if ($null -eq $v -or $v.Count -eq 0) { return [double]::NaN }
  $s = $v | Sort-Object
  $n = $s.Count
  if ($n -eq 1) { return $s[0] }
  $rank = $p / 100.0 * ($n - 1)
  $lo = [int][Math]::Floor($rank)
  $hi = [int][Math]::Ceiling($rank)
  if ($lo -eq $hi) { return $s[$lo] }
  return $s[$lo] + ($s[$hi] - $s[$lo]) * ($rank - $lo)
}

function Get-Column($row, [string]$name) {
  foreach ($p in $row.PSObject.Properties) {
    if ($p.Name -ieq $name) { return $p.Value }
  }
  return $null
}

function Get-DiagStats([string]$Path) {
  if (-not (Test-Path $Path)) { throw "diag log not found: $Path" }
  $text = [System.IO.File]::ReadAllText($Path)
  $fps = [System.Collections.Generic.List[double]]::new()
  $ms = [System.Collections.Generic.List[double]]::new()
  foreach ($m in [regex]::Matches($text,
    '(?<fps>\d+\.\d+) fps, (?<ms>\d+\.\d+) ms/frame \((?<cpu>\d+\.\d+) preparing: copies (?<copyMs>\d+\.\d+) ms (?<copyMB>\d+\.\d+) MB, textures (?<texMs>\d+\.\d+) ms (?<texMB>\d+\.\d+) MB; (?<gpuWait>\d+\.\d+) ms host wait for submissions; (?<backMs>\d+\.\d+) copying back (?<backMB>\d+\.\d+) MB\), (?<draws>\d+) draws/frame')) {
    $fps.Add([double]$m.Groups['fps'].Value)
    $ms.Add([double]$m.Groups['ms'].Value)
  }
  $busy = [System.Collections.Generic.List[double]]::new()
  $wall = [System.Collections.Generic.List[double]]::new()
  $gap = [System.Collections.Generic.List[double]]::new()
  $noted = [System.Collections.Generic.List[double]]::new()
  $resid = [System.Collections.Generic.List[double]]::new()
  foreach ($m in [regex]::Matches($text,
    'GPU cost by pipeline: (?<busy>\d+\.\d+) ms/frame summed busy over (?<lists>\d+) lists in (?<subs>\d+) submissions \((?<cmds>\d+) commands/frame; wall (?<wall>\d+\.\d+) ms/frame, gap (?<gap>\d+\.\d+), util (?<util>\d+)%; noted (?<noted>\d+\.\d+), residual (?<resid>\d+\.\d+), dropped (?<dropped>\d+)\)')) {
    $busy.Add([double]$m.Groups['busy'].Value)
    $wall.Add([double]$m.Groups['wall'].Value)
    $gap.Add([double]$m.Groups['gap'].Value)
    $noted.Add([double]$m.Groups['noted'].Value)
    $resid.Add([double]$m.Groups['resid'].Value)
  }
  return @{
    File = $Path
    FpsN = $fps.Count; CostN = $busy.Count
    FpsMed = (Get-Median $fps.ToArray())
    Ms = $ms.ToArray()
    MsMed = (Get-Median $ms.ToArray())
    MsP50 = (Get-Percentile $ms.ToArray() 50)
    MsP95 = (Get-Percentile $ms.ToArray() 95)
    MsP99 = (Get-Percentile $ms.ToArray() 99)
    BusyMed = (Get-Median $busy.ToArray())
    WallMed = (Get-Median $wall.ToArray())
    GapMed = (Get-Median $gap.ToArray())
    NotedMed = (Get-Median $noted.ToArray())
    ResidMed = (Get-Median $resid.ToArray())
  }
}

function Get-PresentStats([string]$Path, [string]$AppFilter, [double]$T0, [double]$T1, [int]$DropFirst, [int]$DropLast) {
  if (-not (Test-Path $Path)) { throw "PresentMon CSV not found: $Path" }
  $rows = Import-Csv -Path $Path
  if ($null -eq $rows -or @($rows).Count -eq 0) { throw "PresentMon CSV is empty: $Path" }
  $rows = @($rows)
  if ($AppFilter -ne '') {
    $kept = @($rows | Where-Object {
      $a = Get-Column $_ 'Application'
      $a -ne $null -and ([string]$a -like "*$AppFilter*")
    })
    if ($kept.Count -eq 0) { throw "no PresentMon rows match -App '$AppFilter'" }
    $rows = $kept
  }
  # Resolve time + interval columns (schema varies by PresentMon release).
  $first = $rows[0]
  $hasInterval = (Get-Column $first 'MsBetweenPresents') -ne $null
  if (-not $hasInterval) { throw 'PresentMon CSV has no MsBetweenPresents column' }
  $timeKind = 'TimeInSeconds'
  if ((Get-Column $first 'TimeInSeconds') -eq $null) {
    if ((Get-Column $first 'QpcTime') -ne $null) { $timeKind = 'QpcTime' } else { $timeKind = '' }
  }
  $freq = 0.0
  if ($timeKind -eq 'QpcTime') {
    $f = Get-Column $first 'QpcFrequency'
    if ($f -eq $null) { throw 'PresentMon CSV has QpcTime but no QpcFrequency; use a capture with a TimeInSeconds column' }
    $freq = [double]$f
  }
  $times = [System.Collections.Generic.List[double]]::new()
  $gaps = [System.Collections.Generic.List[double]]::new()
  foreach ($r in $rows) {
    $tRaw = Get-Column $r $timeKind
    $gRaw = Get-Column $r 'MsBetweenPresents'
    if ($timeKind -eq '') {
      # No usable time column: keep order, synthesize 0-based times later.
      $times.Add([double]::NaN)
    }
    elseif ($timeKind -eq 'QpcTime') {
      $times.Add([double]$tRaw / $freq)
    }
    else {
      $times.Add([double]$tRaw)
    }
    $g = 0.0
    if ([double]::TryParse([string]$gRaw, [ref]$g)) { $gaps.Add($g) } else { $gaps.Add([double]::NaN) }
  }
  # Align to the requested window, then drop transients.
  $idx = 0..($rows.Count - 1)
  if ($timeKind -ne '' -and (-not [double]::IsNaN($T0))) { $idx = $idx | Where-Object { $times[$_] -ge $T0 } }
  if ($timeKind -ne '' -and (-not [double]::IsNaN($T1))) { $idx = $idx | Where-Object { $times[$_] -le $T1 } }
  $idx = @($idx)
  if ($DropFirst -gt 0) {
    if ($DropFirst -ge $idx.Count) { throw '-SkipFirst drops every PresentMon row' }
    $idx = $idx | Select-Object -Skip $DropFirst
  }
  if ($DropLast -gt 0) {
    if ($DropLast -ge @($idx).Count) { throw '-SkipLast drops every PresentMon row' }
    $idx = @($idx) | Select-Object -SkipLast $DropLast
  }
  $idx = @($idx)
  if ($idx.Count -eq 0) { throw 'no PresentMon rows left after time-window/skip filtering' }
  $takes = @($idx | ForEach-Object { $times[$_] })
  $g = @($idx | ForEach-Object { $gaps[$_] } | Where-Object { -not [double]::IsNaN($_) -and $_ -gt 0 })
  if ($g.Count -eq 0) { throw 'no positive MsBetweenPresents samples left after filtering' }
  $span = 0.0
  $pfps = [double]::NaN
  if ($timeKind -ne '') {
    $tfinite = @($takes | Where-Object { -not [double]::IsNaN($_) })
    if ($tfinite.Count -ge 2) {
      $span = ($tfinite | Measure-Object -Maximum).Maximum - ($tfinite | Measure-Object -Minimum).Minimum
      if ($span -gt 0) { $pfps = ($tfinite.Count - 1) / $span } else { $pfps = [double]::NaN }
    }
  }
  if ([double]::IsNaN($pfps)) {
    # No time column at all: fall back to the mean interval.
    $mean = ($g | Measure-Object -Average).Average
    if ($mean -gt 0) { $pfps = 1000.0 / $mean }
  }
  return @{
    File = $Path; N = $idx.Count; Span = $span; Fps = $pfps
    G = $g
    P50 = (Get-Percentile $g 50); P95 = (Get-Percentile $g 95); P99 = (Get-Percentile $g 99)
  }
}

function Format-Optional([double]$v, [string]$fmt) {
  if ([double]::IsNaN($v)) { return '-' }
  return ('{0:' + $fmt + '}') -f $v
}

if ($SelfTest) {
  $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('presentmon-selftest-' + [Guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  try {
    # Synthetic PresentMon CSV: 100 presents, steady 33.33 ms pacing with one
    # 50 ms hitch. TimeInSeconds matches the gaps exactly.
    $t = 1000.0
    $csvLines = @('Application,ProcessID,SwapChainAddress,PresentMode,TimeInSeconds,MsBetweenPresents,MsUntilDisplayed')
    for ($i = 0; $i -lt 100; $i++) {
      if ($i -eq 0) { $gap = 0.0 } elseif ($i -eq 50) { $gap = 50.0 } else { $gap = 33.333 }
      if ($i -gt 0) { $t += $gap / 1000.0 }
      $csvLines += ('victory.exe,1234,0x1,Hardware: Independent Flip,' + $t.ToString('0.000000', [cultureinfo]::InvariantCulture) + ',' + $gap.ToString('0.000', [cultureinfo]::InvariantCulture) + ',12.000')
    }
    $csv = Join-Path $tmp 'present.csv'
    [System.IO.File]::WriteAllLines($csv, $csvLines)
    # Synthetic diag log: two fps lines (30 fps / 33.3 ms) + one cost report
    # whose wall (34.0 ms) sits just above the present p50, inside agreement.
    $logLines = @(
      '30.0 fps, 33.3 ms/frame (5.0 preparing: copies 1.0 ms 2.0 MB, textures 1.0 ms 3.0 MB; 4.0 ms host wait for submissions; 0.5 copying back 0.1 MB), 1000 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect; 0 root-view slots/frame; 0.0 ms/frame of draws waiting for pipelines'
      '30.0 fps, 33.3 ms/frame (5.0 preparing: copies 1.0 ms 2.0 MB, textures 1.0 ms 3.0 MB; 4.0 ms host wait for submissions; 0.5 copying back 0.1 MB), 1000 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect; 0 root-view slots/frame; 0.0 ms/frame of draws waiting for pipelines'
      'GPU cost by pipeline: 20.0 ms/frame summed busy over 4 lists in 2 submissions (500 commands/frame; wall 34.0 ms/frame, gap 14.0, util 59%; noted 19.0, residual 1.0, dropped 0)'
    )
    $log = Join-Path $tmp 'victory.log'
    [System.IO.File]::WriteAllLines($log, $logLines)
    $d = Get-DiagStats $log
    $p = Get-PresentStats $csv '' ([double]::NaN) ([double]::NaN) 1 0
    $fails = @()
    if ($p.N -ne 99) { $fails += "present N: expected 99 after -SkipFirst 1, got $($p.N)" }
    if ([Math]::Abs($p.P50 - 33.333) -gt 0.01) { $fails += ('present p50: expected 33.333, got {0:N3}' -f $p.P50) }
    if ([Math]::Abs($p.P95 - 33.333) -gt 0.01) { $fails += ('present p95: expected 33.333, got {0:N3}' -f $p.P95) }
    if ([Math]::Abs($p.P99 - 33.666) -gt 0.05) { $fails += ('present p99: expected ~33.67 (linear interpolation over 98x33.333 + 1x50 hitch), got {0:N3}' -f $p.P99) }
    if ([Math]::Abs($p.Fps - 1000.0 / 33.5) -gt 0.5) { $fails += ('present fps: expected ~29.9, got {0:N2}' -f $p.Fps) }
    if ([Math]::Abs($d.FpsMed - 30.0) -gt 1e-9) { $fails += "app fps median: expected 30.0, got $($d.FpsMed)" }
    if ([Math]::Abs($d.WallMed - 34.0) -gt 1e-9) { $fails += "wall median: expected 34.0, got $($d.WallMed)" }
    $diff = $p.P50 - $d.WallMed
    if ([Math]::Abs($diff) -gt 12.0) { $fails += ('reconciliation: expected |diff| <= 12 ms, got {0:N2}' -f $diff) }
    # Mismatch path: a second CSV paced at 100 ms must trip the > 40 ms flag.
    $t2 = 2000.0
    $slow = @('Application,ProcessID,SwapChainAddress,PresentMode,TimeInSeconds,MsBetweenPresents,MsUntilDisplayed')
    for ($i = 0; $i -lt 20; $i++) {
      if ($i -eq 0) { $g2 = 0.0 } else { $g2 = 100.0; $t2 += $g2 / 1000.0 }
      $slow += ('victory.exe,1234,0x1,Hardware: Independent Flip,' + $t2.ToString('0.000000', [cultureinfo]::InvariantCulture) + ',' + $g2.ToString('0.000', [cultureinfo]::InvariantCulture) + ',12.000')
    }
    $csv2 = Join-Path $tmp 'slow.csv'
    [System.IO.File]::WriteAllLines($csv2, $slow)
    $p2 = Get-PresentStats $csv2 '' ([double]::NaN) ([double]::NaN) 1 0
    if (-not (($p2.P50 - $d.WallMed) -gt 40.0)) { $fails += 'mismatch path: expected (slow p50 - wall) > 40 ms' }
    if ($fails.Count -eq 0) {
      Write-Output 'SELF-TEST PASS: present fps/p50/p95/p99, app fps, wall reconciliation, and >40 ms mismatch flag all match synthetic inputs.'
    }
    else {
      foreach ($f in $fails) { Write-Output ("SELF-TEST FAIL: $f") }
      exit 1
    }
  }
  finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  }
  return
}

if ($PresentMonCsv -eq '' -or $DiagLog -eq '') { throw 'need -PresentMonCsv and -DiagLog (or use -SelfTest)' }

$diag = Get-DiagStats $DiagLog
$pres = Get-PresentStats $PresentMonCsv $App $StartTime $EndTime $SkipFirst $SkipLast

if ($diag.FpsN -eq 0) { throw "no Meter::frame fps lines in $DiagLog (was it run with VICTORY_GPU_TIMING=1?)" }
if ($diag.CostN -eq 0) { throw "no 'GPU cost by pipeline:' reports in $DiagLog" }

$dApp = $pres.P50 - $diag.MsMed
$dWall = $pres.P50 - $diag.WallMed

function Flag-For([double]$d) {
  $a = [Math]::Abs($d)
  if ([double]::IsNaN($a)) { return 'UNKNOWN (no data)' }
  if ($a -le 12.0) { return 'AGREE (within +/-12 ms)' }
  if ($a -le 40.0) { return 'DISTORTION-BAND (12-40 ms: timed-mode waits/readback plausibly explain it)' }
  return 'MISMATCH (> 40 ms: beyond the timed-mode distortion band)'
}

$lines = @()
$lines += '== present-to-present reconciliation =='
$lines += ("  PresentMon : {0} (N={1} presents, span {2:N2} s)" -f $PresentMonCsv, $pres.N, $pres.Span)
$lines += ("  victory log: {0} ({1} fps lines, {2} cost reports)" -f $DiagLog, $diag.FpsN, $diag.CostN)
$lines += ''
$lines += 'fps:'
$lines += ("  app fps (Meter::frame median) .... {0}" -f (Format-Optional $diag.FpsMed 'N1'))
$lines += ("  present fps ((N-1)/span) ......... {0}" -f (Format-Optional $pres.Fps 'N2'))
$lines += ''
$lines += 'frame-time percentiles (ms):'
$lines += ("  victory ms/frame .. p50 {0,8} | p95 {1,8} | p99 {2,8}" -f
  (Format-Optional $diag.MsP50 'N2'), (Format-Optional $diag.MsP95 'N2'), (Format-Optional $diag.MsP99 'N2'))
$lines += ("  present gap ....... p50 {0,8} | p95 {1,8} | p99 {2,8}   (MsBetweenPresents)" -f
  (Format-Optional $pres.P50 'N2'), (Format-Optional $pres.P95 'N2'), (Format-Optional $pres.P99 'N2'))
$lines += ''
$lines += 'wall-busy-gap reconciliation (ms/frame medians):'
$lines += ("  summed busy {0} | wall {1} | gap {2} | noted {3} | residual {4}" -f
  (Format-Optional $diag.BusyMed 'N1'), (Format-Optional $diag.WallMed 'N1'),
  (Format-Optional $diag.GapMed 'N1'), (Format-Optional $diag.NotedMed 'N1'), (Format-Optional $diag.ResidMed 'N1'))
$lines += ("  present p50 - app ms/frame median . {0:+0.00;-0.00} ms -> {1}" -f $dApp, (Flag-For $dApp))
$lines += ("  present p50 - victory wall median  {0:+0.00;-0.00} ms -> {1}" -f $dWall, (Flag-For $dWall))
$lines += ''
$worst = [Math]::Max([Math]::Abs($dApp), [Math]::Abs($dWall))
if ([double]::IsNaN($worst)) { $lines += 'Verdict: UNKNOWN - missing data on one side.' }
elseif ($worst -gt 40.0) { $lines += 'Verdict: MISMATCH - the two sides disagree beyond the 12-40 ms timed-mode distortion band; do not claim hardware fps from this pair.' }
elseif ($worst -gt 12.0) { $lines += 'Verdict: DISTORTION-BAND - disagreement fits inside timed-mode waits/readback; hardware fps claims still need an untimed PresentMon run.' }
else { $lines += 'Verdict: AGREE - present pacing matches the victory log within +/-12 ms.' }

$text = $lines -join "`r`n"
Write-Output $text
if ($Out -ne '') {
  [System.IO.File]::WriteAllText($Out, $text)
  Write-Output "wrote $Out"
}
