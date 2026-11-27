<#
.SYNOPSIS
  Compare two sides (baseline vs candidate) of VIctory diagnostics logs.

.DESCRIPTION
  Extracts the per-report-period metrics from logs produced with
  VICTORY_GPU_TIMING=1 (crates/victory-gpu/src/host.rs: costs::log,
  Meter::frame, starve::log_starvation) and the copystat/texture companions,
  then prints:

  - side summaries (median across all report periods in all files per side):
    fps, ms/frame, draws/frame, summed-busy, wall, util, gap, noted,
    residual, upload/copy/texture MB, heap + title-memory step ms
  - a hash-keyed diff table of the per-kind-then-pipeline cost rows
    (crates/victory-gpu/src/host.rs: cost_label/cost_kind), so that a faster
    build reaching a different cutscene shows up as scene variance instead
    of hiding inside a wall-clock average
  - scene-variance warnings when draws/commands/texture footprints differ

  Give each side one or more log files (repeat runs of the same build/scene);
  row costs are medianed across every cost report in every file of the side.

.EXAMPLE
  ./compare-gpu-logs.ps1 -Baseline base1.log,base2.log,base3.log `
    -Candidate cand1.log,cand2.log,cand3.log -Top 25
#>
param(
  [Parameter(Mandatory = $true)][string[]]$Baseline,
  [Parameter(Mandatory = $true)][string[]]$Candidate,
  [int]$Top = 25,
  [string]$Out = ""
)

$ErrorActionPreference = 'Stop'
# Invariant output: log decimals are dots, and [double] casts parse
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

function Get-Reports([string]$Path) {
  $text = [System.IO.File]::ReadAllText($Path)
  $r = @{
    File    = $Path
    Fps     = [System.Collections.Generic.List[double]]::new()
    Ms      = [System.Collections.Generic.List[double]]::new()
    Draws   = [System.Collections.Generic.List[double]]::new()
    CopyMs  = [System.Collections.Generic.List[double]]::new()
    CopyMB  = [System.Collections.Generic.List[double]]::new()
    BackMs  = [System.Collections.Generic.List[double]]::new()
    BackMB  = [System.Collections.Generic.List[double]]::new()
    Busy    = [System.Collections.Generic.List[double]]::new()
    Wall    = [System.Collections.Generic.List[double]]::new()
    Util    = [System.Collections.Generic.List[double]]::new()
    Gap     = [System.Collections.Generic.List[double]]::new()
    Noted   = [System.Collections.Generic.List[double]]::new()
    Resid   = [System.Collections.Generic.List[double]]::new()
    Cmds    = [System.Collections.Generic.List[double]]::new()
    Subs    = [System.Collections.Generic.List[double]]::new()
    BlockMs = [System.Collections.Generic.List[double]]::new()
    HeapMs  = [System.Collections.Generic.List[double]]::new()
    MemMs   = [System.Collections.Generic.List[double]]::new()
    UpMs    = [System.Collections.Generic.List[double]]::new()
    PreMs   = [System.Collections.Generic.List[double]]::new()
    TexMB   = [System.Collections.Generic.List[double]]::new()
    Rows    = @{}
  }
  # Meter::frame fps line (host.rs).
  foreach ($m in [regex]::Matches($text,
    '(?<fps>\d+\.\d+) fps, (?<ms>\d+\.\d+) ms/frame \((?<cpu>\d+\.\d+) preparing: copies (?<copyMs>\d+\.\d+) ms (?<copyMB>\d+\.\d+) MB, textures (?<texMs>\d+\.\d+) ms (?<texMB>\d+\.\d+) MB; (?<gpuWait>\d+\.\d+) ms host wait for submissions; (?<backMs>\d+\.\d+) copying back (?<backMB>\d+\.\d+) MB\), (?<draws>\d+) draws/frame')) {
    $r.Fps.Add([double]$m.Groups['fps'].Value)
    $r.Ms.Add([double]$m.Groups['ms'].Value)
    $r.Draws.Add([double]$m.Groups['draws'].Value)
    $r.CopyMs.Add([double]$m.Groups['copyMs'].Value)
    $r.CopyMB.Add([double]$m.Groups['copyMB'].Value)
    $r.BackMs.Add([double]$m.Groups['backMs'].Value)
    $r.BackMB.Add([double]$m.Groups['backMB'].Value)
  }
  # costs::log summary + per-kind-then-pipeline rows (host.rs).
  foreach ($m in [regex]::Matches($text,
    'GPU cost by pipeline: (?<busy>\d+\.\d+) ms/frame summed busy over (?<lists>\d+) lists in (?<subs>\d+) submissions \((?<cmds>\d+) commands/frame; wall (?<wall>\d+\.\d+) ms/frame, gap (?<gap>\d+\.\d+), util (?<util>\d+)%; noted (?<noted>\d+\.\d+), residual (?<resid>\d+\.\d+), dropped (?<dropped>\d+)\)')) {
    $r.Busy.Add([double]$m.Groups['busy'].Value)
    $r.Wall.Add([double]$m.Groups['wall'].Value)
    $r.Util.Add([double]$m.Groups['util'].Value)
    $r.Gap.Add([double]$m.Groups['gap'].Value)
    $r.Noted.Add([double]$m.Groups['noted'].Value)
    $r.Resid.Add([double]$m.Groups['resid'].Value)
    $r.Cmds.Add([double]$m.Groups['cmds'].Value)
    $r.Subs.Add([double]$m.Groups['subs'].Value)
  }
  foreach ($m in [regex]::Matches($text,
    '(?m)^.*?(?<ms>\d+\.\d+) ms/frame \((?<pct>\d+)% of busy, (?<each>\d+\.\d+) ms each, (?<us>\d+\.\d+) us/draw, (?<n>\d+\.\d+)/frame\): (?<kind>\S+) (?<label>.+?)\s*$')) {
    $key = $m.Groups['kind'].Value + ' ' + $m.Groups['label'].Value.Trim()
    if (-not $r.Rows.ContainsKey($key)) { $r.Rows[$key] = [System.Collections.Generic.List[object]]::new() }
    $r.Rows[$key].Add(@{ Ms = [double]$m.Groups['ms'].Value; N = [double]$m.Groups['n'].Value; Us = [double]$m.Groups['us'].Value })
  }
  foreach ($m in [regex]::Matches($text,
    'GPU starvation blocking: (?<tot>\d+\.\d+) ms/frame')) {
    $r.BlockMs.Add([double]$m.Groups['tot'].Value)
  }
  # copystat::log: the prepare-step and upload/prelude slices (copystat.rs).
  foreach ($m in [regex]::Matches($text, 'heaps (?<heap>\d+\.\d+) ms, title memory (?<mem>\d+\.\d+) ms')) {
    $r.HeapMs.Add([double]$m.Groups['heap'].Value)
    $r.MemMs.Add([double]$m.Groups['mem'].Value)
  }
  foreach ($m in [regex]::Matches($text, 'heap texture uploads (?<up>\d+\.\d+) ms, state preludes (?<pre>\d+\.\d+) ms')) {
    $r.UpMs.Add([double]$m.Groups['up'].Value)
    $r.PreMs.Add([double]$m.Groups['pre'].Value)
  }
  # Meter::frame texture companion (host.rs): live host-texture footprint.
  foreach ($m in [regex]::Matches($text, 'host textures: (?<n>\d+) \((?<mb>\d+) MB')) {
    $r.TexMB.Add([double]$m.Groups['mb'].Value)
  }
  return $r
}

function Merge-Side([string[]]$Files) {
  $agg = @{ N = 0; Scalars = @{}; Rows = @{} }
  foreach ($f in $Files) {
    if (-not (Test-Path $f)) { throw "log file not found: $f" }
    $rep = Get-Reports $f
    $agg.N++
    foreach ($k in @('Fps', 'Ms', 'Draws', 'CopyMs', 'CopyMB', 'BackMs', 'BackMB',
        'Busy', 'Wall', 'Util', 'Gap', 'Noted', 'Resid', 'Cmds', 'Subs',
        'BlockMs', 'HeapMs', 'MemMs', 'UpMs', 'PreMs', 'TexMB')) {
      if (-not $agg.Scalars.ContainsKey($k)) { $agg.Scalars[$k] = [System.Collections.Generic.List[double]]::new() }
      foreach ($v in $rep[$k]) { $agg.Scalars[$k].Add($v) }
    }
    foreach ($key in $rep.Rows.Keys) {
      if (-not $agg.Rows.ContainsKey($key)) { $agg.Rows[$key] = [System.Collections.Generic.List[object]]::new() }
      foreach ($s in $rep.Rows[$key]) { $agg.Rows[$key].Add($s) }
    }
  }
  $med = @{}
  foreach ($k in $agg.Scalars.Keys) { $med[$k] = Get-Median ([double[]]$agg.Scalars[$k].ToArray()) }
  $rows = @{}
  foreach ($key in $agg.Rows.Keys) {
    $ss = $agg.Rows[$key]
    $rows[$key] = @{
      Ms = Get-Median ([double[]]@($ss | ForEach-Object { $_.Ms }))
      N  = Get-Median ([double[]]@($ss | ForEach-Object { $_.N }))
      Us = Get-Median ([double[]]@($ss | ForEach-Object { $_.Us }))
    }
  }
  return @{ Files = $Files; Med = $med; Rows = $rows }
}

function Show-Side([string]$Name, $Side) {
  $m = $Side.Med
  $lines = @()
  $lines += "== $Name ($($Side.Files.Count) file(s)) =="
  $lines += ("  fps {0,6} | ms/frame {1,8} | draws/frame {2,6} | cmds/frame {3,6} | subs/frame {4,5}" -f
    ('{0:N1}' -f $m['Fps']), ('{0:N1}' -f $m['Ms']),
    ('{0:N0}' -f $m['Draws']), ('{0:N0}' -f $m['Cmds']), ('{0:N1}' -f $m['Subs']))
  $lines += ("  GPU busy {0,7} ms/f | wall {1,7} ms/f | gap {2,6} ms/f | util {3,4}% | noted {4,7} | resid {5,6} | block {6,6}" -f
    ('{0:N1}' -f $m['Busy']), ('{0:N1}' -f $m['Wall']), ('{0:N1}' -f $m['Gap']),
    ('{0:N0}' -f $m['Util']), ('{0:N1}' -f $m['Noted']), ('{0:N1}' -f $m['Resid']), ('{0:N2}' -f $m['BlockMs']))
  $lines += ("  CPU copies {0,6} ms {1,6} MB | copy-back {2,6} ms {3,6} MB | heap step {4,6} ms | mem step {5,6} ms | uploads {6,6} ms | preludes {7,6} ms | textures {8,6} MB" -f
    ('{0:N1}' -f $m['CopyMs']), ('{0:N1}' -f $m['CopyMB']),
    ('{0:N1}' -f $m['BackMs']), ('{0:N1}' -f $m['BackMB']),
    ('{0:N1}' -f $m['HeapMs']), ('{0:N1}' -f $m['MemMs']),
    ('{0:N1}' -f $m['UpMs']), ('{0:N1}' -f $m['PreMs']), ('{0:N0}' -f $m['TexMB']))
  return $lines
}

function Get-RelDiff([double]$bs, [double]$cs) {
  if ([double]::IsNaN($bs) -or [double]::IsNaN($cs)) { return [double]::NaN }
  if ([Math]::Abs($bs) -lt 1e-9) {
    if ([Math]::Abs($cs) -lt 1e-9) { return 0.0 }
    return [double]::NaN
  }
  return ($cs - $bs) / [Math]::Abs($bs)
}

$bside = Merge-Side $Baseline
$cside = Merge-Side $Candidate

$linesOut = @()
$linesOut += Show-Side 'BASELINE' $bside
$linesOut += Show-Side 'CANDIDATE' $cside
$linesOut += ''

# Scene-variance gate: footprints that must match before any timing claim.
$warned = $false
foreach ($k in @('Draws', 'Cmds', 'TexMB')) {
  $d = Get-RelDiff $bside.Med[$k] $cside.Med[$k]
  if (-not [double]::IsNaN($d) -and [Math]::Abs($d) -gt 0.10) {
    $linesOut += ('WARNING: scene variance: {0} base={1:N1} cand={2:N1} ({3:+0.0%;-0.0%}) - same-scene medians required before perf claims' -f
      $k, $bside.Med[$k], $cside.Med[$k], $d)
    $warned = $true
  }
}
if (-not $warned) { $linesOut += 'Scene footprints match within 10% (draws, commands, texture MB).' }
$linesOut += ''

$keys = @($bside.Rows.Keys + $cside.Rows.Keys | Sort-Object -Unique)
$table = foreach ($key in $keys) {
  $b = $bside.Rows[$key]; $c = $cside.Rows[$key]
  $bms = $(if ($b) { $b.Ms } else { 0.0 }); $cms = $(if ($c) { $c.Ms } else { 0.0 })
  [pscustomobject]@{
    Label    = $key
    Base_ms  = if ($b) { '{0:N2}' -f $b.Ms } else { '-' }
    Cand_ms  = if ($c) { '{0:N2}' -f $c.Ms } else { '-' }
    Delta_ms = '{0:+0.00;-0.00}' -f ($cms - $bms)
    Rel      = if ($b -and $c) { '{0:+0.0%;-0.0%}' -f (Get-RelDiff $b.Ms $c.Ms) }
               elseif ($c) { 'NEW' } else { 'GONE' }
    Base_n   = if ($b) { '{0:N1}' -f $b.N } else { '-' }
    Cand_n   = if ($c) { '{0:N1}' -f $c.N } else { '-' }
    Base_us  = if ($b) { '{0:N1}' -f $b.Us } else { '-' }
    Cand_us  = if ($c) { '{0:N1}' -f $c.Us } else { '-' }
    Sort     = [Math]::Max($bms, $cms)
  }
}
$shown = $table | Sort-Object -Property Sort -Descending | Select-Object -First $Top |
  Select-Object Label, Base_ms, Cand_ms, Delta_ms, Rel, Base_n, Cand_n, Base_us, Cand_us
$linesOut += 'Per-kind-then-pipeline GPU cost medians (ms/frame), top by max(base, cand):'
$linesOut += ($shown | Format-Table -AutoSize | Out-String)
$onlyB = @($bside.Rows.Keys | Where-Object { -not $cside.Rows.ContainsKey($_) })
$onlyC = @($cside.Rows.Keys | Where-Object { -not $bside.Rows.ContainsKey($_) })
if ($onlyB.Count -gt 0) { $linesOut += ('Rows only in baseline ({0}): {1}' -f $onlyB.Count, ($onlyB -join ' | ')) }
if ($onlyC.Count -gt 0) { $linesOut += ('Rows only in candidate ({0}): {1}' -f $onlyC.Count, ($onlyC -join ' | ')) }
$linesOut += ''
$linesOut += ('Header deltas (cand - base): fps {0:+0.0;-0.0}, ms/frame {1:+0.0;-0.0}, GPU busy {2:+0.0;-0.0} ms/f, wall {3:+0.0;-0.0} ms/f, util {4:+0;-0}%, draws/frame {5:+0;-0}, texture MB {6:+0;-0}' -f
  ($cside.Med['Fps'] - $bside.Med['Fps']), ($cside.Med['Ms'] - $bside.Med['Ms']),
  ($cside.Med['Busy'] - $bside.Med['Busy']), ($cside.Med['Wall'] - $bside.Med['Wall']),
  ($cside.Med['Util'] - $bside.Med['Util']), ($cside.Med['Draws'] - $bside.Med['Draws']),
  ($cside.Med['TexMB'] - $bside.Med['TexMB']))

$text = $linesOut -join "`r`n"
Write-Output $text
if ($Out -ne '') {
  [System.IO.File]::WriteAllText($Out, $text)
  Write-Output "wrote $Out"
}
