<#
.SYNOPSIS
  Bounded unattended WARP smoke run: proves the title path executes headless-ish
  on the software adapter and emits the diagnostics a perf claim needs.

.DESCRIPTION
  Agents have no GPU, but WARP is a software D3D12 adapter, so a diagnostics
  build can still execute the full title path on it (slowly; cannot reset the
  driver - README.md). There is no fixed-frame or headless flag in the title
  path (apps/victory/src/run.rs; no VICTORY_* frame-limit knob exists), so this
  harness runs the title bounded by wall-clock time, kills the process tree,
  and grades the console log:

  PASS requires all of:
  - 'host device on the software adapter (WARP)' (crates/victory-gpu/src/host.rs)
  - at least one 'fps, ... ms/frame (...) draws/frame' line (Meter::frame)
  - at least one 'GPU cost by pipeline:' report (costs::log)

  FAIL/WARN otherwise. Throughput numbers from WARP are NEVER compared against
  hardware or across builds for speed claims: WARP's per-draw cost differs by
  orders of magnitude. The harness only validates that the diagnostics pipeline
  works and captures scene footprints (draws/frame, commands/frame, texture MB)
  that the compare script can use.

  Bounds-era footprint (shadow/bindless host-copy bound, placed shell bound,
  driver-compile permit) is captured beside the PASS verdict: every
  'host copies of title memory: ...' report is parsed into held-MB
  first/last/max, evictions/frame and transient-served; compile-permit waits
  ('pipeline waited for a driver-compile slot', waited_ms) are counted. If
  PASS has no such report the script sets BoundsVerdict WARN - the build under
  test predates the bounds or lost their diagnostics. This never changes PASS:
  bounds signal only, never a throughput comparison.

.EXAMPLE
  ./warp-smoke.ps1 -VictoryExe D:/dev/VIctory-perf/target/diag/victory.exe `
    -TitleExe 'G:/Title.exe' -Seconds 180
#>
param(
  [Parameter(Mandatory = $true)][string]$VictoryExe,
  [Parameter(Mandatory = $true)][string]$TitleExe,
  [int]$Seconds = 180,
  [string]$LogDir = "warp-smoke-logs",
  [string]$ExtraEnv = ""
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $VictoryExe)) { throw "victory binary not found: $VictoryExe" }
if (-not (Test-Path $TitleExe)) { throw "title exe not found: $TitleExe" }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$dir = Join-Path $LogDir $stamp
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$log = Join-Path $dir 'console.log'

$env:VICTORY_D3D12_WARP = '1'
$env:VICTORY_GPU_TIMING = '1'
foreach ($pair in ($ExtraEnv -split ';' | Where-Object { $_ -match '=' })) {
  $k, $v = $pair -split '=', 2
  [System.Environment]::SetEnvironmentVariable($k.Trim(), $v)
  "extra env: $($k.Trim())=$v" | Out-File -Append $log -Encoding utf8
}

"victory: $VictoryExe" | Out-File $log -Encoding utf8
"title:   $TitleExe" | Out-File -Append $log -Encoding utf8
"started:  $(Get-Date -Format o)" | Out-File -Append $log -Encoding utf8

$proc = Start-Process -FilePath $VictoryExe -ArgumentList @('run', '--execute', $TitleExe) `
  -RedirectStandardOutput (Join-Path $dir 'stdout.log') `
  -RedirectStandardError (Join-Path $dir 'stderr.log') `
  -PassThru

$exited = $proc.WaitForExit($Seconds * 1000)
if (-not $exited) {
  # The whole tree: the title path may leave helpers behind, and a survivor
  # would keep writing to the redirected handles the parent's wait is on.
  & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null
  $proc.WaitForExit(15000) | Out-Null
  "bounded run: killed after ${Seconds}s" | Out-File -Append $log -Encoding utf8
}
else {
  "bounded run: exited on its own after $($proc.ExitCode)" | Out-File -Append $log -Encoding utf8
}

$console = ''
foreach ($f in @((Join-Path $dir 'stdout.log'), (Join-Path $dir 'stderr.log'))) {
  if (Test-Path $f) { $console += [System.IO.File]::ReadAllText($f) + "`r`n" }
}
# Strip SGR: tracing's fmt colours field names, so the raw text is
# 'waited_ms' ESC '=' ESC '29' - a field-value regex never sees 'waited_ms='.
# The stored console.log is plain too, so a later replay grades identically.
$console = [regex]::Replace($console, '\x1b\[[0-9;]*m', '')
$console | Out-File -Append $log -Encoding utf8

$warp = $console -match 'host device on the software adapter \(WARP\)'
$fps = [regex]::Matches($console, '\d+\.\d+ fps, \d+\.\d+ ms/frame').Count
$cost = [regex]::Matches($console, 'GPU cost by pipeline:').Count
$removed = $console -match 'device removed|device-removed|host GPU failure'

# --- bounds-era footprint (signal only) --------------------------------------
# shadow::log_shadow, per 5 s report: made/freed/evicted/transient per frame and
# what the host-copy bound still holds. First/last/max held-MB shows whether the
# bound held over the run; it is never a throughput number.
$shadow = [regex]::Matches($console,
  'host copies of title memory: ([\d.]+) made/frame, ([\d.]+) freed/frame, ' +
  '([\d.]+) evicted/frame \((\d+) transient served\); ' +
  '([\d.]+) MB held \(([\d.]+) MB arena in (\d+) chunks, ([\d.]+) MB free\)')
# The log formats numbers with '.'; parse them culture-invariantly (the host
# may be on a comma-decimal locale).
function ConvertTo-InvariantDouble([string]$Text) {
  return [double]::Parse($Text, [System.Globalization.CultureInfo]::InvariantCulture)
}
$held = @($shadow | ForEach-Object { ConvertTo-InvariantDouble $_.Groups[5].Value })
$evicted = @($shadow | ForEach-Object { ConvertTo-InvariantDouble $_.Groups[3].Value })
$transient = @($shadow | ForEach-Object { [int]$_.Groups[4].Value })
# Field order in the rendered line is the subscriber's choice: take waited_ms
# from any line carrying the permit wait, in either order. Match the line body
# without ^/$: with CRLF endings .NET's multiline $ sits after the \r, so an
# anchored [^\r\n]*$ never matches.
$waits = @([regex]::Matches($console, '[^\r\n]*pipeline waited for a driver-compile slot[^\r\n]*') |
  ForEach-Object {
    $m = [regex]::Match($_.Value, 'waited_ms=(\d+)')
    if ($m.Success) { [int]$m.Groups[1].Value }
  })

# Allocation pressure the bounds are supposed to answer: shared-buffer
# allocations the driver refused (E_OUTOFMEMORY, 0x8007000E). Footprint
# signal only - WARP's budget is host RAM, so this is where a late bound
# shows up as failure-to-allocate rather than slow frames.
$allocWarn = [regex]::Matches($console,
  'no shared buffer[^\r\n]*0x8007000E').Count

# Scene footprint from the last report of each kind (compare-script inputs).
$draws = @([regex]::Matches($console, '(\d+) draws/frame,'))
$commands = @([regex]::Matches($console, '(\d+) commands/frame'))
$textureMb = @([regex]::Matches($console, 'host textures: \d+ \(([\d.]+) MB,'))
$leftNull = @([regex]::Matches($console, 'left null: ([^\r\n]*?) \(placed textures'))

$bounds = [ordered]@{
  ShadowReports    = $shadow.Count
  HeldMbFirst      = if ($held.Count) { $held[0] } else { $null }
  HeldMbLast       = if ($held.Count) { $held[-1] } else { $null }
  HeldMbMax        = if ($held.Count) { ($held | Measure-Object -Maximum).Maximum } else { $null }
  EvictedPerFMax   = if ($evicted.Count) { ($evicted | Measure-Object -Maximum).Maximum } else { $null }
  TransientsMax    = if ($transient.Count) { ($transient | Measure-Object -Maximum).Maximum } else { $null }
  CompileWaits     = $waits.Count
  CompileWaitMsSum = if ($waits.Count) { ($waits | Measure-Object -Sum).Sum } else { 0 }
  CompileWaitMsMax = if ($waits.Count) { ($waits | Measure-Object -Maximum).Maximum } else { 0 }
  AllocWarnLines   = $allocWarn
  HeapViewReports  = [regex]::Matches($console, 'heap views:').Count
  LastDrawsPerFrame    = if ($draws.Count) { [int]$draws[-1].Groups[1].Value } else { $null }
  LastCommandsPerFrame = if ($commands.Count) { [int]$commands[-1].Groups[1].Value } else { $null }
  LastTextureMb        = if ($textureMb.Count) { ConvertTo-InvariantDouble $textureMb[-1].Groups[1].Value } else { $null }
  LastLeftNull         = if ($leftNull.Count) { $leftNull[-1].Groups[1].Value.Trim() } else { $null }
}

$result = @{
  Time       = $stamp
  WarpDevice = [bool]$warp
  FpsLines   = $fps
  CostLines  = $cost
  Removed    = [bool]$removed
  Verdict    = 'FAIL'
  Bounds     = $bounds
}
if ($warp -and $fps -gt 0 -and $cost -gt 0 -and -not $removed) { $result.Verdict = 'PASS' }
elseif ($warp) { $result.Verdict = 'WARN' }

# A passing run with no shadow report is not bounds-era (or lost the log);
# report that, but keep PASS exactly as documented above.
if ($shadow.Count -gt 0) { $boundsVerdict = 'OK' }
elseif ($result.Verdict -eq 'PASS') { $boundsVerdict = 'WARN' }
else { $boundsVerdict = 'n/a' }
$result.BoundsVerdict = $boundsVerdict

$result | ConvertTo-Json | Out-File (Join-Path $dir 'result.json') -Encoding utf8
$result | Format-List | Out-String | Write-Output
Write-Output "logs: $dir"
if ($result.Verdict -eq 'FAIL') { exit 1 }
