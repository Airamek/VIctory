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
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
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
$console | Out-File -Append $log -Encoding utf8

$warp = $console -match 'host device on the software adapter \(WARP\)'
$fps = [regex]::Matches($console, '\d+\.\d+ fps, \d+\.\d+ ms/frame').Count
$cost = [regex]::Matches($console, 'GPU cost by pipeline:').Count
$removed = $console -match 'device removed|device-removed|host GPU failure'

$result = @{
  Time       = $stamp
  WarpDevice = [bool]$warp
  FpsLines   = $fps
  CostLines  = $cost
  Removed    = [bool]$removed
  Verdict    = 'FAIL'
}
if ($warp -and $fps -gt 0 -and $cost -gt 0 -and -not $removed) { $result.Verdict = 'PASS' }
elseif ($warp) { $result.Verdict = 'WARN' }

$result | ConvertTo-Json | Out-File (Join-Path $dir 'result.json') -Encoding utf8
$result | Format-List | Out-String | Write-Output
Write-Output "logs: $dir"
if ($result.Verdict -eq 'FAIL') { exit 1 }
