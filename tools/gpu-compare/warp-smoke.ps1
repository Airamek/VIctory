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

  FAIL/WARN otherwise. Grading is by the Verdict field in result.json -
  NOT by this script's process exit code (always 0 when the harness itself
  ran): callers must read result.json and require Verdict PASS.
  Throughput numbers from WARP are NEVER compared against
  hardware or across builds for speed claims: WARP's per-draw cost differs by
  orders of magnitude. The harness only validates that the diagnostics pipeline
  works and captures scene footprints (draws/frame, commands/frame, texture MB)
  that the compare script can use.
  Child env is isolated (knobs set only on the child process) and all
  waits are bounded via the Wait-ProcExit poll wrapper - never infinite.

.EXAMPLE
  ./warp-smoke.ps1 -VictoryExe D:/victory-target/diag/victory.exe `
    -TitleExe 'G:/Title.exe' -Seconds 600
#>
param(
  [Parameter(Mandatory = $true)][string]$VictoryExe,
  [Parameter(Mandatory = $true)][string]$TitleExe,
  [ValidateRange(30, 1800)][int]$Seconds = 600,
  [string]$LogDir = "D:/victory-target/warp-smoke-logs",
  [string]$ExtraEnv = ""
)

$ErrorActionPreference = 'Stop'

# Bounded wait wrapper: polls HasExited in short slices up to TimeoutMs.
# Never blocks forever - the parameterless WaitForExit() overload is banned here.
function Wait-ProcExit($proc, [int]$TimeoutMs) {
  $elapsed = 0
  $slice = 250
  while ($elapsed -lt $TimeoutMs) {
    try { if ($proc.HasExited) { return $true } }
    catch { return $true }
    Start-Sleep -Milliseconds $slice
    $elapsed += $slice
  }
  try { return $proc.HasExited } catch { return $true }
}

if (-not (Test-Path $VictoryExe)) { throw "victory binary not found: $VictoryExe" }
if (-not (Test-Path $TitleExe)) { throw "title exe not found: $TitleExe" }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$dir = Join-Path $LogDir $stamp
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$log = Join-Path $dir 'console.log'

# Child-process env isolation: VICTORY_* knobs live ONLY on the child's
# ProcessStartInfo environment. This process's $env: is never touched.
# Launch goes through cmd.exe /c with OS-level file redirects, so no
# PowerShell pipe readers can deadlock on a chatty child during long runs.
$stdoutLog = Join-Path $dir 'stdout.log'
$stderrLog = Join-Path $dir 'stderr.log'
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = 'cmd.exe'
$psi.Arguments = '/c ""' + $VictoryExe + '" run --execute "' + $TitleExe + '" > "' + $stdoutLog + '" 2> "' + $stderrLog + '"' + '"'
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false
$psi.EnvironmentVariables['VICTORY_D3D12_WARP'] = '1'
$psi.EnvironmentVariables['VICTORY_GPU_TIMING'] = '1'
foreach ($pair in ($ExtraEnv -split ';' | Where-Object { $_ -match '=' })) {
  $k, $v = $pair -split '=', 2
  $psi.EnvironmentVariables[$k.Trim()] = $v
  "extra env (child only): $($k.Trim())=$v" | Out-File -Append $log -Encoding utf8
}

"victory: $VictoryExe" | Out-File $log -Encoding utf8
"title:   $TitleExe" | Out-File -Append $log -Encoding utf8
"started:  $(Get-Date -Format o)" | Out-File -Append $log -Encoding utf8

$proc = [System.Diagnostics.Process]::Start($psi)

$exited = Wait-ProcExit $proc ($Seconds * 1000)
if (-not $exited) {
  # Kill the whole tree (victory may spawn helpers); /T = tree, /F = force.
  & taskkill /PID $proc.Id /T /F 2>&1 | Out-File -Append $log -Encoding utf8
  # Generous grace: WARP shutdown + child teardown can take a while.
  [void](Wait-ProcExit $proc 30000)
  "bounded run: tree-killed after ${Seconds}s" | Out-File -Append $log -Encoding utf8
}
else {
  # Generous settle so trailing diagnostics flush into the redirected logs.
  [void](Wait-ProcExit $proc 10000)
  try { "bounded run: exited on its own (code $($proc.ExitCode))" | Out-File -Append $log -Encoding utf8 }
  catch { "bounded run: exited on its own" | Out-File -Append $log -Encoding utf8 }
}
# OS-level redirects already flushed to disk by the exited/killed tree.
try { $proc.Refresh() } catch {}

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
Write-Output "grade on result.json Verdict (PASS required), not on exit code."
