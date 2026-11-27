#Requires -Version 5.1
<#
.SYNOPSIS
  Runs the amdxc64 diag31 repro: one OS process per VS+PS candidate.

.DESCRIPTION
  Each candidate gets its own process running
  `cargo run --example amdxc64_repro`, because a driver crash
  (0xC0000005 in amdxc64.dll, as in diag31) takes down the whole process.
  Every candidate has a 600 s watchdog; on timeout the process tree is
  killed and the candidate is recorded as TIMED-OUT.

  Reads:  C:\Games\GTA6\shaders.rpf (read-only, never written).
  Writes: D:\victory-target\amdxc64-repro\runs\<tag>\{stdout.txt,result.txt}
          only. Nothing under C:\Games, nothing in the repo.
#>
param(
    [string]$Worktree = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$RunsDir = 'D:\victory-target\amdxc64-repro\runs',
    [int]$TimeoutSec = 600
)

$ErrorActionPreference = 'Stop'

# Hardware device only: Host::new(false) honours VICTORY_WARP=1, and WARP
# cannot reproduce an amdxc64 crash.
Remove-Item Env:\VICTORY_WARP -ErrorAction SilentlyContinue
Remove-Item Env:\VICTORY_EMIT_DUMP -ErrorAction SilentlyContinue

$candidates = @(
    @{ tag = 'vs9975+ps7637-prime'; vs = '0xeff29';   vshash = '9975d5335335cc9c'; ps = '0x4f96cf6'; pshash = '76376a2ee6c53ffe' },
    @{ tag = 'vs9975+psbf72-near';  vs = '0xeff29';   vshash = '9975d5335335cc9c'; ps = '0x4f8d8a6'; pshash = 'bf7278cd8d6d6ff3' },
    @{ tag = 'vs9975+ps9763-near';  vs = '0xeff29';   vshash = '9975d5335335cc9c'; ps = '0x4f85da2'; pshash = '9763863ad4cd2579' }
)

$game = 'C:\Games\GTA6\shaders.rpf'
if (-not (Test-Path $game)) { throw "game file missing: $game" }
New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null

Write-Output "worktree=$Worktree"
Write-Output "runs=$RunsDir timeout=${TimeoutSec}s candidates=$($candidates.Count)"

Push-Location $Worktree
try {
    Write-Output "--- building repro (release) ---"
    # Native stderr under `$ErrorActionPreference = 'Stop'` raises, so
    # build with 'Continue' and judge by $LASTEXITCODE instead.
    $prevEA = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        cargo build --release --example amdxc64_repro -p victory 2>&1 | ForEach-Object { Write-Output "$_" }
    } finally {
        $ErrorActionPreference = $prevEA
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed: $LASTEXITCODE" }
    $exe = Join-Path $Worktree 'target\release\examples\amdxc64_repro.exe'
    if (-not (Test-Path $exe)) {
        # cargo may place examples at target\release\examples\<name>.exe; fall back to a search
        $exe = (Get-ChildItem (Join-Path $Worktree 'target\release') -Recurse -Filter 'amdxc64_repro.exe' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
    }
    if (-not $exe -or -not (Test-Path $exe)) { throw 'repro binary not found after build' }
    Write-Output "exe=$exe"
} finally {
    Pop-Location
}

$summary = @()
foreach ($c in $candidates) {
    $dir = Join-Path $RunsDir $c.tag
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $out = Join-Path $dir 'stdout.txt'
    $resultPath = Join-Path $dir 'result.txt'
    if (Test-Path $out) { Remove-Item $out }
    Write-Output ""
    Write-Output "=== candidate $($c.tag) (timeout ${TimeoutSec}s, own process) ==="

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = "--game `"$game`" --vs-offset $($c.vs) --vs-hash $($c.vshash) --ps-offset $($c.ps) --ps-hash $($c.pshash) --tag $($c.tag)"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.WorkingDirectory = $Worktree
    $proc = [System.Diagnostics.Process]::Start($psi)
    $exited = $proc.WaitForExit($TimeoutSec * 1000)
    if (-not $exited) {
        try { $proc.Kill($true) } catch { try { $proc.Kill() } catch {} }
        $proc.WaitForExit(10000) | Out-Null
        $verdict = 'TIMED-OUT'
        "[$([DateTime]::UtcNow.ToString('o'))] TIMED-OUT after ${TimeoutSec}s" | Out-File -Append -FilePath $out -Encoding utf8
    } else {
        $code = $proc.ExitCode
        $verdict = switch ($code) {
            0 { 'PASS' }
            2 { 'REFUSED' }
            3 { 'SETUP-FAILED' }
            -1073741819 { 'CRASH-0xC0000005 (reproduces diag31 driver crash)' }
            default { "EXIT-$code" }
        }
        "[$([DateTime]::UtcNow.ToString('o'))] exit=$code verdict=$verdict" | Out-File -Append -FilePath $out -Encoding utf8
    }
    # Capture whatever the process managed to flush before exit/crash.
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    if ($stdout) { $stdout | Out-File -Append -FilePath $out -Encoding utf8 }
    if ($stderr) { "--- stderr ---`n$stderr" | Out-File -Append -FilePath $out -Encoding utf8 }
    $summaryLine = "$($c.tag): $verdict"
    Write-Output $summaryLine
    Get-Content $out | ForEach-Object { Write-Output "  $_" }
    "$summaryLine" | Out-File -FilePath $resultPath -Encoding utf8
    $summary += $summaryLine
}

Write-Output ""
Write-Output "=== summary ==="
$summary | ForEach-Object { Write-Output $_ }
