# PresentMon present-to-present ground truth

Methodology context: `docs/gpu-profiling-methodology.md` §4 item 2.
Until this exists, the wall reference is `Meter::frame`'s fps line plus the
window caption (methodology §1.7); PresentMon measures what `Meter` cannot:
true present-to-present pacing independent of our instrumentation waits.
It gates hardware fps claims (methodology §3), not GPU-busy claims.

This directory adds tooling only. No game/GPU code is touched.

## 1. Install PresentMon

Upstream: https://github.com/GameTechDev/PresentMon.

1. Open that repo's Releases page and download a Windows x64 build
   (`PresentMon-<version>-x64.exe`, naming varies by release).
2. No install needed: it is a single exe. Place it somewhere stable, e.g.
   `D:\tools\PresentMon\PresentMon.exe`, and record the exact version +
   exe hash next to every capture (CSV schema varies by release).
3. Confirm it runs: `PresentMon.exe --help` (flag names differ between the
   classic console build and newer builds; the examples below use the
   long-standing `-capture` / `-process_name` / `-output_file` style - if
   your build rejects a flag, its `--help` is authoritative).

## 2. Capture flags for victory runs

The presenting process is `victory.exe` (the title runs under
`victory run --execute <title>`; PresentMon must target the process that
presents, not the title exe).

Steady-state capture (timed diagnostics run):

```powershell
PresentMon.exe -capture -process_name victory.exe -output_file run.csv
victory run --execute G:/Title.exe   # with VICTORY_GPU_TIMING=1 in the env
# stop PresentMon after victory exits (Ctrl+C / stop the capture)
```

Notes:

- Start PresentMon **before** launching victory and stop it **after** victory
  exits, so the CSV brackets the whole run; the compare script trims to the
  steady window (section 3).
- Write down the wall-clock start/end of the victory run and the PresentMon
  version/exe hash with the CSV. The victory log has no per-line timestamps,
  so these notes are the alignment record.
- Keep one CSV per run; do not concatenate runs into one CSV.
- For a hardware fps claim, capture at least one **untimed** run
  (`VICTORY_GPU_TIMING` unset): timed mode adds waits + stamp-buffer
  readbacks that distort the wall by ~12-40 ms (methodology §1.2), and that
  distortion also sits in the wall PresentMon sees.
- WARP runs (`VICTORY_D3D12_WARP=1`) never need PresentMon: WARP numbers are
  not comparable to hardware or across builds (methodology §1.9).

## 3. Time-align a PresentMon CSV with a victory diag log

1. Identify the steady window: skip title load and the first seconds of the
   scene (transient uploads, pipeline creation), and the teardown tail.
2. Trim the PresentMon side with the compare script:
   `-SkipFirst N -SkipLast M` drops load-transient presents by count, or
   `-StartTime T0 -EndTime T1` keeps a `TimeInSeconds` window. Either is
   fine; state which you used alongside the result.
3. The victory side contributes all `Meter::frame` fps lines and all
   `GPU cost by pipeline:` reports in the log (medians/p50-p99). If the log
   spans scenes, split it first - mixed scenes are apples-to-oranges
   (methodology §3 rule 1).
4. Optionally filter the CSV to the game process with `-App victory` when
   the capture caught other processes' presents.

Run:

```powershell
./compare-presentmon.ps1 -PresentMonCsv run.csv -DiagLog victory.log `
  -SkipFirst 30 -SkipLast 10 -Out recon.txt
```

## 4. Reading the output

- **app fps vs present fps**: `Meter::frame` fps median vs PresentMon
  `(N-1)/span` over the aligned window.
- **frame-time p50/p95/p99, both sides**: victory `ms/frame` samples vs
  `MsBetweenPresents` samples. p95/p99 catch hitches either side hides.
- **wall-busy-gap reconciliation**: victory `wall` median vs PresentMon p50,
  printed with summed-busy / gap / noted / residual context. The gap plus
  residual is where unstamped work lives (preludes + present-copy list,
  methodology §1.3); a present-vs-wall delta near `gap + residual` points at
  unstamped work, not at a measurement bug.
- **mismatch flag** (documented timed-mode distortion band, 12-40 ms):
  `|present p50 - victory median| <= 12 ms` is agreement; 12-40 ms fits
  inside timed-mode waits/readback (need an untimed run before any fps
  claim); `> 40 ms` is a MISMATCH - do not claim hardware fps from that pair.

## 5. Self-test (no live game needed)

```powershell
./compare-presentmon.ps1 -SelfTest
```

Builds synthetic CSV + diag-log inputs (steady 33.33 ms pacing with one
50 ms hitch; wall just above the present p50), checks present fps /
p50/p95/p99, the app-fps and wall medians, the <= 12 ms reconciliation,
and that a synthetic 100 ms-paced CSV trips the > 40 ms MISMATCH flag.

## 6. Post-resync follow-up (NOT implemented here - needs in-tree work)

Ground truth strictly needs no in-tree hook to be useful: CSV-vs-log
reconciliation works today. A per-present join (matching one CSV row to one
victory frame) would need this, specified exactly so it can land after the
resync without redesign:

- Location: `present()` in `crates/victory-gpu/src/host.rs` (the unstamped
  present-copy path described in methodology §1.3), gated the same way the
  other diagnostics lines are (diagnostics cfg / `VICTORY_GPU_TIMING=1`).
- Behaviour: emit one `tracing::info!` line per presented frame carrying a
  monotonically increasing present sequence id and the current frame number,
  e.g. `victory present seq=<n> frame=<m>`.
- The compare script would then gain a `-SeqPattern` join: match CSV row
  index (within the aligned window) to `seq`, and report per-frame
  `MsBetweenPresents - ms/frame` residuals instead of window medians only.
- Do not implement this on this branch: the resync owns `host.rs`.
