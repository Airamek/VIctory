# GPU perf methodology without a GPU (unattended agents)

Branch: `eval/gpu-perf-methodology`. No accuracy changes, no env-knob hacks:
this branch adds only this doc and two PowerShell tools under
`tools/gpu-compare/`. All timings below are host-GPU (RX 6600 XT) numbers taken
from session logs and `D:\dev\ROADMAP_30FPS.md`; agents have no GPU and must
not invent replacement numbers.

The problem being solved: intro-cutscene-only testing compares wall-clock
frame averages across builds that reach *different* cutscene frames, so the
average is apples-to-oranges; per-unit `VICTORY_GPU_TIMING` costs vary ~3x
run-to-run for the same dispatch hash (27.4/17.7/8.2 ms for `dispatch
3600x1x1 [39f0e9...]`); and some GPU work is unstamped while the CPU wall
includes it. This doc surveys what exists, defines what a perf claim needs,
and ranks the remaining instrumentation work.

## 1. Survey: what already exists

### 1.1 `VICTORY_GPU_TIMING` grouping and stamps

- Timestamp heap: `STAMPS = 16384` slots per list
  (`crates/victory-gpu/src/host.rs:1043`), one `EndQuery` per noted command
  plus the list busy pair (`BUSY_START`/`BUSY_END`, host.rs:1047-1048).
  Commands past `MAX_TIMED_COUNT` are not timed and surface as a `dropped`
  count (host.rs:1053, `timed()` host.rs:2604-2667, `close()` host.rs:2722-2747).
- Grouping: default timing keeps the normal cuts grouping - the lists run
  exactly as with timing off (host.rs:1087-1089, 1868-1885). Per-list alone
  execution (each list run and fence-waited on its own) is opt-in via
  `VICTORY_GPU_TIMING_ALONE=1` (host.rs:1096-1104, groups built at
  host.rs:1877-1885).
- Cost table: `cost_label` drops addresses so the same pass adds up,
  `cost_kind` normalizes to `draw|indexed|dispatch|mesh|copy|resolve|texture|
  DROPPED|other`, `parse_draws` counts instances/groups for us-per-draw
  (host.rs:4945-5014). `costs::add` tallies per list, `add_submission` tallies
  summed-busy vs CPU wall per submission piece, `log` prints the top 16 rows
  with `% of busy`, `ms each`, `us/draw`, `/frame`, plus `wall/gap/util`,
  `noted/residual/dropped` (host.rs:5038-5151). Slow lists (> 50 ms busy) are
  logged with their 6 slowest commands (host.rs:4927-4943).
- The 3x run-to-run variance on one dispatch hash is expected behavior, not a
  tool bug: each stamp delta is that run's real GPU time (clocks, contention
  with the async timeline, memory pressure). Handle it with medians over
  repeat runs and `us/draw` + `% of busy`, never with a single sample.

### 1.2 Timed-mode distortion (the +/-12-40 ms)

Timed mode adds work the untimed path does not do: a `wait_gpu` + `settle`
(serialize the async timeline) per submission group, per-group readback
`Map`/`Unmap` of the stamp buffer, and string formatting of every note
(host.rs:1927-1940, 1984, 2604-2667). Default (non-alone) timing preserves
batching, so the distortion is wait/readback overhead, not regrouping;
`VICTORY_GPU_TIMING_ALONE=1` additionally regroups per list and serializes
everything. Consequence for the gate (section 3): the primary metric is
summed busy + per-kind medians (GPU-side, batching preserved); wall/fps is
sanity only, and no wall-clock-only win smaller than the distortion band is
claimable.

### 1.3 Unstamped GPU work (no copy queue exists)

Correction to the session premise: there is **no copy queue** in the
codebase. The host device creates exactly one `DIRECT` queue
(host.rs:330-331); every `ExecuteCommandLists` in the tree targets it.
The GPU work that lands in the CPU wall but not in busy is:

- Prelude lists: transitions only, deliberately not bracketed; their GPU
  time stays out of busy and inside the wall-minus-busy gap (host.rs:2731-2736
  comment in `close()`, gap definition host.rs:5081-5085).
- The present-copy list: `present()` records barriers + `CopyResource` into
  `p.copy.list` and runs it with `self.execute(&[&p.copy])` on the same
  direct queue with no timestamps (host.rs:4684-4715).
- Barriers inside otherwise-noted lists (covered by the residual: busy minus
  noted, clamped at zero, host.rs:2664).

So "busy excludes copy, wall includes it" is precisely "busy excludes
preludes + present-copy, wall includes their CPU-side submission + waiting".
The log already separates this: `noted` vs `residual` vs `gap`
(host.rs:5126-5138). Any change touching uploads, preludes, or present must
show noted + residual + gap all flat-or-better (section 3).

### 1.4 Starvation counters and timeline depth

`starve` module (host.rs:101-145): depth histogram over 17 buckets
(`DEPTHS = 17`, host.rs:109), per-cause CPU blocking ms over 5 causes -
in-flight-full, list-still-running, range conflict, staging/table conflict,
drain/present wait (host.rs:113-117). Depth is noted per timed submission at
the submission's timeline length (host.rs:1978); blocking is measured around
waits that already happen, never added (host.rs:104-105). `log_starvation`
drains both every report even when no lists were timed (host.rs:5156-5182).

### 1.5 `copystat` steps and companions

Counters enum `C` (crates/victory-gpu/src/copystat.rs:11-107) includes
`StepHeapUs`/`StepMemoryUs`/`FlushUs`, `UploadsUs`, `PreludeUs`,
`SyncHeapsUs`/`SyncSlots`, `MirrorWaitUs`. `copystat::log` prints the whole
line per frame over the report window (copystat.rs:214-307), plus top-6
snapshot copies by bytes and top-10 source ranges with per-range
x/frame, MB/frame, ms/frame (copystat.rs:313-333). Companion lines in the
same report: `host textures` footprint and make/free flow (host.rs:4517-4535),
`video memory` budgets (host.rs:4537-4551), `host texture MB` census
(host.rs:4552-4557), frame log (frame.rs via host.rs:4559), the older
`gpu_times` per-kind summary (host.rs:466-476, only meaningful when a
submission grouped several lists), bindless/shadow reports, texture
read-backs (host.rs:4565-4572), and pipeline-drop reasons (host.rs:4573-4578).

### 1.6 Pipeline / capture / trace / skip knobs

- `VICTORY_PIPELINE_LOG=<file>`: one line per title pipeline created, first
  used, made or loaded (crates/victory-gpu/src/pipeline/make.rs:217-228).
- `VICTORY_CAPTURE_FRAMES=<list>` also saves those frames (and with
  `VICTORY_DUMP_TARGETS=1` dumps every render target they touch, in
  first-use order); `VICTORY_CAPTURE_WORLD=<list>` counts from the first
  frame that ran 500 draws, the world's first, whose number depends on load
  speed (crates/victory-gpu/src/capture.rs:19-61, 227-279, `dump_submission`
  capture.rs:361-367). Frames are saved at half size (capture.rs:67-...).
- `VICTORY_TRACE_SHADERS=<hash>,...`: every draw/dispatch of a listed
  shader logged with bindings (first 500, all in listed capture frames);
  indirect-execution records read back every frame
  (host.rs:5223-5252).
- `VICTORY_SKIP_PIPELINES=<hash>,...`: omit draws/dispatches (incl.
  indirect) whose pipeline contains a listed shader hash
  (crates/victory-gpu/src/pipeline.rs:280-292).
- Full env table: README.md:200-221.

### 1.7 PresentMon ground truth

Planned, not built: roadmap section 6 (`D:\dev\ROADMAP_30FPS.md:59-66`)
lists PresentMon ground truth as a profiling step. No PresentMon reference
exists anywhere under `crates/` or `apps/` (grep: zero hits). Until it is
integrated, the wall reference is `Meter::frame`'s fps line
(host.rs:4468-4516) plus the window-caption text (host.rs:4590-4615).
PresentMon would measure what `Meter` cannot: true present-to-present
pacing independent of our instrumentation waits.

### 1.8 Microbenchmarks (GPU-less runnable)

- Snapshot fill patterns: `snapshot_fill_throughput`
  (crates/victory-gpu/src/shadow/snapshot_tests.rs:76-135, `#[ignore =
  "benchmark"]`): serial extend vs reserve + parallel raw fill (release:
  ~12.8 GB/s parallel vs ~3.0 serial on the roadmap machine). Runs without a
  GPU (pure memcpy).
- `bench_write` (crates/victory-shader/src/emit/bitcode.rs:1637-1638),
  `bench_bools` (crates/victory-shader/src/structure.rs:1817-1818),
  `benchmark_private_data_release` (crates/victory-gpu/src/object.rs:463-464):
  `#[ignore]` release benchmarks via `cargo test --release ... -- --ignored
  --nocapture`.
- `texture_layout_bench` example (crates/victory-formats/examples/),
  XMA inverse-transform bench (crates/victory-formats/src/xma.rs:1371-1375).

### 1.9 WARP support and its limits

`VICTORY_D3D12_WARP=1` creates the host device on the software adapter in
diagnostics builds (host.rs:317-321), refusing a non-software adapter
(host.rs:410-427); compare procedure in README.md:237-241. Limits: slow;
cannot reset the driver (README.md:216); wild shader accesses AV-crash the
process instead of faulting the GPU; no DRED breadcrumbs (only last-notes,
host.rs:5270-5283); multiply-add fusing differs from hardware
(RX 6700 XT fuses, WARP does not - host.rs:361-370), so pixel-level output
can differ legitimately. WARP numbers are never comparable to hardware
numbers or across builds for speed claims; WARP is for executing the title
path and validating diagnostics (section 2).

## 2. Tooling added here (`tools/gpu-compare/`)

### 2.1 `compare-gpu-logs.ps1` - hash-keyed log compare

Parses one or more baseline logs and one or more candidate logs (repeat runs
of the same build/scene), medians every metric across all report periods,
and prints:

- side summaries: fps, ms/frame, draws/commands/submissions per frame,
  GPU busy/wall/gap/util, noted/residual/block ms, CPU copy + copy-back
  ms/MB, heap + title-memory step ms, upload + prelude ms, live texture MB;
- the per-kind-then-pipeline diff table (`kind label` key from `cost_label`/
  `cost_kind`): base/candidate ms/frame, delta, relative change, n/frame and
  us/draw both sides, NEW/GONE rows, top 25 by max(base, candidate);
- header deltas and scene-variance warnings.

Scene variance is surfaced, not hidden: draws/frame, commands/frame and
texture MB differing by more than 10% between sides print a WARNING and no
timing claim may be made from that pair (section 3). Verified against
synthetic logs reproducing the exact `Meter::frame`, `costs::log`,
starvation, copystat, and `host textures` formats.

### 2.2 `warp-smoke.ps1` - bounded WARP smoke harness

There is no fixed-frame or headless flag in the title path
(`apps/victory/src/run.rs`; no `VICTORY_*` frame-limit/timeout knob exists -
section 1.9 scope confirmed by grep), so the harness runs
`victory run --execute <title>` bounded by wall-clock time (default 180 s)
with `VICTORY_D3D12_WARP=1 VICTORY_GPU_TIMING=1`, kills the process tree,
and grades the console log. PASS requires all three: the `host device on
the software adapter (WARP)` line, at least one fps line, at least one `GPU
cost by pipeline:` report; any device-removed/host-failure text fails it.
Captured scene footprints (draws/commands/texture MB) feed the compare
script. Explicit non-goal: no throughput comparison on WARP, ever.

## 3. Gate for perf claims without user gameplay

A "win" is a GPU-busy or per-kind claim from `VICTORY_GPU_TIMING` reports,
never a wall-clock average across different cutscene frames. All four
conditions are required:

1. **Same scene.** Draws/frame, commands/frame, texture MB medians within
   10% (the compare script warns otherwise). Different cutscene frames mean
   different work; re-run to the same frame (capture-frame numbers,
   `VICTORY_CAPTURE_WORLD` offsets) instead of claiming.
2. **Repeat runs, medians, GPU-side metrics.** At least 3 runs per side;
   compare medians of summed busy, per-kind ms/frame, `% of busy`, and
   `us/draw`. A win must exceed twice the baseline's run-to-run spread on
   that metric - the 27.4/17.7/8.2 ms single-hash observation is the reason
   single samples and wall averages are inadmissible.
3. **Distortion and unstamped-work accounting.** Default timing (batching
   preserved) for all claim runs; `VICTORY_GPU_TIMING_ALONE=1` runs are
   diagnostic only. Wall/fps deltas smaller than the ~12-40 ms distortion
   band prove nothing alone. Changes touching uploads, preludes, or present
   must show noted + residual + gap flat-or-better, and no NEW unexplained
   rows or `dropped` growth.
4. **No regressions in the companions.** Starvation blocking flat-or-better,
   copystat heap/memory/upload/prelude steps flat-or-better, no new
   pipeline-drop reasons, no device removals, no visual-corruption reports
   from capture frames. Standard repo gates still apply to any code change
   (`cargo test --workspace --locked`, `clippy -D warnings`, `cargo xtask
   policy` - ROADMAP_30FPS.md:40-45).

WARP runs may support correctness and footprint claims only. PresentMon (not
yet integrated) is required before any hardware fps claim, not before
GPU-busy claims.

## 4. Ranked profiling investments

1. **Stamp the preludes and the present-copy list.** The only structural
   hole: two direct-queue lists run unstamped while their wait sits in the
   wall. Cost: one timestamp pair + resolve each (same pattern as the busy
   pair, host.rs:2737-2746); report them as explicit `prelude`/`present`
   rows so `residual` + `gap` become pure idle/overlap. Small, highest value:
   closes the "unstamped copy work" question without a copy queue.
2. **PresentMon integration.** External present-to-present pacing as ground
   truth for `Meter::frame`; gates hardware fps claims. Medium value, needed
   exactly once, no per-iteration use.
3. **Per-cause starvation attribution: keep, extend only if uploads move.**
   The 5-cause blocking + depth histogram already answer "starved vs
   saturated" (roadmap section 6's 15-20% util question). Add an upload-stall
   cause only when/if buffer copies move to the GPU; otherwise no work.
4. **PIX/WARP capture flow.** Highest debug value (single-frame deep dive,
   shader-level attribution), lowest perf-gate value: it does not fix
   apples-to-oranges averaging or run-to-run variance. Build after 1-2, on
   demand.

Explicitly de-prioritized (per roadmap section 6): snapshot/copy-back
thread fan-out, storm caps beyond the current ones, submission
consolidation, and present-depth work - all wait until CPU-side traffic is
near the prep <= 10 ms budget, where they become measurable.
