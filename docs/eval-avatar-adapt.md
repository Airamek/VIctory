# eval/avatar-adapt — ADAPT survey verdict (take / adapt-or-park)

Base: `eval/1828-rebase @ f70bef7` (fork `Airamek/VIctory`, origin `Marvisak/VIctory` READ-ONLY).
Avatar ref: `https://github.com/avatarmanz/VIctory-7900XTX` fetched anonymously read-only
into `refs/remotes/avatar/*` (main + `pr-10` + `pr-4`). No push to avatar or origin;
this branch pushes to fork only. No GPU to agents: no perf runs here.

Our tree has since landed overlapping work — never regress it:
- `8eebf0e` widened `helpers_read` (`quad && pixel`) + upstream `4015a18` quad reads at triangle edges.
- view-marks + `Part` snapshots + parallel `run_back` + `submission_gen`
  (upstream `cd94d35`+`940d32f` model; live logs showed 2.5–3.8 GB snapshots + 2.3–3.9 GB compares/frame before marks).
- `Batch::hull` + `submission_gen` heap-hull prepare cache (`b7aa55e`), + in-review class-b slices 1–9.
- landed async-#1828 timeline/`execute_inner` + class-b + drawkill + huge-PSO gating (`eabd767`).

Policy for this branch: no knobs (`VICTORY_*` gates), no accuracy cuts, tests mandatory,
gates per commit: workspace + `clippy --all-targets --all-features -D warnings` + `cargo xtask policy`.
`CARGO_TARGET_DIR=D:\victory-target\avatar-adapt`.

## A1 — 1c8dd31 quad-op neighbour reads — PARK here (already covered elsewhere)

Avatar `1c8dd31` ("Pixel programs read their quad's neighbours with quad operations"):
`lower.rs` gains `helpers_read() = quad && pixel` + `quad_read(x, lanes)` (identity /
single-lane `QuadReadLaneAt` / across-X/Y/diagonal `quadOp` + lane-index selects),
wired into `ds_swizzle` (offset `0x8000` perm / and-or/xor masks) and `Dpp::Row ctrl<=0xFF`;
`mod.rs` gains `quad_neighbours_at_a_triangles_edge_are_read` + `triangle_mesh_to(reach)`.
Class: shadow holes (slope-bias depth), dashed triangle edges — helper lanes read as 0
through wave reads on AMD.

Ours already subsumes correctness:
- `8eebf0e` widened `helpers_read` to `quad && pixel` (same predicate, different comment).
- Upstream `4015a18` routes the same two sites through `quad_pick` (dynamic-index
  `QuadReadLaneAt` ×4 + selects). Correctness hole closed; `perf ~0` per brief.

Their delta over ours is perf-only: constant-pattern `across` ops instead of 4× laneAt.
That exact optimisation already landed in `eval/drawkill @ 764c172`
("Pixel quad patterns take one quad op; in-quad DPP8 reads through quads":
`quad_pick(x, src, pattern)`, `0xE4` identity / `0x00/0x55/0xAA/0xFF` broadcast /
`0xB1/0x4E/0x1B` across, plus in-quad DPP8 via quad path, + `quad_tests.rs` 283 lines).
Porting `1c8dd31` here would duplicate/conflict with drawkill, not with base.

Verdict: TAKE nothing in `eval/avatar-adapt`. Correctness: ours subsumes. Perf delta:
drawkill subsumes. Revisit trigger: none needed; drawkill owns this.

## A2 (0d5e4c8 LAND) + A3 (c651346 / 18c0129 dword refresh) — PARK, do NOT port

- `0d5e4c8` LAND (`VICTORY_LAND`, on): 1,961 insertions, new `land.rs` + `land/tests.rs`
  (556+433), `safecopy` AVX2 + never-guarded views, `shadow`/`watch`/`host`/`inplace`
  rework. Claim: relay 1,500×64KiB (98 MB)/frame 45 ms → 8 ms on 4 threads.
- `c651346` SEND_CHANGED (`VICTORY_SEND_CHANGED`, was on): refresh sends changed dwords
  only (`dword_runs`, `card_changed_runs`, `SentRuns`), not whole 64 KiB pages.
- `18c0129` SEND_OWN (on): per-submission bytes at prepare time + `SentRuns::near/forget`,
  `stale_pages` diagnostic, `VICTORY_CARD_ALL` / `VICTORY_PICKED_WRITES` / `VICTORY_UPCOPY_CHECK`.

Warning label (decisive): avatar itself reverted SEND_CHANGED to off-by-default in
`6ae0696` — "run tile1-0 lost light and geometry with it; run bis1-0, the same but
for this, was right". Accuracy-risk, knob-gated, on a stale base that predates
view-marks: avatar has no `view_marks`, no `copy_back_marks`, no `Part` snapshots
(`git grep view_marks/copy_back_marks` on `avatar/main`: empty; `VICTORY_LAND` only).

What remains MB-scale post-view-marks (verified in-tree, no port needed):
- `shadow::writing()` fast path: marked pages arm, no snapshot; unmarked path keeps
  parallel fill. `copy_back_marks` runs before the parallel `run_back` loop
  (`host.rs:2658/2664`, `run_back` groups by copy + `copy_threads` scope).
- `copy_back_marked` compares/writes only marked 64 KiB pages; `Drop` merges
  snapshot-forget + mark give-back. The old "declared GBs snapshot+compare" path
  only fires for unmarked ranges.
- `submission_gen` (`Batch::hull`, `host.rs:2052/4391/6067`) already isolates
  per-submission prepare/refresh staleness.

Stacking LAND/SEND_* blindly with in-review class-b slices (3/5/6/7/8: refresh spans,
write-back merge, guard planning, settle/conflict caches, page-table prep cache)
risks ordering conflicts (marks-first vs land-beside-guards, `SentRuns` vs `hull` gen,
per-submission own-bytes vs `submission_done`).

Verdict: PARK all of A2/A3. Take zero lines. Revisit trigger: if post-marks logs still
show GB-scale snapshot/compare per frame, re-evaluate a *narrow* dword-run take
*inside marked pages only*, unconditional (no knob), with HW A/B + tile1-0/bis1-0
regression proof. Until then, class-b owns this area.

## A4 — 44629de / c3e643e / 53c5256 descriptor threading — PARK, keep hull + class-b

Avatar trio (all knob-gated, stale base):
- `53c5256`: `HEAP_SYNC_ASIDE` (beside preparing) + `QUICK_MIRROR` (made-slot told
  without lock); 16-record compare batches.
- `c3e643e`: slot sets as bits, heaps' records get own threads, mirrored counting.
- `44629de`: records copied as bytes, pages read/compared/sorted on threads,
  empty `shows_none` slots skipped, card-step pages asked once.

Ours (no knobs): `Batch::hull` (`HashMap<copyPtr,(from,to,gen,dirty)>`,
`add_cached` skips redundant pending/current merge on same gen/dirtiness/covered range,
`title_written` clears) + `submission_gen` + in-review class-b prepare/heap caches
(classb-1 coalesce Copy/Fill + committed-pages cache; classb-7 settle/conflict cache;
classb-8 page-table prep cache). Same prepare/heap wall, newer base, no `unsafe Sync`
(`Copied` holds raw `*mut u8` + `unsafe impl Sync`).

Disjoint bits examined: `Copied::changes` parallel compare, `shows_none` skip,
`quick_mirror` lock-free tell. All are gated (`VICTORY_HEAP_SYNC_ASIDE=0`,
`VICTORY_QUICK_MIRROR=0`, `VICTORY_VIEW_THREADS`) and assume no hull/gen; porting
would fork the heap path and fight class-b 7/8.

Verdict: PARK. Port nothing. Revisit trigger: after class-b 1–9 land, if heap-prep
wall remains, re-evaluate a knobless `shows_none`-style skip only, with tests.

## A5 — THIN / ASYNC_SUBMIT / JOIN_PARTS — PARK with keep-analysis (no implementation)

Rival async design on stale base, knob-gated everywhere:
- `3a99ee6` THIN (`VICTORY_THIN=1`, off): 45–60k list calls/frame appended to a stream,
  8 threads translate; 912-line `list/thin.rs`.
- `b8b30d1` ASYNC_SUBMIT (`VICTORY_ASYNC_SUBMIT=1`, off): one line over all queues +
  presents in call order; 758-line `queue/behind.rs`. Idea from fxceless `f85101b6`.
- `37332f4` JOIN_PARTS (`VICTORY_JOIN_PARTS=8`, off): runner takes following Run entries
  with self-met waits/signals (fxceless `39731795` idea), refresh front in one go
  (`VICTORY_REFRESH_FRONT` measured at nothing); `VICTORY_PLACED_ORDER` fill cut.

Landed coverage it would duplicate: async-#1828 (timeline, `execute_inner`, fence/host
ordering kept in `eb9c95e` resync), per-draw clamp-chunk hoisting, class-b batching,
drawkill. Avatar design predates `submission_gen`/`hull`/marks ordering and carries
its own queue/behind/host/card_steps fork (673+868+1096 lines) that cannot land
separably without a rebase.

What, if anything, it offers beyond landed async-#1828 + class-b + drawkill: only the
*join of consecutive Run entries with fronted refresh* as an idea; everything else
(list-thread translation, submit thread, placed-order cuts) is a whole-queue fork.

Verdict: PARK, analysis only, no code. Revisit trigger: if prep wall survives
slices 1–9 (measure host prepare ms/frame after class-b), re-evaluate a narrow
knobless join of adjacent Runs with refresh-front only — rebased, with ordering proof.

## A8 — 20b9834 RDNA_FLOAT (0.8 ms claim) — PARK, accuracy-risk, NO knob, tests mandatory

Change (`VICTORY_RDNA_FLOAT`, on; `rdna::ALL=15`): max/min as single FMax/FMin
(drop the 7-op −0 ordering), legacy mul as plain product + bounded-check
(`0×anything=0`, "a legacy product's zero may have the other sign"), med3 as min/max
of ordered constants, fract as raw Frc. Cache keys on `host_rdna()` (`host_settings`
3→4, fingerprint += rdna bits): every shader retranslates.

Claim: yard replay turn-by-turn 12 loops: dispatches+draws 18.41→17.58 ms, lists
29.2→28.4 of 36.6 ms; 0.4 without LEGACY; picture same (79 dB; two same-set replays
differ by 68). Left-out list (mad, fast rcp, 4-dword scalar loads, heap-kind skip,
per-compute wave size, no barriers) measured ~nothing.

Paper evaluation (no GPU to agents):
- MAX_MIN: dropping −0<+0 ordering is observable via `signbit`/division; needs proof
  no title shader distinguishes (79 dB on one capture ≠ proof).
- LEGACY: `bounded(a,12)` heuristic + `float_k` non-zero-finite fast path; fallback
  still changes signed-zero and may hide `0×inf/NaN→0` vs IEEE. Commit message
  admits sign flip. Console has `0×anything=0` (D3D9 rule) but host driver FMA/contraction
  and NaN payload propagation differ; `bounded` depth-12 is not a proof.
- MED3/FRACT: narrower, but still float-rule changes keyed into every pipeline.
- Tests added (`probe_tests`, `valu_tests`, bounded/float_k unit) are host-side value
  checks, not cross-adapter (6600-XT vs 7900-XTX) differential + WARP image proof.

Policy: accuracy-risk needs 6600-XT A/B + correctness proof. Implement unconditional
or not at all — NO `VICTORY_RDNA_FLOAT` knob (knobs for float rules fork the shader
cache and hide accuracy deltas). Tests mandatory, and they must be differential HW.

Verdict: PARK (do not implement). Keep spelled-out forms.

## A10 — a30829c slow-pipeline quarter — PARK, keep ours (likely dupe)

- Ours `eabd767`: predictive — past 256 KiB containers (xcl huge-PSO size) a pipeline
  waits for the single huge-compile slot; ordinary pipelines never wait; log carries
  container bytes. No knob. Fixes commit-charge exhaustion + driver contention upfront.
- Theirs `a30829c` (`VICTORY_PIPELINE_SLOW=0`): reactive — pipelines observed >1 s go to
  `slow-pipelines.txt` (`victory warm` notes too) and wait in their own list served by
  a quarter of workers. First-run with fresh caches: 3,020 pipelines, 16 workers stuck
  seconds–tens of seconds, 0.5–1.6 s/frame + ≤2,900 dropped draws 60→150 s.

Analysis: same wall (multi-second driver compiles stall all workers), opposite trigger
(size vs observed time). Ours prevents the stall; theirs recovers after one stall and
helps slow-but-small pipelines ours misses. Knob-gated, needs `warm.rs` + library/make
fork (147 lines). No proof ours misses real slow pipelines in steady state.

Verdict: PARK. Keep `eabd767`. Revisit trigger: if logs show slow-but-small pipelines
stalling workers despite huge-gating, take the slow-list as a *complement* (knobless),
not a replacement.

## Open PRs #10 (wide-dispatch split) + #4 (LDS flags) — review-first, route out

- PR #10 (avatar `pr-10`, drive-by `gomeaw`/`chsantanna05`): `32dd6ae` split wide
  indirect dispatches on 32-lane hosts (`VICTORY_DISPATCH_SPLIT=1`, `USER_GROUP_BASE`,
  `bound_split`, second `ExecuteIndirect` past 65,535 x-groups; `split_group_base()`)
  + `42d6c52`/`d2bd5c9` test scoping + `d78024b` replay `SPLIT_FLAG`/split-lists kept
  intact. Touches `draw`/`indirect`/`lower`/`shader`/`replay` — knob-gated, 32-lane-only,
  replay format change. Do NOT take in avatar-adapt.
  Route: `eval/drawkill` (indirect/clamp owner) + `eval/storm` (shader owner) for
  review-first; needs 32-lane HW proof + replay-compat sign-off.
- PR #4 (avatar `pr-4`, drive-by `gomeaw`): "Check the final LDS flag from every lane
  on a 32-lane adapter" — test-only (`wave64_tests.rs` +64/−6: per-lane UAV offsets via
  `offen`, 14 dispatches over active×skip, skips off 32-lane/WARP-less hosts).
  Shape is safe (test-only) but asserts `1+active!=0` semantics of split-wave LDS join.
  Route: `eval/storm` for shader-wave review; take only as test-only commit after ack.

## Bottom line for lead

No code deltas taken in this branch (doc-only). Everything with a perf claim is either
already subsumed (A1→drawkill, A10→huge-PSO, async→#1828/class-b/drawkill) or
knob-gated on a stale pre-marks base with an accuracy warning (A2/A3, A4, A5, A8).
Porting any of them here would regress landed work or stack with in-review class-b
slices. Revisit triggers above are all measurement-gated (prep/copy-back walls after
class-b 1–9; HW A/B for float rules; slow-small pipeline evidence; 32-lane HW for PRs).
