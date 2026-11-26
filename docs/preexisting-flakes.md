# Pre-existing flaky tests (census)

Branch: `eval/flake-census`, off `fork/eval/1828-rebase @ 78b823f`
(fork `Airamek/VIctory`; `origin` read-only, never pushed).
Doc-only: no code changes, no gate weakening (no skips, no test edits).

Reviewers kept asking whether a red suite means the branch under review
broke something. This file is the in-repo answer: every test known to fail
intermittently on clean trees, what it asserts, why it flakes, and where it
was proven pre-existing (base SHA + who proved it). A gate failure matching
an entry here is adjudicated per the standard below — never by skipping the
test. The perf-claim gate (`docs/gpu-profiling-methodology.md` §3) points
here for the flake half of its verdict.

Conventions: paths are `cargo test -p victory-gpu --lib <module>::<test>`
unless another package is named. `solo` = run alone in the test binary
(`--exact <path>`); `in-suite` = full parallel run; `serial` =
`--test-threads=1`. `base` = a tree with none of the branch's diff applied.

## Adjudication standard (classb precedent §57/59 + survey-followups §69)

Source: `D:\dev\TRIAGE_PROGRESS.md` §§57/59 (classb-into-eval) and §69
(survey-followups), full table in `docs/eval-survey-followups.md`
("Gates — adjudication").

1. **Census flake = passes alone.** A test listed here fails a full parallel
   run but passes solo (and usually serial / on rerun). That alone
   adjudicates it: re-run solo, record, move on.
2. **Non-census failure needs three legs:** solo on the branch, solo on the
   clean base, and — if both pass — a group-run-together decider (the
   suspects run together in one binary, §60: 4/4 together + solo each +
   rotation across runs/thread counts). If pristine controls flake
   identically, it is full-suite parallel-load interference, not merge
   breakage.
3. **Base-rate comparison for new suspects** (§57): e.g. merge only if the
   suspect fails at most ~1 in ~6 total runs and nothing but census flakes
   appears otherwise (`ec7ad37`: eviction 1/~8, aliased 1/~8, read_marks ~4
   with base also failing; base `dac689b` 275/0; solo everything green —
   verdict MERGE ACCEPTED, §59).
4. **No gate-weakening, ever** (§69): capped retries, then adjudication with
   the table committed. No `#[ignore]`, no skips, no thread/job games to
   green a branch. Solo runs are diagnostics, not gate replacements.

## Confirmed census (base-vs-branch evidence on record)

| # | Test (package `victory-gpu` lib unless noted) | Asserts (1 line) | Root cause | Proven pre-existing (base + prover) |
|---|---|---|---|---|
| 1 | `frame::tests::waits_end_on_ticks_and_count_frames` (`crates/victory-gpu/src/frame.rs:221`) | Flip waits each take their own 50 ms tick; a late wait ends at once on the latest tick | **Tick-timing**: wall-clock asserts (`elapsed < 10 ms`, 110 ms sleeps) vs a box at 100% CPU from sibling builds | Fails on clean base AND unmodified `origin/main` (§22, #2611 rerun); 4/4 fail stretch on pristine base (§39, classb fix team); proven on clean `f70bef7` (§35, avatar-small) |
| 2 | `host::tests::read_marks_of_both_bitmaps_are_taken` (`crates/victory-gpu/src/host.rs:888`) | Read marks of both view bitmaps are taken exactly once per slot until forgotten | **Parallel shared-state**: process-global read-mark registries (`take_reads`) shared with concurrently running sibling tests; holds `submitted` lock but races other tests' takes | Fails on clean base too (§59: ~4 runs incl. base; §71: flake on base too; §49: solo-green; §20: 20/20 alone; survey B1 base run fails, §69 table) |
| 3 | `host::batch::flush_copies_each_copy_once` (`crates/victory-gpu/src/host.rs:8248`) | 20 disjoint copies each flush exactly the title-written bytes, once | **Parallel shared-state / load**: parallel flush/copy-back threads convoy on the title-memory guard lock (§27); global-count assertions polluted by sibling tests | 4/4 branch + **3/3 clean base** (§36, belobug2 reviewer); WARP serial-lib solo-green (§77, lead) |
| 4 | `host::batch::a_copy_is_refreshed_once_per_submission_by_the_pages_written` (`crates/victory-gpu/src/host.rs:8211`) | A whole-copy submission refreshes the one written page once, and nothing twice | **Parallel shared-state / load**: same guard-lock + global watch/shadow state as #3 | WARP serial-lib fail, solo-green (§77, lead: "previously documented"); base-side rate weaker than #3 — see note |
| 5 | `bindless::tests::heap_views_are_current` (`crates/victory-gpu/src/bindless.rs:663`) | A shader reaching buffer+texture only through the heap reads each round's title bytes | **GPU timing / shared device**: live host-device heap slots + GPU round-trips under full-suite concurrency | Known full-suite concurrency flake, passes alone + reruns (§33 gate note, phase-3 port); §15 census list; §27: unserialized (parallel copy-back broke its global counts pre-lock) |
| 6 | `bindless::tests::heap_views_follow_writes_through_another_mapping` (`crates/victory-gpu/src/bindless.rs:686`) | Same as #5 with the title writing through a second (pool) view of the pages | **GPU timing / shared device**: twin of #5, same `rounds()` harness | §15 census "heap_views twins" (implement-round, all pass alone/serial/rerun); same family as #5 |
| 7 | `shadow::snapshot_tests::two_copies_keep_their_snapshots_apart` (`crates/victory-gpu/src/shadow/snapshot_tests.rs:218`) | Two copies' interleaved snapshots each copy back their own bytes | **Parallel shared-state**: global snapshot-block count assertions vs parallel sibling tests | Failed 2x under parallel load; passes alone, serial 171 green, 3/3 lib reruns, full workspace green after (§13, lead); §15/§16 census (memo ruled out); §27 root-caused to missing single-test lock, fixed by holding it |
| 8 | `draw::tests::root_views_reach_their_copy_and_read_zeros_past_it` (`crates/victory-gpu/src/draw.rs:2564`) | Root-view reads reach the copy's end from the host copy; past-end reads are 0 | **GPU timing / load**: host-GPU round-trip + shared device under parallel load | Proven on unmodified `eb9c95e` (§17, heap-hull merge check, passes alone); rotating flake §22 (passes alone) |
| 9 | `pipeline::make::tests::huge_compiles_wait_for_the_slot` (`crates/victory-gpu/src/pipeline/make.rs:813`) | Huge driver compiles take the slot serially; ordinary ones never wait | **Tick-timing**: cross-thread elapsed-time probe (`started.elapsed()`) vs loaded box | §15 census ("huge_slot"); §22 (1x, passes alone); passes solo (§35, classb review gates); survey R3/R4 branch-only in-suite fails, solo-green both sides (§69 table) |
| 10 | `shadow::tests::title_writes_under_a_busy_range_are_copied` (`crates/victory-gpu/src/shadow.rs:1781`) | A title write under a shader-declared range still reaches the copy; post-refresh title writes are not undone | **Parallel shared-state**: global guard/watch/shadow state under parallel load | Survey R7 branch in-suite fail, solo-green both sides (§69 table, accepted environmental) |
| 11 | `shadow::tests::bytes_off_the_runtimes_memory_are_copied_at_each_use` (`crates/victory-gpu/src/shadow.rs:1817`) | Off-runtime bytes are copied at each use, never guarded | **Parallel shared-state**: same family as #10 | Survey R4/R7 branch + **B1 clean-base** fail, solo-green both sides (§69 table) |
| 12 | `shadow::tests::marked_pages_go_back_by_the_bytes_the_gpu_changed` (`crates/victory-gpu/src/shadow.rs`) | Marked pages go back by GPU-changed bytes only; unmarked title bytes stand | **Parallel shared-state**: same family as #10 | Survey R7 branch in-suite fail, solo-green both sides (§69 table) |
| 13 | `draw::tests::a_texture_written_in_one_format_reads_in_another_through_a_placed_view` (`crates/victory-gpu/src/draw.rs:2931`) | Cross-format placed-view reads share one host texture (castable) or bridge by copy | **GPU timing / shared device**: placed-view/host-texture sharing order on the live device | Survey R5 branch in-suite fail once, solo-green both sides (§69 table, accepted environmental; base co-occurrence NOT observed in 2 base runs — weaker base evidence) |
| 14 | `watch::tests::writes_mark_every_tracker_of_their_page` (`crates/victory-gpu/src/watch.rs:972`) | First take reports changed; after taking, only written pages; every tracker of a page sees it | **Parallel shared-state**: process-global watch registries under parallel load | Survey capped-2 branch + **B2 clean-base** fail, solo-green both sides (§69 table) |
| 15 | `watch::tests::writes_through_another_mapping_mark_the_tracker` (`crates/victory-gpu/src/watch.rs:1237`) | Writes through a second mapping (or runtime-reported) mark the tracker | **Parallel shared-state**: same family as #14 | Survey capped-2 branch + **B2 clean-base** fail, solo-green both sides (§69 table) |
| 16 | `watch::tests::kernel_reads_into_guarded_blocks_after_opening` (`crates/victory-gpu/src/watch.rs:1158`) | A kernel read into a guarded block fails; opened first, it lands and counts changed | **Parallel shared-state / guard timing**: guard-page VEH + global watch state | Survey capped-2 branch + **B2 clean-base** fail, solo-green both sides (§69 table) |
| 17 | `watch::tests::a_tracker_dropped_on_one_mapping_leaves_the_other_guarded` (`crates/victory-gpu/src/watch.rs:1262`) | Dropping one mapping's tracker leaves the other mapping guarded | **Guard timing**: guard-page teardown race under load | §73 merge gate 315/1 (SOLO-GREEN, accepted per precedent); survey table: B2 clean-base in-suite fail (§69 table) |
| 18 | `external::tests::eviction_after_capturing_copy_refreshes_the_held_host_texture` (`crates/victory-gpu/src/external.rs:550`) | After eviction clears the cached slot, alias lookup still refreshes and returns the captured copy | **GPU timing / load**: eviction + alias-cache races on the live device | §57: 1x in-suite-only (tiersmalls author flag; 3 subsequent in-suite + alone green); §59 decider 1/~8 runs, solo-green, MERGE ACCEPTED; WARP solo-green §77; cardbuf base comparison identical incl. single-threaded (§61/§62) |
| 19 | `external::tests::aliased_textures_copy_from_their_gpu_writer_on_first_external_read` (`crates/victory-gpu/src/external.rs:346`) | A placed texture aliasing GPU-written memory copies from the writer's host texture on first external read | **GPU timing / load**: same family as #18 | §59 decider 1/~8 runs, solo-green; §60 group-run-together decider 4/4 + pristine controls flake identically — MERGED+pushed |
| 20 | `host::title_copies::copy_back_writes_mapped_pages_only` (`crates/victory-gpu/src/host.rs`) | Copy-backs land only in mapped title-memory pages (bytes read back as written) | **Parallel shared-state**: CPU-side assert on shared title-memory mappings; passes solo, in-module group, and serial full-suite; fails intermittently only in full parallel runs | §109: fix-tree solo-green, base-solo green, title_copies group 4/4 green, merge-gate in-suite fail 1x then serial re-run green - census-listed, MERGE ACCEPTED |
| 21 | `shadow::writeback_merge_matches_bytewise_oracle` (`crates/victory-gpu/src/shadow.rs`) | Merged write-back matches the bytewise oracle | **Parallel shared-state**: CPU-side oracle comparison; solo-green on fix and base tips, group-green; fails intermittently only in full parallel runs (same family as #20) | §131: fix-tip solo-green, base-tip solo-green, shadow::writeback group green, batch-gate in-suite fail 1x - census-listed, MERGE ACCEPTED |
| 22 | `title_copy::tests::guard_pages_stay_guarded` (`crates/victory-gpu/src/title_copy.rs:327`) | A guard page a copy reaches stays a guard page (bytes as the kernel copy gives) | **Parallel shared-state / guard timing**: solo-green on fix and base tips, title_copy group 8/8 green, serial full-suite green; fails intermittently only in full parallel runs; graduated from UNCONFIRMED watchlist | §131: fix-tip solo-green, base-tip solo-green, title_copy group 8/8 green, batch-gate in-suite fail 1x then serial re-run adjudication - census-listed, MERGE ACCEPTED |
| 23 | `host::tests::copy_engine_uploads_leave_common_and_read_back_exact` (`crates/victory-gpu/src/host.rs`) | COPY-queue first upload + re-upload leave COMMON and read back exact (new copy-fix regression test) | **Parallel shared-state**: solo-green repeatedly on fix and merge tips, group-green; fails intermittently ONLY in full parallel lib runs with the upload-missing signature (`upload` None, same as #20) while the serialized twin run on the same tree passes. Revisit as interaction if it recurs deterministically | §135: merge-tip solo + fix-branch solo/group/parallel green, batch-gate in-suite fail 1x then serial re-run green - census-listed, MERGE ACCEPTED |

Notes:

- #4 (`a_copy_is_refreshed`): the task brief lists it as census; direct
  evidence is §77 (serial-lib fail + solo-green on tip) with the lead calling
  it "previously documented". Its base-side rate is less quantified than #3's
  3/3 — treat base attribution as weaker, solo-green as firm.
- #13 (`a_texture_written_...`): accepted environmental in §69, but the two
  clean-base runs did not reproduce it. Keep on census (reviewer-accepted),
  do not cite it as base-proven.
- #18 single-threaded occurrence: cardbuf gates showed base comparison
  identical including single-threaded runs (§61/§62) — i.e. this family is
  not an artifact of thread count alone.

## UNCONFIRMED (listed in the brief; no base-vs-branch pointer found — do not cite as proven)

| Test | Asserts (1 line) | Why listed / status |
|---|---|---|
| `memory::tests::a_submission_that_missed_leaves_the_copies_at_once` (`crates/victory-gpu/src/memory.rs:1215`) | A submission that missed memory leaves host copies at title bytes; one that missed nothing keeps its writes in the copies | Brief-named ("missed_copies/leaves_the_other"); no flake mention in §§13-74 or the survey table. Suspected family: withholding/global-state. Needs solo + base-solo data before census. |
| `watch::tests::unmapped_holes_are_cached_until_remapped` (`crates/victory-gpu/src/watch.rs:1127`) | Unmapped pages are memoized as holes (no `VirtualProtect`/`VirtualQuery` retry) until remapped | Brief-named; no flake mention found. Suspected family: guard timing. UNCONFIRMED. |
| `title_copy::tests::guard_pages_stay_guarded` (`crates/victory-gpu/src/title_copy.rs:327`) | A guard page a copy reaches stays a guard page (bytes as the kernel copy gives) | GRADUATED to census entry #22 (confirmed parallel-only: solo/base/group green, in-suite intermittent fail) — row kept for history, do not cite as unconfirmed. |
| `pipeline::library` worker panic under disk-full | (exact test fn unidentified) | One panic observed only while `C:` was at 0 bytes (§36, belobug2 reviewer). Environmental (disk-full), not branch breakage. Exact test path UNCONFIRMED — do not cite a name. |

## Serialization / ignore markers in the tree (grep, this tip)

- No census test carries `#[ignore]`. Every `#[ignore]` in the tree is a
  benchmark/measurement (`snapshot_fill_throughput`, `bench_write`,
  `bench_bools`, `benchmark_private_data_release`, texture/XMA benches,
  `snapshot_tests` throughput) or needs out-of-repo assets — see also
  methodology §1.8. None of them is a gate test.
- Established serial-only pattern (not skips — the tests still run, just
  never overlapping): `host::submissions::GATE_LOCK`
  (`crates/victory-gpu/src/host.rs:8857-8859`, "serializes tests that
  temporarily gate the shared host queue"), and the snapshot single-test
  lock (`crates/victory-gpu/src/shadow/snapshot_tests.rs`, "one test at a
  time", §27). If an UNCONFIRMED guard/queue test above recurs
  deterministically, this pattern — not a revert, not an ignore — is the
  prescribed fix (§13).

## Explicitly NOT flakes (do not adjudicate as census)

Rule #0 (freeze-class exclusion, §137-138): process aborts
(`0xC0000409`, `0xC0000005`, `0x80000001`), `host::fail()` device-removed
aborts, and any run that wedges the display driver / needs a reboot are
STOP-THE-LINE incidents, never census material. A solo-green rerun after an
abort proves nothing about the abort ("fails parallel, passes solo" applies
to assertion failures only). Each census entry must record abort-vs-assert;
any entry whose evidence includes an abort stays UNCONFIRMED until the
abort's own mechanism is proven.

- `resource::tests::a_null_view_has_no_alias_source` abort (§74): hard
  `0xC0000409` fast-fail under resync bisect, prime suspect upstream
  `219878a` × tiersmalls `#2129`-area alias-source port. Deterministic crash
  under investigation, not load interference.
- Tiersmalls `safe_copy` VEH guard-page crash `0x80000001` (§51): deterministic
  incl. single-threaded, fixed by the guard-page VEH handler (§55).
- COPY-list legality aborts (timestamp queries §63, buffer 0xC1 §65): retail
  deterministic aborts, each fixed by its fix team. Real bugs, not flakes.

## Gates for this doc commit

- `cargo xtask policy` must pass (doc-only change).
- No code changes, so no test gate is claimed or needed (stated, not run).
| 24 | `watch::tests::a_busy_alias_leaves_the_page_changed_and_keeps_saved_bytes` (`crates/victory-gpu/src/watch.rs`) | Busy alias leaves page changed and keeps saved bytes (watch tracker) | **Full-run shared-state**: solo-green on fix and base tips, module group 22/22 green on fix; fails intermittently ONLY in full workspace serial runs. Same family as #18 (serial-only flake). Revisit as interaction if it recurs deterministically or outside full runs | S138: fix-solo + base-solo + watch-group green, gate3 full-workspace fail 1x - census-listed, MERGE ACCEPTED |
| 25 | `shadow::tests::partial_refresh_under_marked_range_updates_saved_bytes` (`crates/victory-gpu/src/shadow.rs`) | Partial refresh under marked range updates saved bytes (shadow write-back) | **Full-run shared-state**: solo-green on fix and base tips, group-green (3 group runs: partial_refresh passed each; one run each failed by big_snapshot sibling flake and one unknown, both unattributed); fails intermittently ONLY in full workspace serial runs. Same family as #20. Revisit as interaction if deterministic | S140: fix-solo + base-solo + 3x group green, ident-gate full-workspace fail 1x - census-listed, MERGE ACCEPTED |
