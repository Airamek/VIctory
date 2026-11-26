# eval/survey-followups — survey follow-ups (take / subsumed / skip)

Base: `fork/eval/1828-rebase @ 5be261f` (fork `Airamek/VIctory`, origin `Marvisak/VIctory` READ-ONLY).
The cool_fox commits cited by the survey (`f4b48ad`, `e05d463`) are not fetchable
from the fork (no such refs; `git ls-remote fork` shows only eval branches), so
each item was verified against the landed tip and ported from the survey's
description, not by cherry-pick. No knobs (`VICTORY_*` gates), no accuracy
changes, no behavior change beyond the stated items. Larger cool_fox reworks
(residency guard, publish-slot-after-upload, per-heap scoping, movie-plane
buffer) are explicitly out of scope (post-resync) and untouched here.
`CARGO_TARGET_DIR=D:\victory-target\survey-follow` (D: only).

## Item 1 — oversized indirect records take nothing (TAKEN)

Survey: cool_fox `f4b48ad` / Avatar `449830f` (`VICTORY_INDIRECT_REST=0`): an
oversized indirect record takes nothing, so later valid draws run.

First verified against our landed drawkill clamp-hoist (always-hoist `f188c83` +
card-resident budget `8834ab8` + first-chunk revert `468ef69`, all ancestors of
the base): the hoist does NOT subsume the behavior. The hoisted single-group
clamp (`crates/victory-gpu/src/indirect.rs`, `SOURCE`) still takes the rest —
`costs[i] > left` sets `left = 0`, so one oversized record ends its list's
budget. The old behavior was even pinned by
`a_record_that_does_not_fit_ends_the_budget` (`[T,T,F,F,F,F,F,F]`).

Port (commit `ddb9233`, unconditional, no knob): drop the `left = 0`, so an
oversized record runs empty but leaves `left` alone. Sibling-test math
re-checked (unaffected: `rejected_records_do_not_wrap_the_budget`,
`the_budget_goes_to_the_first_records`, `a_spent_budget_stays_spent` give
identical results under both rules — verified by reasoning, tests confirm).

Test: replaced with `a_record_that_does_not_fit_takes_nothing`
(`[T,T,F,T,T,T,T,T]` — two 3·2²⁴ draws, one oversized, five small; the small
ones after the oversized one run). Runs on the host GPU (WARP here); skips
without one like the other clamp tests.

## Item 2 — fence WaitOnAddress + WakeByAddressAll (TAKEN)

Survey: cool_fox `e05d463` (`fence.rs`/`queue.rs`): our `fence.rs:113` stub
warned `E_NOTIMPL` instead of blocking.

Port (commit `65611fa`, unconditional, no knob):
- `set()`: after storing the value, `WakeByAddressAll` on the fence's own
  value, releasing guest threads waiting on the fence's GPU-visible address
  (titles read it at `gpu_address`) at once — plus our own waiters below.
- `set_event_on_completion` null-event path: instead of `E_NOTIMPL`, wait in
  250 ms `WaitOnAddress` slices on the value, `pump_held(Why::Fence)` between
  slices (a queue's signal waits for the submissions before it, so held work
  that may run does). Each slice re-reads the value, so a missed wake costs
  at most a slice; an already-reached value still returns above the loop
  (steady state ~0 ms). The waiters'-lock guard is dropped before the loop
  (the pump signals through `set`, which takes it — no nesting). Non-Windows
  keeps `E_NOTIMPL`; the import is `cfg(not(windows))`-gated so Windows
  builds stay warning-free.

Test: new `blocking_waits_end_on_signal` (scoped waiter thread blocks on value
9 with a null event; main thread signals after 100 ms; waiter returns `S_OK`,
value reads 9 — mirrors `signals_set_waiting_events`, 0.10 s wall).
The old `wait(f, 9, null) == E_NOTIMPL` assertion is removed (it would now
block forever); event-based coverage in that test is unchanged.

## Item 3 — cardbuf review nits

- (a) Delete dead `p.card_writes` loop in `complete()` — TAKEN (commit
  `885a47e`). `card_writes` is `mem::take`n at submit when the gate is on, so
  `p.card_writes` is empty at completion; when the gate is off no card shadow
  can exist (placement needs `CardState::try_new`, which returns `None`), so
  it is empty then too. `submission_done` on a card shadow would be a no-op
  anyway (`compared` stays empty: `writing()` early-returns for card,
  `copy_back` diverts to `finish_card_pending`). Comment records the why.
- (b) `debug_assert!(!is_card())` for keep/snapshot paths — TAKEN (commit
  `885a47e`, in `keep_writes` and `copy_back_or_not`, `shadow.rs`). Card
  ranges bypass `back` (they go to the card engine's plan), and `copy_back`
  diverts card shadows before the snapshot path; the asserts pin that
  invariant. Zero release cost.
- (c) `firsts` bookkeeping for before-list GPU time — VERIFIED PRESENT,
  nothing taken. The submit path inserts the card before-list at `lists[0]`
  and bumps `firsts[1..]`/`cuts[1..]` with `debug_assert!(firsts[0]==0)`
  (`host.rs`, card submit block), so firsts-windows (alone timing) and
  `group_members`/cuts grouping (frame timing) stay consistent with the
  before-list inside run 0's window. Covered by `cuts_groups_hold_their_runs`.

## Gates

Per commit: `cargo test --workspace --locked`, `cargo clippy --workspace
--all-targets --all-features --locked -- -D warnings`, `cargo xtask policy`.
Targeted suites green at each step (`indirect::` 7/7 incl. the new GPU-run
test, `fence::` 3/3 incl. the new wake test, `shadow::` 19/19, `card::` 3/3;
clippy and policy clean).

One pre-existing load flake observed and chased, not caused by this branch:
full-workspace runs intermittently fail exactly one timing-sensitive test —
first `frame::tests::waits_end_on_ticks_and_count_frames` (`left: 0, right:
1`), then on re-run `host::tests::read_marks_of_both_bitmaps_are_taken` while
the frame test passed. Both pass in isolation; neither touches the clamp
(frame waits), the fence wait (frame ticks predate it), or card paths. Confirmed pre-existing:
the same full `victory-gpu` suite on the clean base tip (`D:\dev\VIctory` @
`5be261f`, separate target dir) also fails under load — 2 failed
(`read_marks_of_both_bitmaps_are_taken` again, plus
`bytes_off_the_runtimes_memory_are_copied_at_each_use`), each passing in
isolation. No action taken here; the branch's own targeted suites are green.
