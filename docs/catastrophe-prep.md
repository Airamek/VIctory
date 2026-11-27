# Catastrophe prep (survive the five box-killers)

Base: `eval/1828-rebase @ 19062fe` (merge of microbench hot-paths).
Fork-only doc. No code changes, no gate changes.

> See also: `docs/ops-guidelines.md` (task pattern, gameplay protocol,
> storage hygiene — this doc is linked from there), `docs/preexisting-flakes.md`
> (census + Rule #0 freeze-class exclusion + GPU/CPU split + WARP policy),
> `docs/gpu-profiling-methodology.md` (claim gate, `warp-smoke.ps1` 180 s bound,
> compare script), `docs/microbench-hot-paths.md` (hot-loop SSD scratch
> `C:\vt\<name>` runbook), `docs/eval-survey-followups.md` (adjudication table),
> `docs/architecture.md` (code map), `D:\dev\TRIAGE_PROGRESS.md` (lead's live
> persistent memory — tips, verdicts, queue, resume commands).

## 1. The five catastrophes

| # | Killer | Signature | First move |
|---|---|---|---|
| 1 | Frontend crash (opencode TUI / harness dies) | Session gone, worktree + branches intact on disk | Resume commands (§3) from a new session; nothing was pushed unless you pushed it |
| 2 | Wipe / restart (user wipes, Windows Update reboots) | `D:/dev/VIctory-*` worktrees gone, `C:\vt\*` scratch gone, `D:\victory-target\*` may survive (D: is cold storage) | Re-clone fork, re-checkout branches from fork (§2: branches are the only durable work), rebuild targets |
| 3 | Context loss (compact / fresh session, files intact) | You forget everything; disk remembers | Read `docs/ops-guidelines.md` §0 + `D:\dev\TRIAGE_PROGRESS.md` first — they are the persistent memory. Then §3 resume commands |
| 4 | Frozen box (display wedged, TDR storm, driver needs reboot) | Screen frozen, SSH/harness dead, `0xC0000409` / `0xC0000005` / `0x80000001` aborts or `host::fail()` device-removed aborts in logs | STOP-THE-LINE per census Rule #0: never census material, never solo-green adjudicated. Reboot, record abort-vs-assert, keep the diag log |
| 5 | Wedged test binary (a test hangs the suite) | `cargo test` never finishes, no new output for minutes | Kill it, re-run solo with the §4 timers; a hang is a bounded-wait violation until proven otherwise (§5) |

## 2. What to persist (before every risky step)

Durability order, most-durable first:

1. **Branches on the fork.** The ONLY thing that survives a wipe. Push early,
   push to the Airamek fork only — never to `origin` (read-only upstream),
   never to `triage-merge-queue` (read-only). One task = one branch.
   `git push fork <branch>` before any long gate, before gameplay, before sleep.
2. **Worktrees.** `D:/dev/VIctory-*`, one per task. Convenient, NOT durable:
   a wipe deletes them. Re-derivable from fork branches at any time.
3. **Target dirs.** Cold storage on D: (`D:\victory-target\<name>`); HOT-LOOP
   testing (serial runs, parallel measurement, heavy gates, microbenches)
   belongs on SSD scratch in a short path (`C:\vt\<name>` — faster link +
   test-binary load on NVMe). SSD targets are scratch: delete when the task
   closes. NEVER let C: below 30 GB free.
4. **Logs.** Gameplay: `$env:TEMP\victory-diagNN.log` (NN increments; crash
   re-run adds `$env:VICTORY_D3D12_DEBUG=1`). Test gates: keep the console
   tail with the gate excerpt (real excerpts, not summaries). After every
   gameplay run record: binary SHA + timestamp, env combo, frames reached,
   crash signature (HRESULT + debug-layer line), log filename.
5. **Lead memory.** `docs/ops-guidelines.md` §0 + `D:\dev\TRIAGE_PROGRESS.md`:
   update at every task boundary (tips, verdicts, pending queue, resume
   commands). A fresh session must resume from these two files alone.

## 3. Resume commands (fresh session, in order)

```powershell
# Where am I, what survived?
git status -sb
git log --oneline -8
git branch -vv
git ls-remote fork | Select-String 'eval/|ensemble/'
# Worktrees + targets + scratch that made it:
Get-ChildItem D:/dev/VIctory-* -Directory | Select-Object Name
Get-ChildItem D:\victory-target\ | Select-Object Name, @{n='GB';e={[math]::Round((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB,2)}}
Get-ChildItem C:\vt\ -Directory -ErrorAction SilentlyContinue | Select-Object Name
Get-ChildItem $env:TEMP\victory-diag*.log -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime
# Then read memory:
# docs/ops-guidelines.md §0, D:\dev\TRIAGE_PROGRESS.md tail, docs/preexisting-flakes.md census table
```

`git status` tells you what was mid-flight. `git log` tells you the tip.
`git ls-remote fork` tells you what is actually safe (pushed). If a branch
is not on the fork, it exists only on this disk — push it before anything else.

## 4. Generous test timers (never kill a healthy suite)

Loaded-box variance is real (27.4/17.7/8.2 ms for one dispatch hash;
tick-timing census entries #1/#9). Timeouts below kill healthy runs.

- **600 s per test binary.** Full `victory-gpu` lib / workspace gates get
  10 minutes wall-clock before you call it wedged. Open a live progress
  window (tail the log) for anything expected past ~2 minutes — no silent
  long runs.
- **120 s solo.** A single `--exact <path>` test gets 2 minutes. Past that it
  is either a bounded-wait violation (§5) or a freeze-class incident (Rule #0),
  not a slow test.
- **180 s warp-smoke.** `tools/gpu-compare/warp-smoke.ps1` default bound:
  `victory run --execute <title>` under `VICTORY_D3D12_WARP=1
  VICTORY_GPU_TIMING=1`, kill the tree at 180 s, grade the log. PASS needs
  all three: the `host device on the software adapter (WARP)` line, ≥1 fps
  line, ≥1 `GPU cost by pipeline:` report; any device-removed/host-failure
  text fails it. WARP is correctness + footprint only — never speed, never
  hardware claims.

Quiet box for gates that matter. Loaded-box GPU runs are suspect by default.

## 5. No-infinite-spin rules (tests never wait forever)

Determinism items 2+5 (`aab3fa35` TDR fail-fast + test-only panic routing;
`ecd9e702` item 4 bounded waits) are the law:

- **No `INFINITE` waits in tests.** `wait_gpu` / `wait_copy` (and the card
  fetch wait) go through the `cfg(test)` path: `wait_for_submission` slices
  of 250 ms, removal check each slice, 10 s / 20 s warns, 25 s stall budget.
  `Removed` and `Stalled` fail the test via `fail()` instead of piling
  behind `INFINITE`. `finish_loop` keeps the production path.
- **Freeze-class is stop-the-line, never census.** Process aborts
  (`0xC0000409`, `0xC0000005`, `0x80000001`), `host::fail()`
  device-removed aborts, anything that wedges the display driver / needs a
  reboot: record abort-vs-assert, halt the gate, investigate the mechanism.
  A solo-green rerun after an abort proves nothing.
- **Serialization, not skips.** Every `victory-gpu` test holds
  `crate::host::GPU_TEST_LOCK` for its whole body (first line). No
  `#[ignore]`, no skips, no thread-count games to green a branch. Solo runs
  are diagnostics + census adjudication (`docs/preexisting-flakes.md`:
  census flake = passes alone; non-census needs solo + base-solo +
  group-run-together decider), never gate replacements.
- **TDR fail-fast in tests** refuses the hardware device at TdrLevel 0/1
  instead of freezing the box. Tests fail red on WARP exactly as on hardware.

If you find an unbounded wait in a test, that is the bug — bound it per the
`wait_for_submission` pattern, do not raise the timeout and move on.

## 6. Storage hygiene (after EVERY task closes)

- Check: `D:\victory-target\*` sizes, `D:/dev/VIctory-*` worktrees,
  `$env:TEMP\victory-diag*.log`, `C:\vt\*` scratch.
- Reclaim: delete stale target dirs of merged/abandoned tasks; remove
  superseded diag logs (keep the current chain); drop SSD scratch
  (`C:\vt\<name>`) and report reclaimed GB; drop worktrees of merged tasks
  only after fork-push verified; never touch unknown-owner dirs blindly
  (identify first).
- Report reclaimed GB in the task-close note (`TRIAGE_PROGRESS.md` section).
- **Floors and exemptions:** NEVER let C: below 30 GB free. `C:\Games` is
  exempt — never touched by cleanup. The `Games` folders hold this project's
  title files: exempt from audits, never touched by cleanup.
- Disk-full is its own catastrophe (one `pipeline::library` worker panic was
  observed only while C: was at 0 bytes): treat a red gate on a full disk as
  environmental, free space, re-run before any other adjudication.
