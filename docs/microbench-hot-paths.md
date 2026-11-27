# Micro-benchmarks for the new memory/compile bounds

Branch: `ensemble-victory-no-game-testing#max0f5-microbench` (fork-only).
Design + implementation. No accuracy change, no env knobs, no new dependencies.

This note maps the hot paths added by the two bounding changes on `eval`
(`e9b2ef8f` "bound bindless host copies (evict idle + pressure transients)"
and `26095486` "cap concurrent driver compiles at 3"), says where synthetic
benches belong and why, and records the first measurements. Counts in the
bench output are synthetic registry sizes, not gameplay footprints; they
isolate CPU bookkeeping from driver/GPU work that needs a title and a device.

## 1. Code map (what the bounds added)

### 1.1 Host-texture eviction (`crates/victory-gpu/src/resource.rs`)

- `evict_unused()` (`resource.rs:1620`) runs once per presented frame from
  `Host::execute` (`host.rs:2647`). It calls `video_memory()`, subtracts the
  placement free bytes, and passes an `over` byte count to `evict`.
- `evict()` (`resource.rs:1645`) reads shader read marks, builds `candidates()`
  (`resource.rs:1679`: every title texture with an evictable host texture),
  and picks victims with `choose` (`resource.rs:1694`).
- `choose` is the pure CPU half: partition idle (both `IDLE_FRAMES` and
  `IDLE_MS`) from the rest, then sort the rest by last-use milliseconds and
  take until `over` is covered. It also allocates two `Vec<usize>`s.
- Cost scales with the number of live host textures: `O(N log N)` per pass,
  on the submitting thread that runs the frame's last submission.

### 1.2 Host-copy eviction (`crates/victory-gpu/src/shadow.rs`)

- `evict_unused()` (`shadow.rs:1804`) runs beside the texture pass from
  `Host::execute` (`host.rs:2649`), with the same per-frame gate.
- `evict()` (`shadow.rs:1920`) first calls `arena().release_empty()` (trailing
  fully free arena chunks), then `candidates()` (`shadow.rs:1831`: buffer
  registry holds, loose windows, live exact copies), `plan(...)` (new, below),
  and the per-victim `evictable` re-check (`shadow.rs:1862`: no heap-slot
  viewers, no recorded/in-flight submission refs, not card-resident).
- The planning half was inline; it is now `plan()` (`shadow.rs:1897`) with the
  same idle-partition + LRU-sort algorithm as `resource::choose`. Extracting it
  is behaviour-preserving (same indices, same order, same `over`
  subtraction) and makes the hot loop measurable without constructing D3D12
  resources.
- Cost here scales with live buffers + loose windows + live exact copies
  (the bound log reports the counts).

### 1.3 Pressure-served transients (`crates/victory-gpu/src/shadow.rs`)

- `pressured(ask)` (`shadow.rs:255`) = two atomic loads + `video_memory()`;
  `relieve(ask)` (`shadow.rs:260`) runs an evict pass first.
- The pressure branch in `loose()` (`shadow.rs:1743`), `exact()`
  (`shadow.rs:2155`), `find()` (`shadow.rs:1648`)/`Arena::take`
  (`shadow.rs:2244`) either returns
  a cached copy or `transient_dedicated(...)` (`shadow.rs:2065`): a copy that
  is never registered, pinned by its lists, remade whole per submission.
- Pooling: `SPARE_TRANSIENT` (count/bytes-bounded), `take_transient_buffer`,
  `return_transient_buffer`; the best-fit selection is now `spare_fit()`
  (`shadow.rs:2028`). The CPU side is a scan of at most `SPARE_TRANSIENT_MAX`
  (16) entries.
- The dominant cost is the D3D12 buffer make/map and the title-memory copy —
  device work, not measurable in a CPU microbench. Its signal is already
  in-engine: `SHADOW_STATS[SHADOW_STATS_TRANSIENT]`, `made/freed/evicted`
  per frame, pool size and held MB in `log_shadow()` (`shadow.rs:1986`).

### 1.4 Driver-compile permit (`crates/victory-gpu/src/pipeline/make.rs`)

- `COMPILE_PERMITS = 3` (`make.rs:151`), `COMPILE_SLOTS` (`make.rs:154`),
  `CompileGuard::take()` / `Drop` (`make.rs:159`/`178`).
- Taken in `Pending::make()` (`make.rs:104`, `CompileGuard::take()` at :108)
  before `HostPipeline::new`, and
  in the pre-build (`pipeline.rs:696`), so both share the cap.
- The permit is a `Mutex<usize>` + `Condvar`: per compile it costs one
  lock/unlock when uncontended and a wait when all three slots are held. Real
  compiles take milliseconds to seconds, so the interesting property is the
  queueing shape, not the lock cost.

## 2. Where the benches belong, and why

All four benches are in-crate libtest tests marked
`#[ignore = "benchmark"]`, in the module that owns the code:

| Bench | Location |
| --- | --- |
| `resource::tests::benchmark_evict_choose` | `crates/victory-gpu/src/resource.rs` |
| `shadow::tests::benchmark_evict_plan` | `crates/victory-gpu/src/shadow.rs` |
| `shadow::tests::benchmark_spare_transient_fit` | `crates/victory-gpu/src/shadow.rs` |
| `pipeline::make::tests::benchmark_compile_permit` | `crates/victory-gpu/src/pipeline/make.rs` |

Why this shape rather than `benches/` + criterion:

1. **Private access.** `choose`, `plan`, `spare_fit` and `CompileGuard` are
   `pub(crate)`/private and several touch crate statics. A `benches/` target is
   a separate crate and would need new public hooks or a feature-gated
   re-export. In-module tests reach the real code with no API surface change.
2. **No new dependency.** The workspace has no criterion in `Cargo.lock`; adding
   it is lockfile churn and a network fetch for a three-loop synthetic bench.
   The repo already benchmarks this way: `object::tests::
   benchmark_private_data_release`, `shadow::snapshot_tests::
   snapshot_fill_throughput`, `pipeline.rs::bench_prebuild_key_vs_full`,
   `draw.rs`, `texture.rs`, `title_copy.rs` (all `#[ignore = "benchmark"]`).
3. **Same runbook.** `cargo test --release ... -- --ignored --nocapture` is the
   documented command in `docs/gpu-profiling-methodology.md` §1.8.
4. `crates/victory-shader/benches/guard_runs.rs` (`harness = false`, hand
   timed) is the right pattern only for a *public* API (`victory_shader::
   structure`); it does not help for private GPU-crate internals. Criterion's
   statistical harness is overkill for deterministic planning loops and would
   still need the same private-access workaround.

Caveat: these are CPU-only benches. They must not be read as gameplay or GPU
costs. The registry sizes are inputs, not observations; take the observed
sizes from `log_shadow` (`buffers`, `loose windows`, `live exact copies`) and
the host-texture census in the diagnostics report.

## 3. What each bench measures

### 3.1 `benchmark_evict_choose` (host textures)

Times `choose` over 1,024 / 4,096 / 16,384 synthetic uses, one in four idle,
`over` = 10% of bytes. Reports median ns per pass and how many victims. This is
the per-frame CPU selection cost; `candidates()` registry scanning and the
`give_up` driver calls are outside it.

### 3.2 `benchmark_evict_plan` (host copies)

The same shape over `plan`, with 2 KiB–33 KiB copies. Same algorithm as 3.1 but
the shadow registry's mix (buffers, loose windows, exact copies).

### 3.3 `benchmark_spare_transient_fit` (transient pool)

Times `spare_fit` over a full 16-entry pool for 2 KiB / 8 KiB / 1 MiB / 8 MiB
asks. Shows the CPU overhead a pressure-served copy adds; the buffer make and
copy that follow are device work and belong to the in-engine counters.

### 3.4 `benchmark_compile_permit` (cap 3)

- Phase 1: 1,000,000 uncontended `take()`/drop cycles, ns each — the whole
  per-compile permit cost.
- Phase 2: 1 / 3 / 6 / 10 workers (the real pool is at most 8 + 2) each make
  16 modelled 200 µs compiles, and the wall time is compared with the ideal
  (`workers·compiles·hold / min(workers, 3)`). This is a model: a real compile
  is seconds and needs the title's shaders. It quantifies serialisation, not
  driver throughput.

## 4. How to run

Hot-loop measurement runs on the C: scratch target (standing rule: D: is a
spinning disk; keep C: ≥ 30 GB free), and the target is deleted afterwards:

```powershell
$env:CARGO_TARGET_DIR='C:\vt\microbench'
cargo test --release -p victory-gpu --lib --locked -- `
  --ignored --nocapture --test-threads=1 benchmark_
Remove-Item C:\vt\microbench -Recurse -Force
```

`--test-threads=1` matters for `benchmark_compile_permit` (the permit is a
process-global) and keeps the timing loops from competing with each other.
`benchmark_` also matches the pre-existing `benchmark_private_data_release`.

Cheap gates for the branch itself stay on D:
`cargo check -p victory-gpu --all-targets --locked`,
`cargo clippy -p victory-gpu --all-targets --all-features --locked -- -D warnings`,
`cargo xtask policy`.

## 5. First measurements (synthetic counts, this box)

Release, Windows, C: scratch target, `--test-threads=1`, medians of 9 samples.
Not a gameplay run and not a GPU comparison.

```
test resource::tests::benchmark_evict_choose ...
  1024 candidates:   6275.0 ns/pass median (256 chosen)
  4096 candidates:  12981.2 ns/pass median (1024 chosen)
 16384 candidates: 173850.0 ns/pass median (4096 chosen)
test shadow::tests::benchmark_evict_plan ...
  1024 candidates:   4915.6 ns/pass median (256 chosen)
  4096 candidates:  17987.5 ns/pass median (1024 chosen)
 16384 candidates: 140200.0 ns/pass median (4096 chosen)
test shadow::tests::benchmark_spare_transient_fit ...
    2048 bytes over 16 pooled:    9.9 ns/call median (fit: Some(0))
    8192 bytes over 16 pooled:   12.4 ns/call median (fit: Some(1))
 1048576 bytes over 16 pooled:   12.6 ns/call median (fit: Some(8))
 8388608 bytes over 16 pooled:   10.7 ns/call median (fit: Some(11))
test pipeline::make::tests::benchmark_compile_permit ...
permit acquire+release:   29.1 ns median
 1 workers x 16 compiles:   3.69 ms wall,   3.20 ms ideal (+15% queueing)
 3 workers x 16 compiles:   3.65 ms wall,   3.20 ms ideal (+14% queueing)
 6 workers x 16 compiles:   7.37 ms wall,   6.40 ms ideal (+15% queueing)
10 workers x 16 compiles:  12.72 ms wall,  10.67 ms ideal (+19% queueing)
```

Reading:

- **Eviction planning** is negligible up to a few thousand live copies
  (~5–18 µs/pass) and becomes visible at 16k (~0.14–0.17 ms/pass, ~0.4–0.5%
  of a 30 fps frame) on the submitting thread. The number that matters is the
  *observed* registry size; `log_shadow` already reports it. If a real scene
  keeps 16k+ live copies, a cheaper selection (size-aware bucket or a
  partial-select instead of a full sort) is worth considering; otherwise this
  is not a frame-time risk. `choose` and `plan` are the same algorithm; the
  difference at 4k is input/noise, not a regression.
- **Transient pool** selection is ~10–13 ns/call. It cannot be the reason a
  pressure-served copy is slow; the allocation/copy is.
- **Compile permit** is 29 ns uncontended. The phase-2 deltas are dominated by
  the spin model's own floor (a 1-worker run is already +15% over the ideal
  because a 200 µs spin is not exactly 200 µs); 10 workers add only ~4
  percentage points over that floor. With real compiles at 10^3–10^4 × the
  modelled hold, the permit's own overhead is far below measurement noise. Its
  value is the bounded commit charge, not speed.

## 6. Merge recommendation

Merge the branch after ordinary review: it adds four ignored benches plus two
behaviour-preserving pure extractions (`shadow::plan`, `shadow::spare_fit`)
that the compiler, clippy and the existing unit tests cover. No production
behaviour, no frame-time change, no new dependency. If a reviewer insists on a
zero-production-diff benches branch, drop the two extractions and keep the
choose/permit benches; the shadow/transient benches then cannot call the real
helpers and should be deferred rather than replaced by a copy of the algorithm.

Follow-up (not this branch): use the in-engine `log_shadow` counts to pick
realistic registry sizes, and re-run `benchmark_evict_choose`/`_plan` with
those counts; log the permit's `waited_ms` line (`make.rs:172`) in a diag run
to see how often compiles actually queue at cap 3.

## 7. Hygiene

Measurement target `C:\vt\microbench` was 0.27 GB and was deleted on close
(standing rule). C: free before the build 42.7 GB, after cleanup 35.3 GB —
the drop is another teammate's `C:\vt\warp-smoke` / `C:\vt\test-hygiene-gate`
builds, not this one; C: never went below 30 GB during this task.
