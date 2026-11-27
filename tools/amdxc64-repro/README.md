> **See also:** README.md · docs/architecture.md · docs/asset-policy.md
> · docs/ops-guidelines.md · docs/gpu-profiling-methodology.md
> · docs/preexisting-flakes.md · docs/eval-survey-followups.md
> · D:/dev/TRIAGE_PROGRESS.md · D:/dev/ROADMAP_30FPS.md · D:/dev/TASK_QUEUE.md
> (index only; content below unchanged — doc-linker pass, eval/1828-rebase @ 19062fe)

# amdxc64 repro (diag31)

Standalone `CreateGraphicsPipelineState` repro for the diag31 crash, on
top of `accc1afc`.

## The crash

`%TEMP%\victory-diag31.log` (analysis in
`%TEMP%\diag31-crash-shader.txt`):

- First-chance `0xC0000005` (null-read, `rbx=0x0 r9=0x0`) at
  `amdxc64.dll+0x671f78`, stack `#0-#19` all in `amdxc64.dll`,
  `#20-#28` in `D3D12Core.dll`/`d3d12.dll`, `#29-#35` in `victory.exe`
  (unhandled repeats at the same offset).
- Immediately before the first chance, the runtime translated pixel
  shader `76376a2ee6c53ffe` (`user_data=2 samplers=6 cached=true`,
  `ms=0`); the two previous translations are the sibling 6-sampler
  pixels `bf7278cd8d6d6ff3`, `9763863ad4cd2579`.
- Interleaving `host pipeline` lines bind the same vertex
  `9975d5335335cc9c` with other pixels (`ms` 2786/3025/4212,
  `rectangles=false`); no `host pipeline` line for the `7637...` pair
  exists, so the crash lands inside its pipeline creation.
- The run was otherwise healthy (31 fps, 63 being made, 0 dropped, no
  `host pipeline refused` anywhere), i.e. the permit/shader path was
  fine and the driver died compiling our DXIL. Same family as belobug
  #324 (AMD's compiler crashes on our DXIL).

## Container-hash key (`VICTORY_NAME_SHADERS`)

The 16-hex runtime hashes name *containers*, not sources. The mapping
was produced (read-only over the game, listing written outside the repo)
with e.g.:

```powershell
$env:VICTORY_NAME_SHADERS = 'D:\victory-target\amdxc64-repro\names.txt'
# then: victory translate <game paths...>  (read-only walk; the env var
# appends "<hash>`t<source>" per container — see apps/victory/src/translate.rs)
```

Result (`D:\victory-target\amdxc64-repro\names.txt`):

| runtime hash       | source                                              |
|--------------------|-----------------------------------------------------|
| `9975d5335335cc9c` | `C:\Games\GTA6\shaders.rpf@0xeff29 vs_classfied_fullScreen_60f7e46b_Wrapped` |
| `bf7278cd8d6d6ff3` | `C:\Games\GTA6\shaders.rpf@0x4f8d8a6 PS_MIRROR_PBR_MATERIAL_GLINTS_5f34c83e_Wrapped` |
| `9763863ad4cd2579` | `C:\Games\GTA6\shaders.rpf@0x4f85da2 PS_MIRROR_PBR_MATERIAL_SHEEN_5f34c83e_Wrapped` |
| `76376a2ee6c53ffe` | `C:\Games\GTA6\shaders.rpf@0x4f96cf6 PS_MIRROR_PBR_MATERIAL_RETROREFLECTION_5f34c83e_Wrapped` |

Translated DXIL for all four (game-derived, kept outside the repo) is
under `D:\victory-target\amdxc64-repro\emit\`. Neither the DXIL nor the
game file is committed (asset policy, rule #1); the repro re-translates
from the game file at run time and asserts the hash prefixes first.

## Candidates (one process each)

| tag                  | VS     | PS     | note                                              |
|----------------------|--------|--------|---------------------------------------------------|
| `vs9975+ps7637-prime` | `0xeff29` | `0x4f96cf6` | prime suspect: translation immediately precedes the crash |
| `vs9975+psbf72-near`  | `0xeff29` | `0x4f8d8a6` | nearest 6-sampler sibling (control)               |
| `vs9975+ps9763-near`  | `0xeff29` | `0x4f85da2` | nearest 6-sampler sibling (control)               |

## Run

```powershell
powershell -ExecutionPolicy Bypass -File tools\amdxc64-repro\run-amdxc64-repro.ps1
```

- Builds `apps/victory/examples/amdxc64_repro.rs` (release) and runs it
  once per candidate, each in its own process with a **600 s** watchdog
  (`Process.WaitForExit(600_000)`; kill on timeout → `TIMED-OUT`).
- Reads `C:\Games\GTA6\shaders.rpf` read-only. Writes only to
  `D:\victory-target\amdxc64-repro\runs\<tag>\{stdout.txt,result.txt}`.
  Never writes to `C:\Games`, never writes to the repo.
- Forces the hardware device (`VICTORY_WARP` scrubbed; WARP cannot
  reproduce an `amdxc64` crash).

Verdicts per candidate: `PASS` (exit 0, pipeline created),
`CRASH-0xC0000005` (reproduces the diag31 driver crash),
`REFUSED` (exit 2, host refused with reason, no crash),
`SETUP-FAILED` (exit 3, bad container/hash/device),
`TIMED-OUT` (killed at 600 s), else `EXIT-<code>`.

## Workaround (bf72 quarantine)

Result (runs in `D:\victory-target\amdxc64-repro\runs\`):

- `vs9975+psbf72-near`: `CRASH-0xC0000005` (dies at the "creating graphics
  pipeline" marker — reproduces diag31).
- `vs9975+ps7637-prime`, `vs9975+ps9763-near`: `PASS`.

Same vertex shader and same 6-sampler shape in all three, so the crash is
the bf72 pixel DXIL (`PS_MIRROR_PBR_MATERIAL_GLINTS`), not the vertex
shader or the pattern. A solo in-process `CreateGraphicsPipelineState`
crashes, so no serialisation or retry avoids it.

`crates/victory-gpu/src/pipeline.rs` hard-codes a quarantine for container
hash `bf7278cd8d6d6ff3` (`QUARANTINED_SHADERS`): pipelines holding it are
never queued or compiled — their draws are dropped like an untranslated
shader's, logged loudly once (`note_quarantined`), never wrong pixels.
Deliberately unconditional (not `VICTORY_SKIP_PIPELINES`, which is
diagnostics-gated): every build must avoid the crashing compile.

## Fidelity notes / limitations

- The repro translates with the same function and options the shader
  cache uses (`translate_with_used`, `view_marks: true`) and creates the
  pipeline through `victory_shader::emit::host::Host::pipeline`, i.e.
  the same `CreateGraphicsPipelineState` the offline
  `victory translate --pipelines` path uses.
- The live runtime additionally wraps creation in the pipeline library
  (`crates/victory-gpu/src/pipeline/library.rs`) and uses the title's
  graphics state (blend/raster/targets/depth from the PSO stream);
  `Host::pipeline` derives targets from the pixel shader's outputs.
  If the prime candidate passes here, the next step is a second repro
  that replays the exact title `GraphicsState` for the pair.
- Exit `CRASH-0xC0000005` on `vs9975+ps7637-prime` with `PASS` on the two
  near siblings isolates the crash to the `7637...` DXIL; a crash on all
  three points at the shared vertex or the 6-sampler pattern instead.
