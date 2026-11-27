// Standalone amdxc64 repro (diag31): one `CreateGraphicsPipelineState`
// for a translated VS+PS pair on the hardware adapter.
//
// Why this exists: diag31 crashed with a first-chance 0xC0000005
// (null-read) at amdxc64.dll+0x671f78, immediately after the runtime
// translated pixel shader 76376a2ee6c53ffe (cached DXIL, 6 samplers) and
// while creating its host pipeline with vertex 9975d5335335cc9c. The
// process kept running (post-crash host summary 3 s later), so the fault
// is a user-mode driver crash inside pipeline creation on our DXIL -- the
// same family as belobug #324 (AMD's compiler crashes on our DXIL).
//
// What it does, in this order:
//   1. Reads the game file READ-ONLY and unpacks the two SGD5 shader
//      records (the offsets in `VICTORY_NAME_SHADERS` listings are record
//      markers; `unpack` yields the exact container the title hands the
//      runtime).
//   2. Checks each container's BLAKE3-128 prefix against the expected
//      hash from the runtime log, so a game-version drift fails loudly
//      instead of testing the wrong pair (falls back to a hash scan when
//      the offsets moved).
//   3. Translates each container exactly like the runtime's shader cache
//      does (`translate_with_used` with `view_marks: true`), with no
//      cache and no writes anywhere.
//   4. Creates the hardware device (`Host::new(false)`) and calls
//      `Host::pipeline(&vs, &ps)`, i.e. one `CreateGraphicsPipelineState`.
//      Exit 0 + "pipeline created" means the driver compiled it;
//      a 0xC0000005 process crash at the "creating graphics pipeline"
//      marker reproduces diag31; exit 2 means the host refused it
//      (message on stdout) rather than crashing.
//
// One process per candidate: the PowerShell wrapper
// (tools/amdxc64-repro/run-amdxc64-repro.ps1) runs this binary once per
// VS+PS pair with a 600 s watchdog, so a driver crash in one candidate
// cannot take down the others. Nothing is ever written to the game
// directory; logs go to D:/victory-target/amdxc64-repro/runs/.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use victory_game::sgd5;
use victory_shader::emit::{self, host::Host};

/// Prime suspect from diag31 (`%TEMP%\victory-diag31.log`):
/// vertex 9975d5335335cc9c + pixel 76376a2ee6c53ffe.
const DEFAULT_GAME: &str = r"C:\Games\GTA6\shaders.rpf";
const DEFAULT_VS_OFFSET: &str = "0xeff29";
const DEFAULT_VS_HASH: &str = "9975d5335335cc9c";
const DEFAULT_PS_OFFSET: &str = "0x4f96cf6";
const DEFAULT_PS_HASH: &str = "76376a2ee6c53ffe";

#[derive(Parser, Debug)]
#[command(name = "amdxc64_repro", about = "Standalone CreateGraphicsPipelineState repro for diag31")]
struct Args {
    /// Game file holding the SGD5 shader records (opened read-only).
    #[arg(long, default_value = DEFAULT_GAME)]
    game: PathBuf,
    /// File offset of the vertex record's marker (hex, e.g. 0xeff29).
    #[arg(long, default_value = DEFAULT_VS_OFFSET)]
    vs_offset: String,
    /// Expected BLAKE3-128 hex prefix of the unpacked vertex container.
    #[arg(long, default_value = DEFAULT_VS_HASH)]
    vs_hash: String,
    /// File offset of the pixel record's marker (hex).
    #[arg(long, default_value = DEFAULT_PS_OFFSET)]
    ps_offset: String,
    /// Expected BLAKE3-128 hex prefix of the unpacked pixel container.
    #[arg(long, default_value = DEFAULT_PS_HASH)]
    ps_hash: String,
    /// Candidate label for the log (e.g. vs9975+ps7637).
    #[arg(long, default_value = "vs9975+ps7637")]
    tag: String,
}

fn parse_offset(s: &str) -> Result<usize, String> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    usize::from_str_radix(s, 16).map_err(|e| format!("bad offset {s:?}: {e}"))
}

/// Unpacks the SGD5 record at `offset` (falling back to a hash scan when
/// the game moved it): the exact container the title hands the runtime,
/// the same bytes `VICTORY_NAME_SHADERS` hashes.
fn container_at(data: &[u8], offset: usize, expected: &str, what: &str) -> Result<(Vec<u8>, String), String> {
    let records = sgd5::records(data);
    if records.is_empty() {
        return Err(format!("{what}: no SGD5 records in {}", data.len()));
    }
    // First the recorded offset (fast, exact); then a hash scan.
    let mut ordered: Vec<_> = records.iter().collect();
    ordered.sort_by_key(|r| if r.offset == offset { 0 } else { 1 });
    for r in ordered {
        let Some(container) = r.unpack() else { continue };
        let hash = blake3::hash(&container).to_hex()[..16].to_string();
        if hash.starts_with(&expected.to_lowercase()) {
            let via = if r.offset == offset { "offset" } else { "hash-scan" };
            println!(
                "{what} container record={:#x} name={} size={} hash={hash} (found via {via})",
                r.offset,
                r.name.unwrap_or("?"),
                container.len()
            );
            return Ok((container, hash));
        }
    }
    Err(format!(
        "{what}: no record with offset {offset:#x} or hash prefix {expected:?} \
         ({} records; game version drift? re-run with VICTORY_NAME_SHADERS, see tools/amdxc64-repro/README.md)",
        records.len()
    ))
}

fn describe(s: &emit::HostShader) -> String {
    format!(
        "stage={:?} dxil={} user_data={} samplers={} static_samplers={} \
         root_cbvs={:?} root_srvs={:?} root_uavs={:?} memory={} memory_writes={} \
         view_addresses={} heap={} heap_writes={} view_marks={}",
        s.stage,
        s.dxil.len(),
        s.user_data.len(),
        s.samplers.len(),
        s.static_samplers.len(),
        s.root_cbvs,
        s.root_srvs,
        s.root_uavs,
        s.memory,
        s.memory_writes,
        s.view_addresses,
        s.heap,
        s.heap_writes,
        s.view_marks,
    )
}

fn main() {
    let code = run();
    std::process::exit(code);
}

fn run() -> i32 {
    let args = Args::parse();
    println!("amdxc64-repro tag={}", args.tag);
    println!("game={} (read-only)", args.game.display());

    let data = match std::fs::read(&args.game) {
        Ok(d) => d,
        Err(e) => {
            println!("READ-FAILED game={}: {e}", args.game.display());
            return 3;
        }
    };
    println!("game bytes={}", data.len());

    let vs_offset = match parse_offset(&args.vs_offset) {
        Ok(o) => o,
        Err(e) => {
            println!("BAD-ARGS {e}");
            return 3;
        }
    };
    let ps_offset = match parse_offset(&args.ps_offset) {
        Ok(o) => o,
        Err(e) => {
            println!("BAD-ARGS {e}");
            return 3;
        }
    };

    let (vs_container, vs_hash) = match container_at(&data, vs_offset, &args.vs_hash, "vs") {
        Ok(c) => c,
        Err(e) => {
            println!("CONTAINER-FAILED {e}");
            return 3;
        }
    };
    let (ps_container, ps_hash) = match container_at(&data, ps_offset, &args.ps_hash, "ps") {
        Ok(c) => c,
        Err(e) => {
            println!("CONTAINER-FAILED {e}");
            return 3;
        }
    };
    let _ = (vs_container.len(), ps_container.len());

    // Same translation the runtime's shader cache runs (`view_marks: true`),
    // but with no cache and no writes anywhere.
    let options = emit::Options { view_marks: true };
    let vs = match emit::translate_with_used(&vs_container, options) {
        Ok((s, _)) => s,
        Err(e) => {
            println!("TRANSLATE-FAILED vs {vs_hash}: {e}");
            return 3;
        }
    };
    let ps = match emit::translate_with_used(&ps_container, options) {
        Ok((s, _)) => s,
        Err(e) => {
            println!("TRANSLATE-FAILED ps {ps_hash}: {e}");
            return 3;
        }
    };
    println!("vs translated {}", describe(&vs));
    println!("ps translated {}", describe(&ps));

    // Hardware device only: WARP would not reproduce an amdxc64 crash.
    // (`Host::new(false)` still honours VICTORY_WARP=1, so the wrapper
    // scrubs that variable before launching this binary.)
    let host = match Host::new(false) {
        Ok(h) => h,
        Err(e) => {
            println!("NO-DEVICE {e}");
            return 3;
        }
    };
    println!("device=hardware (Host::new ok)");

    println!("creating graphics pipeline VS {vs_hash} + PS {ps_hash} ...");
    let _ = std::io::stdout().flush();
    let started = Instant::now();
    match host.pipeline(&vs, &ps) {
        Ok(_) => {
            println!("pipeline created ms={} (PASS: driver compiled it)", started.elapsed().as_millis());
            0
        }
        Err(e) => {
            println!("host refused ms={}: {e} (no driver crash)", started.elapsed().as_millis());
            2
        }
    }
}
