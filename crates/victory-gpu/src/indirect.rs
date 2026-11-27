//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1) {
            dest.Store3(at + 8, args.Load3(at + 8));
        } else {
            dest.Store2(at + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // A record that does not fit takes nothing, so the
                    // records after it that fit still run (an oversized
                    // record no longer ends its list's budget).
                    costs[i] = PAST;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// One hoisted clamp: `records` records of `kind` (PC argument numbering)
/// `stride` bytes apart at host address `gpu` (a title buffer's copy, only
/// read), clamped in recorded order with the rest of its chunk (sequential
/// order preserved; nothing reordered).
/// `args` is the title buffer's host copy the clamp reads, `count` the one
/// holding the title's record count, if any (for the pre-clamp barriers).
pub(crate) struct ClampReq {
    pub(crate) gpu: u64,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) records: u32,
    pub(crate) args: ID3D12Resource,
    pub(crate) count: Option<ID3D12Resource>,
}

/// Records, in `list`, the clamps of `reqs` back to back as one sequential
/// chunk: unordered-access barriers scoped to the title buffers the clamps
/// read (their argument copies, and any distinct count buffer) instead of
/// a global flush, then one compute root signature and pipeline, then per
/// request its constants, views and dispatch (none with no records), each
/// followed by an unordered-access barrier on the shared budget so it is
/// charged in request order. The requests clamp into slices of one of the
/// list's argument buffers ([`HostList::arguments`]), at increasing
/// offsets: the (buffer, offset) pairs, in request order, for the
/// executions (and [`done_chunk`] after). Replaces the list's compute root
/// signature and pipeline (callers bind theirs after). `None` if the clamp
/// is not available, or a request's arguments are not 4-byte aligned
/// (alignment checked by the caller when hoisted; nothing recorded).
pub(crate) fn bound_chunk(list: &HostList, reqs: &[ClampReq]) -> Option<Vec<(ID3D12Resource, u64)>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let c = clamp()?;
    let l = &list.list;
    // One buffer for the chunk, a slice per request: executions of a chunk
    // run after all its clamps, so sharing one buffer at offset 0 would
    // leave every execution reading the last clamp's records. Offsets
    // first: a request the clamp refuses records nothing, including the
    // budget's zeroing.
    const ALIGN: u64 = 16;
    let mut offsets = Vec::with_capacity(reqs.len());
    let mut total = 0;
    for req in reqs {
        if !req.gpu.is_multiple_of(4) || !req.stride.is_multiple_of(4) {
            return None;
        }
        offsets.push(total);
        total += u64::from(req.stride)
            .checked_mul(u64::from(req.records.max(1)))?
            .checked_next_multiple_of(ALIGN)?;
    }
    let out = list.arguments(total.max(16))?;
    let was_zeroed = list.budget_zeroed();
    let (budget_res, budget, card) = list.budget()?;
    let first = !was_zeroed;
    // SAFETY: a live buffer.
    let base = unsafe { out.GetGPUVirtualAddress() };
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // title buffers' copies the caller's list holds.
    unsafe {
        // Order earlier title shader writes into the clamps' argument
        // copies (and any distinct count buffer) first, with barriers
        // scoped to those buffers instead of a global flush: one per
        // buffer, deduplicated.
        let mut seen: Vec<*mut core::ffi::c_void> = Vec::new();
        let mut pre: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for req in reqs {
            for buffer in [&req.args].into_iter().chain(req.count.as_ref()) {
                let raw = buffer.as_raw();
                if !seen.contains(&raw) {
                    seen.push(raw);
                    pre.push(uav_barrier_for(buffer));
                }
            }
        }
        if !pre.is_empty() {
            l.ResourceBarrier(&pre);
        }
        l.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for (req, &at) in reqs.iter().zip(offsets.iter()) {
            if req.records == 0 {
                // Zero records clamp nothing and execute nothing: no
                // dispatch and no unordered-access barriers; the slice is
                // never read.
                continue;
            }
            let constants = [req.kind, req.stride, req.records, 0];
            l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, req.gpu);
            l.SetComputeRootUnorderedAccessView(2, budget);
            l.SetComputeRootUnorderedAccessView(3, base + at);
            // One group walks every record.
            l.Dispatch(req.records.min(1), 1, 1);
            // The shared budget reaches the next clamp in order.
            l.ResourceBarrier(&[uav_barrier_for(&budget_res)]);
        }
        l.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
        // The card's copy of the budget goes back to common: the next
        // recording zeroes it again. The mapped word never leaves common.
        // Only the first chunk promotes the word (its zeroing leaves it
        // unordered access); later chunks reach it by implicit promotion
        // from common, like the mapped word, so only the first reverts it.
        if card && first {
            l.ResourceBarrier(&[transition(
                &budget_res,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }
    Some(offsets.into_iter().map(|at| (out.clone(), at)).collect())
}

/// After the executions a chunk clamped: its argument buffers back to
/// common (one per buffer; a chunk shares one).
pub(crate) fn done_chunk(list: &HostList, clamped: &[ID3D12Resource]) {
    if clamped.is_empty() {
        return;
    }
    let barriers: Vec<D3D12_RESOURCE_BARRIER> = clamped
        .iter()
        .map(|out| {
            transition(
                out,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COMMON,
            )
        })
        .collect();
    // SAFETY: an open list and live buffers.
    unsafe { list.list.ResourceBarrier(&barriers) };
}
/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// Test-only single-record path (production clamps hoisted chunks with
/// [`bound_chunk`]).
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
#[cfg(test)]
pub(crate) fn bound(list: &HostList, gpu: u64, (kind, stride, records): (u32, u32, u32)) -> Option<ID3D12Resource> {
    let was_zeroed = list.budget_zeroed();
    let (Some(c), Some((budget_res, budget, card))) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
        // As in [`bound_chunk`]: only the recording's first clamp promotes
        // the card's budget, so only it reverts it (later lists reusing one
        // find it common).
        if card && !was_zeroed {
            l.ResourceBarrier(&[transition(
                &budget_res,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }
    Some(out)
}

/// After the execution: the list's argument buffer back to common.
///
/// Test-only (production reverts chunks with [`done_chunk`]).
#[cfg(test)]
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Test-only single clamps ([`bound`]) still flush globally.
#[cfg(test)]
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

/// An unordered-access barrier on one resource: earlier shader writes
/// into it finish before later work reads or writes it.
fn uav_barrier_for(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::UAV_BARRIERS.with(|l| l.borrow_mut().push(resource.as_raw() as usize));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed COM pointer; `ManuallyDrop` never
                // releases it, and the caller keeps `resource` alive through
                // `ResourceBarrier`.
                pResource: std::mem::ManuallyDrop::new(Some(unsafe { std::mem::transmute_copy(resource) })),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    thread_local! {
        pub(super) static UAV_BARRIERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    fn take_uav_barriers() -> Vec<usize> {
        UAV_BARRIERS.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        let _gpu = crate::host::gpu_test_lock();
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            host.drain();
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let _gpu = crate::host::gpu_test_lock();
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let _gpu = crate::host::gpu_test_lock();
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit takes nothing: after two draws of
    /// 3 * 2^24 vertex instances a third runs empty but leaves the budget,
    /// and the small ones after it run.
    #[test]
    fn a_record_that_does_not_fit_takes_nothing() {
        let _gpu = crate::host::gpu_test_lock();
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, true, true, true, true, true]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        let _gpu = crate::host::gpu_test_lock();
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1)).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            host.drain();
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let _gpu = crate::host::gpu_test_lock();
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// A clamp chunk scopes its unordered-access barriers to the buffers
    /// its clamps read (each argument copy once, any distinct count buffer
    /// once, the list's budget once per clamping request) and never flushes
    /// globally; requests with no records move their buffers straight to
    /// indirect arguments, still paired with `done_chunk`. A second chunk
    /// leaves the card's budget as it found it: only the first chunk
    /// promotes (zeroes) and reverts it.
    #[test]
    fn chunk_scopes_uav_barriers_and_skips_zero_records() {
        let _gpu = crate::host::gpu_test_lock();
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (args, args_data) = host.shared_buffer(256).unwrap();
        let (count, _) = host.shared_buffer(256).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            args_data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            args.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        let list = host.new_list().unwrap();
        let req =
            |(kind, stride, records, counter): (u32, u32, u32, Option<super::ID3D12Resource>)| {
                super::ClampReq {
                    gpu,
                    kind,
                    stride,
                    records,
                    args: args.clone(),
                    count: counter,
                }
            };

        // Two clamping requests sharing one argument copy, one with a
        // distinct count buffer, one with none, one with no records.
        let reqs = [
            req((2, 12, 1, Some(count.clone()))),
            req((0, 16, 1, None)),
            req((1, 20, 1, None)),
            req((2, 12, 0, None)),
        ];
        take_uav_barriers();
        issued();
        let outs = super::bound_chunk(&list, &reqs).unwrap();
        // One buffer for the chunk, a slice per request: distinct,
        // 16-byte-aligned offsets in request order.
        assert!(outs.iter().all(|(o, _)| key(o) == key(&outs[0].0)));
        assert_eq!(
            outs.iter().map(|(_, at)| *at).collect::<Vec<_>>(),
            [0, 16, 32, 64]
        );
        let buffers = [outs[0].0.clone()];
        super::done_chunk(&list, &buffers);
        // A second chunk, with records: it finds the card's budget common
        // and leaves it so (only the first chunk reverts it).
        let more = [req((2, 12, 1, None))];
        let outs2 = super::bound_chunk(&list, &more).unwrap();
        assert_eq!(outs2.len(), 1);
        let buffers2 = [outs2[0].0.clone()];
        super::done_chunk(&list, &buffers2);
        // The budget's key only after the chunks: asking before would spend
        // this recording's first (its zeroing) ahead of them.
        let budget_key = key(&list.budget().unwrap().0);
        // Scoped barriers only: the argument copy, the distinct count
        // buffer, then the budget once per clamping request (three clamping
        // requests in the first chunk, one in the second; the zero-record
        // one clamps nothing).
        assert_eq!(
            take_uav_barriers(),
            [
                key(&args),
                key(&count),
                budget_key,
                budget_key,
                budget_key,
                key(&args),
                budget_key,
            ]
        );
        // Every transition starts from the buffer's current state, and the
        // chunks' buffers and the budget all end common again: had the
        // second chunk reverted unordered access to common on a common
        // buffer, replay would fail as the debug layer does.
        let mut states: std::collections::HashMap<usize, u32> =
            [(key(&outs[0].0), 0), (key(&outs2[0].0), 0), (budget_key, 0)]
                .into_iter()
                .collect();
        replay(&issued(), &mut states).unwrap();
        assert!(states.values().all(|&s| s == 0));
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
