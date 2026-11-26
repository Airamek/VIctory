//! Title buffers resident in device-local VRAM (`D3D12_HEAP_TYPE_DEFAULT`)
//! with GPU-side Wave64 64 KiB page hashing, behind `VICTORY_CARD_BUFFERS`.
//!
//! Instead of keeping large buffers (4-75+ MiB) in system RAM (`L0`) over
//! PCIe and taking multi-megabyte CPU snapshots + CPU diffs every
//! submission:
//! - Shaders read and write `Shadow::buffer` directly in GDDR6 VRAM.
//! - `CardState::staging` (`L0` write-back, mapped as `Shadow::data`)
//!   receives CPU title-memory updates and simultaneously serves as the
//!   exact pre-submission snapshot (zero snapshot `memcpy` in
//!   `Shadow::writing`).
//! - Before a submission (`record_before`), only CPU-modified ranges are
//!   uploaded via `CopyBufferRegion` and re-hashed on the GPU (if the buffer
//!   has hashes).
//! - At the end of a submission (`record_after`), a Wave64 compute shader
//!   dispatches one 64-thread workgroup per declared 64 KiB piece in VRAM
//!   (64 coalesced 1024-byte bursts per page), writing a 4-byte dirty flag
//!   per page to a small readback buffer.
//! - After the submission has run on both queues (`finish_back`), ONLY
//!   pages flagged dirty are copied from VRAM to a readback buffer and
//!   merged into title memory + `staging`.
//!
//! Staging lifetime: a card shadow's staging is its own persistent mapping
//! (never pooled), held by the submission's [`CardPlan`] until that
//! submission finishes on both queues, so there is no staging double-buffer
//! question and no pool to coordinate: retirement reuses the submission's
//! own `retire` path. Placement stalls do not exist either: where VRAM (or
//! the engine) is unavailable the buffer stays a host copy, exactly as
//! before. The one serialization is GPU-side: at most one card submission
//! runs unfinished at a time (a submission with card writes settles the
//! previous card submission first), so the engine's reusable lists and
//! differs buffers are never reset or overwritten while in flight. If that
//! settle ever measures hot, the follow-up is per-plan differs slices
//! instead of one shared buffer, not weaker ordering.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::host::Host;
use crate::watch::PAGE;

/// Wave64 / 64-thread-per-page 64 KiB hashing compute shader (`cs_5_1`).
/// One workgroup (`SV_GroupID.x`) hashes one 64 KiB buffer-relative piece
/// using 64 threads reading coalesced 1024-byte strides (64 iterations per
/// page).
const CARD_COMPARE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // first 64 KiB piece index
    uint pad0;     // reserved (0)
    uint size;     // buffer size in bytes (multiple of 65536)
    uint pieces;   // number of pieces in this dispatch
    uint out_at;   // byte offset in `differs` for gid.x == 0
    uint compare;  // 0: store hashes; 1: compare only; 2: compare and store new hash
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);

groupshared uint4 s_acc[64];

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (gid.x >= pieces) return;
    uint piece = first + gid.x;
    uint lo = piece * 65536u;
    uint hi = min(lo + 65536u, size);

    uint4 h = uint4(
        0x9E3779B9u ^ (tid * 0x85EBCA6Bu),
        0x85EBCA6Bu ^ (tid * 0xC2B2AE35u),
        0xC2B2AE35u ^ (tid * 0x27D4EB2Fu),
        0x27D4EB2Fu ^ (tid * 0x165667B1u)
    );
    [loop]
    for (uint at = lo + tid * 16u; at < hi; at += 1024u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    s_acc[tid] = h;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            uint4 a = s_acc[tid];
            uint4 b = s_acc[tid + stride];
            uint4 m = (a ^ (b * uint4(0x27D4EB2Fu, 0x165667B1u, 0x9E3779B9u, 0x85EBCA6Bu)))
                    * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            s_acc[tid] = m.yzwx ^ (m >> 15);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        uint4 final_h = s_acc[0];
        if (compare != 0u) {
            uint4 prev = hashes.Load4(piece * 16u);
            differs.Store(out_at + gid.x * 4u, any(final_h != prev) ? 1u : 0u);
            if (compare == 2u) {
                hashes.Store4(piece * 16u, final_h);
            }
        } else {
            hashes.Store4(piece * 16u, final_h);
        }
    }
}
"#;

/// Bytes currently allocated in VRAM for card buffers (`[0]`) and their
/// hash tables (`[1]`).
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A card buffer with its merged CPU-written byte ranges, drained for
/// one submission's upload.
type DirtyCards = Vec<(Arc<CardState>, Vec<(usize, usize)>)>;

/// Card buffers with CPU-to-VRAM uploads pending (`dirty_up`).
static DIRTY_CARDS: Mutex<Vec<Weak<CardState>>> = Mutex::new(Vec::new());

/// Whether VRAM card buffers are placed (`VICTORY_CARD_BUFFERS=1`;
/// rollout switch, default off: no behavior change unless a run opts in).
/// Removal criteria: delete the gate (placement unconditional) once WARP
/// and hardware runs show the copy-back shrinking on the frame-rate line
/// with no correctness reports for four weeks; until then the default-off
/// gate keeps every title on host copies.
pub(crate) fn card_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CARD_BUFFERS"))
}

/// Smallest buffer placed in VRAM (below it the hash tables and dispatches
/// cost more than the CPU diffs they replace), and the most bytes of VRAM
/// card buffers and hash tables may hold together. Constants, not knobs:
/// the only switch is [`card_enabled`].
const MIN_BYTES: u64 = 4 << 20;
const MAX_TOTAL_BYTES: u64 = 2048 << 20;
const MAX_SINGLE_BYTES: u64 = 1 << 30;

pub(crate) fn hash_table_bytes(size: u64) -> u64 {
    ((size / PAGE as u64 + 1) * 16).next_multiple_of(256)
}

/// Per-`Shadow` state when a buffer lives in device-local VRAM (`DEFAULT`).
pub(crate) struct CardState {
    /// CPU-mapped staging mirror (`L0` write-back; `Shadow::data` points
    /// here).
    pub(crate) staging: ID3D12Resource,
    /// Device-local VRAM buffer (`DEFAULT`, `ALLOW_UNORDERED_ACCESS`; the
    /// same COM resource as `Shadow::buffer`).
    pub(crate) vram: ID3D12Resource,
    /// Buffer width in bytes (a multiple of `PAGE`).
    pub(crate) size: u64,
    /// Lazily created `DEFAULT` UAV buffer holding 16-byte hashes per 64
    /// KiB piece.
    hashes: Mutex<Option<ID3D12Resource>>,
    /// True once `hashes` exists and has been populated for the whole
    /// buffer.
    hashes_ready: AtomicBool,
    /// Buffer-relative byte ranges `[from, to)` written to `staging` by the
    /// CPU that must be copied to `vram` (and re-hashed if `hashes_ready`)
    /// before the next GPU submission.
    dirty_up: Mutex<Vec<(usize, usize)>>,
    /// Deduplication flag for [`DIRTY_CARDS`].
    queued: AtomicBool,
    /// Ourselves, for queueing uploads without an `Arc` at hand.
    this: Weak<CardState>,
}

// SAFETY: D3D12 resources are free-threaded; mutable fields are
// synchronized.
unsafe impl Send for CardState {}
// SAFETY: as above.
unsafe impl Sync for CardState {}

impl Drop for CardState {
    fn drop(&mut self) {
        CARD_BYTES[0].fetch_sub(self.size, Ordering::Relaxed);
        if self.hashes.get_mut().ok().and_then(Option::take).is_some() {
            CARD_BYTES[1].fetch_sub(hash_table_bytes(self.size), Ordering::Relaxed);
        }
        // Kept alive until the GPU work running now (or the next
        // submission's) has run, like any other retired resource.
        crate::host::retire(self.staging.clone());
        crate::host::retire(self.vram.clone());
    }
}

impl CardState {
    /// A VRAM card buffer for `size` bytes (a multiple of `PAGE`), if the
    /// gate is on and VRAM has room: else `None` and the buffer stays a host
    /// copy, exactly as before.
    pub(crate) fn try_new(
        host: &Host,
        staging: &ID3D12Resource,
        size: u64,
    ) -> Option<Arc<Self>> {
        if !card_enabled() {
            return None;
        }
        if !(MIN_BYTES..MAX_SINGLE_BYTES).contains(&size) || !size.is_multiple_of(PAGE as u64) {
            return None;
        }
        if !host.card_ready() {
            return None;
        }
        if let Some([(used_mb, budget_mb), _]) = crate::host::video_memory()
            && budget_mb > 0.0
            && used_mb * 1e6 + size as f64 > budget_mb * 1e6 * 75.0 / 100.0
        {
            return None;
        }
        if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > MAX_TOTAL_BYTES {
            CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
            return None;
        }
        Self::new_unchecked(host, staging, size, true)
    }

    /// A `CardState` directly (used by `try_new` and WARP tests).
    pub(crate) fn new_unchecked(
        host: &Host,
        staging: &ID3D12Resource,
        size: u64,
        counted: bool,
    ) -> Option<Arc<Self>> {
        let Some(vram) = host.device_buffer(size) else {
            if counted {
                CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
            }
            return None;
        };
        if !counted {
            CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
        }
        Some(Arc::new_cyclic(|me| Self {
            staging: staging.clone(),
            vram,
            size,
            hashes: Mutex::new(None),
            hashes_ready: AtomicBool::new(false),
            dirty_up: Mutex::new(Vec::new()),
            queued: AtomicBool::new(false),
            this: me.clone(),
        }))
    }

    /// Notes that the CPU copied new bytes into `staging[from..to]`.
    pub(crate) fn mark_dirty(&self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        {
            let mut dirty = self.dirty_up.lock().unwrap_or_else(|e| e.into_inner());
            dirty.push((from, to));
        }
        if !self.queued.swap(true, Ordering::AcqRel)
            && let Some(me) = self.this.upgrade()
        {
            DIRTY_CARDS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(Arc::downgrade(&me));
        }
    }

    /// Whether any card buffer has uploads pending (without draining them).
    pub(crate) fn has_dirty() -> bool {
        DIRTY_CARDS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|w| {
                w.upgrade().is_some_and(|c| {
                    !c.dirty_up.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
                })
            })
    }

    /// All card buffers with CPU-to-VRAM uploads pending, with their merged
    /// ranges.
    pub(crate) fn take_dirty_cards() -> DirtyCards {
        let mut dirty_cards: DirtyCards = Vec::new();
        let mut q = DIRTY_CARDS.lock().unwrap_or_else(|e| e.into_inner());
        for weak in q.drain(..) {
            if let Some(card) = weak.upgrade() {
                let ranges = card.take_dirty_up();
                if !ranges.is_empty() {
                    dirty_cards.push((card, ranges));
                }
            }
        }
        dirty_cards
    }

    /// Ensures the GPU hash table exists; returns it and whether it still
    /// needs a full initialization.
    fn ensure_hashes(&self, host: &Host) -> Option<(ID3D12Resource, bool)> {
        let mut h = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(res) = h.as_ref() {
            return Some((res.clone(), !self.hashes_ready.load(Ordering::Acquire)));
        }
        let bytes = hash_table_bytes(self.size);
        let res = host.device_buffer(bytes)?;
        crate::host::name(&res, || format!("card hashes {bytes:#x}"));
        CARD_BYTES[1].fetch_add(bytes, Ordering::Relaxed);
        *h = Some(res.clone());
        Some((res, true))
    }

    fn take_dirty_up(&self) -> Vec<(usize, usize)> {
        self.queued.store(false, Ordering::Release);
        let mut dirty = self.dirty_up.lock().unwrap_or_else(|e| e.into_inner());
        if dirty.is_empty() {
            return Vec::new();
        }
        let mut ranges = std::mem::take(&mut *dirty);
        merge_byte_ranges(&mut ranges);
        ranges
    }

    pub(crate) fn total_pieces(&self) -> u32 {
        (self.size / PAGE as u64) as u32
    }
}

/// Merges overlapping or adjacent `[from, to)` ranges in place.
fn merge_byte_ranges(ranges: &mut Vec<(usize, usize)>) {
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable();
    let mut out = 0usize;
    for i in 1..ranges.len() {
        let (a, b) = ranges[i];
        if a <= ranges[out].1 {
            ranges[out].1 = ranges[out].1.max(b);
        } else {
            out += 1;
            ranges[out] = (a, b);
        }
    }
    ranges.truncate(out + 1);
}

/// Converts byte ranges `[from, to)` into sorted, merged 64 KiB piece runs
/// `(first_piece, count)`.
fn byte_ranges_to_piece_runs(ranges: &[(usize, usize)], max_pieces: u32) -> Vec<(u32, u32)> {
    if ranges.is_empty() || max_pieces == 0 {
        return Vec::new();
    }
    let mut spans: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges {
        if from >= to {
            continue;
        }
        let first = (from / PAGE) as u32;
        let end = (to.div_ceil(PAGE) as u32).min(max_pieces);
        if first < end {
            spans.push((first, end));
        }
    }
    if spans.is_empty() {
        return Vec::new();
    }
    spans.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(spans.len());
    for (a, b) in spans {
        if let Some(last) = out.last_mut()
            && a <= last.0 + last.1
        {
            let end = (last.0 + last.1).max(b);
            last.1 = end - last.0;
        } else {
            out.push((a, b - a));
        }
    }
    out
}

/// Per-submission plan built in `record_before` and consumed in
/// `record_after` + `finish_back`.
pub(crate) struct CardPlan {
    entries: Vec<CardWriteEntry>,
    total_pieces: u32,
    /// The `differs` readback `record_after` copied into, and its bytes:
    /// read back in `finish_back` (the engine's buffer may have grown
    /// since, for a later submission, retiring this one).
    differs: Option<(ID3D12Resource, u64)>,
}

struct CardWriteEntry {
    shadow: Arc<crate::shadow::Shadow>,
    card: Arc<CardState>,
    hashes: ID3D12Resource,
    /// Original declared title ranges `(address, len)` for clearing
    /// `pending`.
    declared: Vec<(usize, usize)>,
    /// Merged 64 KiB buffer-relative piece runs `(first_piece, count)`.
    runs: Vec<(u32, u32)>,
    /// First dword index in `differs` for this entry.
    out_start: u32,
}

impl CardPlan {
    fn empty() -> Self {
        CardPlan {
            entries: Vec::new(),
            total_pieces: 0,
            differs: None,
        }
    }

    pub(crate) fn has_writes(&self) -> bool {
        self.total_pieces > 0
    }

    pub(crate) fn clear_pending(self) {
        for entry in &self.entries {
            entry.shadow.finish_card_pending(&entry.declared);
        }
    }
}

/// Per-`Host` reusable GPU pipelines, command lists, and scratch buffers
/// for VRAM card buffer uploads, Wave64 hashing, and dirty-page fetch.
pub(crate) struct CardEngine {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    before_alloc: ID3D12CommandAllocator,
    before_list: ID3D12GraphicsCommandList,
    after_alloc: ID3D12CommandAllocator,
    after_list: ID3D12GraphicsCommandList,
    fetch_alloc: ID3D12CommandAllocator,
    fetch_list: ID3D12GraphicsCommandList,
    differs_gpu: ID3D12Resource,
    differs_rb: (ID3D12Resource, u64),
    dummy_uav: ID3D12Resource,
    fetch_rb: Option<(ID3D12Resource, u64)>,
    fence: ID3D12Fence,
    fence_value: u64,
    event: HANDLE,
}

// SAFETY: accessed only under the submission lock or the engine mutex, one
// at a time; D3D12 interfaces are free-threaded.
unsafe impl Send for CardEngine {}

impl Drop for CardEngine {
    fn drop(&mut self) {
        // SAFETY: our own handle, no longer waited on.
        let _ = unsafe { CloseHandle(self.event) };
    }
}

impl CardEngine {
    pub(crate) fn new(host: &Host) -> windows::core::Result<Self> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: valid HLSL source and entry point.
        unsafe {
            D3DCompile(
                CARD_COMPARE_HLSL.as_ptr().cast(),
                CARD_COMPARE_HLSL.len(),
                windows::core::s!("card_compare_wave64"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: valid compiled DXBC blob from D3DCompile.
        let bytecode = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: bytecode embeds `[RootSignature(RS)]`.
        let root_sig: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, bytecode)? };
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytecode.as_ptr().cast(),
                BytecodeLength: bytecode.len(),
            },
            ..Default::default()
        };
        // SAFETY: valid compute pipeline description.
        let pso: ID3D12PipelineState =
            unsafe { host.device.CreateComputePipelineState(&pso_desc)? };

        let make_pair = || -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
            // SAFETY: plain direct command allocator and list creation.
            unsafe {
                let alloc: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &alloc,
                    None,
                )?;
                list.Close()?;
                Ok((alloc, list))
            }
        };
        let (before_alloc, before_list) = make_pair()?;
        let (after_alloc, after_list) = make_pair()?;
        let (fetch_alloc, fetch_list) = make_pair()?;

        let init_diff_bytes = 256 << 10; // 65,536 pieces (4 GiB of ranges)
        let differs_gpu = host
            .device_buffer(init_diff_bytes)
            .ok_or_else(windows::core::Error::empty)?;
        let differs_rb = host
            .readback_buffer(init_diff_bytes)
            .ok_or_else(windows::core::Error::empty)?;
        let dummy_uav = host
            .device_buffer(256)
            .ok_or_else(windows::core::Error::empty)?;
        crate::host::name(&differs_gpu, || "card differs gpu".into());
        crate::host::name(&differs_rb, || "card differs readback".into());
        crate::host::name(&dummy_uav, || "card dummy uav".into());
        // SAFETY: plain D3D12 object creation.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        // SAFETY: plain event creation; closed in `Drop`.
        let event = unsafe { CreateEventW(None, false, false, None)? };

        Ok(Self {
            root_sig,
            pso,
            before_alloc,
            before_list,
            after_alloc,
            after_list,
            fetch_alloc,
            fetch_list,
            differs_gpu,
            differs_rb: (differs_rb, init_diff_bytes),
            dummy_uav,
            fetch_rb: None,
            fence,
            fence_value: 0,
            event,
        })
    }

    fn ensure_differs_capacity(&mut self, host: &Host, needed_bytes: u64) -> bool {
        if needed_bytes <= self.differs_rb.1 {
            return true;
        }
        let size = needed_bytes.next_multiple_of(256 << 10);
        let (Some(gpu), Some(rb)) = (host.device_buffer(size), host.readback_buffer(size)) else {
            return false;
        };
        let old_gpu = std::mem::replace(&mut self.differs_gpu, gpu);
        let old_rb = std::mem::replace(&mut self.differs_rb, (rb, size));
        crate::host::retire(old_gpu);
        crate::host::retire(old_rb.0);
        true
    }

    fn ensure_fetch_capacity(&mut self, host: &Host, needed_bytes: u64) -> Option<ID3D12Resource> {
        if let Some((ref res, cap)) = self.fetch_rb
            && needed_bytes <= cap
        {
            return Some(res.clone());
        }
        let size = needed_bytes.next_multiple_of(8 << 20);
        let rb = host.readback_buffer(size)?;
        crate::host::name(&rb, || format!("card fetch readback {size:#x}"));
        if let Some((old, _)) = self.fetch_rb.replace((rb.clone(), size)) {
            crate::host::retire(old);
        }
        Some(rb)
    }

    /// Runs one command list on the direct queue at once and waits for it:
    /// the completion-time fetch runs outside any submission (the caller
    /// may hold the submission lock, so this never touches it).
    fn run_and_wait(&mut self, host: &Host, cmd: ID3D12CommandList) {
        self.fence_value += 1;
        let value = self.fence_value;
        // SAFETY: a closed list; our fence and event.
        let waited = unsafe {
            host.direct_queue().ExecuteCommandLists(&[Some(cmd)]);
            host.direct_queue()
                .Signal(&self.fence, value)
                .and_then(|()| self.fence.SetEventOnCompletion(value, self.event))
        };
        match waited {
            Ok(()) => {
                // SAFETY: our own event, signalled above.
                unsafe { WaitForSingleObject(self.event, INFINITE) };
            }
            Err(e) => tracing::warn!(error = %e, "card fetch submit failed"),
        }
    }

    /// Dispatches `CARD_COMPARE` for `count` contiguous 64 KiB pieces
    /// starting at `first`.
    #[allow(clippy::too_many_arguments)] // the shader's own root constants, views and dispatch shape
    unsafe fn dispatch_hash(
        &self,
        list: &ID3D12GraphicsCommandList,
        vram: &ID3D12Resource,
        hashes: &ID3D12Resource,
        differs: &ID3D12Resource,
        first: u32,
        size: u32,
        count: u32,
        out_at_bytes: u32,
        compare: u32,
    ) {
        if count == 0 {
            return;
        }
        let constants = [first, 0u32, size, count, out_at_bytes, compare];
        // SAFETY: caller ensures `list` is open with `root_sig` and `pso`
        // bound and all 3 buffers are live with valid states.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, vram.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, differs.GetGPUVirtualAddress());
            list.Dispatch(count, 1, 1);
        }
    }

    /// Uploads all CPU-modified `Card` ranges to VRAM, initializes hashes
    /// for any `Card` shadow about to be written by shaders, and re-hashes
    /// uploaded pieces of hashed shadows. Returns the before command list
    /// (runs first in the submission) and the plan. On a recording failure
    /// the drained uploads are re-queued, initialized hashes forgotten, and
    /// an empty plan returned: the submission does no card work rather than
    /// half of it.
    pub(crate) fn record_before(
        &mut self,
        host: &Host,
        writes: &[(Arc<crate::shadow::Shadow>, usize, usize)],
        dirty_cards: DirtyCards,
    ) -> (Option<ID3D12CommandList>, CardPlan) {
        // Group declared `Step::GpuWrite` ranges by `Card` shadow and
        // ensure each written card shadow has its hash table allocated.
        let mut entries: Vec<CardWriteEntry> = Vec::new();
        let mut need_full_hash: Vec<(Arc<CardState>, ID3D12Resource)> = Vec::new();
        let mut total_pieces = 0u32;

        for (shadow, address, len) in writes {
            let Some(card) = shadow.card_state() else {
                continue;
            };
            if let Some(entry) = entries.iter_mut().find(|e| Arc::ptr_eq(&e.shadow, shadow)) {
                entry.declared.push((*address, *len));
            } else if let Some((hashes, full_init)) = card.ensure_hashes(host) {
                if full_init {
                    need_full_hash.push((card.clone(), hashes.clone()));
                }
                entries.push(CardWriteEntry {
                    shadow: shadow.clone(),
                    card: card.clone(),
                    hashes,
                    declared: vec![(*address, *len)],
                    runs: Vec::new(),
                    out_start: 0,
                });
            }
        }

        for entry in &mut entries {
            let mut byte_ranges = Vec::with_capacity(entry.declared.len());
            for &(addr, len) in &entry.declared {
                let (from, to) = entry.shadow.inside(addr, len);
                if from < to {
                    byte_ranges.push((from, to));
                }
            }
            entry.runs = byte_ranges_to_piece_runs(&byte_ranges, entry.card.total_pieces());
            entry.out_start = total_pieces;
            total_pieces += entry.runs.iter().map(|&(_, c)| c).sum::<u32>();
        }

        // Also check which dirty-uploaded shadows already have hashes (so
        // the re-hash below covers their newly uploaded pieces).
        #[allow(clippy::type_complexity)]
        let mut rehash_uploads: Vec<(Arc<CardState>, ID3D12Resource, Vec<(u32, u32)>)> =
            Vec::new();
        for (card, ranges) in &dirty_cards {
            // Skip partial rehash if this card is already getting a full
            // hash below.
            if need_full_hash.iter().any(|(c, _)| Arc::ptr_eq(c, card)) {
                continue;
            }
            if card.hashes_ready.load(Ordering::Acquire)
                && let Some(hashes) = card
                    .hashes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
            {
                let runs = byte_ranges_to_piece_runs(ranges, card.total_pieces());
                if !runs.is_empty() {
                    rehash_uploads.push((card.clone(), hashes, runs));
                }
            }
        }

        let plan = CardPlan {
            entries,
            total_pieces,
            differs: None,
        };

        if dirty_cards.is_empty() && need_full_hash.is_empty() {
            return (None, plan);
        }

        // SAFETY: at most one card submission runs un finished at a time
        // (the submission settled the previous card one first), so the
        // allocator and list are idle.
        let reset = unsafe {
            self.before_alloc.Reset().is_ok()
                && self.before_list.Reset(&self.before_alloc, None).is_ok()
        };
        if !reset {
            self.abandon(plan, dirty_cards, need_full_hash);
            return (None, CardPlan::empty());
        }
        let list = &self.before_list;

        // Step A: Copy dirty staging ranges -> VRAM.
        for (card, ranges) in &dirty_cards {
            for &(from, to) in ranges {
                // SAFETY: both buffers cover `[from, to)`; implicit COMMON
                // promotion.
                unsafe {
                    list.CopyBufferRegion(
                        &card.vram,
                        from as u64,
                        &card.staging,
                        from as u64,
                        (to - from) as u64,
                    );
                }
            }
        }

        // Step B: Transition uploaded VRAM buffers COPY_DEST ->
        // NON_PIXEL_SHADER_RESOURCE (or COMMON if no hashing follows).
        let mut to_srv = Vec::new();
        let mut to_common = Vec::new();
        for (card, _) in &dirty_cards {
            let will_hash = need_full_hash.iter().any(|(c, _)| Arc::ptr_eq(c, card))
                || rehash_uploads.iter().any(|(c, _, _)| Arc::ptr_eq(c, card));
            let after = if will_hash {
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            } else {
                D3D12_RESOURCE_STATE_COMMON
            };
            to_srv.push(crate::host::transition(
                &card.vram,
                D3D12_RESOURCE_STATE_COPY_DEST,
                after,
            ));
            if will_hash {
                to_common.push(crate::host::transition(
                    &card.vram,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
            }
        }
        if !to_srv.is_empty() {
            // SAFETY: valid resource transitions.
            unsafe { list.ResourceBarrier(&to_srv) };
        }

        // Step C: Dispatch Wave64 CARD_COMPARE (compare = 0) to store
        // hashes.
        if !need_full_hash.is_empty() || !rehash_uploads.is_empty() {
            // SAFETY: live root signature and PSO.
            unsafe {
                list.SetComputeRootSignature(&self.root_sig);
                list.SetPipelineState(&self.pso);
            }
            for (card, hashes) in &need_full_hash {
                // SAFETY: buffer is readable as SRV (either transitioned
                // above or COMMON).
                unsafe {
                    self.dispatch_hash(
                        list,
                        &card.vram,
                        hashes,
                        &self.dummy_uav,
                        0,
                        card.size as u32,
                        card.total_pieces(),
                        0,
                        0,
                    );
                }
                card.hashes_ready.store(true, Ordering::Release);
            }
            for (card, hashes, runs) in &rehash_uploads {
                for &(first, count) in runs {
                    // SAFETY: buffer is in NON_PIXEL_SHADER_RESOURCE.
                    unsafe {
                        self.dispatch_hash(
                            list,
                            &card.vram,
                            hashes,
                            &self.dummy_uav,
                            first,
                            card.size as u32,
                            count,
                            0,
                            0,
                        );
                    }
                }
            }
            // Flush hash UAV writes and return transitioned VRAM buffers to
            // COMMON.
            to_common.push(uav_barrier());
        }

        if !to_common.is_empty() {
            // SAFETY: valid barriers.
            unsafe { list.ResourceBarrier(&to_common) };
        }

        // SAFETY: closing the recorded list.
        if unsafe { list.Close() }.is_err() {
            self.abandon(plan, dirty_cards, need_full_hash);
            return (None, CardPlan::empty());
        }
        (list.cast().ok(), plan)
    }

    /// Undoes a failed `record_before`: drained uploads are re-queued for
    /// the next submission, never-initialized hashes forgotten, and declared
    /// ranges released (their GPU writes go unobserved this submission, and
    /// the next submission's compare catches them if declared again).
    fn abandon(
        &self,
        plan: CardPlan,
        dirty_cards: DirtyCards,
        need_full_hash: Vec<(Arc<CardState>, ID3D12Resource)>,
    ) {
        for (card, ranges) in &dirty_cards {
            for &(from, to) in ranges {
                card.mark_dirty(from, to);
            }
        }
        for (card, _) in &need_full_hash {
            card.hashes_ready.store(false, Ordering::Release);
        }
        plan.clear_pending();
    }

    /// Records the post-submission GPU hash compare (`compare = 2`) and
    /// copies the tiny `differs` array to `differs_rb` (runs in the same
    /// submission, under its fence).
    pub(crate) fn record_after(
        &mut self,
        host: &Host,
        plan: &mut CardPlan,
    ) -> Option<ID3D12CommandList> {
        if plan.total_pieces == 0 {
            return None;
        }
        let needed_bytes = u64::from(plan.total_pieces) * 4;
        if !self.ensure_differs_capacity(host, needed_bytes) {
            return None;
        }
        // SAFETY: at most one card submission runs un finished at a time,
        // so the allocator and list are idle.
        unsafe {
            self.after_alloc.Reset().ok()?;
            self.after_list.Reset(&self.after_alloc, None).ok()?;
            // Global UAV barrier ensures all shader writes from the runs
            // are visible.
            self.after_list.ResourceBarrier(&[uav_barrier()]);
            self.after_list.SetComputeRootSignature(&self.root_sig);
            self.after_list.SetPipelineState(&self.pso);

            for entry in &plan.entries {
                let mut out_idx = entry.out_start;
                for &(first, count) in &entry.runs {
                    self.dispatch_hash(
                        &self.after_list,
                        &entry.card.vram,
                        &entry.hashes,
                        &self.differs_gpu,
                        first,
                        entry.card.size as u32,
                        count,
                        out_idx * 4,
                        2, // compare against previous hash AND store updated hash
                    );
                    out_idx += count;
                }
            }

            let to_copy = [crate::host::transition(
                &self.differs_gpu,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )];
            self.after_list.ResourceBarrier(&to_copy);

            self.after_list.CopyBufferRegion(
                &self.differs_rb.0,
                0,
                &self.differs_gpu,
                0,
                needed_bytes,
            );

            let back_common = [crate::host::transition(
                &self.differs_gpu,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )];
            self.after_list.ResourceBarrier(&back_common);

            self.after_list.Close().ok()?;
            plan.differs = Some((self.differs_rb.0.clone(), self.differs_rb.1));
            self.after_list.cast().ok()
        }
    }

    /// Reads the `differs` results after the submission ran on both queues,
    /// fetches ONLY the 64 KiB pages the GPU actually modified, and merges
    /// them into title memory + `staging`. Returns total bytes written to
    /// title memory.
    pub(crate) fn finish_back(&mut self, host: &Host, plan: CardPlan) -> usize {
        if plan.entries.is_empty() {
            return 0;
        }
        let Some((differs_rb, _)) = &plan.differs else {
            plan.clear_pending();
            return 0;
        };
        let bytes = u64::from(plan.total_pieces) * 4;
        // SAFETY: `differs_rb` is a readback buffer of at least `bytes`
        // bytes; the submission that filled `[0, bytes)` has run on both
        // queues.
        // SAFETY: `differs_rb` is a readback buffer of at least `bytes`
        // bytes; the submission that filled `[0, bytes)` has run on both
        // queues.
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: bytes as usize,
        };
        // SAFETY: as above (the map itself borrows the live buffer).
        if unsafe { differs_rb.Map(0, Some(&range), Some(&mut data)) }.is_err() {
            plan.clear_pending();
            return 0;
        }
        // SAFETY: mapped above, `total_pieces` dwords.
        let differs: Vec<u32> = unsafe {
            std::slice::from_raw_parts(data.cast::<u32>(), plan.total_pieces as usize)
        }
        .to_vec();
        // SAFETY: the same buffer, unmapped once.
        unsafe { differs_rb.Unmap(0, None) };

        // Collect coalesced runs of dirty 64 KiB pieces per entry.
        struct DirtyRun {
            entry_idx: usize,
            first_piece: u32,
            count: u32,
            rb_offset: u64,
        }
        let mut dirty_runs: Vec<DirtyRun> = Vec::new();
        let mut total_bytes = 0u64;

        for (entry_idx, entry) in plan.entries.iter().enumerate() {
            let mut out_idx = entry.out_start as usize;
            let mut cur: Option<(u32, u32)> = None;
            for &(first, count) in &entry.runs {
                for k in 0..count {
                    let piece = first + k;
                    if differs.get(out_idx).copied().unwrap_or(0) != 0 {
                        match &mut cur {
                            Some((s, c)) if *s + *c == piece => *c += 1,
                            Some((s, c)) => {
                                let bytes = u64::from(*c) * PAGE as u64;
                                dirty_runs.push(DirtyRun {
                                    entry_idx,
                                    first_piece: *s,
                                    count: *c,
                                    rb_offset: total_bytes,
                                });
                                total_bytes += bytes;
                                cur = Some((piece, 1));
                            }
                            None => cur = Some((piece, 1)),
                        }
                    }
                    out_idx += 1;
                }
            }
            if let Some((s, c)) = cur {
                let bytes = u64::from(c) * PAGE as u64;
                dirty_runs.push(DirtyRun {
                    entry_idx,
                    first_piece: s,
                    count: c,
                    rb_offset: total_bytes,
                });
                total_bytes += bytes;
            }
        }

        let mut written = 0usize;
        if total_bytes > 0
            && let Some(fetch_res) = self.ensure_fetch_capacity(host, total_bytes)
        {
            // SAFETY: allocator and list are idle (at most one card
            // submission runs unfinished at a time).
            let ok = unsafe {
                self.fetch_alloc.Reset().is_ok()
                    && self.fetch_list.Reset(&self.fetch_alloc, None).is_ok()
            };
            if ok {
                for run in &dirty_runs {
                    let entry = &plan.entries[run.entry_idx];
                    // SAFETY: `vram` is in COMMON (promotes to COPY_SOURCE,
                    // decays to COMMON).
                    unsafe {
                        self.fetch_list.CopyBufferRegion(
                            &fetch_res,
                            run.rb_offset,
                            &entry.card.vram,
                            u64::from(run.first_piece) * PAGE as u64,
                            u64::from(run.count) * PAGE as u64,
                        );
                    }
                }
                // SAFETY: the recorded list closes once.
                if unsafe { self.fetch_list.Close() }.is_ok()
                    && let Ok(cmd) = self.fetch_list.cast::<ID3D12CommandList>()
                {
                    self.run_and_wait(host, cmd);

                    // SAFETY: the fetch filled the readback buffer, which
                    // just ran.
                    let mut fetch_data = std::ptr::null_mut();
                    let fetch_range = D3D12_RANGE {
                        Begin: 0,
                        End: total_bytes as usize,
                    };
                    // SAFETY: `fetch_res` is a readback buffer of at
                    // least `total_bytes` bytes; the fetch just ran.
                    if unsafe { fetch_res.Map(0, Some(&fetch_range), Some(&mut fetch_data)) }.is_ok()
                    {
                        let fetch_base = fetch_data as usize;
                        // Group dirty runs by entry so distinct shadows
                        // merge in parallel.
                        #[allow(clippy::type_complexity)]
                        let mut per_entry: Vec<(usize, Vec<(u32, u32, usize)>)> = Vec::new();
                        for run in &dirty_runs {
                            let ptr = fetch_base + run.rb_offset as usize;
                            if let Some(g) = per_entry.iter_mut().find(|(i, _)| *i == run.entry_idx) {
                                g.1.push((run.first_piece, run.count, ptr));
                            } else {
                                per_entry.push((run.entry_idx, vec![(run.first_piece, run.count, ptr)]));
                            }
                        }
                        let total_written = AtomicU64::new(0);
                        // At most 4 threads (the title's threads need the
                        // rest); one entry's ranges merge in order on one
                        // thread, disjoint entries share nothing but locks.
                        if per_entry.len() < 2 {
                            for (entry_idx, runs) in &per_entry {
                                let entry = &plan.entries[*entry_idx];
                                total_written.fetch_add(
                                    entry.shadow.merge_card_pieces(&entry.card, runs) as u64,
                                    Ordering::Relaxed,
                                );
                            }
                        } else {
                            std::thread::scope(|s| {
                                let chunk =
                                    per_entry.len().div_ceil(4.min(per_entry.len()));
                                let mut parts = per_entry.chunks(chunk);
                                // The finishing thread takes the first chunk
                                // itself.
                                let mine = parts.next();
                                let mut handles = Vec::new();
                                // Shared refs, so the `move` closures below
                                // copy them instead of moving the locals.
                                let plan_ref = &plan;
                                let written_ref = &total_written;
                                for part in parts {
                                    handles.push(s.spawn(move || {
                                        let mut subtotal = 0u64;
                                        for (entry_idx, runs) in part {
                                            let entry = &plan_ref.entries[*entry_idx];
                                            subtotal += entry.shadow.merge_card_pieces(
                                                &entry.card,
                                                runs,
                                            ) as u64;
                                        }
                                        written_ref.fetch_add(subtotal, Ordering::Relaxed);
                                    }));
                                }
                                if let Some(part) = mine {
                                    let mut subtotal = 0u64;
                                    for (entry_idx, runs) in part {
                                        let entry = &plan.entries[*entry_idx];
                                        subtotal += entry
                                            .shadow
                                            .merge_card_pieces(&entry.card, runs)
                                            as u64;
                                    }
                                    total_written.fetch_add(subtotal, Ordering::Relaxed);
                                }
                                for h in handles {
                                    h.join().expect("a card merge thread panicked");
                                }
                            });
                        }
                        written = total_written.into_inner() as usize;
                        // SAFETY: the same buffer, unmapped once.
                        unsafe { fetch_res.Unmap(0, None) };
                    }
                }
            }
        }

        // Clear all declared pending ranges on each card shadow.
        plan.clear_pending();
        written
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_helpers_merge_and_convert_pieces() {
        let mut r = vec![(100, 200), (50, 120), (300, 400), (200, 300)];
        merge_byte_ranges(&mut r);
        assert_eq!(r, vec![(50, 400)]);

        let runs = byte_ranges_to_piece_runs(
            &[(0, 10), (PAGE - 1, PAGE + 1), (3 * PAGE, 4 * PAGE)],
            8,
        );
        assert_eq!(runs, vec![(0, 2), (3, 1)]);
    }

    /// The merge of GPU-modified bytes: dirty spans go to the title,
    /// staging follows, concurrent CPU writes are reported for re-upload.
    #[test]
    fn card_merge_reports_dirty_spans_and_cpu_writes() {
        let n = 32usize;
        let mut title = vec![1u8; n];
        let mut staging = vec![1u8; n];
        let mut gpu = vec![1u8; n];
        gpu[4..8].copy_from_slice(&[9, 9, 9, 9]);
        title[16..20].copy_from_slice(&[7, 7, 7, 7]);
        let (span, cpu) = crate::shadow::merge_gpu_writes_card_for_test(
            &mut title,
            &mut staging,
            &gpu,
        );
        assert_eq!(span, Some((4, 8)));
        assert!(cpu);
        assert_eq!(&title[4..8], &[9, 9, 9, 9]);
        assert_eq!(&staging[4..8], &[9, 9, 9, 9]);
        assert_eq!(&staging[16..20], &[7, 7, 7, 7]);
    }

    /// End to end on WARP: initial upload + Wave64 hash, GPU-only writes
    /// detected per page, concurrent CPU writes merged and re-uploaded.
    /// Early-returns with no GPU here (the gate defaults off, so CI without
    /// WARP exercises only the pure tests above).
    #[test]
    fn warp_card_roundtrip_upload_wave64_hash_and_concurrent_merge() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut engine = CardEngine::new(host).expect("CardEngine should build on WARP");
        let pages = 4usize;
        let size = (pages * PAGE) as u64;
        let mut title = vec![0u8; size as usize];
        for (i, b) in title.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31) ^ (i >> 8)) & 0xFF) as u8;
        }
        let base = title.as_ptr() as usize;

        let (staging, staging_ptr) = host.shared_buffer(size).expect("staging buffer");
        let card = CardState::new_unchecked(host, &staging, size, false).expect("card buffer");
        let shadow =
            crate::shadow::Shadow::new_card_test(card.clone(), staging_ptr, base as u64, size);
        assert!(shadow.is_card());

        // Phase 1: Initial CPU copy -> upload to VRAM + Wave64 hash, no GPU writes.
        shadow.copy_for_test(base, base + size as usize);
        assert!(CardState::has_dirty());
        shadow.writing(base, size as usize, false);
        let dirty = CardState::take_dirty_cards();
        assert!(!CardState::has_dirty());
        let (before, plan) = engine.record_before(host, &[(shadow.clone(), base, size as usize)], dirty);
        assert!(before.is_some(), "initial upload + hash must emit before list");
        assert_eq!(plan.total_pieces, pages as u32);
        let mut plan = plan;
        let after = engine.record_after(host, &mut plan).expect("after list");
        host.execute_and_wait_for_test(before.unwrap());
        host.execute_and_wait_for_test(after);

        let differs_1 = read_differs(&engine, pages);
        assert_eq!(differs_1, vec![0, 0, 0, 0]);
        let written_1 = engine.finish_back(host, plan);
        assert_eq!(written_1, 0);
        shadow.finish_card_pending(&[(base, size as usize)]);

        // Phase 2: GPU writes 4 bytes on page 2 while the CPU concurrently
        // writes 4 different bytes on the same page 2.
        let (patch_buf, patch_ptr) = host.shared_buffer(256).expect("patch buffer");
        let gpu_bytes = [0xDEu8, 0xAD, 0xBE, 0xEF];
        // SAFETY: our mapped patch buffer.
        unsafe { std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), patch_ptr, 4) };
        // SAFETY: plain direct command allocator creation.
        let patch_alloc: ID3D12CommandAllocator = unsafe {
            host.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .expect("alloc");
        // SAFETY: plain direct command list creation.
        let patch_list: ID3D12GraphicsCommandList = unsafe {
            host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &patch_alloc,
                None,
            )
        }
        .expect("list");
        // SAFETY: an open list over live buffers.
        unsafe {
            patch_list.CopyBufferRegion(
                &card.vram,
                (2 * PAGE + 128) as u64,
                &patch_buf,
                0,
                4,
            );
            patch_list.Close().expect("close");
        }
        let patch_cmd: ID3D12CommandList = patch_list.cast().expect("cast");
        // Execute after record_before while the alloc stays alive.
        shadow.writing(base, size as usize, false);
        let dirty = CardState::take_dirty_cards();
        let (before2, plan2) =
            engine.record_before(host, &[(shadow.clone(), base, size as usize)], dirty);
        assert!(before2.is_none(), "no CPU dirty uploads expected in phase 2");
        let mut plan2 = plan2;
        let after2 = engine.record_after(host, &mut plan2).expect("after2");
        // Simulate a concurrent CPU modification on page 2 while the GPU runs.
        title[2 * PAGE + 256..2 * PAGE + 260].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        host.execute_and_wait_for_test(patch_cmd);
        drop((patch_list, patch_alloc));
        host.execute_and_wait_for_test(after2);

        let differs_2 = read_differs(&engine, pages);
        assert_eq!(differs_2, vec![0, 0, 1, 0], "only page 2 should differ");
        let written_2 = engine.finish_back(host, plan2);
        assert_eq!(written_2, 4, "tight dirty span should write exactly 4 GPU bytes");
        assert_eq!(&title[2 * PAGE + 128..2 * PAGE + 132], &gpu_bytes);
        assert_eq!(&title[2 * PAGE + 256..2 * PAGE + 260], &[0x11, 0x22, 0x33, 0x44]);

        // Phase 3: Next submission uploads the merged concurrent CPU
        // write on page 2 back to VRAM and re-hashes it.
        shadow.writing(base, size as usize, false);
        let dirty = CardState::take_dirty_cards();
        let (before3, plan3) =
            engine.record_before(host, &[(shadow.clone(), base, size as usize)], dirty);
        assert!(
            before3.is_some(),
            "concurrent CPU write must queue page 2 for re-upload and re-hash"
        );
        let mut plan3 = plan3;
        let after3 = engine.record_after(host, &mut plan3).expect("after3");
        host.execute_and_wait_for_test(before3.unwrap());
        host.execute_and_wait_for_test(after3);
        let written_3 = engine.finish_back(host, plan3);
        assert_eq!(written_3, 0, "after re-upload + re-hash, 0 bytes should differ");

        // Read back page 2 directly from VRAM and verify both GPU and CPU writes exist there.
        let rb = host.readback_buffer(PAGE as u64).expect("verify rb");
        // SAFETY: plain direct command allocator creation.
        let v_alloc: ID3D12CommandAllocator = unsafe {
            host.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .expect("v_alloc");
        // SAFETY: plain direct command list creation.
        let v_list: ID3D12GraphicsCommandList = unsafe {
            host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &v_alloc,
                None,
            )
        }
        .expect("v_list");
        // SAFETY: an open list over live buffers.
        unsafe {
            v_list.CopyBufferRegion(&rb, 0, &card.vram, (2 * PAGE) as u64, PAGE as u64);
            v_list.Close().expect("v_close");
        }
        let v_cmd: ID3D12CommandList = v_list.cast().expect("v_cast");
        host.execute_and_wait_for_test(v_cmd);
        drop((v_list, v_alloc));
        let mut verify = std::ptr::null_mut();
        let v_range = D3D12_RANGE {
            Begin: 0,
            End: PAGE,
        };
        // SAFETY: a readback buffer of `PAGE` bytes; the copy just ran.
        unsafe { rb.Map(0, Some(&v_range), Some(&mut verify)) }.expect("map");
        // SAFETY: mapped above, `PAGE` bytes.
        let vram_page2 = unsafe { std::slice::from_raw_parts(verify as *const u8, PAGE) };
        assert_eq!(&vram_page2[128..132], &gpu_bytes);
        assert_eq!(&vram_page2[256..260], &[0x11, 0x22, 0x33, 0x44]);
        // SAFETY: the same buffer, unmapped once.
        unsafe { rb.Unmap(0, None) };
    }

    /// Reads the first `n` differs dwords (tests).
    fn read_differs(engine: &CardEngine, n: usize) -> Vec<u32> {
        let (rb, _) = &engine.differs_rb;
        // SAFETY: a readback buffer; the submission that filled it finished
        // (every probe above waited).
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: n * 4,
        };
        // SAFETY: as above (the map itself borrows the live buffer).
        unsafe { rb.Map(0, Some(&range), Some(&mut data)) }.expect("differs map");
        // SAFETY: mapped above, `n` dwords.
        let out: Vec<u32> = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), n) }.to_vec();
        // SAFETY: the same buffer, unmapped once.
        unsafe { rb.Unmap(0, None) };
        out
    }
}
