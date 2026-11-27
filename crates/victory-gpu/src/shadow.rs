//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

#[cfg(test)]
pub(crate) mod snapshot_tests;

fn merge_writeback(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut changed = 0;
    for ((title_byte, host_byte), old_byte) in title.iter_mut().zip(host).zip(old) {
        let mask = 0u8.wrapping_sub(u8::from(*host_byte != *old_byte));
        let delta = (*title_byte ^ *host_byte) & mask;
        *title_byte ^= delta;
        *host_byte = *title_byte;
        changed |= delta;
    }
    changed != 0
}

#[test]
fn writeback_merge_matches_bytewise_oracle() {
    for len in [0, 1, 63, 64, 65, 193, 65_536] {
        let old = vec![0x11; len];
        for stride in [1, 5, 67, len + 1] {
            let mut title = old.clone();
            let mut host = old.clone();
            for index in (0..len).step_by(3) {
                title[index] = 0x22;
            }
            for index in (0..len).step_by(stride) {
                host[index] = if index % 2 == 0 { 0x22 } else { 0x33 };
            }
            let expected: Vec<_> = host
                .iter()
                .zip(&old)
                .zip(&title)
                .map(|((&gpu, &previous), &cpu)| if gpu == previous { cpu } else { gpu })
                .collect();
            let changed = expected != title;
            assert_eq!(merge_writeback(&mut title, &mut host, &old), changed);
            assert_eq!(title, expected);
            assert_eq!(host, expected);
        }
    }
}
/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    /// D3D12 state after the last submitted list using this copy.
    pub(crate) state: AtomicU32,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<crate::fx::Set<(usize, usize)>>,
    /// The copy's bytes before the submission ran, where its shaders write:
    /// what the GPU wrote differs. Parts of the copy that do not overlap, in
    /// order, with where each starts in [`SNAPSHOTS`], and the 64 KiB pieces
    /// (by title address) already compared.
    before: Mutex<(Vec<Part>, crate::fx::Set<usize>)>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// Device-local VRAM mirror (`DEFAULT`) + Wave64 GPU hash state
    /// (`crate::card`), when the buffer is card-resident: shaders read and
    /// write `buffer` directly in VRAM, `data` maps `card.staging`, and no
    /// snapshot or CPU compare ever runs for it.
    card: Option<Arc<crate::card::CardState>>,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
    /// The last presented frame and milliseconds the title used it (a list
    /// named it, or a submission refreshed it): [`evict_unused`] gives up
    /// idle copies first.
    last: [AtomicU64; 2],
    /// A pressure-served copy, never in [`BUFFERS`], [`LOOSE`] or [`EXACT`]:
    /// made whole from title memory at every submission ([`Shadow::update`]
    /// with no tracker), pinned by its lists until they reset.
    transient: bool,
    /// A pooled buffer's bytes, when its buffer came from the transient
    /// pool: given back there (still charged) when it goes.
    pool_cap: Option<u64>,
}

/// `[from, to)` of a copy, kept from `at` of [`SNAPSHOTS`]' bytes on.
#[derive(Clone, Copy)]
struct Part {
    from: usize,
    to: usize,
    at: usize,
}

/// The running submission's snapshots, one after the other: in reused
/// memory (fresh memory costs page faults), emptied with its last part.
struct Snapshots {
    bytes: Vec<u8>,
    parts: usize,
}

static SNAPSHOTS: RwLock<Snapshots> = RwLock::new(Snapshots {
    bytes: Vec::new(),
    parts: 0,
});

/// The running submission's snapshots, for comparing what its shaders
/// wrote: many threads hold this while copying back, each its own copy's
/// bytes (disjoint parts of one block).
fn snapshots() -> std::sync::RwLockReadGuard<'static, Snapshots> {
    SNAPSHOTS.read().unwrap_or_else(|e| e.into_inner())
}

/// The snapshots, to keep more bytes or give them up: one thread at a
/// time, never while another copies back (preparation and copy-back are
/// separate phases of a submission).
fn snapshots_mut() -> std::sync::RwLockWriteGuard<'static, Snapshots> {
    SNAPSHOTS.write().unwrap_or_else(|e| e.into_inner())
}

/// Threads filling one snapshot keep: big keeps split by bytes (a 1 MB
/// keep is already worth a second thread); half the logical processors,
/// as the pipeline workers, so the title's threads still have cores. A
/// compile storm is running the workers on the other cores, so the fill
/// stays mostly out of their way (diag9: snapshots fell to ~2 GB/s
/// filling against six busy compile workers).
fn snapshot_threads(total: usize) -> usize {
    const PARALLEL_SNAPSHOT_BYTES: usize = 1 << 20;
    if total < PARALLEL_SNAPSHOT_BYTES {
        return 1;
    }
    let mut threads = std::thread::available_parallelism()
        .map_or(2, |n| (n.get() / 2).clamp(2, 8));
    if crate::pipeline::pipelines_waiting() > 0 {
        threads = threads.min(2);
    }
    threads
}

/// Gives up the snapshots of `parts`.
fn forget(parts: &mut Vec<Part>) {
    let mut snapshots = snapshots_mut();
    snapshots.parts -= parts.len();
    parts.clear();
    if snapshots.parts == 0 {
        snapshots.bytes.clear();
    }
}

impl Drop for Shadow {
    /// With a range not copied back (its list was reset meanwhile), with its
    /// shared-buffer run to hand back, and with mark bits to hand back.
    /// Pooled transient buffers go back to the pool (still charged: the
    /// commit is still held); everything else is uncharged, except arena
    /// runs (charged with their chunk, not the copy) and card shadows (the
    /// card engine's VRAM accounting owns those: never in the bound, so the
    /// bound only ever holds evictable bytes).
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        forget(&mut self.before.get_mut().unwrap_or_else(|e| e.into_inner()).0);
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        } else if self.transient {
            if let Some(cap) = self.pool_cap.filter(|&c| c >= self.len) {
                SHADOW_BYTES.fetch_add(cap - self.len, Relaxed);
                if return_transient_buffer(self.buffer.clone(), self.data as usize, self.gpu, cap) {
                    SHADOW_STATS[SHADOW_STATS_FREED].fetch_add(1, Relaxed);
                    return;
                }
                SHADOW_BYTES.fetch_sub(cap - self.len, Relaxed);
            }
            SHADOW_BYTES.fetch_sub(self.len, Relaxed);
        } else if !self.is_card() {
            SHADOW_BYTES.fetch_sub(self.len, Relaxed);
        }
        SHADOW_STATS[SHADOW_STATS_FREED].fetch_add(1, Relaxed);
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// Host bytes held by copies of title memory: [`BUFFERS`] and [`LOOSE`]
/// copies' title bytes plus pooled transient buffers (see [`SHADOW_ARENA_BYTES`]
/// for the arena chunks [`EXACT`] copies live in). Card-resident shadows are
/// not charged here (the card engine's VRAM accounting owns those), so the
/// bound only ever holds evictable bytes. Bounded by [`evict_unused`]
/// and the transient fallback in [`find`]/[`loose`]/[`exact`]/[`Arena::take`].
pub(crate) static SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);
/// Of the host commit copies hold, in arena chunks: charged per chunk, with
/// the per-chunk breakdown in [`log_shadow`].
static SHADOW_ARENA_BYTES: AtomicU64 = AtomicU64::new(0);
/// Copies made, freed, given up by [`evict_unused`], and pressure-served
/// transients ([`SHADOW_STATS_TRANSIENT`]).
pub(crate) static SHADOW_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const SHADOW_STATS_MADE: usize = 0;
const SHADOW_STATS_FREED: usize = 1;
const SHADOW_STATS_EVICTED: usize = 2;
const SHADOW_STATS_TRANSIENT: usize = 3;

/// Past this share of the system (non-local) video-memory budget, host
/// copies of title memory are given up least recently used first, until use
/// is back under [`SHADOW_LOW`]: mirrors `crate::resource` (`BUDGET_HIGH` /
/// `BUDGET_LOW`), but against the budget host copies actually commit.
const SHADOW_HIGH: f64 = 0.90;
const SHADOW_LOW: f64 = 0.85;

/// The system-memory budget's bytes, when the host tells it.
fn shadow_budget() -> Option<u64> {
    crate::host::video_memory().map(|[_, (_, budget)]| (budget * 1e6) as u64)
}

/// Host bytes copies hold now (buffers, loose windows, transients, pooled
/// transient buffers, and arena chunks; not card-resident shadows, which the
/// card engine's VRAM accounting owns).
fn shadow_total() -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    SHADOW_BYTES.load(Relaxed).saturating_add(SHADOW_ARENA_BYTES.load(Relaxed))
}

/// Whether `ask` more bytes would put copies past [`SHADOW_HIGH`] of the
/// budget: the caller gives up idle copies first, else serves a transient.
fn pressured(ask: u64) -> bool {
    shadow_budget().is_some_and(|budget| shadow_total().saturating_add(ask) > (SHADOW_HIGH * budget as f64) as u64)
}

/// Gives up idle copies until `ask` more bytes fit under [`SHADOW_LOW`].
fn relieve(ask: u64) {
    let Some(budget) = shadow_budget() else { return };
    let low = (SHADOW_LOW * budget as f64) as u64;
    let over = shadow_total().saturating_add(ask).saturating_sub(low);
    if over > 0 {
        evict(crate::queue::presented(), crate::resource::now_ms(), over);
    }
}

/// What `parts` (in order, not overlapping) leave out of `[from, to)`.
fn uncovered(parts: &[Part], from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut gaps = Vec::new();
    let mut at = from;
    for p in &parts[parts.partition_point(|p| p.to <= from)..] {
        if p.from >= to {
            break;
        }
        if p.from > at {
            gaps.push((at, p.from));
        }
        at = at.max(p.to);
    }
    if at < to {
        gaps.push((at, to));
    }
    gaps
}

/// The bytes `parts` keep of `[from, to)`, from one part or put together in
/// `room`; `None` if they do not cover it.
fn kept<'a>(
    parts: &[Part],
    bytes: &'a [u8],
    (from, to): (usize, usize),
    room: &'a mut Vec<u8>,
) -> Option<&'a [u8]> {
    let parts = &parts[parts.partition_point(|p| p.to <= from)..];
    if let Some(p) = parts.first()
        && p.from <= from
        && to <= p.to
    {
        return Some(&bytes[p.at + (from - p.from)..][..to - from]);
    }
    room.clear();
    for p in parts {
        let at = from + room.len();
        if at >= to || p.from > at {
            break;
        }
        room.extend_from_slice(&bytes[p.at + (at - p.from)..][..p.to.min(to) - at]);
    }
    (room.len() == to - from).then_some(&room[..])
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// whose disjoint ranges different submission threads refresh and compare
// (one thread per host copy in a parallel batch, one range at a time per
// copy), whose snapshot slots different threads fill (disjoint bytes of one
// reservation), and which the thread finishing an earlier submission writes
// only in ranges that one's shaders wrote (which the submitting thread
// leaves alone until it has finished); title memory is read through the
// kernel.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Whether `[address, address + len)` lies wholly inside the buffer.
    pub(crate) fn covers(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        start <= address && address.checked_add(len).is_some_and(|to| to <= end)
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        self.touch();
        list.defer_refresh(self, address as usize, len as usize);
        if self.transient {
            list.keep_transient(self);
        }
    }

    fn new(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        card: Option<Arc<crate::card::CardState>>,
    ) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None, card)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's).
    fn new_at(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        slice: Option<Slice>,
        card: Option<Arc<crate::card::CardState>>,
    ) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        // Card-resident shadows never take the marked-pages path (their
        // writes are declared ranges, compared by GPU hashes), so they
        // hold no view-bitmap bits.
        let marks = if card.is_some() {
            None
        } else {
            tracked.as_ref().and_then(|_| marks().take(pages(start, len)))
        };
        let (frame, now) = (crate::queue::presented(), crate::resource::now_ms());
        let shadow = Arc::new(Shadow {
            buffer,
            state: AtomicU32::new(0),
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            base: slice.map_or(0, |s| s.offset),
            slice,
            card,
            last: [AtomicU64::new(frame), AtomicU64::new(now)],
            transient: false,
            pool_cap: None,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        if slice.is_none() && !shadow.is_card() {
            // An arena run's bytes are charged with its chunk
            // ([`Arena::take`]), not with the copy, and a card shadow's with
            // the card engine's VRAM accounting: the bound only ever holds
            // evictable bytes.
            use std::sync::atomic::Ordering::Relaxed;
            SHADOW_BYTES.fetch_add(len, Relaxed);
        }
        // Every copy counts as made (freed in [`Drop`]): even an arena run,
        // whose bytes ride with its chunk.
        {
            use std::sync::atomic::Ordering::Relaxed;
            SHADOW_STATS[SHADOW_STATS_MADE].fetch_add(1, Relaxed);
        }
        shadow
    }

    /// A pressure-served copy of `[start, start + len)` over `buffer` (a
    /// pooled or fresh upload buffer, never an arena run): untracked, so
    /// every submission copies it whole from title memory; never inserted
    /// into [`BUFFERS`], [`LOOSE`] or [`EXACT`], never hull-cached.
    fn new_transient(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        pool_cap: Option<u64>,
    ) -> Arc<Shadow> {
        use std::sync::atomic::Ordering::Relaxed;
        SHADOW_BYTES.fetch_add(len, Relaxed);
        SHADOW_STATS[SHADOW_STATS_MADE].fetch_add(1, Relaxed);
        SHADOW_STATS[SHADOW_STATS_TRANSIENT].fetch_add(1, Relaxed);
        Arc::new(Shadow {
            buffer,
            state: AtomicU32::new(0),
            data,
            gpu,
            start,
            len,
            tracked: None,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks: None,
            base: 0,
            slice: None,
            card: None,
            last: [
                AtomicU64::new(crate::queue::presented()),
                AtomicU64::new(crate::resource::now_ms()),
            ],
            transient: true,
            pool_cap,
        })
    }

    /// Whether it is pressure-served ([`Shadow::new_transient`]).
    pub(crate) fn is_transient(&self) -> bool {
        self.transient
    }

    /// The title used it now: a list named it, or a submission refreshed it.
    pub(crate) fn touch(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.last[0].store(crate::queue::presented(), Relaxed);
        self.last[1].store(crate::resource::now_ms(), Relaxed);
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// Whether the buffer lives in device-local VRAM (`crate::card`).
    pub(crate) fn is_card(&self) -> bool {
        self.card.is_some()
    }

    /// Its VRAM state, when card-resident.
    pub(crate) fn card_state(&self) -> Option<Arc<crate::card::CardState>> {
        self.card.clone()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        self.touch();
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
        if self.transient {
            list.keep_transient(self);
        }
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// each byte once per submission, however many ranges declare it, and
    /// none between them (the world declares overlapping views of its 174 MB
    /// heap dozens of times), in reused memory (fresh snapshots cost ~2.3
    /// million page faults per world frame). Big keeps are filled on
    /// threads: the slots are disjoint, so they share nothing but a read
    /// hold on the block (which also pins its allocation while filling).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if self.card.is_some() {
            // Card-resident: `staging` is the exact pre-submission
            // snapshot (zero snapshot `memcpy`); the card engine uploads
            // it and hashes VRAM around the submission instead.
            return;
        }
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let parts = &mut before.0;
        let gaps = uncovered(parts, from, to);
        if gaps.is_empty() {
            // Refresh the covered slots from the copy: an earlier submission
            // may have copied back since they were kept (submissions overlap;
            // `settle` keeps overlapping writers out, so no in-flight
            // submission still reads them), and the next compare must see
            // the copy as it is now, not as it was then.
            let mut snapshots = snapshots_mut();
            for p in parts.iter() {
                let (s, e) = (p.from.max(from), p.to.min(to));
                if s < e {
                    // SAFETY: inside the mapping and the snapshots block.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    snapshots.bytes[p.at + (s - p.from)..p.at + (e - p.from)]
                        .copy_from_slice(host);
                }
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        {
            let bytes = gaps.iter().map(|(a, b)| (b - a) as u64).sum();
            crate::copystat::add(crate::copystat::C::SnapshotBytes, bytes);
            crate::copystat::snapshot(self.start, self.len, bytes);
        }
        let total: usize = gaps.iter().map(|(a, b)| b - a).sum();
        // The slots, reserved under one short lock; filled after it (each
        // byte once: the gaps partition the kept bytes).
        let base = {
            let mut snapshots = snapshots_mut();
            let base = snapshots.bytes.len();
            snapshots.bytes.reserve(total);
            // SAFETY: every reserved byte is filled below before any reader
            // runs (copy-back is a later phase; tests sharing the block hold
            // its single-test lock, and the read hold below pins this
            // allocation meanwhile). The lint fires on the pattern, not on
            // a use: the fill covers exactly `total` bytes by construction.
            #[allow(clippy::uninit_vec)]
            unsafe {
                snapshots.bytes.set_len(base + total)
            };
            snapshots.parts += gaps.len();
            base
        };
        let mut jobs: Vec<(usize, usize, usize)> = Vec::with_capacity(gaps.len());
        let mut at = base;
        for (from, to) in gaps {
            parts.push(Part { from, to, at });
            jobs.push((from, at - base, to - from));
            at += to - from;
        }
        parts.sort_unstable_by_key(|p| p.from);
        self.fill_snapshot(jobs, base, total);
    }

    /// Copies the kept bytes (`(offset in the copy, offset in the reserved
    /// block, length)`) into the snapshot block at `base`: on threads where
    /// the keep is big enough, each thread disjoint bytes of one
    /// reservation.
    fn fill_snapshot(&self, jobs: Vec<(usize, usize, usize)>, base: usize, total: usize) {
        // The block, held for reading while filling (no reallocation
        // meanwhile: that takes the write lock, which this hold excludes;
        // the fill itself needs no lock, its bytes are disjoint).
        let hold = snapshots();
        let dst = hold.bytes.as_ptr() as usize;
        if snapshot_threads(total) <= 1 {
            for (from, at, len) in jobs {
                // SAFETY: `from` is inside the mapping and `base + at` in
                // the reservation, both `len` bytes, disjoint per job.
                unsafe {
                    std::ptr::copy_nonoverlapping(self.data.add(from), (dst + base + at) as *mut u8, len);
                }
            }
            return;
        }
        // Split the jobs into per-thread byte quotas (a single big gap is
        // split across threads, not stuck on one).
        let quota = total.div_ceil(snapshot_threads(total));
        let mut chunks: Vec<Vec<(usize, usize, usize)>> = Vec::new();
        let mut current: Vec<(usize, usize, usize)> = Vec::new();
        let mut used = 0;
        for (from, at, len) in jobs {
            let (mut from, mut at, mut len) = (from, at, len);
            while len > 0 {
                if used == quota {
                    chunks.push(std::mem::take(&mut current));
                    used = 0;
                }
                let n = (quota - used).min(len);
                current.push((from, at, n));
                from += n;
                at += n;
                len -= n;
                used += n;
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        let data = self.data as usize;
        std::thread::scope(|s| {
            let mut parts = chunks.into_iter();
            // The submitting thread takes the first chunk itself.
            let mine = parts.next();
            let mut handles = Vec::new();
            for chunk in parts {
                handles.push(s.spawn(move || {
                    for (from, at, len) in chunk {
                        // SAFETY: as above; chunks partition the jobs.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (data as *const u8).add(from),
                                (dst + base + at) as *mut u8,
                                len,
                            );
                        }
                    }
                }));
            }
            if let Some(chunk) = mine {
                for (from, at, len) in chunk {
                    // SAFETY: as above.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.data.add(from),
                            (dst + base + at) as *mut u8,
                            len,
                        );
                    }
                }
            }
            for h in handles {
                h.join().expect("a snapshot thread panicked");
            }
        });
        drop(hold);
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`Shadow::copy_back_marked`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        if self.card.is_some() {
            // Card-resident: the card engine merges GPU writes
            // (`merge_card_pieces`), not the snapshot compare.
            self.finish_card_pending(&[(address, len)]);
            return 0;
        }
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        // Card-resident ranges never reach the snapshot path: `copy_back`
        // diverts them to the card engine, and only snapshot `back` entries
        // call this.
        debug_assert!(!self.is_card());
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        // As above: card shadows are diverted in `copy_back`, so reaching
        // here with one would mix the card engine's ranges into snapshots.
        debug_assert!(!self.is_card());
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (parts, compared) = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if back && from < to {
            let snapshots = snapshots();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut room = Vec::new();
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && compared.contains(&(start + at)) {
                    at += n;
                    continue;
                }
                let Some(old) = kept(parts, &snapshots.bytes, (at, at + n), &mut room) else {
                    at += n;
                    continue;
                };
                if n == PIECE {
                    compared.insert(start + at);
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    if merge_writeback(title, host, old) {
                        let _writing = crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    // Keep the snapshot current for this range in case another
                    // disjoint in-flight submission keeps `snapshot` alive.
                    // (Parts are immutable once written: disjoint submissions
                    // use different slots, and overlapping ones are settled
                    // first, so no update is needed.)
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it
        // (submissions in flight do not declare overlapping ranges of a
        // copy: a later one finishes the earlier first), and the snapshot
        // with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            forget(parts);
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, page: usize) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        // Our own write below must not save the page for this copy again.
        t.unarm(page / PAGE);
        let saved = t.saved(page / PAGE);
        if saved.is_none() && t.changed(page / PAGE) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) {
            return 0;
        }
        let written = match &saved {
            // The title's bytes are the copy's from before: what differs is
            // the shaders'.
            None => (*host != title[..]).then_some(host.as_ptr()),
            Some(before) => {
                let before = &before[from - page..to - page];
                merge_writeback(&mut title, host, before).then_some(title.as_ptr())
            }
        };
        if let Some(bytes) = written {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, bytes, n);
        }
        // Equal to title memory now: taken, unless it holds the page in part.
        if from == page && to == page + PAGE {
            t.taken(&[page / PAGE]);
        }
        written.map_or(0, |_| n)
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.touch();
        if self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty());
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them nothing: what the
            // title wrote there is not known.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    fn update_tracked(&self, from: usize, to: usize, busy: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        let mut sync_before = |a: usize, b: usize| {
            let Some(before) = &mut before else { return };
            let (a, b) = (a - start, b - start);
            let mut snapshots = snapshots_mut();
            for p in before.0.iter() {
                let (s, e) = (p.from.max(a), p.to.min(b));
                if s < e {
                    // SAFETY: inside the mapping and the snapshots block.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    snapshots.bytes[p.at + (s - p.from)..p.at + (e - p.from)]
                        .copy_from_slice(host);
                }
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            sync_before(a, b);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in t.untracked(from, to) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += self.copy(a, b);
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WholeBytes, (b - a) as u64);
                add(C::WholeUs, started.elapsed().as_micros() as u64);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        copied
    }

    /// Whether the title may have written a page of `[from, to)` (title
    /// addresses) since the copy last took it (always, for memory that is
    /// not tracked).
    pub(crate) fn stale(&self, from: usize, to: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return false;
        }
        self.tracked
            .as_ref()
            .is_none_or(|t| t.any_changed(from, to))
    }

    /// Once a submission's ranges of the copy have all gone back: the next
    /// submission compares its pieces afresh (several may be in flight
    /// while the snapshot lives).
    pub(crate) fn submission_done(&self) {
        self.before
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .clear();

    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if let Some(card) = &self.card {
            card.mark_dirty(from - start, to - start);
        }
        if let Some(t) = &self.tracked {
            // SAFETY: `to - from` bytes inside the mapping, just copied.
            t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
        }
        to - from
    }

    /// Merges GPU-modified 64 KiB pieces (fetched from VRAM into `runs`:
    /// `(first_piece, count, readback_ptr)`) into title memory and
    /// `card.staging`. Returns bytes written to title memory.
    pub(crate) fn merge_card_pieces(
        &self,
        card: &Arc<crate::card::CardState>,
        runs: &[(u32, u32, usize)],
    ) -> usize {
        let start = self.start as usize;
        let shadow_len = self.len as usize;
        let mut written = 0usize;
        let mut title_buf = vec![0u8; PAGE];
        let mut synced: Vec<usize> = Vec::new();

        for &(first_piece, count, rb_ptr) in runs {
            for k in 0..count {
                let piece = (first_piece + k) as usize;
                let off = piece * PAGE;
                if off >= shadow_len {
                    break;
                }
                let n = PAGE.min(shadow_len - off);
                let title_addr = start + off;
                // SAFETY: `rb_ptr` holds `count * PAGE` bytes fetched from
                // VRAM; `self.data` maps `card.staging` of at least
                // `shadow_len` bytes.
                let gpu_new = unsafe {
                    std::slice::from_raw_parts((rb_ptr + (k as usize) * PAGE) as *const u8, n)
                };
                // SAFETY: `self.data` maps `card.staging` of at least
                // `shadow_len` bytes; `off + n` is inside the buffer.
                let staging_old =
                    unsafe { std::slice::from_raw_parts_mut(self.data.add(off), n) };
                if gpu_new == staging_old {
                    continue;
                }
                let title_slice = &mut title_buf[..n];
                if !crate::host::read_into(title_addr as *const u8, title_slice) {
                    // Unmapped in title memory: still sync staging to VRAM
                    // so future diffs compare against the latest GPU bytes.
                    staging_old.copy_from_slice(gpu_new);
                    continue;
                }
                let (dirty_span, cpu_modified) =
                    merge_gpu_writes_card(title_slice, staging_old, gpu_new);
                if let Some((dirty_start, dirty_end)) = dirty_span {
                    let dirty_len = dirty_end - dirty_start;
                    let _writing = crate::watch::writing(title_addr + dirty_start, dirty_len);
                    crate::host::write_mapped(
                        (title_addr + dirty_start) as *mut u8,
                        title_slice[dirty_start..].as_ptr(),
                        dirty_len,
                    );
                    written += dirty_len;
                }
                if cpu_modified {
                    // Concurrent CPU write on the same page: upload the
                    // merged `staging` back to VRAM before the next
                    // submission.
                    card.mark_dirty(off, off + n);
                }
                let page_lo = title_addr.div_ceil(PAGE);
                let page_hi = (title_addr + n) / PAGE;
                for p in page_lo..page_hi {
                    synced.push(p);
                }
            }
        }

        // Pages made equal to title memory are current in the copy.
        if let Some(t) = &self.tracked {
            let (first, last) = (start.div_ceil(PAGE), (start + shadow_len) / PAGE);
            let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
            t.taken(&whole);
        }
        written
    }

    /// A submitted card range is done with: it no longer blocks refreshes.
    pub(crate) fn finish_card_pending(&self, declared: &[(usize, usize)]) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        for &(address, len) in declared {
            pending.retain(|&r| r != (address, address + len));
        }
    }

    /// A card shadow for tests: `data` maps `card.staging`.
    #[cfg(test)]
    pub(crate) fn new_card_test(
        card: Arc<crate::card::CardState>,
        data: *mut u8,
        start: u64,
        len: u64,
    ) -> Arc<Self> {
        let buffer = card.vram.clone();
        // SAFETY: live VRAM resource.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, start, len, Some(card));
        if let Some(t) = &shadow.tracked {
            t.disarm();
        }
        shadow
    }

    /// Copies title memory into the copy (tests).
    #[cfg(test)]
    pub(crate) fn copy_for_test(&self, from: usize, to: usize) -> usize {
        self.copy(from, to)
    }
}

/// Merges GPU-modified bytes (`gpu_new` differs from `staging_old`) into
/// `title` and syncs `staging_old` to the merged bytes. Returns the dirty
/// span for the title and whether any byte unchanged by the GPU differed in
/// `title` (a concurrent title CPU write that must be uploaded back to
/// VRAM).
fn merge_gpu_writes_card(
    title: &mut [u8],
    staging_old: &mut [u8],
    gpu_new: &[u8],
) -> (Option<(usize, usize)>, bool) {
    let n = title.len().min(staging_old.len()).min(gpu_new.len());
    let mut first = usize::MAX;
    let mut last = 0usize;
    let mut cpu_modified = false;
    let mut i = 0;
    while i + 8 <= n {
        let gw = u64::from_ne_bytes(gpu_new[i..i + 8].try_into().unwrap());
        let ow = u64::from_ne_bytes(staging_old[i..i + 8].try_into().unwrap());
        if gw == ow {
            let tw = u64::from_ne_bytes(title[i..i + 8].try_into().unwrap());
            if tw != ow {
                staging_old[i..i + 8].copy_from_slice(&title[i..i + 8]);
                cpu_modified = true;
            }
        } else {
            for k in i..i + 8 {
                let g = gpu_new[k];
                let o = staging_old[k];
                if g != o {
                    if title[k] != g {
                        title[k] = g;
                        first = first.min(k);
                        last = k + 1;
                    }
                    staging_old[k] = g;
                } else if title[k] != o {
                    staging_old[k] = title[k];
                    cpu_modified = true;
                }
            }
        }
        i += 8;
    }
    for k in i..n {
        let g = gpu_new[k];
        let o = staging_old[k];
        if g != o {
            if title[k] != g {
                title[k] = g;
                first = first.min(k);
                last = k + 1;
            }
            staging_old[k] = g;
        } else if title[k] != o {
            staging_old[k] = title[k];
            cpu_modified = true;
        }
    }
    ((first < last).then_some((first, last)), cpu_modified)
}

/// [`merge_gpu_writes_card`], for the card engine's tests.
#[cfg(test)]
pub(crate) fn merge_gpu_writes_card_for_test(
    title: &mut [u8],
    staging_old: &mut [u8],
    gpu_new: &[u8],
) -> (Option<(usize, usize)>, bool) {
    merge_gpu_writes_card(title, staging_old, gpu_new)
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index.
            let bits = victory_shader::emit::VIEW_READS + victory_shader::emit::HEAP_VIEWS;
            let bytes = u64::from(bits / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // Both bitmaps: shaders that use title memory mark their reads in theirs.
    let mut maps: Vec<&mut [u32]> = [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks; the GPU is done with them
        // (one submission runs at a time).
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter_mut().for_each(|bits| bits[w] &= !bit);
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    for (w, s) in seen.iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps), whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (w, word) in bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared) and, with `memory_withheld`, pages that shaders using
/// title memory marked; the bitmaps are cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)], memory_withheld: bool) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        // Marks of offsets past the bitmap land in its last bit, no copy's.
        bits[words - 1] = 0;
        let used = (marks().end as usize).div_ceil(32).min(words);
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            let into = if memory && memory_withheld { &mut held } else { &mut marked };
            while x != 0 {
                into.push((w * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    if !held.is_empty() {
        let m = marks();
        for (_, page) in held.into_iter().filter_map(|n| page_of(&m, n)) {
            crate::memory::drop_withheld(page, PAGE);
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked.iter().filter_map(|&n| page_of(&m, n)).collect()
    };
    for (copy, page) in pages {
        if withheld
            .iter()
            .any(|(c, a, n)| Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a + n)
        {
            continue;
        }
        let written = copy.copy_back_marked(page);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let key = buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        return loose(address, end);
    };
    if let Some(shadow) = buffers().get(&key).and_then(|e| e.shadow.clone()) {
        shadow.touch();
        return Some(shadow);
    }
    if buffers().get(&key).is_some() {
        let len = buffers().get(&key).map(|e| e.end - key.0).unwrap_or(0);
        let size = len.next_multiple_of(1 << 16);
        if pressured(size) {
            // The evict pass takes the registries itself: never under one.
            relieve(size);
        }
        if pressured(size) {
            // Bound, not refused: a copy that is never cached, made whole
            // from title memory at every submission, until its lists reset.
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer over the host-copy bound: pressure-served copy"
            );
            return transient_dedicated(key.0, len);
        }
    }
    let mut map = buffers();
    let entry = map.get_mut(&key)?;
    if let Some(shadow) = entry.shadow.clone() {
        shadow.touch();
        return Some(shadow);
    }
    {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        let h = host()?;
        let (staging, data) = h.shared_buffer(size)?;
        crate::host::name(&staging, || format!("title buffer {:#x}+{len:#x}", key.0));
        let (buffer, card) = match crate::card::CardState::try_new(h, &staging, size) {
            Some(card) => {
                let vram = card.vram.clone();
                crate::host::name(&vram, || format!("title buffer {:#x}+{len:#x} VRAM", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "large buffer placed in VRAM (card buffer)"
                );
                (vram, Some(card))
            }
            None => {
                if len >= 16 << 20 {
                    tracing::info!(
                        start = format_args!("{:#x}", key.0),
                        len,
                        "large buffer copied to the host"
                    );
                } else {
                    tracing::debug!(
                        start = format_args!("{:#x}", key.0),
                        len,
                        "buffer copied to the host"
                    );
                }
                (staging, None)
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len, card));
        changed();
    }
    let shadow = entry.shadow.clone()?;
    shadow.touch();
    Some(shadow)
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches: idle ones are given up by
/// [`evict_unused`] and remade whole on next use (this supersedes the old
/// never-free shortcut).
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    if let Some(shadow) = LOOSE.lock().unwrap_or_else(|e| e.into_inner()).get(&(start, len)) {
        shadow.touch();
        return Some(shadow.clone());
    }
    if pressured(len) {
        // The evict pass takes the registries itself: never under one.
        relieve(len);
    }
    let h = host()?;
    if pressured(len) {
        // Bound, not refused: a copy that is never cached, made whole from
        // title memory at every submission, until its lists reset.
        tracing::debug!(
            start = format_args!("{start:#x}"),
            len,
            "loose memory over the host-copy bound: pressure-served copy"
        );
        return transient_dedicated(start, len);
    }
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        shadow.touch();
        return Some(shadow.clone());
    }
    let (staging, data) = h.shared_buffer(len)?;
    crate::host::name(&staging, || format!("title memory {start:#x}+{len:#x}"));
    let (buffer, card) = match crate::card::CardState::try_new(h, &staging, len) {
        Some(card) => {
            let vram = card.vram.clone();
            crate::host::name(&vram, || format!("title memory {start:#x}+{len:#x} VRAM"));
            tracing::info!(
                start = format_args!("{start:#x}"),
                len,
                "loose memory placed in VRAM (card buffer)"
            );
            (vram, Some(card))
        }
        None => (staging, None),
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len, card);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// Once per presented frame, before a submission (beside
/// [`crate::resource::evict_unused`], same site in `Host::execute`): gives up
/// host copies of title memory the title has not used for
/// [`crate::resource::IDLE_MS`] and [`crate::resource::IDLE_FRAMES`], and past
/// [`SHADOW_HIGH`] of the system budget the least recently used ones until
/// under [`SHADOW_LOW`].
pub(crate) fn evict_unused() {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: AtomicU64 = AtomicU64::new(u64::MAX);
    let frame = crate::queue::presented();
    if LAST.swap(frame, Relaxed) == frame {
        return;
    }
    let total = shadow_total();
    let over = match shadow_budget() {
        Some(budget) if total > (SHADOW_HIGH * budget as f64) as u64 => {
            total.saturating_sub((SHADOW_LOW * budget as f64) as u64)
        }
        _ => 0,
    };
    evict(frame, crate::resource::now_ms(), over);
}

/// A candidate copy: where the registry holds it (never for transients:
/// they are never inserted, never evicted, only freed with their lists).
enum Victim {
    Buffer((u64, usize)),
    Loose((u64, u64)),
    Exact((u64, u64, u64)),
}

/// The copies that may be given up ([`evictable`]): buffers' and loose
/// windows' registry holds, and live exact copies (weak entries upgraded).
fn candidates() -> Vec<(Arc<Shadow>, Victim)> {
    let mut out = Vec::new();
    for ((start, resource), entry) in buffers().iter() {
        if let Some(shadow) = entry.shadow.clone() {
            out.push((shadow, Victim::Buffer((*start, *resource))));
        }
    }
    for (key, shadow) in LOOSE.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        out.push((shadow.clone(), Victim::Loose(*key)));
    }
    for (key, weak) in EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.iter() {
        if let Some(shadow) = weak.upgrade() {
            out.push((shadow, Victim::Exact(*key)));
        }
    }
    out
}

/// Whether `shadow` may be given up: no heap slot views it (bindless ranges
/// or a heap slot's kept copy), no recorded or in-flight submission names it
/// (only the registry and this pass hold it, nothing declared unwritten, no
/// snapshot parts), and it is not card-resident (the card engine's plans, and
/// VRAM accounting, own those).
///
/// A use racing this (a title thread recording) remakes the copy whole from
/// title memory on next use: eviction only drops bytes title memory already
/// holds. Nothing outstanding needs writing back: every declared shader
/// write goes back through `pending`/`before` before its submission
/// finishes, and no submission naming this copy is recorded or in flight
/// (its steps would hold it). `marks` disarm with `pending`: arming only
/// happens in [`Shadow::writing`], which also declares the range.
fn evictable(shadow: &Arc<Shadow>, victim: &Victim) -> bool {
    if shadow.is_card() {
        return false;
    }
    // The registry's holds, plus this pass's own clone: Buffer/Loose hold
    // one `Arc` each in their maps, Exact holds only a `Weak`.
    let registry = match victim {
        Victim::Buffer(_) | Victim::Loose(_) => 1,
        Victim::Exact(_) => 0,
    };
    if Arc::strong_count(shadow) != registry + 1 {
        return false;
    }
    if !shadow.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
        return false;
    }
    if !shadow.before.lock().unwrap_or_else(|e| e.into_inner()).0.is_empty() {
        return false;
    }
    let ptr = Arc::as_ptr(shadow) as usize;
    if crate::bindless::viewing(ptr) {
        return false;
    }
    if crate::descriptor::host::keeping(ptr) {
        return false;
    }
    true
}

/// Of copies last used at (frame, ms) and taking (bytes) each, those to give
/// up at (`frame`, `now`): every one idle for [`crate::resource::IDLE_FRAMES`]
/// and [`crate::resource::IDLE_MS`], then the least recently used others until
/// `over` bytes are gone (indices into `uses`). The planning half of
/// [`evict`], apart from the registries so it can be measured
/// (`benchmark_evict_plan`).
fn plan(uses: &[(u64, u64, u64)], frame: u64, now: u64, over: u64) -> Vec<usize> {
    use crate::resource::{IDLE_FRAMES, IDLE_MS};
    let idle = |&(f, ms, _): &(u64, u64, u64)| {
        frame.saturating_sub(f) >= IDLE_FRAMES && now.saturating_sub(ms) >= IDLE_MS
    };
    let (mut gone, mut rest): (Vec<usize>, Vec<usize>) =
        (0..uses.len()).partition(|&i| idle(&uses[i]));
    let mut over = over.saturating_sub(gone.iter().map(|&i| uses[i].2).sum());
    rest.sort_by_key(|&i| uses[i].1);
    for i in rest {
        if over == 0 {
            break;
        }
        over = over.saturating_sub(uses[i].2);
        gone.push(i);
    }
    gone
}

/// Gives up the copies the title has not used for [`crate::resource::IDLE_MS`]
/// and [`crate::resource::IDLE_FRAMES`] at (`frame`, `now` ms), then the least
/// recently used ones until `over` bytes are gone. Returns how many and their
/// bytes.
fn evict(frame: u64, now: u64, over: u64) -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    // Trailing fully free chunks go first (no title bytes in them), so the
    // pass below only gives up copies holding title bytes.
    arena().release_empty();
    let found = candidates();
    let uses: Vec<(u64, u64, u64)> = found
        .iter()
        .map(|(s, _)| (s.last[0].load(Relaxed), s.last[1].load(Relaxed), s.len))
        .collect();
    let gone = plan(&uses, frame, now, over);
    let mut count = 0u64;
    let mut freed = 0u64;
    for i in gone {
        let (shadow, victim) = &found[i];
        // Re-checked: a title thread may have named it meanwhile.
        if !evictable(shadow, victim) {
            continue;
        }
        let removed = match victim {
            Victim::Buffer(key) => buffers()
                .get_mut(key)
                .filter(|e| e.shadow.as_ref().is_some_and(|s| Arc::ptr_eq(s, shadow)))
                .map(|e| {
                    e.shadow = None;
                    changed();
                })
                .is_some(),
            Victim::Loose(key) => LOOSE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(key)
                .is_some_and(|s| Arc::ptr_eq(&s, shadow)),
            Victim::Exact(key) => {
                let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
                let live = guard.1.get(key).and_then(|w| w.upgrade()).is_some_and(|s| Arc::ptr_eq(&s, shadow));
                if live {
                    guard.1.remove(key);
                }
                live
            }
        };
        if !removed {
            continue;
        }
        // No declared write is outstanding (checked above): title memory
        // already holds every byte. Close the snapshot bookkeeping all the
        // same; the next use remakes the copy whole from title memory.
        shadow.submission_done();
        freed += shadow.len;
        count += 1;
        // The registry's hold is gone (`found`'s clones drop at the end of
        // the pass): the copy is freed, and uncharged, then — unless a
        // racing title thread named it after the check, which only remakes
        // whole what it reads.
    }
    if count > 0 {
        SHADOW_STATS[SHADOW_STATS_EVICTED].fetch_add(count, Relaxed);
    }
    (count, freed)
}

/// Logs [`SHADOW_STATS`] and [`SHADOW_BYTES`] per frame over `frames`, with
/// the arena chunks' breakdown and what still lives: beside
/// [`bindless::log`]/[`log_views`].
#[cfg(feature = "diagnostics")]
pub(crate) fn log_shadow(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [made, freed, evicted, transient] = [0, 1, 2, 3].map(|i| SHADOW_STATS[i].swap(0, Relaxed));
    let arena_bytes = SHADOW_ARENA_BYTES.load(Relaxed);
    let (chunks, free_bytes) = {
        let a = arena();
        let free_bytes: u64 = a.chunks.iter().map(|c| c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), free_bytes)
    };
    let live_buffers = buffers().values().filter(|e| e.shadow.is_some()).count();
    let live_loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner()).len();
    let (live_exact, dead_exact) = {
        let guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
        let live = guard.1.values().filter(|w| w.strong_count() > 0).count();
        (live, guard.1.len() - live)
    };
    let (kept_slots, kept_copies) = crate::descriptor::host::kept_census();
    tracing::info!(
        "host copies of title memory: {:.0} made/frame, {:.0} freed/frame, {:.0} evicted/frame ({} transient served); {:.1} MB held ({:.1} MB arena in {} chunks, {:.1} MB free) in {live_buffers} buffers, {live_loose} loose windows, {live_exact} live exact copies ({dead_exact} dead); heap slots keep {kept_slots} copies in {kept_copies} slots",
        made as f64 / frames,
        freed as f64 / frames,
        evicted as f64 / frames,
        transient,
        shadow_total() as f64 / 1e6,
        arena_bytes as f64 / 1e6,
        chunks,
        free_bytes as f64 / 1e6
    );
}

/// Finished transient buffers for reuse ([`transient_dedicated`]): buffer,
/// mapped pointer, host address, bytes. Bounded like `SPARE_UPLOADS` (count
/// and bytes); over cap the buffer drops, and the next pressure ask
/// allocates afresh.
static SPARE_TRANSIENT: Mutex<Vec<(ID3D12Resource, usize, u64, u64)>> = Mutex::new(Vec::new());
const SPARE_TRANSIENT_MAX: usize = 16;
const SPARE_TRANSIENT_BYTES: u64 = 64 << 20;

/// The pooled buffer to reuse for `size` bytes: the smallest capacity that is
/// at least `size` and within twice it (best fit), if any. The scan of the
/// pool ([`SPARE_TRANSIENT_MAX`] entries at most) is the CPU cost of a
/// pressure-served copy (`benchmark_spare_transient_fit`).
fn spare_fit(caps: impl Iterator<Item = u64>, size: u64) -> Option<usize> {
    let size = size.max(1);
    let max_fit = size.saturating_mul(2);
    caps.enumerate()
        .filter(|(_, cap)| *cap >= size && *cap <= max_fit)
        .min_by_key(|(_, cap)| *cap)
        .map(|(i, _)| i)
}

/// A pooled transient buffer of at least `size` bytes (capacity within
/// twice that, best fit): still charged to [`SHADOW_BYTES`] while pooled.
fn take_transient_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8, u64, u64)> {
    let mut pool = SPARE_TRANSIENT.lock().unwrap_or_else(|e| e.into_inner());
    let idx = spare_fit(pool.iter().map(|(_, _, _, cap)| *cap), size)?;
    let (buffer, data, gpu, cap) = pool.swap_remove(idx);
    Some((buffer, data as *mut u8, gpu, cap))
}

/// Keeps a finished transient buffer for reuse, bounded in count and bytes;
/// false when over cap (it drops instead).
fn return_transient_buffer(buffer: ID3D12Resource, data: usize, gpu: u64, cap: u64) -> bool {
    let mut pool = SPARE_TRANSIENT.lock().unwrap_or_else(|e| e.into_inner());
    let total: u64 = pool.iter().map(|(_, _, _, c)| *c).sum();
    if pool.len() >= SPARE_TRANSIENT_MAX || total.saturating_add(cap) > SPARE_TRANSIENT_BYTES {
        return false;
    }
    pool.push((buffer, data, gpu, cap));
    true
}

/// A pressure-served copy of `[start, start + len)`: never inserted into
/// [`BUFFERS`], [`LOOSE`] or [`EXACT`], never hull-cached (queued per
/// submission), pinned by its lists until they reset. Untracked, so every
/// submission copies it whole from title memory: accuracy-neutral, only
/// busier. `view_copy`/`buffer_srv`/`buffer_uav` keep returning `Some`; only
/// the backing and lifetime change. `None` only with no host at all (the
/// old bug path, never the pressure path).
fn transient_dedicated(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let len = len.max(1);
    if let Some((buffer, data, gpu, cap)) = take_transient_buffer(len) {
        let shadow = Shadow::new_transient(buffer, data, gpu, start, len, Some(cap));
        shadow.touch();
        // The pool held `cap` bytes charged; the copy holds `len`: the
        // constructor charged `len` on top of the pool's still-charged `cap`,
        // so the pool's whole `cap` goes back here.
        use std::sync::atomic::Ordering::Relaxed;
        SHADOW_BYTES.fetch_sub(cap, Relaxed);
        return Some(shadow);
    }
    let h = host()?;
    let (buffer, data) = h.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("pressure copy {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new_transient(buffer, data, gpu, start, len, None);
    shadow.touch();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it. Over the host-copy bound the copy is
/// pressure-served ([`transient_dedicated`], still `true`): heap slots keep
/// its buffer but never the copy itself (each resolution mints a new one),
/// so the slot's buffer and the refreshed copy cannot diverge behind a pin.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        shadow.touch();
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    if let Some(shadow) = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .get(&(address, len, element))
        .and_then(std::sync::Weak::upgrade)
    {
        shadow.touch();
        return Some(shadow);
    }
    // The arena grows no new chunk past the bound: idle copies go first
    // (the evict pass takes the registries itself: never under one), then
    // the view gets a dedicated copy instead of an arena run.
    if pressured(CHUNK) {
        relieve(CHUNK);
    }
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        shadow.touch();
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    // Over the host-copy bound the arena grows no new chunk: the view gets
    // a copy of its own bytes over a dedicated buffer instead (never
    // inserted here, never hull-cached).
    let (buffer, data, gpu, slice) = match arena().take(len, element) {
        Some(run) => run,
        None => {
            drop(guard);
            tracing::debug!(
                address = format_args!("{address:#x}"),
                len,
                "view over the host-copy bound: pressure-served copy"
            );
            return transient_dedicated(address, len);
        }
    };
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice), None);
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                if pressured(size) {
                    // Fully free chunks go before growing past the bound
                    // (no other lock is held under the arena's own: the
                    // evict pass runs before, in `exact`, never here).
                    self.release_empty();
                }
                if pressured(size) {
                    // The caller serves a dedicated copy instead.
                    return None;
                }
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                use std::sync::atomic::Ordering::Relaxed;
                SHADOW_ARENA_BYTES.fetch_add(size, Relaxed);
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }

    /// Drops trailing fully free chunks (their bytes uncharged): the last
    /// resort before growing past the bound. Only trailing chunks go: live
    /// slices name their chunk by index, so only popping the end keeps every
    /// live index valid. A fully free chunk holds no live run by
    /// construction (every live run is out of the free map).
    fn release_empty(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        while let Some((buffer, _, _, free)) = self.chunks.last() {
            // SAFETY: a live buffer.
            let size = unsafe { buffer.GetDesc() }.Width;
            if free.len() != 1 || free.get(&0) != Some(&size) {
                break;
            }
            self.chunks.pop();
            SHADOW_ARENA_BYTES.fetch_sub(size, Relaxed);
        }
    }
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
        // holds the submission lock).
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).expect("not pressured in tests");
        let (_, _, _, b) = arena.take(5000, 16).expect("not pressured in tests");
        let (_, _, _, c) = arena.take(4, 4).expect("not pressured in tests");
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).expect("not pressured in tests");
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let _one = snapshot_tests::ONE.lock().unwrap_or_else(|e| e.into_inner());
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|p| p.to - p.from).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two parts");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "each byte once");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }


    #[test]
    fn buffer_states_follow_submission_order_and_combine_reads() {
        use std::sync::atomic::Ordering;
        let Some(host) = host() else { return };
        let Some((buffer, data)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, 0, 256, None);

        let write = host.new_list().unwrap();
        write.buffer_state(&shadow, crate::host::BUFFER_UNORDERED_ACCESS);
        write.close();
        host.execute(&[&write]);
        assert_eq!(
            shadow.state.load(Ordering::Relaxed),
            crate::host::BUFFER_UNORDERED_ACCESS
        );

        let read = host.new_list().unwrap();
        read.buffer_state(&shadow, crate::host::BUFFER_VERTEX);
        read.buffer_state(&shadow, crate::host::BUFFER_SHADER_RESOURCE);
        read.close();
        host.execute(&[&read]);
        assert_eq!(
            shadow.state.load(Ordering::Relaxed),
            crate::host::BUFFER_VERTEX | crate::host::BUFFER_SHADER_RESOURCE
        );

        let next_write = host.new_list().unwrap();
        next_write.buffer_state(&shadow, crate::host::BUFFER_UNORDERED_ACCESS);
        next_write.close();
        host.execute(&[&next_write]);
        assert_eq!(
            shadow.state.load(Ordering::Relaxed),
            crate::host::BUFFER_UNORDERED_ACCESS
        );
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// A partial refresh of an armed page updates its saved bytes too (both
    /// under and beside a pending marked range): copy-back neither puts the
    /// refreshed byte back over a later title write nor drops a shader write
    /// back to the pre-fault value.
    #[test]
    fn partial_refresh_under_marked_range_updates_saved_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(p, len);
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(p + 16, 16) > 0 && copy.update(p + PAGE + 16, 16) > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p);
        copy.copy_back_marked(p + PAGE);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }

    /// Submissions in flight on one copy: a range declared again while an
    /// earlier submission's snapshot lives starts from the copy as it is
    /// now, so what went back meanwhile is not taken for the GPU's again.
    #[test]
    fn snapshot_stays_fresh_across_disjoint_in_flight_submissions() {
        let _one = snapshot_tests::ONE.lock().unwrap_or_else(|e| e.into_inner());
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, 256);
        let Some(copy) = find(p as u64, 256) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, 256);
        // SAFETY: our pool.
        let title = || unsafe { (p as *const u8).read_volatile() };
        // S1 declares range A, S2 the disjoint range B while S1 is in flight.
        copy.writing(p, 64, false);
        copy.writing(p + 128, 64, false);
        // S1's shaders write byte 0 of A; it goes back while S2 runs.
        copy.write_host(p as u64, &[0x11]);
        copy.copy_back(p, 64);
        copy.submission_done();
        assert_eq!(title(), 0x11);
        // S3 declares A again while S2 keeps the snapshot; the title then
        // writes byte 0 while S3 runs (S3's shaders leave it alone).
        copy.writing(p, 64, false);
        // SAFETY: our pool.
        unsafe { (p as *mut u8).write_volatile(0x22) };
        copy.copy_back(p, 64);
        copy.submission_done();
        assert_eq!(title(), 0x22, "the title's byte, not S1's again");
        // S2 finishes last and releases the snapshot.
        copy.copy_back(p + 128, 64);
        assert!(copy.before.lock().unwrap().0.is_empty());
        unregister(p, p as u64);
    }

    /// Cost of the CPU half of [`evict`] ([`plan`]): idle partition plus LRU
    /// sort over the copies a frame's registry holds. Isolates the new bound's
    /// per-frame bookkeeping from the driver calls that make/free the copies.
    /// Run with
    /// `cargo test --release -p victory-gpu --lib -- --ignored --nocapture benchmark_evict_plan`.
    #[test]
    #[ignore = "benchmark"]
    fn benchmark_evict_plan() {
        use std::hint::black_box;
        use std::time::Instant;

        use crate::resource::{IDLE_FRAMES, IDLE_MS};
        // A pass well after the idle thresholds: three candidates in four
        // used recently, one idle for longer than both thresholds. The
        // budget asks for about a tenth of the bytes (a pressure pass).
        let (frame, now) = (10_000u64, 600_000u64);
        let (idle_frame, idle_ms) = (frame - IDLE_FRAMES - 1, now - IDLE_MS - 1);
        for count in [1024usize, 4096, 16_384] {
            let uses: Vec<(u64, u64, u64)> = (0..count)
                .map(|i| {
                    let bytes = 2048 + (i as u64 % 32) * 1024;
                    if i % 4 == 0 {
                        (idle_frame, idle_ms, bytes)
                    } else {
                        (frame - (i as u64 % 60), now - (i as u64 % 20_000), bytes)
                    }
                })
                .collect();
            let total: u64 = uses.iter().map(|u| u.2).sum();
            let over = total / 10;
            let batch = (64 * 1024 / count).max(1);
            let mut samples = [0.0; 9];
            for sample in &mut samples {
                let start = Instant::now();
                for _ in 0..batch {
                    black_box(plan(black_box(&uses), frame, now, over));
                }
                *sample = start.elapsed().as_secs_f64() * 1e9 / batch as f64;
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{count:6} candidates: {:8.1} ns/pass median ({} chosen)",
                samples[4],
                plan(&uses, frame, now, over).len()
            );
        }
    }

    /// Cost of the CPU half of a pressure-served copy: the pool's best-fit
    /// scan ([`spare_fit`], at most [`SPARE_TRANSIENT_MAX`] entries), against
    /// the buffer make/copy that dominates the path. Run with
    /// `cargo test --release -p victory-gpu --lib -- --ignored --nocapture benchmark_spare_transient_fit`.
    #[test]
    #[ignore = "benchmark"]
    fn benchmark_spare_transient_fit() {
        use std::hint::black_box;
        use std::time::Instant;

        // A full pool, capacities spread as the copy sizes are (a few KB up
        // to tens of MB).
        let caps: Vec<u64> = (0..SPARE_TRANSIENT_MAX)
            .map(|i| 4096u64 << (i % 13))
            .collect();
        for size in [2048u64, 8192, 1 << 20, 8 << 20] {
            let mut samples = [0.0; 9];
            for sample in &mut samples {
                const CALLS: u32 = 200_000;
                let start = Instant::now();
                for _ in 0..CALLS {
                    black_box(spare_fit(black_box(caps.iter().copied()), black_box(size)));
                }
                *sample = start.elapsed().as_secs_f64() * 1e9 / CALLS as f64;
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{size:8} bytes over {} pooled: {:6.1} ns/call median (fit: {:?})",
                caps.len(),
                samples[4],
                spare_fit(caps.iter().copied(), size)
            );
        }
    }
}
