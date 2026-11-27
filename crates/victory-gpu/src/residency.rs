//! Host-memory pressure admission and pool-growth backoff for texture staging.
//!
//! Re-anchored, knob-free core of two cool_fox takes onto this tree (which
//! has no `pool.rs`/`residency.rs`, no texture prefetch pass and no
//! `texture/ahead.rs`; its staging pools are `texture.rs` `BUFFERS`/`CHUNKS`
//! and `host.rs` `SPARE_UPLOADS`):
//!
//! - Pool-backoff core (`ab75ac3e` "Back off refused GPU pool growth and
//!   defer uploads under low commit"): a refused growth or upload allocation
//!   (`E_OUTOFMEMORY`) stops *new* growth for 500 ms while existing blocks
//!   stay usable. Only the core is ported; the draw-pass gates
//!   (`admit_texture_pass`, `BAKE_SERVED`, mid-sweep `upload_refused`
//!   breaks) have no anchor here (no texture-pass budgets) and stay parked
//!   per the spec verdict.
//! - Residency pressure/admission (`e05d4633` `residency.rs`, thresholds from
//!   `706305b4`): hysteresis pressure flag from `GlobalMemoryStatusEx`,
//!   sampled at most once a second from the per-submission sweep
//!   (`texture::recycle`), plus a soft per-frame prefetch cap. Only the
//!   pressure/admission half is ported; the `NOT_RESIDENT` + `WAIT_RESIDENT`
//!   fence-wait path (`0322b39a`/`f1d3eb2b`) stays parked pending a retail
//!   A/B per the spec verdict.
//!
//! Everything here is unconditional: no env knobs. Callers that need new
//! device memory (`texture::staging_buffer`, `texture::staging_space`) ask
//! [`allow_growth`]; reuse of already-live blocks is never gated (a refused
//! pool still serves its blocks, and a sampled draw's upload only defers to
//! a later submission when growth is held).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

const GIB: u64 = 1 << 30;
/// A refused allocation stops new staging growth this long (fork: 500 ms).
/// Existing blocks stay usable throughout.
const BACKOFF_MS: u64 = 500;
/// Pressure samples at most this often (fork: the sweep asks once a second).
const SAMPLE_MS: u64 = 1000;

static PRESSURED: AtomicBool = AtomicBool::new(false);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static NEXT_SAMPLE_AT: AtomicU64 = AtomicU64::new(0);
static RETRY_GROWTH_AT: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

#[cfg(windows)]
fn memory_status() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: the API initializes the fields of a correctly sized structure,
    // and only writes through the pointer we hand it for the call.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is a live, correctly sized `MEMORYSTATUSEX`.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        return None;
    }
    Some((status.ullAvailPhys, status.ullAvailPageFile))
}

#[cfg(not(windows))]
fn memory_status() -> Option<(u64, u64)> {
    None
}

/// `E_OUTOFMEMORY` (`0x8007000E`): the only allocation failure that backs
/// off growth. Other failures (e.g. invalid parameters) are caller bugs, not
/// memory pressure, and must not still the pools.
#[cfg(windows)]
const E_OUTOFMEMORY: i32 = windows_sys::Win32::Foundation::E_OUTOFMEMORY;
#[cfg(not(windows))]
const E_OUTOFMEMORY: i32 = 0x8007000E_u32 as i32;

/// Records a refused device-memory allocation (an HRESULT `code()` value):
/// only `E_OUTOFMEMORY` holds new staging growth for [`BACKOFF_MS`].
pub(crate) fn allocation_error(hresult: i32) {
    if hresult == E_OUTOFMEMORY {
        RETRY_GROWTH_AT.fetch_max(now_ms().saturating_add(BACKOFF_MS), Ordering::Relaxed);
    }
}

/// Whether new staging growth is held after a refused allocation. Reuse of
/// live blocks is never held; growth callers return `None` (the upload stays
/// due for a later submission) instead of asking the driver again.
pub(crate) fn upload_refused() -> bool {
    now_ms() < RETRY_GROWTH_AT.load(Ordering::Relaxed)
}

/// Hysteresis from the fork: pressure enters at 4 GiB free RAM or 6 GiB
/// commit, and only clears past 6 GiB free *and* 8 GiB commit, so a sample
/// straddling the line does not flap the pools every second.
fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was {
        physical < 6 * GIB || commit < 8 * GIB
    } else {
        physical <= 4 * GIB || commit <= 6 * GIB
    }
}

/// Whether the host is under memory pressure (last [`refresh`]).
pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// A soft per-frame staging cap: one texture may cross it, and the pools
/// retain unserved work for later frames (fork: 16 MiB under pressure).
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool) -> u64 {
    if pressure { bytes.min(16 << 20) } else { bytes }
}

/// Called by the per-submission sweep ([`crate::texture::recycle`]): samples
/// the system's memory at most once a second. A failed sample keeps the last
/// good state. Logs only on transitions.
pub(crate) fn refresh() {
    let now = now_ms();
    let next = NEXT_SAMPLE_AT.load(Ordering::Relaxed);
    if now < next {
        return;
    }
    if NEXT_SAMPLE_AT
        .compare_exchange(
            next,
            now.saturating_add(SAMPLE_MS),
            Ordering::Relaxed,
            Ordering::Relaxed,
        )
        .is_err()
    {
        return;
    }
    let Some((physical, commit)) = memory_status() else {
        return;
    };
    AVAILABLE_COMMIT.store(commit, Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                "host memory pressure: trimming idle staging pools"
            );
        } else {
            tracing::info!(
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                "host memory pressure cleared"
            );
        }
    }
}

/// Commit that must remain for live game allocations (fork: 2 GiB for a
/// bake-grade pass, 4 GiB for lookahead-grade growth).
fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    commit >= if bake { 2 * GIB } else { 4 * GIB }
}

/// Whether staging may grow a new block now: never inside the post-refusal
/// backoff window, and only while enough commit remains for the game and
/// submissions already in flight. Until the first sample the commit is
/// unknown and growth is allowed (failures then arm the backoff instead).
pub(crate) fn allow_growth() -> bool {
    !upload_refused() && has_upload_reserve(AVAILABLE_COMMIT.load(Ordering::Relaxed), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure(false, 12 * GIB, 6 * GIB));
        assert!(next_pressure(false, 4 * GIB, 16 * GIB));
        assert!(next_pressure(true, 5 * GIB, 9 * GIB));
        assert!(next_pressure(true, 7 * GIB, 7 * GIB));
        assert!(!next_pressure(true, 6 * GIB, 8 * GIB));
        assert!(!next_pressure(false, 5 * GIB, 7 * GIB));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit(0, true), 0);
        assert_eq!(prefetch_limit(8 << 20, true), 8 << 20);
        assert_eq!(prefetch_limit(512 << 20, true), 16 << 20);
        assert_eq!(prefetch_limit(192 << 20, false), 192 << 20);
    }

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead) in [
            (277 << 20, false, false),
            (2 * GIB, true, false),
            (4 * GIB, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
        }
    }

    #[test]
    fn refused_uploads_hold_growth_for_half_a_second() {
        // Mutates the shared backoff clock, so restore it at once: the
        // window below lasts only for these synchronous assertions (no
        // sleeps), and production only arms it on a real E_OUTOFMEMORY.
        let prev = RETRY_GROWTH_AT.load(Ordering::Relaxed);
        RETRY_GROWTH_AT.store(0, Ordering::Relaxed);
        assert!(allow_growth());
        assert!(!upload_refused());
        // Anything but E_OUTOFMEMORY is a caller bug, not pressure.
        allocation_error(0);
        assert!(allow_growth());
        allocation_error(E_OUTOFMEMORY);
        assert!(upload_refused());
        assert!(!allow_growth());
        RETRY_GROWTH_AT.store(prev, Ordering::Relaxed);
    }
}
