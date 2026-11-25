//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! Submissions are not waited for: each signals the next value of the host
//! fence and joins the [`Timeline`], and is finished (what its shaders
//! wrote copied back into title memory, then its bottom-of-pipe writes and
//! the title fence signals queued behind it) in submission order once the
//! GPU has run it: by our completion thread, or by whichever thread first
//! needs it (a later submission's step touching memory it writes, a reset
//! of one of its lists, a present).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::timeline::{Timeline, Touches};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// The copy engine for texture uploads (`VICTORY_COPY_QUEUE`): its
    /// queue and fence, and the last value signalled on it, behind its own
    /// lock. `None` where the device has none (or the switch is off and no
    /// copy work was ever submitted): the direct queue runs uploads as
    /// before. The graphics queue and fence above are untouched.
    copy_queue: Option<ID3D12CommandQueue>,
    copy_fence: Option<ID3D12Fence>,
    copy_submitted: Mutex<u64>,
    copy_lock: Mutex<()>,
    /// Submissions not finished yet, oldest first.
    timeline: Timeline<Pending>,
    /// Whether the completion thread runs.
    finisher: std::sync::Once,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the timeline's lists are pinned while in it (`Pending`).
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Starvation counters (diagnostics, `VICTORY_GPU_TIMING`): the timeline
/// depth histogram (submissions pushed at each depth since the last
/// frame-rate line) and per-cause CPU blocking milliseconds (time in
/// `settle` waits by what waited). Timing-only: measured around the waits
/// that already happen, never added.
#[cfg(feature = "diagnostics")]
pub(crate) mod starve {
    /// Depth buckets `0..=16` (submissions in flight after a push).
    pub(crate) const DEPTHS: usize = 17;
    /// Blocking causes: in-flight full, list still running, range conflict,
    /// staging/table conflict, drain/present wait, copy-queue wait.
    pub(crate) const CAUSES: usize = 6;
    pub(crate) const IN_FLIGHT: usize = 0;
    pub(crate) const RUNNING: usize = 1;
    pub(crate) const CONFLICT: usize = 2;
    pub(crate) const STAGING_TABLE: usize = 3;
    pub(crate) const DRAIN_PRESENT: usize = 4;
    pub(crate) const COPY_WAIT: usize = 5;

    static HIST: [std::sync::atomic::AtomicU64; DEPTHS] =
        [const { std::sync::atomic::AtomicU64::new(0) }; DEPTHS];
    static BLOCK_US: [std::sync::atomic::AtomicU64; CAUSES] =
        [const { std::sync::atomic::AtomicU64::new(0) }; CAUSES];

    pub(crate) fn note_depth(len: usize) {
        HIST[len.min(DEPTHS - 1)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn note_block(cause: usize, us: u64) {
        if cause < CAUSES {
            BLOCK_US[cause].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub(crate) fn take() -> ([u64; DEPTHS], [u64; CAUSES]) {
        let mut hist = [0u64; DEPTHS];
        for (i, h) in hist.iter_mut().enumerate() {
            *h = HIST[i].swap(0, std::sync::atomic::Ordering::Relaxed);
        }
        let mut block = [0u64; CAUSES];
        for (i, b) in block.iter_mut().enumerate() {
            *b = BLOCK_US[i].swap(0, std::sync::atomic::Ordering::Relaxed);
        }
        (hist, block)
    }
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// How a wait for a submission ended (tests: the async path waits on the
/// host fence instead, and the completion thread finishes what has run).
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Waited {
    /// The host signalled the submission's fence.
    Done,
    /// The device was removed while the submission ran.
    Removed,
    /// Neither after [`STALL_SECS`]: the GPU has faulted or hung and the
    /// host has not said so.
    Stalled,
}

/// How long one wait for a submission lasts before the device is asked
/// whether it is still there.
#[cfg(test)]
const WAIT_SLICE_MS: u32 = 250;

/// A submission still running after this many seconds is said so, and again
/// each time that time has doubled.
#[cfg(test)]
const LONG_WAIT_SECS: u64 = 10;

/// A submission still running after this many seconds is given up. A host
/// whose GPU faults may neither finish the work nor report the device
/// removed: vkd3d-proton waits on the driver without an end unless
/// `VKD3D_CONFIG=fault` or `breadcrumbs` is set, and says itself that such
/// a wait of over 5 s is a lost device. Windows resets a device whose
/// submission runs for about 2 s. The title ends its own run when its main
/// thread has waited 30 s for the render thread, which a submission that
/// never finishes brings about; this is below that, so the run ends with
/// our report of the cause and not with the title's.
#[cfg(test)]
const STALL_SECS: u64 = 25;

/// Waits for a submission: `signalled` waits one slice ([`WAIT_SLICE_MS`])
/// for its fence and says whether it came. Between slices `removed` asks
/// the device whether it was removed, because a host that loses its device
/// need not signal the fences of the work it dropped (vkd3d-proton after a
/// GPU fault does not), and a wait without an end then holds the title's
/// submit thread until the title ends the run, with no report of ours.
/// `long` is told the seconds waited so far, at [`LONG_WAIT_SECS`] and each
/// doubling of it; at [`STALL_SECS`] the wait is given up.
#[cfg(test)]
fn wait_for_submission(
    mut signalled: impl FnMut() -> bool,
    mut removed: impl FnMut() -> bool,
    mut long: impl FnMut(u64),
) -> Waited {
    let mut slices = 0u64;
    let mut report_at = LONG_WAIT_SECS;
    loop {
        if signalled() {
            return Waited::Done;
        }
        if removed() {
            return Waited::Removed;
        }
        slices += 1;
        let secs = slices * u64::from(WAIT_SLICE_MS) / 1000;
        if secs >= STALL_SECS {
            return Waited::Stalled;
        }
        if secs >= report_at {
            long(secs);
            report_at *= 2;
        }
    }
}

static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU if it was made (never makes it).
pub(crate) fn made() -> Option<&'static Host> {
    HOST.get()?.as_ref()
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        // The copy engine for texture uploads, alongside the graphics queue
        // (created here, used only behind `VICTORY_COPY_QUEUE`).
        // SAFETY: plain D3D12 object creation with a valid description.
        let (copy_queue, copy_fence) = match device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            ..Default::default()
        }) {
            Ok(q) => {
                let f = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok();
                (Some(q), f)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host D3D12 copy queue; uploads stay on the direct queue");
                (None, None)
            }
        };
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        // Self-identifying diagnostics: one line proving from the log alone
        // which gates were active (a run with VICTORY_COPY_QUEUE=1 left no
        // marker). Diagnostics only; no behavior change.
        #[cfg(feature = "diagnostics")]
        tracing::info!(
            "host diagnostics self-id v1: VICTORY_COPY_QUEUE={} VICTORY_GPU_TIMING={} profile={} WARP={} GBV={} DRED={} DEBUG={}",
            crate::switch::on("VICTORY_COPY_QUEUE"),
            crate::switch::on("VICTORY_GPU_TIMING"),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            crate::switch::on("VICTORY_D3D12_WARP"),
            crate::switch::on("VICTORY_D3D12_GBV"),
            crate::switch::on("VICTORY_D3D12_DRED"),
            crate::switch::on("VICTORY_D3D12_DEBUG"),
        );
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match (victory_shader::emit::host::Host {
            device: device.clone(),
        })
        .fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            copy_queue,
            copy_fence,
            copy_submitted: Mutex::new(0),
            copy_lock: Mutex::new(()),
            timeline: Timeline::new(),
            finisher: std::sync::Once::new(),
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// Diagnostics: our process's private commit and working set, bytes.
#[cfg(feature = "diagnostics")]
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.PagefileUsage as u64, c.WorkingSetSize as u64)
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: GPU time per kind of command and
/// pipeline (ms, commands) and lists' wall time against the sum of their
/// commands' (ms), since the last frame-rate line.
#[cfg(feature = "diagnostics")]
type GpuTimes = (std::collections::HashMap<String, (f64, u64)>, f64, f64);
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<GpuTimes>> = Mutex::new(None);

/// Adds a timed list's commands (from [`HostList::slowest`]) to [`GPU_TIMES`],
/// by the command's kind and pipeline: its note before the first ` | `,
/// without counts and addresses.
#[cfg(feature = "diagnostics")]
fn gpu_times(wall: Duration, commands: &[(f64, String)]) {
    let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let (by, wall_ms, sum) = times.get_or_insert_with(Default::default);
    *wall_ms += wall.as_secs_f64() * 1e3;
    for (ms, note) in commands {
        // "draw 3x1 [mesh 1f.., pixel 2e..] | ...": the kind and the pipeline.
        let head = note.split(" | ").next().unwrap_or_default();
        let kind = head.split(' ').next().unwrap_or_default();
        let key = match head.find('[') {
            Some(at) => format!("{kind} {}", &head[at..]),
            None => kind.to_owned(),
        };
        let e = by.entry(key).or_default();
        (e.0, e.1) = (e.0 + ms, e.1 + 1);
        *sum += ms;
    }
}

/// Logs [`GPU_TIMES`] per frame over `frames` frames: the 25 costliest.
#[cfg(feature = "diagnostics")]
fn log_gpu_times(frames: f64) {
    let Some((by, wall, sum)) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    tracing::info!(
        "GPU timing: lists {:.1} ms/frame of wall time, their commands {:.1} ms/frame",
        wall / frames,
        sum / frames
    );
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (key, (ms, n)) in by.iter().take(25) {
        tracing::info!("  GPU {:.2} ms/frame in {:.1} commands/frame: {key}", ms / frames, *n as f64 / frames);
    }
}

/// Diagnostics: the host device's video memory use and budget, in MB, local
/// (on the card) and non-local (system memory the GPU reads), as the
/// system's memory manager sees them: use over budget makes it page
/// allocations at submissions.
pub(crate) fn video_memory() -> Option<[(f64, f64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let adapter = ADAPTER
        .get_or_init(|| {
            let device = &host()?.device;
            // SAFETY: plain device and DXGI queries.
            unsafe {
                let luid = device.GetAdapterLuid();
                let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid(luid).ok()
            }
        })
        .as_ref()?;
    let info = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a valid out pointer.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage as f64 / 1e6, info.Budget as f64 / 1e6))
    };
    Some([info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?, info(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?])
}

#[cfg(test)]
pub(crate) mod tests {
    /// A submission's wait ends when its fence comes, or when the device is
    /// gone and the fence never will.
    #[test]
    fn a_wait_for_a_submission_ends_when_the_device_is_removed() {
        use super::{LONG_WAIT_SECS, STALL_SECS, WAIT_SLICE_MS, Waited, wait_for_submission};
        let slices_per_sec = 1000 / u64::from(WAIT_SLICE_MS);
        // The fence comes in the third slice: the device is asked twice.
        let (mut waits, mut asked) = (0, 0);
        let done = wait_for_submission(
            || {
                waits += 1;
                waits == 3
            },
            || {
                asked += 1;
                false
            },
            |_| panic!("not a long wait"),
        );
        assert_eq!((done, waits, asked), (Waited::Done, 3, 2));
        // The fence never comes and the device is removed in the fifth slice.
        let mut asked = 0;
        let lost = wait_for_submission(
            || false,
            || {
                asked += 1;
                asked == 5
            },
            |_| panic!("not a long wait"),
        );
        assert_eq!((lost, asked), (Waited::Removed, 5));
        // A fence signalled together with the removal is a finished wait:
        // the caller's check after the submission reports the removal.
        assert_eq!(wait_for_submission(|| true, || true, |_| ()), Waited::Done);
        // A long submission on a live device is waited for, and said so at
        // 10 s and 20 s.
        let (mut waits, mut said) = (0, Vec::new());
        let done = wait_for_submission(
            || {
                waits += 1;
                waits > 22 * slices_per_sec
            },
            || false,
            |secs| said.push(secs),
        );
        assert_eq!(done, Waited::Done);
        assert_eq!(said, [LONG_WAIT_SECS, 2 * LONG_WAIT_SECS]);
        // One that neither finishes nor loses its device is given up, after
        // every slice of the time allowed and no sooner.
        let mut waits = 0;
        let stalled = wait_for_submission(
            || {
                waits += 1;
                false
            },
            || false,
            |_| (),
        );
        assert_eq!((stalled, waits), (Waited::Stalled, STALL_SECS * slices_per_sec));
    }

    thread_local! {
        /// Every transition barrier built on this thread (resource, before,
        /// after), in order.
        pub(crate) static ISSUED: std::cell::RefCell<Vec<(usize, u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The barriers built on this thread since the last call.
    pub(crate) fn issued() -> Vec<(usize, u32, u32)> {
        ISSUED.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    /// Replays transitions in the order the GPU runs them from the states
    /// in `states` (resources not in it are not checked), by the debug
    /// layer's rule: each barrier's before-state is the resource's current
    /// one. The first that is not, else the states after.
    pub(crate) fn replay(issued: &[(usize, u32, u32)], states: &mut std::collections::HashMap<usize, u32>) -> Result<(), String> {
        for (k, &(resource, before, after)) in issued.iter().enumerate() {
            if let Some(state) = states.get_mut(&resource) {
                if *state != before {
                    return Err(format!("barrier {k} on {resource:#x}: before {before:#x}, but the resource is in {state:#x}"));
                }
                *state = after;
            }
        }
        Ok(())
    }

    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    /// Cross-queue upload order: the COPY engine leaves a texture in
    /// COPY_DEST, and the DIRECT side makes it SHADER_RESOURCE after the
    /// copy Wait. Replays the two barriers the queues record (as
    /// [`transition`] issues them) from COMMON: the copy barrier then the
    /// direct one runs, and the texture ends readable by shaders. No GPU
    /// required: the barrier values mirror `record_upload` (COPY) and the
    /// DIRECT prelude/first `move_texture(SHADER_RESOURCE)`.
    #[test]
    fn copy_queue_barrier_order_leaves_textures_readable() {
        const COMMON: u32 = 0x1;
        const COPY_DEST: u32 = 0x400;
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let resource = 0x1234_5000usize;
        // COPY then DIRECT: runs from COMMON to readable.
        let barriers = vec![
            (resource, COMMON, COPY_DEST),
            (resource, COPY_DEST, SHADER_RESOURCE),
        ];
        let mut states = std::collections::HashMap::from([(resource, COMMON)]);
        replay(&barriers, &mut states).expect("copy then direct barriers run");
        assert_eq!(states[&resource], SHADER_RESOURCE);
        // DIRECT without the COPY barrier first does not run: the resource
        // is still in COPY_DEST only after the copy engine ran.
        let mut left_in_copy = std::collections::HashMap::from([(resource, COPY_DEST)]);
        let direct_only = vec![(resource, COMMON, SHADER_RESOURCE)];
        assert!(
            replay(&direct_only, &mut left_in_copy).is_err(),
            "a DIRECT SHADER_RESOURCE barrier from COMMON does not match a texture the copy engine left in COPY_DEST"
        );
        // Dropped COPY_DEST barrier: the texture never becomes readable.
        let mut states = std::collections::HashMap::from([(resource, COMMON)]);
        let copy_only = vec![(resource, COMMON, COPY_DEST)];
        replay(&copy_only, &mut states).expect("the copy barrier runs");
        assert_eq!(states[&resource], COPY_DEST, "left for the DIRECT side");
        let taken = issued();
        let _ = taken;
    }

    /// The copy-queue rollout switch defaults off, the timeline waits for
    /// both queues, and staging retires on both completeds. No GPU required
    /// except for the list-kind checks (early-return without one, as the
    /// other host tests do).
    #[test]
    fn copy_queue_offload_is_gated_and_timeline_waits_for_both_queues() {
        // Rollout switch: default off (no `VICTORY_COPY_QUEUE` in the test
        // environment). When on, uploads move to the copy list; when off,
        // they stay on the direct queue exactly as before.
        assert!(
            !crate::switch::on("VICTORY_COPY_QUEUE") || copy_enabled(),
            "gate reflects VICTORY_COPY_QUEUE"
        );
        // The timeline finishes a submission only once both its queues ran
        // it: direct alone does not finish copy work.
        let t: crate::timeline::Timeline<()> = crate::timeline::Timeline::new();
        t.push(1, 1, crate::timeline::Touches::default(), ());
        t.finish_both(1, 0, |()| panic!("copy work not yet run"));
        assert_eq!(t.len(), 1);
        t.finish_both(1, 1, |()| {});
        assert_eq!(t.len(), 0);
        // Staging retirement on both completeds is covered by `recycle` (which
        // waits for both queues); here we only check the counters exist.
        // Unknown resources uploaded/copied nothing.
        assert_eq!(crate::texture::uploads(std::ptr::null()), 0);
        assert_eq!(crate::texture::copied(std::ptr::null()), 0);
        let Some(host) = host() else { return };
        // Without the switch the copy fence is already passed (MAX), so the
        // direct fence alone decides; list kinds never mix.
        if !copy_enabled() {
            assert_eq!(host.copy_completed(), u64::MAX);
        }
        let direct = host.new_list().expect("a direct list");
        assert!(!direct.is_copy());
        let copy = host.new_copy_list().expect("a copy list");
        assert!(copy.is_copy());
    }

    use super::*;

    /// Shaders that use title memory mark their reads in their own bitmap:
    /// read marks of both bitmaps are taken, each slot once until it is
    /// forgotten, and a forgotten slot's next read is taken again.
    #[test]
    fn read_marks_of_both_bitmaps_are_taken() {
        let Some(host) = host() else { return };
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let (plain, memory) = (victory_shader::emit::HEAP_VIEWS - 2, victory_shader::emit::HEAP_VIEWS - 1);
        crate::shadow::tests::mark_read(false, plain);
        crate::shadow::tests::mark_read(true, memory);
        let taken = crate::shadow::take_reads();
        assert!(taken.contains(&plain), "a plain shader's read");
        assert!(taken.contains(&memory), "a read by a shader that uses title memory");
        let again = crate::shadow::take_reads();
        assert!(!again.contains(&plain) && !again.contains(&memory), "each once");
        crate::shadow::forget_read(memory);
        assert!(!crate::shadow::take_reads().contains(&memory), "forgotten: its mark cleared");
        crate::shadow::tests::mark_read(true, memory);
        assert!(crate::shadow::take_reads().contains(&memory), "read again after it was forgotten");
    }

    /// The textures shaders read through the heap since the last eviction
    /// pass are named again at each pass (a slot's read is reported as new
    /// only once), and the marks are cleared for the next.
    #[test]
    fn heap_reads_since_the_last_pass_name_their_textures() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(texture).unwrap().texture) as usize;
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..2 {
            crate::shadow::tests::mark_read(false, 0);
            let slots = crate::shadow::reads_since();
            assert!(slots.contains(&0), "the slot read");
            assert!(crate::descriptor::read_textures(&slots).contains(&raw), "its texture named");
            assert!(!crate::shadow::reads_since().contains(&0), "the marks cleared");
        }
    }

    /// The idle rule against the heap reads that thrashed (run evict1): a
    /// texture shaders read through the heap in every frame stays over more
    /// frames and time than the idle limits; one read once is given up. A
    /// list of the title's naming a texture (a table read, a copy) stamps it
    /// as used; a list of ours (uploads) does not.
    #[test]
    fn textures_read_every_frame_stay_and_title_lists_stamp_theirs() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let step = 32;
        // Two 4x1 RGBA8 textures, linear, in memory of their own, viewed at
        // slots 5 and 6.
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let heap_host = title_heap.host().expect("a host view heap");
        let textures: Vec<crate::object::This> = [5u32, 6]
            .iter()
            .map(|&slot| {
                let address = memory.as_ptr() as u64 + u64::from(slot) * (1 << 13);
                let mut view = [0u32; 16];
                view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                let at = title_heap.start + u64::from(slot) * step;
                crate::descriptor::create_placed_srv(null, &words, &view, at);
                assert!(heap_host.mirror(slot, at, true).is_some());
                crate::resource::texture_at(address, &words).expect("its resource")
            })
            .collect();
        let owned: Vec<_> = textures.iter().map(|&t| crate::resource::owned(t).expect("may be given up")).collect();
        let (idle_frames, idle_ms) = (crate::resource::IDLE_FRAMES, crate::resource::IDLE_MS);
        {
            // No submission takes the marks meanwhile.
            let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Passes a third of the idle limits apart, over three times them.
            for k in 0..10 {
                crate::shadow::tests::mark_read(false, 5);
                if k == 0 {
                    crate::shadow::tests::mark_read(false, 6);
                }
                let read = crate::descriptor::read_textures(&crate::shadow::reads_since());
                crate::resource::evict_among(&owned, &read, 1000 + k * idle_frames / 3, 1_000_000 + k * idle_ms / 3, 0);
                assert!(crate::resource::made_copy(textures[0]).is_some(), "read every frame: stays (frame {k})");
            }
        }
        assert!(crate::resource::made_copy(textures[1]).is_none(), "read once: given up");
        drop(owned);
        // A title list naming the texture stamps it, an upload list does not.
        let copy = crate::resource::host_copy(textures[0]).unwrap();
        for (ours, stamped) in [(true, false), (false, true)] {
            copy.unstamp();
            let list = host.new_list().unwrap();
            if ours {
                list.set_ours();
            }
            let _ = list.transition(textures[0], 0x40);
            assert_eq!(copy.last_used().0 != u64::MAX, stamped, "ours: {ours}");
        }
    }

    /// Textures the heaps view are brought up to date only if the title used
    /// them lately: one whose pages changed but that no shader read is not
    /// uploaded; read in the last pass, it is.
    #[test]
    fn heap_textures_unread_lately_are_not_brought_up() {
        let Some(host) = host() else { return };
        let address = crate::watch::tests::pool(1) as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut object = std::ptr::null();
        let null = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(null, address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
        let copy = crate::resource::host_copy(object).expect("a host texture");
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        viewed.set(0, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        copy.unstamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 0, "not read lately: not brought up");
        copy.stamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 1, "read in the last pass: brought up");
    }

    /// Heap textures brought up in one submission are refreshed side by
    /// side, each into its own staging and its own upload.
    #[test]
    fn heap_textures_brought_up_together_each_get_their_upload() {
        let Some(host) = host() else { return };
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let pool = crate::watch::tests::pool(6);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        let objects: Vec<crate::object::This> = (0..6u32)
            .map(|k| {
                // RGBA8 128x128 (one page each), linear, each its own bytes.
                let address = (pool + k as usize * crate::watch::PAGE) as u64;
                // SAFETY: the texture's own page of the pool.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x10 * k as u8, crate::watch::PAGE) };
                let words: [u32; 14] = [3, 0, 0, 0, 128, 0, 128, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                let mut object = std::ptr::null();
                assert_eq!(crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
                crate::resource::host_copy(object).expect("a host texture").stamp();
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                viewed.set(k, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
                object
            })
            .collect();
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "each brought up once");
        }
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "unchanged: not again");
        }
    }

    #[test]
    fn list_scratch_grows_to_fit_and_reuses_fenced_chunks() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        host.drain();
        list.reset();
        // The finished submission's chunks are reused from the start.
        assert_eq!(list.scratch(512).unwrap(), first);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        list.close();
        host.execute(&[&list]);
        host.drain();
        list.reset();
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }
}

/// Which host queue a list runs on: the graphics queue, or the copy engine
/// (`VICTORY_COPY_QUEUE`) for texture uploads. COPY lists allow only copy
/// and barrier ops; draws, dispatches and bindings are a bug on them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ListKind {
    Direct,
    Copy,
}

/// Whether texture uploads run on the copy engine (`VICTORY_COPY_QUEUE=1`;
/// rollout switch, default off until validated). When off, uploads run on
/// the direct queue exactly as before.
pub(crate) fn copy_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_COPY_QUEUE"))
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    /// Which queue the list runs on.
    kind: ListKind,
    /// Swapped for a fresh one when the list is reset while it runs.
    allocator: Mutex<ID3D12CommandAllocator>,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Buffer states have the same submission ordering, but live on the
    /// persistent shadow buffers rather than on title texture copies.
    buffer_states: Mutex<BufferStates>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// A list of ours (uploads), whose uses are no use of the title's
    /// (`crate::resource::evict`).
    ours: std::sync::atomic::AtomicBool,
    /// The draw or dispatch being recorded runs a program using title
    /// memory ([`HostList::writer_uses_memory`]).
    memory_writer: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: Mutex<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: Mutex<Option<(ID3D12Resource, usize)>>,
    /// The host fence value of its last submission (0: none): it is not
    /// reset, resubmitted or dropped before that one has finished.
    last: AtomicU64,
    /// Up to which of its submissions the list's memory was handed to the
    /// timeline by a reset: its recording since uses none of theirs.
    fresh_after: AtomicU64,
    /// Clamped indirect arguments (`crate::indirect`), reused by each
    /// execution in turn; replaced by a bigger one when too small.
    arguments: Mutex<Option<ID3D12Resource>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

impl Drop for HostList {
    /// Its allocators and upload memory may not go while the GPU runs it,
    /// and a submission in flight still names it.
    fn drop(&mut self) {
        self.settle();
    }
}

// SAFETY: D3D12 objects are free-threaded; a host list is recorded by the
// title list's one recording thread, then run by one submission at a time
// (the queues hold it between, `crate::hold`).
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

/// Timestamps per list (diagnostics); commands past them are not timed.
/// The last two slots are the list's busy pair (start at open, end at
/// close), so the GPU time of un-noted work is an explicit residual.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 16384;

/// The busy pair's slots in a list's timestamp heap (diagnostics, timing).
#[cfg(feature = "diagnostics")]
const BUSY_START: u32 = STAMPS - 2;
#[cfg(feature = "diagnostics")]
const BUSY_END: u32 = STAMPS - 1;

/// Notes with timestamps at most (diagnostics, timing): indices `0` through
/// this for `note_op`, this for the end stamp in `close`.
#[cfg(feature = "diagnostics")]
const MAX_TIMED_COUNT: u32 = STAMPS - 3;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

type BufferStates =
    std::collections::HashMap<usize, (u32, u32, std::sync::Arc<crate::shadow::Shadow>)>;

pub(crate) const BUFFER_VERTEX: u32 = 0x1;
pub(crate) const BUFFER_INDEX: u32 = 0x2;
pub(crate) const BUFFER_SHADER_RESOURCE: u32 = 0x40 | 0x80;
pub(crate) const BUFFER_UNORDERED_ACCESS: u32 = 0x8;
pub(crate) const BUFFER_INDIRECT_ARGUMENT: u32 = 0x200;

const BUFFER_READ_STATES: u32 =
    BUFFER_VERTEX | BUFFER_INDEX | BUFFER_SHADER_RESOURCE | BUFFER_INDIRECT_ARGUMENT;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether the GPU time of each list's noted commands is read back, and slow
/// lists are logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` or the timing on).
/// Timing keeps the normal cuts grouping for execution (the lists run as they
/// do with the timing off); per-list alone execution, which is also what lets
/// a device removal name the list it happened in, is opt-in (below).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Whether timed lists each run alone in their own submission piece
/// (`VICTORY_GPU_TIMING_ALONE=1`, diagnostics builds with `VICTORY_GPU_TIMING`
/// on). Opt-in: the default timing keeps the normal cuts grouping, so the
/// GPU executes the same batching with the timing off or on.
#[cfg(feature = "diagnostics")]
fn timing_alone() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING_ALONE"))
}

/// Whether recorded commands are noted at all: with device-removed data (the
/// markers) or with GPU timing (the per-command timestamps) on.
#[cfg(feature = "diagnostics")]
fn noting() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| dred_on() || timing())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// What a list reset while it ran leaves to its running submission.
struct Retired {
    kind: ListKind,
    allocator: ID3D12CommandAllocator,
    prelude: Prelude,
    deferred: Deferred,
    scratch: Scratch,
    states: States,
    buffer_states: BufferStates,
    twins: std::collections::HashMap<usize, crate::twin::Use>,
    budget: Option<(ID3D12Resource, usize)>,
}

// SAFETY: D3D12 objects are free-threaded; the scratch and budget pointers
// are persistent mappings of their buffers, which go with them.
unsafe impl Send for Retired {}

type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Allocators, preludes and scratch chunks whose submissions finished,
/// for lists reset while they run (dozens per world frame). DIRECT and COPY
/// allocators are never mixed: a COPY allocator resets only a COPY list.
static SPARE_ALLOCATORS: Mutex<Vec<ID3D12CommandAllocator>> = Mutex::new(Vec::new());
static SPARE_ALLOCATORS_COPY: Mutex<Vec<ID3D12CommandAllocator>> = Mutex::new(Vec::new());
static SPARE_PRELUDES: Mutex<Vec<Prelude>> = Mutex::new(Vec::new());
static SPARE_PRELUDES_COPY: Mutex<Vec<Prelude>> = Mutex::new(Vec::new());
static SPARE_SCRATCH: Mutex<Vec<(ID3D12Resource, usize, u64)>> = Mutex::new(Vec::new());
/// Spares kept of each kind at most.
const SPARES: usize = 256;

fn pool<T>(spares: &'static Mutex<Vec<T>>) -> std::sync::MutexGuard<'static, Vec<T>> {
    spares.lock().unwrap_or_else(|e| e.into_inner())
}

impl Retired {
    /// Once the submission that used them has finished.
    fn retire(self) {
        // DIRECT and COPY pools stay apart: an allocator resets only its own kind.
        let (alloc_pool, prelude_pool) = match self.kind {
            ListKind::Direct => (&SPARE_ALLOCATORS, &SPARE_PRELUDES),
            ListKind::Copy => (&SPARE_ALLOCATORS_COPY, &SPARE_PRELUDES_COPY),
        };
        // SAFETY: nothing running uses the allocators any more.
        unsafe {
            if self.allocator.Reset().is_ok() && pool(alloc_pool).len() < SPARES {
                pool(alloc_pool).push(self.allocator);
            }
            if self.prelude.0.Reset().is_ok() && pool(prelude_pool).len() < SPARES {
                pool(prelude_pool).push(self.prelude);
            }
        }
        let mut spare = pool(&SPARE_SCRATCH);
        let room = SPARES.saturating_sub(spare.len());
        spare.extend(self.scratch.chunks.into_iter().take(room));
        drop((self.deferred, self.states, self.buffer_states, self.twins, self.budget));
    }
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory: the range it writes, the range it
    /// reads (address, bytes).
    Run(Box<dyn Fn() + Send>, (usize, usize), (usize, usize)),
    /// Work that writes host memory only: the title memory it reads, and
    /// the list's own memory it writes (pointer, bytes).
    Host(Box<dyn Fn() + Send>, (usize, usize), (usize, usize)),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then), and
    /// whether their program uses title memory (then the range is withheld
    /// when such a list misses, `crate::memory::withholding`).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

/// A piece of a query result: the title's address, the offset it has in
/// the read-back buffer, and its width.
type Target = (usize, u64, usize);

/// What a repeat of a list in one submission writes into the list's own
/// memory, which its earlier run has not read yet: the bytes in an upload
/// buffer of the submission's, and the copies its prelude records (list
/// buffer, offset, offset in `staging`, bytes).
struct Repeat {
    staging: ID3D12Resource,
    copies: Vec<(ID3D12Resource, u64, u64, u64)>,
}

/// Dynamic descriptor table slots leased by a recording; released when the
/// recording drops (after its async submission completes if reset while
/// running).
struct DynamicLease(u64, u32, u32);

impl Drop for DynamicLease {
    fn drop(&mut self) {
        match (self.0, crate::memory::heap()) {
            (crate::memory::OURS_ID, Some(ours)) => ours.release(self.1, self.2),
            (crate::memory::OURS_ID, None) => {}
            _ => crate::descriptor::release_dynamic(self.0, self.1, self.2),
        }
    }
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// The list's upload buffers: buffer, mapped pointer, bytes.
    uploads: Vec<(ID3D12Resource, usize, usize)>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Dynamic descriptor ranges leased by this recording, retired with it
    /// if the list resets before its submission finishes.
    dynamic: Vec<DynamicLease>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Query results the list resolved into a read-back buffer, written to
    /// the title's memory once the list has run: the buffer, then each
    /// (destination, offset in the buffer, bytes).
    resolves: Vec<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, Vec<Target>)>,
    /// The memory values the list waits for before it starts (a list the
    /// title records with a wait after commands is cut there,
    /// `crate::list`).
    waits: Vec<crate::hold::ValueWait>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<std::sync::Arc<dyn Fn() + Send + Sync>>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<std::sync::Arc<dyn Fn() -> bool + Send + Sync>>,
}

/// Submissions in flight at most: a submission past them waits for the
/// oldest first.
const IN_FLIGHT: usize = 16;

/// A list a submission in flight names.
#[derive(Clone, Copy)]
struct ListRef(*const HostList);

// SAFETY: host lists are used from any thread (their state is behind
// locks), and one is not dropped while a submission naming it is in flight
// (`HostList::last`); a reset hands the memory such a submission uses to
// the timeline instead of reusing it.
unsafe impl Send for ListRef {}

impl ListRef {
    fn get<'a>(self) -> &'a HostList {
        // SAFETY: as above, the list outlives the submission.
        unsafe { &*self.0 }
    }
}

/// A range shaders write (host copy, address, bytes) and the runs (by
/// index) that declared it.
type Back = (
    std::sync::Arc<crate::shadow::Shadow>,
    usize,
    usize,
    Vec<(usize, bool)>,
);

/// What is left of a submission once the GPU has run it.
struct Pending {
    lists: Vec<ListRef>,
    /// The host fence value of its direct submission (0: none) and of its
    /// copy submission (0: none; uploads stayed on the direct queue).
    value: u64,
    copy_value: u64,
    /// The ranges its shaders write (host copy, address, bytes), each once,
    /// and the runs that declared each.
    back: Vec<Back>,
    /// Its bottom-of-pipe writes, per run in list order: address, what,
    /// bytes. Query results go back before the writes of their own run.
    writes: Vec<Vec<(usize, crate::queue::Written, usize)>>,
    /// Per run, the query results it resolved into read-back buffers, in
    /// list order: the buffer, then each (destination, offset, bytes).
    resolves: Vec<Vec<(ID3D12Resource, Vec<Target>)>>,
    /// The preludes of lists it names more than once, past their first:
    /// each recording has its own, back to the spares once it has run.
    preludes: Vec<(Prelude, ListKind)>,
    submitted: std::time::Instant,
    /// Per run, asked once it has run: whether its shaders read title
    /// memory with no copy (then what they wrote does not go back).
    missed: Vec<Vec<std::sync::Arc<dyn Fn() -> bool + Send + Sync>>>,
    /// Work once it has run, before its bottom-of-pipe writes, in run
    /// order.
    after: Vec<std::sync::Arc<dyn Fn() + Send + Sync>>,
    /// What the repeats of lists it names more than once copy into the
    /// lists' memory, past their first (`Repeat`), kept until it has run.
    repeats: Vec<Repeat>,
    /// The list it ran alone in when the device was removed (diagnostics).
    removed_in: Option<usize>,
    /// Resources retired before it was submitted (`retire`).
    retired: Vec<ID3D12Resource>,
}

impl Pending {
    /// A submission's, after its steps ran, with the title memory its GPU
    /// work writes (copied back, written at the bottom of the pipe) and
    /// reads (host copies and textures brought up to date for it).
    fn of(runs: &[&HostList]) -> (Pending, Touches) {
        let mut back: Vec<Back> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> =
            std::collections::HashMap::new();
        let (mut pending_writes, mut pending_resolves, mut touches) = (Vec::new(), Vec::new(), Touches::default());
        let (mut missed, mut after) = (Vec::with_capacity(runs.len()), Vec::new());
        // One list's lock at a time: a submission may name a list twice.
        for (run, list) in runs.iter().enumerate() {
            let deferred = list.deferred();
            missed.push(deferred.missed.clone());
            after.extend(deferred.after.iter().cloned());
            for step in &deferred.copies {
                match step {
                    Step::GpuWrite(shadow, address, len, _, memory) => {
                        let copy = std::sync::Arc::as_ptr(shadow) as usize;
                        touches.shaders.push((copy, *address, address + len));
                        match index.entry((copy, *address, *len)) {
                            std::collections::hash_map::Entry::Occupied(o) => {
                                let by = &mut back[*o.get()].3;
                                if by.last() != Some(&(run, *memory)) {
                                    by.push((run, *memory));
                                }
                            }
                            std::collections::hash_map::Entry::Vacant(v) => {
                                v.insert(back.len());
                                back.push((shadow.clone(), *address, *len, vec![(run, *memory)]));
                            }
                        }
                    }
                    Step::Refresh(_, address, len) => touches.reads.push((*address, address + len)),
                    Step::Texture(staging, _) => {
                        touches.reads.push(staging.range());
                    }
                    _ => {}
                }
            }
            let mut writes = Vec::new();
            for &(at, what, bytes) in &deferred.writes {
                writes.push((at, what, bytes));
                touches.labels.push((at, at + bytes));
            }
            pending_writes.push(writes);
            pending_resolves.push(deferred.resolves.clone());
        }
        let pending = Pending {
            lists: runs.iter().map(|&l| ListRef(l)).collect(),
            value: 0,
            copy_value: 0,
            back,
            writes: pending_writes,
            resolves: pending_resolves,
            preludes: Vec::new(),
            submitted: std::time::Instant::now(),
            missed,
            after,
            repeats: Vec::new(),
            removed_in: None,
            retired: Vec::new(),
        };
        (pending, touches)
    }
}

/// A thread's event for waiting on the host fence.
struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: our own handle, no longer waited on.
        let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.0) };
    }
}

thread_local! {
    /// Null if it could not be made: the fence wait then blocks by itself.
    // SAFETY: plain event creation.
    static EVENT: Event = Event(unsafe { CreateEventW(None, false, false, None) }.unwrap_or_default());
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        self.new_list_kind(ListKind::Direct)
    }

    /// A new copy-engine list, open for recording (`VICTORY_COPY_QUEUE`).
    pub(crate) fn new_copy_list(&self) -> windows::core::Result<HostList> {
        self.new_list_kind(ListKind::Copy)
    }

    fn new_list_kind(&self, kind: ListKind) -> windows::core::Result<HostList> {
        let list_type = match kind {
            ListKind::Direct => D3D12_COMMAND_LIST_TYPE_DIRECT,
            ListKind::Copy => D3D12_COMMAND_LIST_TYPE_COPY,
        };
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator =
                self.device.CreateCommandAllocator(list_type)?;
            let list: ID3D12GraphicsCommandList =
                self.device.CreateCommandList(0, list_type, &allocator, None)?;
            let prelude_allocator: ID3D12CommandAllocator =
                self.device.CreateCommandAllocator(list_type)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                list_type,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            let made = HostList {
                kind,
                allocator: Mutex::new(allocator),
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                buffer_states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                ours: false.into(),
                memory_writer: false.into(),
                prelude: Mutex::new((prelude_allocator, prelude)),
                pipelines: Mutex::default(),
                id,
                budget: Mutex::default(),
                last: AtomicU64::new(0),
                fresh_after: AtomicU64::new(0),
                arguments: Mutex::new(None),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            };
            #[cfg(feature = "diagnostics")]
            made.emit_busy_start();
            Ok(made)
        }
    }

    /// A command allocator nothing runs from: a spare, or a new one. DIRECT
    /// and COPY allocators are never mixed.
    fn fresh_allocator(&self, kind: ListKind) -> Option<ID3D12CommandAllocator> {
        let spare = match kind {
            ListKind::Direct => pool(&SPARE_ALLOCATORS).pop(),
            ListKind::Copy => pool(&SPARE_ALLOCATORS_COPY).pop(),
        };
        spare.or_else(|| {
            let list_type = match kind {
                ListKind::Direct => D3D12_COMMAND_LIST_TYPE_DIRECT,
                ListKind::Copy => D3D12_COMMAND_LIST_TYPE_COPY,
            };
            // SAFETY: plain object creation.
            let made = unsafe { self.device.CreateCommandAllocator(list_type) };
            made.ok()
        })
    }

    /// A closed prelude list and its allocator, nothing running from them.
    fn fresh_prelude(&self, kind: ListKind) -> Option<Prelude> {
        let spare = match kind {
            ListKind::Direct => pool(&SPARE_PRELUDES).pop(),
            ListKind::Copy => pool(&SPARE_PRELUDES_COPY).pop(),
        };
        if spare.is_some() {
            return spare;
        }
        let allocator = self.fresh_allocator(kind)?;
        let list_type = match kind {
            ListKind::Direct => D3D12_COMMAND_LIST_TYPE_DIRECT,
            ListKind::Copy => D3D12_COMMAND_LIST_TYPE_COPY,
        };
        // SAFETY: plain object creation.
        unsafe {
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, list_type, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            Some((allocator, list))
        }
    }

    /// Runs closed lists. Their submission joins the timeline, finished
    /// once the GPU has run it.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // A list named twice would execute twice: its earlier run's memory
        // is still in flight when the later run submits, so the submission
        // splits there -- and, worse, some drivers lose the lists between
        // two executions of one object (their writes never land). Each
        // piece names every list once; the fence between pieces keeps the
        // later run from starting before the earlier finished, so its
        // submit-time steps can write the list's memory directly again.
        let mut start = 0;
        while start < runs.len() {
            let mut seen: Vec<*const HostList> = Vec::with_capacity(runs.len() - start);
            let mut end = runs.len();
            for (i, list) in runs.iter().enumerate().skip(start) {
                let id: *const HostList = *list;
                if seen.contains(&id) {
                    end = i;
                    break;
                }
                seen.push(id);
            }
            self.execute_inner(
                &runs[start..end],
                #[cfg(test)]
                None,
            );
            if end < runs.len() {
                // The next piece reuses a list this one ran: wait for it.
                self.drain();
            }
            start = end;
        }
        if runs.is_empty() {
            self.execute_inner(
                &[],
                #[cfg(test)]
                None,
            );
        }
    }

    /// Test helper: arms `gate` on the host queue atomically inside the
    /// submission lock right before `ExecuteCommandLists` so other parallel
    /// tests cannot deadlock behind `gate` while holding `submitted`.
    #[cfg(test)]
    fn execute_gated(&self, runs: &[&HostList], gate: &ID3D12Fence) {
        self.execute_inner(runs, Some(gate));
    }

    fn execute_inner(&self, runs: &[&HostList], #[cfg(test)] gate: Option<&ID3D12Fence>) {
        self.finisher.call_once(|| {
            let spawned = std::thread::Builder::new()
                .name("victory-gpu-finisher".into())
                .spawn(|| made().into_iter().for_each(Host::finish_loop));
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no completion thread; submissions finish when needed");
            }
        });
        // One submission at a time from preparation to submission: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // What has run is finished first. A list still in flight is waited
        // for: its upload memory, budget word and prelude are rewritten
        // below.
        self.finish();
        for list in runs {
            if list.running() {
                #[cfg(feature = "diagnostics")]
                let block = std::time::Instant::now();
                self.settle(list.last.load(Ordering::Acquire));
                #[cfg(feature = "diagnostics")]
                if timing() {
                    starve::note_block(starve::RUNNING, block.elapsed().as_micros() as u64);
                }
            }
        }
        if self.timeline.len() >= IN_FLIGHT
            && let Some((oldest, _)) = self.timeline.span()
        {
            #[cfg(feature = "diagnostics")]
            let block = std::time::Instant::now();
            self.settle(oldest);
            #[cfg(feature = "diagnostics")]
            if timing() {
                starve::note_block(starve::IN_FLIGHT, block.elapsed().as_micros() as u64);
            }
        }
        // Host textures the title has not used for a while (or past the
        // card's budget) are given up first: the last submission has run.
        crate::resource::evict_unused();
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        #[cfg(feature = "diagnostics")]
        let syncing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncHeapsUs);
        crate::descriptor::sync_heaps();
        #[cfg(feature = "diagnostics")]
        drop(syncing);
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // And textures read through tables, uploaded by the same list.
        let textures: Vec<_> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Texture(staging, copy) => Some((staging.clone(), copy.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = (!heaps.is_empty() || !textures.is_empty())
            .then(|| crate::bindless::uploads(&heaps, &textures))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        // Copy offload: the upload list runs on the copy engine, not prepended
        // to the direct runs. Otherwise (gate off) it runs first on DIRECT.
        // Without a copy queue the COPY list still runs first on DIRECT (its
        // copy-only commands are valid there); the DIRECT side still makes
        // the textures readable.
        let upload_is_copy = uploads.as_ref().is_some_and(|u| u.list().is_copy());
        let use_copy_queue =
            upload_is_copy && self.copy_queue.is_some() && self.copy_fence.is_some();
        let runs: Vec<&HostList> = if use_copy_queue {
            runs.to_vec()
        } else {
            uploads
                .as_ref()
                .map(|u| u.list())
                .into_iter()
                .chain(runs.iter().copied())
                .collect()
        };
        let runs = &runs[..];
        // The copy upload list, if any (COPY kind): submitted on the copy
        // queue below, before the direct runs.
        let upload_list: Option<&HostList> = if use_copy_queue {
            uploads.as_ref().map(|u| u.list())
        } else {
            None
        };
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in runs.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        *submitted + 1,
                        runs.len()
                    );
                }
            }
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len() + 2);
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch {
            host: Some(self),
            ..Batch::default()
        };
        // The copy-engine upload, if any: recorded and submitted on the copy
        // queue first (single submission, single Wait on DIRECT below, not
        // one per texture). Its prelude moves textures from their real
        // states to COPY_DEST; the DIRECT side below moves them to
        // SHADER_RESOURCE after the Wait.
        let mut copy_value: u64 = 0;
        // A list named again in this submission: its earlier prelude is
        // not yet run, so the repeat records into a spare one, and starts
        // a new `ExecuteCommandLists` call (the queue runs them in order, so
        // nothing waits). Where each such call starts in `lists`.
        let mut preludes: Vec<(Prelude, ListKind)> = Vec::new();
        if let Some(up) = upload_list {
            let mut copy_batch = Batch {
                host: Some(self),
                ..Batch::default()
            };
            up.run_deferred(&mut copy_batch);
            copy_batch.flush();
            let mut copy_lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2);
            if let Some((prelude, spare)) = up.prelude(None, None) {
                copy_lists.push(Some(prelude));
                if let Some(spare) = spare {
                    // Uploads never repeat in one submission (single use), so
                    // no spare prelude is expected; kept for symmetry.
                    preludes.push((spare, up.kind));
                }
            }
            copy_lists.push(up.list.cast().ok());
            let (queue, fence) = match (&self.copy_queue, &self.copy_fence) {
                (Some(q), Some(f)) => (q, f),
                _ => fail(format_args!("a copy-engine upload without a copy queue")),
            };
            let _copy_guard = self.copy_lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut submitted_copy = self.copy_submitted.lock().unwrap_or_else(|e| e.into_inner());
            *submitted_copy += 1;
            copy_value = *submitted_copy;
            drop(submitted_copy);
            // SAFETY: closed COPY lists, not running (waited for above); our
            // copy fence.
            let signalled = unsafe {
                queue.ExecuteCommandLists(&copy_lists);
                queue.Signal(fence, copy_value)
            };
            if let Err(e) = signalled {
                fail(format_args!(
                    "a host copy submission of {} lists was not signalled ({e})",
                    1
                ));
            }
            // The DIRECT side makes the uploaded textures readable after the
            // Wait below: every texture the COPY list leaves in COPY_DEST
            // starts the first DIRECT run in SHADER_RESOURCE, so its prelude
            // records COPY_DEST to SHADER_RESOURCE.
            if let Some(&first) = runs.first() {
                let uploaded: Vec<std::sync::Arc<crate::resource::HostTexture>> = up
                    .states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .values()
                    .map(|(_, _, copy)| copy.clone())
                    .collect();
                for copy in &uploaded {
                    first.ensure_shader_resource(copy);
                }
            }
        }
        let mut repeats = Vec::new();
        let mut cuts = vec![0];
        let mut since_cut: Vec<*const HostList> = Vec::with_capacity(runs.len());
        for (i, list) in runs.iter().enumerate() {
            let id: *const HostList = *list;
            let repeated = runs[..i].iter().any(|&l| std::ptr::eq(l, id));
            if since_cut.contains(&id) {
                cuts.push(lists.len());
                since_cut.clear();
            }
            since_cut.push(id);
            // A repeat's steps would rewrite the list's own memory (upload
            // memory, scratch, budget word) before its earlier run has read
            // it: they write into memory of the submission instead, which
            // the repeat's prelude copies into the list's on the GPU, in
            // queue order.
            batch.redirect = repeated.then(Vec::new);
            list.run_deferred(&mut batch);
            let repeat = batch
                .redirect
                .take()
                .filter(|r| !r.is_empty())
                .and_then(|r| list.repeat(self, r));
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed =
                crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some((prelude, spare)) = list.prelude(repeated.then_some(self), repeat.as_ref())
            {
                lists.push(Some(prelude));
                if let Some(spare) = spare {
                    preludes.push((spare, list.kind));
                }
            }
            repeats.extend(repeat);
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        // For copy offload the pending submission names the upload list too
        // (kept alive until both queues ran it); its staging reads are the
        // direct runs' texture steps below.
        let pending_runs: Vec<&HostList> = match upload_list {
            Some(up) => std::iter::once(up).chain(runs.iter().copied()).collect(),
            None => runs.to_vec(),
        };
        let (mut pending, mut touches) = Pending::of(&pending_runs);
        pending.preludes = preludes;
        pending.repeats = repeats;
        // What steps brought up to date through view heaps and page tables.
        touches.reads.append(&mut batch.reads);
        touches.shaders.append(&mut batch.shaders);
        touches.tables.extend(batch.tables.drain());
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        cuts.push(lists.len());
        // Diagnostics timing (`VICTORY_GPU_TIMING`): stamps stay in the normal
        // cuts grouping (the lists run as with the timing off). Per-list alone
        // execution is opt-in (`VICTORY_GPU_TIMING_ALONE=1`): each list runs
        // and is waited for on its own, as before.
        #[cfg(feature = "diagnostics")]
        let timed = timing();
        #[cfg(feature = "diagnostics")]
        let alone = timed && timing_alone();
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if alone {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            cuts.windows(2).map(|c| &lists[c[0]..c[1]]).collect()
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups: Vec<&[Option<ID3D12CommandList>]> =
            cuts.windows(2).map(|c| &lists[c[0]..c[1]]).collect();
        // Each list ran alone and was waited for: a removal is its.
        #[cfg(feature = "diagnostics")]
        let per_list = alone;
        #[cfg(not(feature = "diagnostics"))]
        let per_list = false;
        // Which runs each cuts group holds (their `firsts` lie in it).
        #[cfg(feature = "diagnostics")]
        let group_runs: Vec<Vec<usize>> = if alone {
            (0..runs.len()).map(|i| vec![i]).collect()
        } else {
            group_members(&firsts, &cuts)
        };
        let mut removed_in = None;
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
            // SAFETY: closed lists, not running (waited for above); our fence.
            // With copy offload the DIRECT queue waits once for the copy
            // submission before its first dependent group (not per texture).
            let signalled = unsafe {
                #[cfg(test)]
                if i == 0
                    && let Some(gate) = gate
                {
                    let _ = self.queue.Wait(gate, 1);
                }
                if i == 0 && copy_value != 0
                    && let (Some(copy_fence), true) = (&self.copy_fence, true)
                {
                    let _ = self.queue.Wait(copy_fence, copy_value);
                }
                self.queue.ExecuteCommandLists(group);
                self.queue.Signal(&self.fence, *submitted)
            };
            if let Err(e) = signalled {
                // Nothing would finish it.
                fail(format_args!(
                    "a host submission of {} lists was not signalled ({e}){}",
                    pending_runs.len(),
                    self.removal_data(&pending_runs)
                ));
            }
            #[cfg(feature = "diagnostics")]
            if timed {
                self.wait_gpu(*submitted, true);
                crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
                let wall_ms = started.elapsed().as_secs_f64() * 1e3;
                let member_runs: &[usize] = &group_runs[i];
                let frequency = self.frequency();
                let mut group_busy = 0.0;
                for &r in member_runs {
                    let (times, busy, residual, dropped) = runs[r].timed(frequency);
                    group_busy += busy;
                    report_list(&times, busy, residual, dropped, wall_ms / member_runs.len().max(1) as f64);
                }
                costs::add_submission(group_busy, wall_ms);
            }
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                gpu_times(started.elapsed(), &runs[i].slowest(frequency, usize::MAX));
            }
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                // A slow submission grouped several lists: a removal is its.
                removed_in = Some(i);
            }
            if per_list
                && removed_in.is_none()
                // SAFETY: a plain query.
                && unsafe { self.device.GetDeviceRemovedReason() }.is_err()
            {
                removed_in = Some(i);
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        for list in pending_runs.iter().copied() {
            list.last.store(*submitted, Ordering::Release);
        }
        pending.removed_in = removed_in;
        pending.value = *submitted;
        pending.copy_value = copy_value;
        // Resources retired since the last submission go once this one has
        // run, and with it every one before.
        pending.retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
        self.timeline.push(*submitted, copy_value, touches, pending);
        // Staging buffers and chunks this submission's uploads took are
        // its submission's: recyclable once it has run (`texture::recycle`
        // frees only finished ones; submissions run async here). With copy
        // offload both queues must have run them.
        crate::texture::submitted(*submitted, copy_value);
        #[cfg(feature = "diagnostics")]
        if timed {
            starve::note_depth(self.timeline.len());
        }
        #[cfg(feature = "diagnostics")]
        // One list's lock at a time: a submission may name a list twice.
        // (Copy-back runs when the submission finishes: `complete`.)
        if timed {
            self.settle(*submitted);
        }
        #[cfg(feature = "diagnostics")]
        if traced {
            let sub = *submitted;
            drop(submitted);
            self.settle(sub);
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            crate::capture::dump_submission(sub, &targets);
        }
    }

    /// The queue's timestamp frequency (diagnostics).
    #[cfg(feature = "diagnostics")]
    fn frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1)
    }

    /// Finishes, in order, the submissions the GPU has run (no waiting).
    /// A submission with copy work finishes once both queues ran it.
    fn finish(&self) {
        // SAFETY: plain queries (all ones once the device is removed).
        let reached = self.completed();
        let copy_reached = self.copy_completed();
        self.timeline.finish_both(reached, copy_reached, |p| self.complete(p));
    }

    /// The newest submission value the GPU has run (no waiting).
    pub(crate) fn completed(&self) -> u64 {
        // SAFETY: a plain query (all ones once the device is removed).
        unsafe { self.fence.GetCompletedValue() }
    }

    /// The newest copy submission the copy engine has run (no waiting), or
    /// `u64::MAX` where uploads stay on the direct queue: then every
    /// `copy_value` (0) is already passed and the direct fence alone decides.
    pub(crate) fn copy_completed(&self) -> u64 {
        if !copy_enabled() {
            return u64::MAX;
        }
        let Some(fence) = &self.copy_fence else {
            return u64::MAX;
        };
        // SAFETY: a plain query.
        unsafe { fence.GetCompletedValue() }
    }

    /// Finishes every submission up to the one that signals `value`,
    /// waiting for the GPU to run them (both queues where the submission
    /// has copy work).
    pub(crate) fn settle(&self, value: u64) {
        if self
            .timeline
            .span()
            .is_some_and(|(oldest, _)| oldest <= value)
        {
            self.wait_gpu(value, true);
            if copy_enabled() {
                // The submission may carry copy work: finish needs the copy
                // fence past it too. Waiting for the newest copy covers it
                // (copy submissions are in order); with no copy work in
                // flight the wait returns at once.
                let copy_value = *self.copy_submitted.lock().unwrap_or_else(|e| e.into_inner());
                if copy_value != 0 {
                    #[cfg(feature = "diagnostics")]
                    let block = std::time::Instant::now();
                    self.wait_copy(copy_value, true);
                    #[cfg(feature = "diagnostics")]
                    if timing() {
                        starve::note_block(starve::COPY_WAIT, block.elapsed().as_micros() as u64);
                    }
                }
            }
            self.finish();
        }
    }

    /// Before a frame is presented: finishes the previous frame's
    /// submissions, `previous` from [`Host::frame_done`] at its present.
    fn previous_frame(&self, previous: Option<u64>) {
        if let Some(value) = previous {
            #[cfg(feature = "diagnostics")]
            let block = std::time::Instant::now();
            self.settle(value);
            #[cfg(feature = "diagnostics")]
            if timing() {
                starve::note_block(starve::DRAIN_PRESENT, block.elapsed().as_micros() as u64);
            }
        }
    }

    /// After a frame is presented: the newest submission, which the next
    /// present waits for (the frame's own may still run until then).
    fn frame_done(&self) -> Option<u64> {
        self.timeline.span().map(|(_, newest)| newest)
    }

    /// Finishes every submission made so far.
    pub(crate) fn drain(&self) {
        if let Some((_, newest)) = self.timeline.span() {
            #[cfg(feature = "diagnostics")]
            let block = std::time::Instant::now();
            self.settle(newest);
            #[cfg(feature = "diagnostics")]
            if timing() {
                starve::note_block(starve::DRAIN_PRESENT, block.elapsed().as_micros() as u64);
            }
        }
    }

    /// Before a submission's step reads title memory `[address, address +
    /// len)` (or, with `writing`, writes it): finishes the submissions in
    /// flight whose GPU work writes it (or reads it), so that the step sees
    /// title memory and host copies as if they had run when submitted.
    pub(crate) fn settle_range(&self, address: usize, len: usize, writing: bool) {
        if let Some(value) = self
            .timeline
            .conflict(address, address.saturating_add(len), writing)
        {
            #[cfg(feature = "diagnostics")]
            let block = std::time::Instant::now();
            self.settle(value);
            #[cfg(feature = "diagnostics")]
            if timing() {
                starve::note_block(starve::CONFLICT, block.elapsed().as_micros() as u64);
            }
        }
    }

    /// Before view heap `key`'s page table is rewritten: finishes the
    /// submissions in flight whose shaders read it.
    fn settle_table(&self, key: usize) {
        if let Some(value) = self.timeline.table_conflict(key) {
            #[cfg(feature = "diagnostics")]
            let block = std::time::Instant::now();
            self.settle(value);
            #[cfg(feature = "diagnostics")]
            if timing() {
                starve::note_block(starve::STAGING_TABLE, block.elapsed().as_micros() as u64);
            }
        }
    }

    /// Runs `signal` once every submission made so far has finished.
    pub(crate) fn after(&self, signal: impl FnOnce() + Send + 'static) {
        self.timeline.after(signal);
    }

    /// Blocks until the GPU has run what signals `value`; the time counts
    /// as waiting for the GPU if `counted`.
    fn wait_gpu(&self, value: u64, counted: bool) {
        // SAFETY: plain calls on our fence and this thread's event.
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            let started = std::time::Instant::now();
            let waited = EVENT.with(|e| {
                self.fence
                    .SetEventOnCompletion(value, e.0)
                    .map(|()| WaitForSingleObject(e.0, INFINITE))
            });
            if let Err(e) = waited {
                tracing::warn!(error = %e, "host fence wait failed; polling");
                while self.fence.GetCompletedValue() < value {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            if counted {
                SUBMISSION_US[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }

    /// Blocks until the copy engine has run what signals `value` (no-op
    /// without a copy queue); the time counts as waiting for the GPU if
    /// `counted`.
    fn wait_copy(&self, value: u64, counted: bool) {
        let Some(fence) = &self.copy_fence else {
            return;
        };
        // SAFETY: plain calls on our fence and this thread's event.
        unsafe {
            if fence.GetCompletedValue() >= value {
                return;
            }
            let started = std::time::Instant::now();
            let waited = EVENT.with(|e| {
                fence
                    .SetEventOnCompletion(value, e.0)
                    .map(|()| WaitForSingleObject(e.0, INFINITE))
            });
            if let Err(e) = waited {
                tracing::warn!(error = %e, "host copy fence wait failed; polling");
                while fence.GetCompletedValue() < value {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            if counted {
                SUBMISSION_US[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }

    /// The completion thread: finishes each submission as soon as the GPU
    /// has run it, so that title fences and bottom-of-pipe writes move on
    /// their own (the title waits for its events and reads its memory).
    fn finish_loop(&self) {
        loop {
            let value = self.timeline.next();
            self.wait_gpu(value, false);
            if copy_enabled() {
                // The submission may carry copy work: finish needs both.
                // Waiting for the newest copy covers it; without copy work
                // the wait returns at once.
                let copy_value = *self.copy_submitted.lock().unwrap_or_else(|e| e.into_inner());
                if copy_value != 0 {
                    self.wait_copy(copy_value, false);
                }
            }
            self.finish();
        }
    }

    /// Once the GPU has run a submission: what its shaders wrote goes back
    /// into title memory, then its bottom-of-pipe writes, which tell the
    /// title the GPU is done.
    fn complete(&self, p: Pending) {
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            // Any submission in flight may have faulted.
            let mut runs: Vec<&HostList> = p.lists.iter().map(|l| l.get()).collect();
            self.timeline
                .each_work(|w| runs.extend(w.lists.iter().map(|l| l.get())));
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                p.lists.len(),
                self.removal_data(&runs),
                last_notes(&runs),
                removed_during(&runs, p.removed_in)
            ));
        }
        // About how long it ran: from its submission or the end of the one
        // before, whichever is later.
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now = std::time::Instant::now();
        let ran = {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            let from = last.map_or(p.submitted, |l| l.max(p.submitted));
            *last = Some(now);
            now - from
        };
        // The host resets a device whose submission runs for about 2 s.
        if ran > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = p
                .lists
                .iter()
                .flat_map(|l| {
                    l.get()
                        .pipelines
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
                .collect();
            tracing::warn!(
                ms = ran.as_millis(),
                lists = p.lists.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // What shaders wrote goes back before the bottom-of-pipe writes;
        // each range once per submission, however many dispatches declared
        // it (the world's passes declare the same big ranges dozens of
        // times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // Shaders that read memory with no copy computed something else
        // than the console's: what a list's shaders wrote then stays out of
        // title memory, where it broke the title's own data (runs c1, c4).
        let missed: Vec<bool> = p
            .missed
            .iter()
            .map(|checks| checks.iter().any(|check| check()))
            .collect();
        // Upstream withholding, at the submission's completion (as upstream
        // does at its submission): lists that read memory with no copy
        // withhold what their shaders wrote, and a withholding frame widens
        // it to lists whose checks exist but passed.
        crate::memory::note_submission(missed.contains(&true));
        let withholding = crate::memory::withholding();
        let missed: Vec<bool> = missed
            .into_iter()
            .enumerate()
            .map(|(run, m)| m || withholding && !p.missed[run].is_empty())
            .collect();
        let mut copies: Vec<&std::sync::Arc<crate::shadow::Shadow>> = Vec::new();
        let mut back: Vec<(
            std::sync::Arc<crate::shadow::Shadow>,
            usize,
            usize,
            bool,
        )> = Vec::with_capacity(p.back.len());
        for (shadow, address, len, by) in &p.back {
            back.push((
                shadow.clone(),
                *address,
                *len,
                !by.iter().any(|&(run, memory)| memory && missed[run]),
            ));
            if !copies.iter().any(|c| std::sync::Arc::ptr_eq(c, shadow)) {
                copies.push(shadow);
            }
        }
        // A range overlapping a withheld one of the same copy is withheld
        // too: its copy-back would take the other's bytes along (pages a
        // program using title memory and another both wrote).
        let held: Vec<(usize, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (std::sync::Arc::as_ptr(s) as usize, *a, *n))
            .collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = std::sync::Arc::as_ptr(&b.0) as usize;
            b.3 = !held
                .iter()
                .any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        crate::shadow::copy_back_marks(&withheld, withholding);
        // Texture views left null that shaders read: made for what follows.
        let read = crate::shadow::take_reads();
        if !read.is_empty() {
            crate::descriptor::make_read(&read);
        }
        run_back(&back);
        copies.iter().for_each(|c| c.submission_done());
        for work in &p.after {
            work();
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for (writes, resolves) in p.writes.iter().zip(p.resolves.iter()) {
            // Query results first: a bottom-of-pipe write may name the same
            // memory (a buffer the title both resolves into and advances).
            for (readback, targets) in resolves {
                let mut data = std::ptr::null_mut();
                let end = targets
                    .iter()
                    .map(|&(_, offset, bytes)| offset + bytes as u64)
                    .max()
                    .unwrap_or(0);
                let range = D3D12_RANGE {
                    Begin: 0,
                    End: end as usize,
                };
                // SAFETY: a read-back buffer the finished list resolved
                // into; `end` is inside it (the resolve checked the
                // title's own buffer, and the buffer was made that size).
                if unsafe { readback.Map(0, Some(&range), Some(&mut data)) }.is_err() {
                    continue;
                }
                for &(at, offset, bytes) in targets {
                    // SAFETY: reading a u64 at a checked offset of the
                    // mapped buffer, and writing into title memory.
                    unsafe {
                        let value =
                            std::ptr::read_unaligned(data.add(offset as usize).cast::<u64>());
                        if bytes == 4 {
                            (at as *mut u32).write_unaligned(value as u32);
                        } else {
                            (at as *mut u64).write_unaligned(value);
                        }
                    }
                }
                // SAFETY: the same buffer, unmapped once.
                unsafe { readback.Unmap(0, None) };
            }
            for &(at, what, bytes) in writes {
                what.write(at as *mut u8, bytes);
            }
        }
        drop(p.repeats);
        for (prelude, kind) in p.preludes {
            let spare_pool = match kind {
                ListKind::Direct => &SPARE_PRELUDES,
                ListKind::Copy => &SPARE_PRELUDES_COPY,
            };
            // SAFETY: the submission that ran from it has finished.
            if unsafe { prelude.0.Reset() }.is_ok() && pool(spare_pool).len() < SPARES {
                pool(spare_pool).push(prelude);
            }
        }
        // It has run, and every submission before it: what was retired
        // before it was submitted goes.
        drop(p.retired);
        self.log_messages();

    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// Records the list's busy-pair start stamp (diagnostics, timing on):
    /// the GPU time before any command, so un-noted work is explicit.
    /// Called with an open list (after `Reset` or creation). Timing-only:
    /// a timestamp query, no barriers or copies.
    #[cfg(feature = "diagnostics")]
    fn emit_busy_start(&self) {
        if let Some((heap, _)) = self.stamps() {
            // SAFETY: an open list and our query heap.
            unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, BUSY_START) };
        }
    }

    /// The noted commands' GPU time with the list's busy pair (diagnostics,
    /// timing on, after the list ran): per-note milliseconds longest first,
    /// the list's GPU busy milliseconds, the residual (busy minus noted,
    /// un-noted work, clamped at zero) and notes dropped past the stamps.
    #[cfg(feature = "diagnostics")]
    fn timed(&self, frequency: u64) -> (Vec<(f64, String)>, f64, f64, usize) {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let total_notes = notes.len();
        let count = (total_notes as u32).min(MAX_TIMED_COUNT);
        let Some((_, back)) = self.stamps() else {
            return (Vec::new(), 0.0, 0.0, total_notes);
        };
        // Only what this list wrote: the note stamps `0..=count` ((count + 1)
        // qwords) plus the busy pair at `BUSY_START` (2 qwords). The 128 KB
        // buffer is never mapped or copied whole; diagnostics-only path.
        let needed = count as usize + 1;
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the finished list resolved into; both
        // ranges lie inside it (the resolves above wrote them).
        let prefix: Vec<u64> = unsafe {
            let range = D3D12_RANGE {
                Begin: 0,
                End: needed * 8,
            };
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return (Vec::new(), 0.0, 0.0, total_notes);
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), needed).to_vec();
            back.Unmap(0, None);
            s
        };
        // SAFETY: the same read-back buffer, offset into its busy slots.
        let busy_pair: [u64; 2] = unsafe {
            let range = D3D12_RANGE {
                Begin: BUSY_START as usize * 8,
                End: STAMPS as usize * 8,
            };
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return (Vec::new(), 0.0, 0.0, total_notes);
            }
            let s = std::slice::from_raw_parts(
                data.cast::<u64>().add(BUSY_START as usize),
                2,
            )
            .to_vec();
            back.Unmap(0, None);
            [s[0], s[1]]
        };
        let stamps = prefix;
        let to_ms = |ticks: u64| ticks as f64 * 1e3 / frequency.max(1) as f64;
        let mut times: Vec<(f64, String)> = (0..count as usize)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (to_ms(ticks), notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        let noted: f64 = (0..count as usize)
            .map(|i| stamps[i + 1].saturating_sub(stamps[i]) as f64)
            .sum::<f64>()
            * 1e3
            / frequency.max(1) as f64;
        let busy_ticks = busy_pair[1].saturating_sub(busy_pair[0]);
        // An unwritten busy pair (no start/end recorded) reads zero: no busy.
        let busy = if busy_ticks == 0 { 0.0 } else { to_ms(busy_ticks) };
        let residual = (busy - noted).max(0.0);
        let dropped = total_notes.saturating_sub(count as usize);
        (times, busy, residual, dropped)
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran),
    /// at most `n` of them.
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(MAX_TIMED_COUNT);
            // SAFETY: an open list, our heap and read-back buffer. Timing-only:
            // timestamp queries and their resolve; no barriers or copies.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
                // The list's busy-pair end: with the start in `reset`
                // (`emit_busy_start`), the submission's reported busy is the sum
                // of its lists' pairs (same GPU clock across heaps). Prelude
                // lists are not bracketed, so their transition work stays out
                // of busy; the gap to the CPU wall is idle plus that untimed
                // work (and any overlap), the ratio busy over wall.
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, BUSY_END);
                self.list.ResolveQueryData(
                    heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    BUSY_START,
                    2,
                    back,
                    u64::from(BUSY_START) * 8,
                );
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Waits for the list's last submission and finishes it (and those
    /// before it).
    pub(crate) fn settle(&self) {
        let last = self.last.load(Ordering::Acquire);
        if last != 0
            && let Some(host) = made()
        {
            host.settle(last);
        }
    }

    /// Whether the list's last submission, with the memory the list
    /// records into now, has not finished yet.
    pub(crate) fn running(&self) -> bool {
        let last = self.last.load(Ordering::Acquire);
        last > self.fresh_after.load(Ordering::Acquire)
            && made().is_some_and(|h| h.timeline.span().is_some_and(|(oldest, _)| oldest <= last))
    }

    /// Opens the list for recording again. D3D12 lets a list be reset as
    /// soon as it is submitted (titles record the next frame into it while
    /// the GPU runs this one): what its running submission uses (allocator,
    /// prelude, upload and scratch memory, the objects and textures it
    /// keeps, its budget word) then goes to the timeline, freed or reused
    /// once that submission has finished, and the list records into fresh
    /// ones.
    pub(crate) fn reset(&self) {
        let running = if self.running() { made() } else { None };
        let allocator = {
            let mut allocator = self.allocator.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(host) = running {
                let (Some(fresh), Some(prelude)) =
                    (host.fresh_allocator(self.kind), host.fresh_prelude(self.kind))
                else {
                    fail(format_args!(
                        "no fresh host command allocator for a running list"
                    ));
                };
                let mut old_prelude = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
                let mut buffer_states = self.buffer_states.lock().unwrap_or_else(|e| e.into_inner());
                let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
                let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
                let retired = Retired {
                    kind: self.kind,
                    allocator: std::mem::replace(&mut *allocator, fresh),
                    prelude: std::mem::replace(&mut *old_prelude, prelude),
                    deferred: std::mem::take(&mut *self.deferred()),
                    scratch: std::mem::take(&mut *scratch),
                    states: std::mem::take(&mut *states),
                    buffer_states: std::mem::take(&mut *buffer_states),
                    twins: std::mem::take(&mut *twins),
                    budget: budget.take(),
                };
                host.after(move || retired.retire());
                // Still named by that submission: dropping waits for it.
                self.fresh_after
                    .store(self.last.load(Ordering::Acquire), Ordering::Release);
            } else {
                *self.deferred() = Deferred::default();
                self.states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                self.buffer_states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                (scratch.current, scratch.used) = (0, 0);
                // SAFETY: nothing running uses the allocator.
                if let Err(e) = unsafe { allocator.Reset() } {
                    fail(format_args!("a host command allocator did not reset ({e})"));
                }
            }
            allocator.clone()
        };
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        // SAFETY: a closed list (D3D12 allows resetting one that still
        // runs) and an allocator nothing running uses.
        if let Err(e) = unsafe { self.list.Reset(&allocator, None) } {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        #[cfg(feature = "diagnostics")]
        self.emit_busy_start();
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(ID3D12Resource, u32)> {
        // COPY lists move only to copy states: never to shader, unordered or
        // render-target reads (those run on DIRECT, after the copy Wait).
        if self.is_copy() {
            const COPY_STATES: u32 = 0x400 | 0x800 | 0x1;
            debug_assert!(
                wanted & !(COPY_STATES | 0x40 | 0x80) == 0 || wanted & COPY_STATES != 0,
                "COPY list transition to {wanted:#x}: copy states only"
            );
            let sr_uav_rt = 0x4 | 0x8 | 0x40 | 0x80 | 0x10 | 0x20;
            if wanted & sr_uav_rt != 0 && wanted & COPY_STATES == 0 {
                crate::host::fail(format_args!("a COPY list transition to {wanted:#x}: copy states only"));
            }
        }
        // One host texture for the whole move (`crate::resource::evict` may
        // give it up meanwhile; this list then holds it).
        let copy = crate::resource::host_copy(resource)?;
        let texture = copy.texture.clone();
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = legal_state(&texture, wanted);
        self.track(&copy, wanted).map(|before| (texture, before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                if !self.ours.load(Ordering::Relaxed) {
                    copy.stamp();
                }
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        if self.is_copy() {
            // The copy engine leaves uploads in COPY_DEST for the DIRECT side
            // to make readable: COPY lists move only between copy states.
            const COPY_OK: u32 = 0x400 | 0x800 | 0x1;
            if wanted & !COPY_OK != 0 {
                debug_assert!(
                    false,
                    "COPY list move_texture to {wanted:#x}: copy states only"
                );
                crate::host::fail(format_args!("a COPY list move_texture to {wanted:#x}: copy states only"));
            }
        }
        let wanted = legal_state(&copy.texture, wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        self.assert_direct("twin_use");
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// The list is one of ours (uploads): what it uses is no use of the
    /// title's.
    pub(crate) fn set_ours(&self) {
        self.ours.store(true, Ordering::Relaxed);
    }

    /// Whether the list runs on the copy engine.
    pub(crate) fn is_copy(&self) -> bool {
        self.kind == ListKind::Copy
    }

    /// COPY lists allow only copy and barrier ops: draws, dispatches and
    /// bindings on one are a bug. Called by the direct-only entry points.
    pub(crate) fn assert_direct(&self, what: &str) {
        debug_assert!(
            self.kind == ListKind::Direct,
            "{what} on a COPY list: draws, dispatches and bindings run on DIRECT lists only"
        );
        if self.kind != ListKind::Direct {
            crate::host::fail(format_args!("{what} on a COPY list"));
        }
    }

    /// Ensures the list starts with texture `copy` in `SHADER_RESOURCE`
    /// (for the DIRECT side of a copy-engine upload, whose COPY list leaves
    /// it in `COPY_DEST`): the prelude then records the COPY_DEST to
    /// SHADER_RESOURCE barrier. Used at submission, before preludes run.
    pub(crate) fn ensure_shader_resource(&self, copy: &std::sync::Arc<crate::resource::HostTexture>) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let wanted = legal_state(&copy.texture, SHADER_RESOURCE);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                v.insert((wanted, wanted, copy.clone()));
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                o.get_mut().1 = wanted;
            }
        }
    }

    /// What the list's shaders write from now until the guard goes is
    /// written by a program using title memory (`uses`) or not.
    pub(crate) fn writer_uses_memory(&self, uses: bool) -> impl Drop + '_ {
        struct Reset<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        self.memory_writer.store(uses, Ordering::Relaxed);
        Reset(&self.memory_writer)
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// Moves a shadow buffer to `wanted` in this list. Read-only states are
    /// combined when a buffer is used for more than one purpose in a draw.
    pub(crate) fn buffer_state(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        wanted: u32,
    ) {
        // Buffers never move on the copy engine: uploads use staging space.
        self.assert_direct("buffer_state");
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        let next = {
            let mut states = self
                .buffer_states
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            match states.entry(key) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert((wanted, wanted, shadow.clone()));
                    return;
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let state = &mut o.get_mut().1;
                    let before = *state;
                    *state = if before != 0
                        && wanted != 0
                        && before & !BUFFER_READ_STATES == 0
                        && wanted & !BUFFER_READ_STATES == 0
                    {
                        before | wanted
                    } else {
                        wanted
                    };
                    (before != *state).then_some((before, *state))
                }
            }
        };
        if let Some((before, after)) = next {
            // SAFETY: this list and its shadow buffer are live.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    &shadow.buffer,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATES(after as i32),
                )])
            };
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. With `repeated` (the list's
    /// earlier prelude in the same submission has not run yet), it records
    /// into a spare prelude from `repeated`, returned for the submission to
    /// keep until it has run.
    /// A repeat's `repeat` bytes go into the list's memory first.
    fn prelude(
        &self,
        repeated: Option<&Host>,
        repeat: Option<&Repeat>,
    ) -> Option<(ID3D12CommandList, Option<Prelude>)> {

        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let buffer_states = self
            .buffer_states
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // The copies start from the real states, then the list's moves from
        // them; only after both do they become the states the list leaves.
        // Read, not swapped, and built when recorded (after the twin
        // copies): the twin copies must see the real states (as the debug
        // layer checks), not the ones the lists leave.
        let up_to_date = |first: &u32, real: u32| real == *first;
        if copies.is_empty()
            && repeat.is_none()
            && states.values().all(|(first, _, copy)| {
                up_to_date(first, copy.state.load(Ordering::Relaxed))
            })
            && buffer_states.values().all(|(first, _, shadow)| {
                up_to_date(first, shadow.state.load(Ordering::Relaxed))
            })
        {

            return None;
        }
        let moves = || -> Vec<D3D12_RESOURCE_BARRIER> {
            let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = states
                .values()
                .filter_map(|(first, _, copy)| {
                    let real = copy.state.load(Ordering::Relaxed);
                    (real != *first).then(|| {
                        transition(
                            &copy.texture,
                            D3D12_RESOURCE_STATES(real as i32),
                            D3D12_RESOURCE_STATES(*first as i32),
                        )
                    })
                })
                .collect();
            barriers.extend(buffer_states.values().filter_map(|(first, _, shadow)| {
                let real = shadow.state.load(Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &shadow.buffer,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            }));
            barriers
        };
        let record = |(allocator, list): &Prelude| {
            // SAFETY: nothing pending or running uses the prelude (`execute`
            // waits for the list's last submission first, and a repeat in
            // one submission records into a spare); the textures and the
            // repeat's buffers are live.
            let recorded = unsafe {
                allocator.Reset().and_then(|()| {
                    list.Reset(allocator, None)?;
                    if let Some(repeat) = repeat {
                        // The list's buffers start the call in the common
                        // state (buffers decay at the end of each), go to
                        // copy destination for these copies and back.
                        let mut written: Vec<&ID3D12Resource> = Vec::new();
                        for (buffer, offset, from, n) in &repeat.copies {
                            list.CopyBufferRegion(buffer, *offset, &repeat.staging, *from, *n);
                            if !written.iter().any(|b| b.as_raw() == buffer.as_raw()) {
                                written.push(buffer);
                            }
                        }
                        let back: Vec<D3D12_RESOURCE_BARRIER> = written
                            .into_iter()
                            .map(|b| transition(b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON))
                            .collect();
                        list.ResourceBarrier(&back);
                    }
                    for (depth, to_depth) in &copies {
                        let Some(Some(twin)) = depth.twin.get() else { continue };
                        // From each texture's real state to the copy's and back.
                        let mut back = Vec::new();
                        twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                            let real = t.state.load(Ordering::Relaxed);
                            let wanted = legal_state(&t.texture, wanted);
                            (real != wanted).then(|| {
                                let there = transition(
                                    &t.texture,
                                    D3D12_RESOURCE_STATES(real as i32),
                                    D3D12_RESOURCE_STATES(wanted as i32),
                                );
                                back.push(transition(
                                    &t.texture,
                                    D3D12_RESOURCE_STATES(wanted as i32),
                                    D3D12_RESOURCE_STATES(real as i32),
                                ));
                                there
                            })
                        });
                        if !back.is_empty() {
                            list.ResourceBarrier(&back);
                        }
                    }
                    let barriers = moves();
                    if !barriers.is_empty() {
                        list.ResourceBarrier(&barriers);
                    }
                    list.Close()
                })
            };
            if let Err(e) = recorded {
                fail(format_args!("a host state prelude did not record ({e})"));
            }
            list.cast::<ID3D12CommandList>().ok()
        };
        let out = match repeated {
            Some(host) => {
                let Some(spare) = host.fresh_prelude(self.kind) else {
                    fail(format_args!("no fresh host prelude for a repeated list"));
                };
                Some((record(&spare)?, Some(spare)))
            }
            None => {
                let prelude = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
                Some((record(&prelude)?, None))
            }
        };
        // The lists' moves are recorded: what they leave is real now.
        for (_, last, copy) in states.values() {
            copy.state.store(*last, Ordering::Relaxed);
        }
        for (_, last, shadow) in buffer_states.values() {
            shadow.state.store(*last, Ordering::Relaxed);
        }
        out
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data or GPU timing on): the device-removed report pairs the
    /// breadcrumbs' marker commands with the texts in order and names the
    /// commands a fault stopped among, and the timing report adds up the
    /// commands' GPU time by pipeline.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        // Timed notes are what `VICTORY_GPU_TIMING` reports on; markers are
        // only what the device-removed report pairs with breadcrumbs. The last
        // two stamp slots are the busy pair, so notes stop before them.
        if noting() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < BUSY_START {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();

            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        self.assert_direct("used_pipeline");
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        let memory = self.memory_writer.load(Ordering::Relaxed);
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked, memory));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + Sync + 'static) {
        self.deferred().missed.push(std::sync::Arc::new(check));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + Sync + 'static) {
        self.deferred().after.push(std::sync::Arc::new(work));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Records that the list resolved query results into `readback`, and
    /// that each (destination, offset, bytes) of `targets` is written from
    /// it once the list has run. The buffer is kept until the list is
    /// reset, because its contents are only valid after the wait.
    pub(crate) fn defer_resolve(
        &self,
        readback: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        targets: Vec<Target>,
    ) {
        self.deferred().resolves.push((readback, targets));
    }

    /// A memory value the list waits for before it starts.
    pub(crate) fn defer_wait(&self, wait: crate::hold::ValueWait) {
        self.deferred().waits.push(wait);
    }

    /// The memory values the list waits for before it starts.
    pub(crate) fn waits(&self) -> Vec<crate::hold::ValueWait> {
        self.deferred().waits.clone()
    }

    /// Whether nothing has been recorded since the list was made or reset
    /// that a wait recorded now would have to follow: no work, texture
    /// use, bottom-of-pipe write or pipeline (state settings alone carry
    /// over).
    pub(crate) fn is_empty(&self) -> bool {
        let d = self.deferred();
        let deferred_empty = d.copies.is_empty() && d.uploads.is_empty() && d.writes.is_empty() && d.after.is_empty() && d.missed.is_empty();
        drop(d);
        deferred_empty
            && self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset. DIRECT only (COPY uses staging space).
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if self.is_copy() {
            debug_assert!(false, "upload on a COPY list: staging_space only");
            return None;
        }
        let (buffer, data) = host.list_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred()
            .uploads
            .push((buffer.clone(), data as usize, size as usize));
        Some((buffer, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.mapped_buffer(size, true)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push((buffer.clone(), data as usize, size as usize));
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer. DIRECT only: the
    /// COPY path uses only `upload_buffer`/`staging_space` memory, never
    /// the list's CUSTOM WC L0 scratch.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        if self.is_copy() {
            debug_assert!(false, "scratch on a COPY list: staging_space only");
            return None;
        }
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.list_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred()
                .uploads
                .push((buffer, data as usize, len as usize));
            return Some((gpu, data));
        }
        while s.chunks.get(s.current).is_some() && len > SCRATCH_CHUNK - s.used {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let spare = pool(&SPARE_SCRATCH).pop();
            let chunk = match spare {
                Some(chunk) => chunk,
                None => {
                    let (buffer, data) = host.list_buffer(SCRATCH_CHUNK)?;
                    name(&buffer, || "list scratch".into());
                    // SAFETY: a live buffer.
                    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                    (buffer, data as usize, gpu)
                }
            };
            s.chunks.push(chunk);

            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let result = (chunk.2 + at, (chunk.1 + at as usize) as *mut u8);
        s.used += len;
        Some(result)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on. DIRECT
    /// only.
    pub(crate) fn budget(&self) -> Option<u64> {
        if self.is_copy() {
            debug_assert!(false, "budget on a COPY list");
            return None;
        }
        let (buffer, data) = {
            let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
            if budget.is_none() {
                let (b, d) = host()?.shared_buffer(256)?;
                *budget = Some((b, d as usize));
            }
            budget.clone()?
        };
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Host(
                Box::new(move || unsafe { (at as *mut u32).write_volatile(0) }),
                (0, 0),
                (at, 4),
            ));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// The list's argument buffer, at least `bytes` long, common between
    /// its uses (`crate::indirect`). DIRECT only.
    pub(crate) fn arguments(&self, bytes: u64) -> Option<ID3D12Resource> {
        if self.is_copy() {
            debug_assert!(false, "arguments on a COPY list");
            return None;
        }
        let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live buffer.
        if let Some(b) = arguments.as_ref().filter(|b| unsafe { b.GetDesc() }.Width >= bytes) {
            return Some(b.clone());
        }
        let (made, _) = host()?.shared_buffer(bytes.next_power_of_two().max(1 << 16))?;
        name(&made, || "indirect arguments".into());
        // Commands recorded earlier may still name the smaller one.
        if let Some(old) = arguments.replace(made.clone()) {
            self.keep(&old);
        }
        Some(made)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Keeps dynamic descriptor slots `[first, first + n)` on title heap
    /// `heap_id` leased until this recording's submission finishes (or the
    /// list resets while idle).
    pub(crate) fn keep_dynamic(&self, heap_id: u64, first: u32, n: u32) {
        self.deferred()
            .dynamic
            .push(DynamicLease(heap_id, first, n));
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again): the
    /// range `writes` (address, bytes); it reads `reads`.
    pub(crate) fn defer(
        &self,
        writes: (usize, usize),
        reads: (usize, usize),
        work: impl Fn() + Send + 'static,
    ) {
        self.deferred()
            .copies
            .push(Step::Run(Box::new(work), writes, reads));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory: it
    /// reads the title's `reads` and writes `writes` (pointer, bytes) of the
    /// list's own memory (upload or scratch), if any.
    pub(crate) fn defer_host(
        &self,
        reads: (usize, usize),
        writes: (usize, usize),
        work: impl Fn() + Send + 'static,
    ) {
        self.deferred()
            .copies
            .push(Step::Host(Box::new(work), reads, writes));
    }

    /// For a repeat of the list in one submission: the bytes its steps
    /// would have written into the list's memory (pointer, bytes) put in an
    /// upload buffer of its own, with the copies into the list's buffers
    /// the repeat's prelude records. Memory not the list's is written at
    /// once, as before.
    fn repeat(&self, host: &Host, writes: Vec<(usize, Vec<u8>)>) -> Option<Repeat> {
        let total: usize = writes.iter().map(|(_, b)| b.len()).sum();
        let (staging, data) = host.upload_buffer(total as u64)?;
        name(&staging, || "list repeat".into());
        let mut copies = Vec::with_capacity(writes.len());
        let mut at = 0;
        let deferred = self.deferred();
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
        for (dst, bytes) in writes {
            let n = bytes.len();
            let inside = |(buffer, start, len): (&ID3D12Resource, usize, usize)| {
                (start <= dst && dst + n <= start + len)
                    .then(|| (buffer.clone(), (dst - start) as u64))
            };
            let found = deferred
                .uploads
                .iter()
                .map(|(b, s, l)| (b, *s, *l))
                .chain(
                    scratch
                        .chunks
                        .iter()
                        .map(|(b, s, _)| (b, *s, SCRATCH_CHUNK as usize)),
                )
                .chain(budget.iter().map(|(b, s)| (b, *s, 256)))
                .find_map(inside);
            match found {
                Some((buffer, offset)) => {
                    // SAFETY: inside our mapped buffer of `total` bytes.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.add(at), n) };
                    copies.push((buffer, offset, at as u64, n as u64));
                    at += n;
                }
                None => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(
                            "a repeated list's step writes memory not its own: written at once"
                        )
                    });
                    // SAFETY: what the step would have written itself.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, n) };
                }
            }
        }
        (!copies.is_empty()).then_some(Repeat { staging, copies })
    }

    /// Brings a texture's staging buffer and host texture `copy` up to date
    /// when the list is submitted (`crate::bindless::uploads`), keeping
    /// both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        copy: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred().copies.push(Step::Texture(staging, copy));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Run(..) | Step::Host(..) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(..) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.settle(src, len, false);
                    batch.settle(dst, len, true);
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.settle(src, len, false);
                    match &mut batch.redirect {
                        Some(redirect) => {
                            let mut bytes = vec![0; len];
                            copy_mapped(bytes.as_mut_ptr(), src as *const u8, len, false);
                            redirect.push((dst, bytes));
                        }
                        None => {
                            copy_mapped(dst as *mut u8, src as *const u8, len, false);
                        }
                    }
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    // Title memory an earlier submission's shaders write is
                    // current only once that one is copied back, if the
                    // title wrote it since the copy took it.
                    batch.settle_refresh(shadow, *address, *len);
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work, _, _) | Step::Host(work, _, _) => {
                    let mut own = (0, 0);
                    match *step {
                        Step::Run(_, (to, written), (from, read)) => {
                            batch.settle(from, read, false);
                            batch.settle(to, written, true);
                            batch.title_written();
                        }
                        Step::Host(_, (from, read), writes) => {
                            batch.settle(from, read, false);
                            own = writes;
                        }
                        _ => {}
                    }
                    match &mut batch.redirect {
                        // A repeat: what the work writes into the list's
                        // memory goes to the repeat's, the list's left as
                        // its earlier run reads it.
                        Some(redirect) if own.1 > 0 => {
                            let (at, n) = own;
                            // SAFETY: the list's own mapped memory, which
                            // nothing running uses (its earlier run in this
                            // submission has not started).
                            let mine = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, n) };
                            let kept = mine.to_vec();
                            work();
                            redirect.push((at, mine.to_vec()));
                            mine.copy_from_slice(&kept);
                        }
                        _ => work(),
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    // The range's snapshot is taken from the host copy, which
                    // an earlier submission's shaders may still be writing;
                    // the shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    batch.settle(*address, *len, false);
                    if !marked {
                        batch.flush();
                    }
                    shadow.writing(*address, *len, *marked);

                    continue;
                }
                // Brought up to date before the submission's lists were made.
                Step::Texture(..) => continue,
                Step::Heap(viewed) => {
                    crate::bindless::refresh(viewed, |s, a, l| {
                        batch.settle_refresh(s, a, l);
                        batch.add(s, a, l);
                        batch.reads.push((a, a + l));
                    });
                    continue;
                }
                &Step::Memory(heap) => {
                    // shortcut: shaders reading title memory through the
                    // page table may write any copy it names (the pages they
                    // wrote are known only once they have run): the
                    // submission counts as writing them all.
                    let host = batch.host;
                    crate::memory::prepare(
                        heap,
                        || {
                            if let Some(host) = host {
                                host.settle_table(heap);
                            }
                        },
                        |s, a, l| {
                            batch.settle_refresh(s, a, l);
                            batch.add(s, a, l);
                            batch
                                .shaders
                                .push((std::sync::Arc::as_ptr(s) as usize, a, a + l));
                        },
                    );
                    batch.tables.insert(heap);
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// One host copy's copy-back ranges (address, length, whether each goes
/// back), in declaration order.
type BackGroup = (
    std::sync::Arc<crate::shadow::Shadow>,
    Vec<(usize, usize, bool)>,
);

/// Copies back what the submission's shaders wrote (each range once per
/// submission, however many dispatches declared it). One thread per host
/// copy where there are enough ranges: entries of one copy run in
/// declaration order on one thread (their ranges may overlap); entries of
/// different copies touch disjoint copy mappings, sharing only locks and
/// atomic counters.
fn run_back(back: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)]) {
    if back.len() < PARALLEL_ENTRIES {
        for (shadow, address, len, goes_back) in back {
            run_one_back(shadow, *address, *len, *goes_back);
        }
        return;
    }
    // Group by host copy, keeping declaration order (first appearance,
    // then each copy's ranges in order).
    let mut groups: Vec<BackGroup> = Vec::new();
    let mut at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (shadow, address, len, goes_back) in back {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        match at.get(&key) {
            Some(&i) => groups[i].1.push((*address, *len, *goes_back)),
            None => {
                at.insert(key, groups.len());
                groups.push((shadow.clone(), vec![(*address, *len, *goes_back)]));
            }
        }
    }
    if copy_threads(groups.len()) <= 1 {
        for (shadow, ranges) in &groups {
            for &(address, len, goes_back) in ranges {
                run_one_back(shadow, address, len, goes_back);
            }
        }
        return;
    }
    let chunk = groups.len().div_ceil(copy_threads(groups.len()));
    std::thread::scope(|s| {
        let mut parts = groups.chunks(chunk);
        // The submitting thread takes the first chunk itself.
        let mine = parts.next();
        let mut handles = Vec::new();
        for part in parts {
            handles.push(s.spawn(move || {
                for (shadow, ranges) in part {
                    for &(address, len, goes_back) in ranges {
                        run_one_back(shadow, address, len, goes_back);
                    }
                }
            }));
        }
        if let Some(part) = mine {
            for (shadow, ranges) in part {
                for &(address, len, goes_back) in ranges {
                    run_one_back(shadow, address, len, goes_back);
                }
            }
        }
        for h in handles {
            h.join().expect("a copy-back thread panicked");
        }
    });
}

/// One copy-back entry: what shaders wrote in the range goes back into
/// title memory, or is kept where shaders read with no copy.
fn run_one_back(
    shadow: &std::sync::Arc<crate::shadow::Shadow>,
    address: usize,
    len: usize,
    goes_back: bool,
) {
    if goes_back {
        let written = shadow.copy_back(address, len);
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
    } else {
        shadow.keep_writes(address, len);
        crate::memory::note_withheld(len as u64);
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
///
/// A copy's range brought up to date is not copied again in the submission
/// until a step writes title memory: every list of a world submission using
/// title memory asks for every copy whole. What the title's threads write
/// meanwhile is for later submissions (on the console it would race the
/// GPU).
#[derive(Default)]
struct Batch<'a> {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// The range of each copy brought up to date since the last title
    /// write, by copy (held, so that its address names no other).
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// The host, whose earlier submissions may still be in flight.
    host: Option<&'a Host>,
    /// For a repeat of a list in the submission: what its steps write into
    /// the list's own memory (pointer, bytes), for its prelude to copy.
    redirect: Option<Vec<(usize, Vec<u8>)>>,
    /// Ranges brought up to date through view heaps (the GPU reads them),
    /// and copies a page table names (its shaders may write them).
    reads: Vec<(usize, usize)>,
    shaders: Vec<(usize, usize, usize)>,
    /// Page tables (by view heap key) the submission's shaders read.
    tables: std::collections::HashSet<usize>,
}

/// Entries that pay back data-parallel copies: at least this many ranges
/// in one batch before threads are worth their spawning.
const PARALLEL_ENTRIES: usize = 16;

/// Threads for a data-parallel batch of copies: half the logical
/// processors, as the pipeline workers, so the title's threads still have
/// cores (one thread per host copy at most). A compile storm is running
/// the workers on the other cores, so the batch stays mostly out of
/// their way.
fn copy_threads(entries: usize) -> usize {
    if entries < PARALLEL_ENTRIES {
        return 1;
    }
    let mut threads = std::thread::available_parallelism()
        .map_or(2, |n| (n.get() / 2).clamp(2, 8));
    if crate::pipeline::pipelines_waiting() > 0 {
        threads = threads.min(2);
    }
    threads.min(entries)
}

impl Batch<'_> {
    /// Before a step reads (or, with `writing`, writes) title memory
    /// `[address, address + len)`: earlier submissions whose GPU work
    /// writes (or reads) it are finished first.
    fn settle(&self, address: usize, len: usize, writing: bool) {
        if let Some(host) = self.host {
            host.settle_range(address, len, writing);
        }
    }

    /// Before a refresh of `[address, address + len)` of `shadow`: as
    /// [`Batch::settle`], but a submission whose shaders write the range
    /// through the same copy is waited for only if the title wrote a page
    /// of the overlap since the copy took it (the world's big heap is both
    /// refreshed and written by shaders in most submissions).
    fn settle_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let Some(host) = self.host else { return };
        let copy = std::sync::Arc::as_ptr(shadow) as usize;
        let to = address.saturating_add(len);
        if let Some(value) = host
            .timeline
            .refresh_conflict(copy, address, to, |a, b| shadow.stale(a, b))
        {
            host.settle(value);
        }
    }


    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        // Nothing to copy: a world submission asks for every copy its heaps'
        // views and page tables reach (~9,000), a few of them written.
        if shadow.clean() {
            return;
        }
        let (key, end) = (std::sync::Arc::as_ptr(shadow) as usize, address + len);
        if self.current.get(&key).is_some_and(|&(_, a, b)| a <= address && end <= b) {
            return;
        }
        match self.pending.get(&key) {
            Some(&i) => {
                let (_, from, to) = &mut self.refreshes[i];
                (*from, *to) = ((*from).min(address), (*to).max(end));
            }
            None => {
                self.pending.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.current.clear();
    }

    /// Does the refreshes; returns the bytes copied. Serial: the takes
    /// share the title-memory guard lock with the title threads' fault
    /// handler, and one take is microseconds — threads convoyed on it for
    /// no wall-time win (diag7: guard waits 20×, per-take cost 4×), so the
    /// updates stay on the submitting thread.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let refreshes = std::mem::take(&mut self.refreshes);
        for (shadow, from, to) in &refreshes {
            self.note_current(shadow, *from, *to);
        }
        let mut total = 0;
        for (shadow, from, to) in refreshes {
            let (copied, us) = Self::refresh_one(&shadow, from, to);
            total += copied;
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, us, 0);
            #[cfg(not(feature = "diagnostics"))]
            let _ = us;
        }
        total
    }

    /// One refresh: the copy's changed pages, and how long it took.
    fn refresh_one(
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        from: usize,
        to: usize,
    ) -> (usize, u64) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        (copied, us)
    }

    /// Remembers a flushed refresh as the copy's range brought up to date:
    /// the union with what was current if they touch, else the larger.
    fn note_current(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        from: usize,
        to: usize,
    ) {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        let (_, a, b) = self.current.entry(key).or_insert((shadow.clone(), from, to));
        if from <= *b && *a <= to {
            (*a, *b) = ((*a).min(from), (*b).max(to));
        } else if to - from > *b - *a {
            (*a, *b) = (from, to);
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FallbackUs);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        made.ok().and(buffer)
    }

    /// A mapped buffer for a list's own memory (uploads, scratch): as an
    /// upload buffer for the CPU, and a copy destination for the GPU (a
    /// repeat of the list in one submission gets its bytes copied in,
    /// `Repeat`).
    pub(crate) fn list_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no list buffer");
                None
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, false)
    }

    /// A mapped buffer of `size` bytes: upload memory, or (`readback`)
    /// readback memory in the copy-destination state.
    fn mapped_buffer(&self, size: u64, readback: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback { D3D12_HEAP_TYPE_READBACK } else { D3D12_HEAP_TYPE_UPLOAD },
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    if readback { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_GENERIC_READ },
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, readback, "no mapped buffer");
                None
            }
        }
    }

    /// A buffer in the read-back heap, not mapped: what the GPU resolved
    /// into it is read after the submission that filled it has finished.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(8),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer; nothing maps it
        // until the submission that resolved into it has finished.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if self.next_attempt.is_some_and(|next| now < next) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    /// The newest submission when the last frame was presented (its copy
    /// to the back buffer included): the next present waits for it, so one
    /// frame's GPU work runs on past its present.
    presented: Option<u64>,
    /// Diagnostics: the last presented frame and its probe, read at the
    /// next present once that frame's GPU work has finished.
    #[cfg(feature = "diagnostics")]
    probed: Option<(u64, ID3D12Resource)>,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let root_views = crate::draw::take_root_view_slots() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} ms host \
                 wait for submissions; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame",
                making = crate::pipeline::making()
            );
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let (made, made_bytes, freed, freed_bytes) = crate::resource::texture_flow();
            let (heaps, heap_used, heap_free) = crate::placement::stats();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 made {:.1}/frame ({:.1} MB/frame), freed {:.1}/frame ({:.1} MB/frame); heaps of textures {heaps} ({:.0} MB used, {:.0} MB free); \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                made as f64 / frames,
                made_bytes as f64 / 1e6 / frames,
                freed as f64 / frames,
                freed_bytes as f64 / 1e6 / frames,
                heap_used as f64 / 1e6,
                heap_free as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                if let Some([(local, budget), (system, system_budget)]) = video_memory() {
                    let (commit, working_set) = process_memory();
                    tracing::info!(
                        "video memory: {local:.0} MB of a {budget:.0} MB budget on the card, \
                         {system:.0} MB of {system_budget:.0} MB in system memory; process commit {:.0} MB, \
                         working set {:.0} MB; Rust heap {:.0} MB; placed textures made {:.0} MB; list upload memory {} blocks {:.0} MB",
                        commit as f64 / 1e6,
                        working_set as f64 / 1e6,
                        crate::copystat::HEAP_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::resource::PLACED_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::list::UPLOADS[0].load(Ordering::Relaxed),
                        crate::list::UPLOADS[1].load(Ordering::Relaxed) as f64 / 1e6,
                    );
                }
                let [gpu, title] = crate::resource::texture_census().map(|r| r.map(|b| b as f64 / 1e6));
                tracing::info!(
                    "host texture MB (used by lists, viewed by heap slots only, neither): GPU-written {:.0} {:.0} {:.0}, \
                     from title memory {:.0} {:.0} {:.0}",
                    gpu[0], gpu[1], gpu[2], title[0], title[1], title[2]
                );
                let copy_frames = std::mem::take(&mut self.copy_stats_frames) as f64;
                crate::frame::log();
                log_gpu_times(copy_frames);
                crate::copystat::log(copy_frames);
                crate::bindless::log(copy_frames);
                crate::shadow::log_views(copy_frames);
                costs::log(copy_frames);
                let [n, bytes] = [0, 1].map(|i| crate::list::READ_BACKS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "texture read backs: {:.1}/frame, {:.1} KB/frame",
                        n as f64 / frames,
                        bytes as f64 / frames / 1e3
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). Waits for the previous
    /// frame's submissions, not this one's. In diagnostics builds: the
    /// previous probed frame and whether it was black (read once its GPU
    /// work has finished); else `None`.
    pub(crate) fn present(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        frame: u64,
    ) -> Option<(u64, bool)> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() {
            *presenter = match retry
                .attempt(Instant::now(), || self.presenter(width, height, format))?
            {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // The back buffers may not be in use.
            self.drain();
            // SAFETY: no back buffer is held; every submission has finished.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };

            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        // The previous frame's GPU work (and its copy, whose list is reused
        // below) finishes before this frame is presented; this frame's may
        // still run. The title waits for its fences, not for the present:
        // its submission thread calls PresentX before it hands its ring
        // slots back (#1118), so a present that waits for the frame it
        // presents holds that thread for the frame's whole GPU time.
        self.previous_frame(p.presented.take());
        #[cfg(feature = "diagnostics")]
        let shown = p.probed.take().map(|(f, b)| (f, crate::capture::black(&b)));
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            // The copy may still run after this returns: it holds the
            // texture, which the title may release before the next present.
            p.copy.keep(texture);
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.presented = self.frame_done();
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                // A saved frame is read back now: its copy has to be done.
                self.drain();
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            {
                p.probed = probe.map(|b| (frame, b));
                shown
            }
            #[cfg(not(feature = "diagnostics"))]
            None
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                presented: None,
                #[cfg(feature = "diagnostics")]
                probed: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// Diagnostics, `VICTORY_COPIES_IN_VIDEO_MEMORY`: [`Host::shared_buffer`]s
    /// (host copies of title memory among them) in video memory the CPU maps
    /// (a GPU upload heap, where the host has one), to measure how much of
    /// the GPU's time is shaders reading the copies over the bus. CPU reads
    /// of them (snapshots, copy-backs) become slow: for that measurement only.
    #[cfg(feature = "diagnostics")]
    fn copies_in_video_memory(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            if !crate::switch::on("VICTORY_COPIES_IN_VIDEO_MEMORY") {
                return false;
            }
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            }
            .is_ok()
                && options.GPUUploadHeapSupported.as_bool();
            tracing::warn!(on = ok, "host copies of title memory in video memory (diagnostics)");
            ok
        })
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        #[cfg(feature = "diagnostics")]
        let heap = if self.copies_in_video_memory() {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
        } else {
            heap
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// With `VICTORY_GPU_TIMING`: adds one list's noted commands to the cost
/// table, naming its slowest ones when its GPU busy took over 50 ms
/// (diagnostics). `busy_ms` is the list's GPU time (its busy-pair end minus
/// start, prelude work excluded), `residual_ms` the busy not in any note,
/// `dropped` the notes past the stamps, `wall_ms` the list's share of its
/// submission's CPU wall.
#[cfg(feature = "diagnostics")]
fn report_list(times: &[(f64, String)], busy_ms: f64, residual_ms: f64, dropped: usize, wall_ms: f64) {
    costs::add(times, residual_ms, dropped);
    if busy_ms > 50.0 {
        tracing::warn!(
            ms = busy_ms,
            noted_ms = times.iter().map(|(ms, _)| ms).sum::<f64>(),
            residual_ms,
            dropped,
            wall_ms,
            commands = times.len(),
            "slow host list"
        );
        for (ms, op) in times.iter().take(6) {
            tracing::warn!("slow list command, {ms:.1} ms: {op}");
        }
    }
}

/// What a note runs, for the GPU cost table: the kind of command, its size,
/// and the pipeline it binds ("draw 36x1 [vs 12ab, ps 34cd]", without the
/// views and tables that follow). Notes that name resources rather than a
/// pipeline keep the first words that are not addresses ("copy to",
/// "texture upload").
#[cfg(feature = "diagnostics")]
fn cost_label(text: &str) -> String {
    let head = text.split(" | ").next().unwrap_or(text);
    if head.contains('[') {
        // Addresses differ per draw; drop them so that the same pass adds up.
        let words: Vec<&str> = head
            .split(' ')
            .filter(|w| !w.starts_with("0x") && *w != "at")
            .collect();
        return words.join(" ").chars().take(160).collect();
    }
    let words: Vec<&str> = head
        .split(' ')
        .take_while(|w| w.chars().all(|c| c.is_ascii_alphabetic()))
        .collect();
    words.join(" ")
}

/// The note's kind for the kind-then-pipeline rows: the first word of its
/// cost label, normalized (`draw`, `indexed`, `dispatch`, `mesh`, `copy`,
/// `resolve`, `texture`, `DROPPED`, else `other`).
#[cfg(feature = "diagnostics")]
fn cost_kind(text: &str) -> String {
    let first = cost_label(text)
        .split(' ')
        .next()
        .unwrap_or("other")
        .to_owned();
    match first.as_str() {
        "draw" | "indexed" | "dispatch" | "mesh" | "copy" | "resolve" | "texture" | "DROPPED" => first,
        _ => "other".to_owned(),
    }
}

/// How many draws (instances, dispatch groups, mesh groups) a note runs,
/// for us-per-draw. `draw 36x1` and `indexed 6x2` count instances (the number
/// after the `x`); `dispatch 4x2x2` counts groups (the product);
/// `mesh 8` counts groups; anything else counts one per command.
#[cfg(feature = "diagnostics")]
fn parse_draws(text: &str) -> u64 {
    let head = text.split(" | ").next().unwrap_or(text);
    let mut words = head.split(' ');
    let kind = words.next().unwrap_or("");
    let size = words.next().unwrap_or("");
    let product = |s: &str| {
        s.split('x').try_fold(1u64, |a, p| {
            p.parse::<u64>().ok().map(|v| a.saturating_mul(v.max(1)))
        })
    };
    match kind {
        "draw" | "indexed" => size
            .split_once('x')
            .and_then(|(_, m)| m.parse::<u64>().ok())
            .map(|m| m.max(1))
            .unwrap_or(1),
        "dispatch" => product(size).unwrap_or(1).max(1),
        "mesh" => size
            .split('x')
            .next()
            .and_then(|n| n.parse::<u64>().ok())
            .map(|n| n.max(1))
            .unwrap_or(1),
        _ => 1,
    }
}

/// Which run indices each cuts group holds: run `i` is in the group whose
/// `cuts` window contains `firsts[i]` (diagnostics, timing). Alone mode does
/// not use this (each run is its own group).
#[cfg(feature = "diagnostics")]
fn group_members(firsts: &[usize], cuts: &[usize]) -> Vec<Vec<usize>> {
    cuts.windows(2)
        .map(|c| {
            (0..firsts.len())
                .filter(|&i| firsts[i] >= c[0] && firsts[i] < c[1])
                .collect()
        })
        .collect()
}

/// With `VICTORY_GPU_TIMING`: the GPU time the noted commands took since the
/// last report, added up by what they ran, so that a frame's cost can be read
/// off by pipeline (diagnostics). Percentages are over summed GPU busy (the
/// sum of the lists' busy-pair end minus start; prelude lists hold transitions
/// only and are not bracketed, so their GPU time is excluded); the residual is
/// the busy in no note, per list and per frame. Log-only, in the frame-rate
/// line's companion report.
#[cfg(feature = "diagnostics")]
mod costs {
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// One kind-then-pipeline row: commands, their GPU milliseconds and the
    /// draws (instances, groups) they ran.
    type Row = (u64, f64, u64);

    /// Commands, their GPU milliseconds, busy/wall and starvation inputs,
    /// since the last report.
    #[derive(Default)]
    struct Tally {
        by_row: HashMap<(String, String), Row>,
        noted: f64,
        residual: f64,
        dropped: u64,
        lists: u64,
        submissions: u64,
        busy: f64,
        wall: f64,
    }

    static TALLY: Mutex<Option<Tally>> = Mutex::new(None);

    /// Adds one list's noted commands (as GPU milliseconds and note, longest
    /// first), its residual busy and its notes dropped past the stamps.
    /// Submissions' busy and wall go through [`add_submission`].
    pub(super) fn add(times: &[(f64, String)], residual_ms: f64, dropped: usize) {
        let mut tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
        let tally = tally.get_or_insert_with(Tally::default);
        tally.lists += 1;
        tally.residual += residual_ms;
        tally.dropped += dropped as u64;
        for (ms, text) in times {
            let key = (super::cost_kind(text), super::cost_label(text));
            let entry = tally.by_row.entry(key).or_default();
            entry.0 += 1;
            entry.1 += ms;
            entry.2 += super::parse_draws(text);
            tally.noted += ms;
        }
    }

    /// Adds one submission piece's summed GPU busy (the sum of its lists'
    /// busy pairs, prelude work excluded) and CPU wall milliseconds (the
    /// `ExecuteCommandLists` to fence-waited wall). The gap is wall minus
    /// summed busy (idle plus untimed prelude work and any overlap); the ratio
    /// is busy over wall.
    pub(super) fn add_submission(busy_ms: f64, wall_ms: f64) {
        let mut tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
        let tally = tally.get_or_insert_with(Tally::default);
        tally.submissions += 1;
        tally.busy += busy_ms;
        tally.wall += wall_ms;
    }

    /// Logs the biggest kind-then-pipeline rows per frame since the last
    /// call (16 rows), with the residual, summed-busy/gap/utilisation and the
    /// timeline-depth histogram and per-cause blocking milliseconds.
    /// Starvation counters are drained even when no lists were timed, so an
    /// early return never carries them into the next period.
    pub(super) fn log(frames: f64) {
        let (hist, block) = super::starve::take();
        let Some(mut tally) = TALLY.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            log_starvation(&hist, &block, frames);
            return;
        };
        if tally.lists == 0 {
            log_starvation(&hist, &block, frames);
            return;
        }
        let mut items: Vec<((String, String), Row)> = tally.by_row.drain().collect();
        items.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        // Percentages below divide by summed busy: guard the near-zero case
        // instead of scaling by `max(1e-6)`, which would report huge shares
        // for a ~zero busy.
        let busy_for_pct = if tally.busy > 1e-9 { tally.busy } else { 0.0 };
        let pct_of_busy = |ms: f64| {
            if busy_for_pct > 0.0 {
                100.0 * ms / busy_for_pct
            } else {
                0.0
            }
        };
        let per_frame = |v: f64| v / frames;
        let commands: u64 = items.iter().map(|(_, (n, _, _))| *n).sum();
        let gap = (tally.wall - tally.busy).max(0.0);
        let util = if tally.wall > 0.0 { 100.0 * tally.busy / tally.wall } else { 0.0 };
        tracing::info!(
            "GPU cost by pipeline: {:.1} ms/frame summed busy over {:.0} lists in {:.0} submissions ({:.0} commands/frame; wall {:.1} ms/frame, gap {:.1}, util {:.0}%; noted {:.1}, residual {:.1}, dropped {:.0})",
            per_frame(tally.busy),
            per_frame(tally.lists as f64),
            per_frame(tally.submissions as f64),
            per_frame(commands as f64),
            per_frame(tally.wall),
            per_frame(gap),
            util,
            per_frame(tally.noted),
            per_frame(tally.residual),
            per_frame(tally.dropped as f64)
        );
        for ((kind, label), (n, ms, draws)) in items.iter().take(16) {
            let us_per_draw = ms * 1000.0 / (*draws).max(1) as f64;
            tracing::info!(
                "{:.2} ms/frame ({:.0}% of busy, {:.3} ms each, {:.1} us/draw, {:.1}/frame): {kind} {label}",
                per_frame(*ms),
                pct_of_busy(*ms),
                ms / (*n).max(1) as f64,
                us_per_draw,
                per_frame(*n as f64)
            );
        }
        log_starvation(&hist, &block, frames);
    }

    /// Logs the drained starvation histogram and per-cause blocking
    /// milliseconds per frame (diagnostics, timing). Called from [`log`]
    /// even when no lists were timed.
    fn log_starvation(hist: &[u64; super::starve::DEPTHS], block: &[u64; super::starve::CAUSES], frames: f64) {
        let hist_line: Vec<String> = hist
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(d, &c)| format!("{d}:{:.1}", c as f64 / frames))
            .collect();
        if !hist_line.is_empty() {
            tracing::info!(
                "GPU starvation: timeline depth pushes/frame by depth ({}): {}",
                super::starve::DEPTHS,
                hist_line.join(" ")
            );
        }
        let block_ms = block.iter().map(|&u| u as f64 / 1e3 / frames).collect::<Vec<_>>();
        if block_ms.iter().any(|&m| m > 0.0) {
            tracing::info!(
                "GPU starvation blocking: {:.2} ms/frame (inflight {:.2}, running {:.2}, conflict {:.2}, staging/table {:.2}, drain/present {:.2}, copy/wait {:.2})",
                block_ms.iter().sum::<f64>(),
                block_ms[super::starve::IN_FLIGHT],
                block_ms[super::starve::RUNNING],
                block_ms[super::starve::CONFLICT],
                block_ms[super::starve::STAGING_TABLE],
                block_ms[super::starve::DRAIN_PRESENT],
                block_ms[super::starve::COPY_WAIT]
            );
        }
    }
}
/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
static RETIRED: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(resource);
}

/// The retired resources go now (tests; submissions take them along).
#[cfg(test)]
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    if !retired.is_empty() {
        release_later(Box::new(retired));
    }
}

/// Drops `what` on a thread of its own: releasing a D3D12 resource is a
/// kernel call (~5 % of a world frame on the title's submitting thread,
/// run perf1). Only for objects no GPU work in flight uses.
#[allow(dead_code)]
fn release_later(what: Box<dyn std::any::Any + Send>) {
    type Release = std::sync::mpsc::Sender<Box<dyn std::any::Any + Send>>;
    static RELEASE: OnceLock<Release> = OnceLock::new();
    let release = RELEASE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn std::any::Any + Send>>();
        // Without the thread the receiver is gone: sends fail and drop here.
        let _ = std::thread::Builder::new()
            .name("victory-release".into())
            .spawn(move || rx.into_iter().for_each(drop));
        tx
    });
    let _ = release.send(what);
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds: all of
/// them in listed capture frames, the first 500 elsewhere.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if is_traced(what)
        && (crate::capture::listed(crate::queue::presented() + 1)
            || LOGGED.fetch_add(1, Ordering::Relaxed) < 500)
    {
        tracing::info!("traced command: {what}");
    }
}

/// The shader hashes `VICTORY_TRACE_SHADERS` lists (comma-separated;
/// diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn traced_shaders() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    })
}

/// Whether `what` names a shader [`traced_shaders`] lists.
#[cfg(feature = "diagnostics")]
pub(crate) fn is_traced(what: &str) -> bool {
    traced_shaders().iter().any(|h| what.contains(h.as_str()))
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::ISSUED.with(|l| l.borrow_mut().push((resource.as_raw() as usize, before.0 as u32, after.0 as u32)));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // `mod tests` is already taken at the top of this file
    use super::Batch;

    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }
}

    /// Enough host copies flush their refreshes in one batch: disjoint
    /// copies copy what the title wrote, each once (serial: the takes
    /// share the guard lock, so threads convoyed — see `Batch::flush`).
    #[test]
    fn flush_copies_each_copy_once() {
        use crate::watch::PAGE;
        const COPIES: usize = 20;
        let mut pools = Vec::with_capacity(COPIES);
        let mut shadows = Vec::with_capacity(COPIES);
        for _ in 0..COPIES {
            let p = crate::watch::tests::pool(1);
            crate::shadow::register(p, p as u64, PAGE as u64);
            let Some(copy) = crate::shadow::find(p as u64, PAGE as u64) else {
                crate::shadow::unregister(p, p as u64);
                crate::watch::tests::free(p);
                for (p, copy) in pools {
                    crate::shadow::unregister(p, p as u64);
                    drop(copy);
                    crate::watch::tests::free(p);
                }
                return; // no host GPU here
            };
            pools.push((p, copy.clone()));
            shadows.push(copy);
        }
        // The title writes one page's byte in every pool.
        for (i, &(p, _)) in pools.iter().enumerate() {
            // SAFETY: our pools, guarded since the copies took them.
            unsafe { ((p + 8) as *mut u8).write_volatile(i as u8) };
        }
        let mut batch = Batch::default();
        for (p, copy) in &pools {
            batch.add(copy, *p, PAGE);
        }
        assert_eq!(batch.flush(), COPIES * PAGE, "every copy once");
        for (i, (p, copy)) in pools.iter().enumerate() {
            assert_eq!(copy.read_host((*p + 8) as u64, 1)[0].to_le_bytes()[0], i as u8);
        }
        // Only three pools change before the next submission.
        for &(p, _) in pools.iter().take(3) {
            // SAFETY: as above.
            unsafe { ((p + 16) as *mut u8).write_volatile(0x5a) };
        }
        let mut batch = Batch::default();
        for (p, copy) in &pools {
            batch.add(copy, *p, PAGE);
        }
        assert_eq!(batch.flush(), 3 * PAGE, "only the pages written");
        for (p, copy) in pools {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }

    /// Enough declared ranges copy back on threads: what the GPU wrote goes
    /// back in every range, overlapping ranges of one copy in order. Holds
    /// the snapshots' test lock: the block is one for all copies.
    #[test]
    fn parallel_copy_back_goes_back_in_every_range() {
        let _one = crate::shadow::snapshot_tests::ONE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        use crate::watch::PAGE;
        const COPIES: usize = 16;
        let mut pools = Vec::with_capacity(COPIES);
        for _ in 0..COPIES {
            let p = crate::watch::tests::pool(1);
            crate::shadow::register(p, p as u64, PAGE as u64);
            let Some(copy) = crate::shadow::find(p as u64, PAGE as u64) else {
                crate::shadow::unregister(p, p as u64);
                crate::watch::tests::free(p);
                for (p, copy) in pools {
                    crate::shadow::unregister(p, p as u64);
                    drop(copy);
                    crate::watch::tests::free(p);
                }
                return; // no host GPU here
            };
            pools.push((p, copy));
        }
        let mut back = Vec::new();
        for (i, &(p, ref copy)) in pools.iter().enumerate() {
            assert_eq!(copy.update(p, PAGE), PAGE);
            // Two ranges (overlapping on the first copy), each twice: the
            // run dedupes exact repeats and keeps declaration order.
            let (a, b) = if i == 0 { ((0, 100), (50, 150)) } else { ((0, 64), (PAGE - 64, 64)) };
            copy.writing(p + a.0, a.1, false);
            copy.writing(p + b.0, b.1, false);
            // The shaders write one byte in each range.
            copy.write_host((p + a.0) as u64, &[i as u8]);
            copy.write_host((p + b.0) as u64, &[(i as u8).wrapping_add(1)]);
            back.push((copy.clone(), p + a.0, a.1, true));
            back.push((copy.clone(), p + a.0, a.1, true));
            back.push((copy.clone(), p + b.0, b.1, true));
            back.push((copy.clone(), p + b.0, b.1, true));
        }
        // More entries than one thread takes, over enough copies.
        assert!(back.len() >= PARALLEL_ENTRIES);
        run_back(&back);
        for (i, &(p, _)) in pools.iter().enumerate() {
            // SAFETY: our pools.
            unsafe {
                if i == 0 {
                    assert_eq!((p as *const u8).read_volatile(), 0);
                    assert_eq!(((p + 50) as *const u8).read_volatile(), 1);
                } else {
                    assert_eq!((p as *const u8).read_volatile(), i as u8);
                    assert_eq!(((p + PAGE - 64) as *const u8).read_volatile(), (i as u8).wrapping_add(1));
                }
            }
        }
        for (p, copy) in pools {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }

    /// Copy threads stay serial under the threshold and fan out above it.
    #[test]
    fn copy_threads_fan_out_above_the_threshold() {
        assert_eq!(copy_threads(0), 1);
        assert_eq!(copy_threads(PARALLEL_ENTRIES - 1), 1);
        assert!(copy_threads(PARALLEL_ENTRIES) >= 2);
        assert!(copy_threads(1 << 20) <= 8);
    }


#[cfg(test)]
mod withholding {
    /// A list in which a program using title memory missed memory: what
    /// that program wrote is withheld from title memory and leaves the
    /// copies at once; what an ordinary program wrote in the same list goes
    /// back and stays in its copy (run light13: the imposter transforms in
    /// the BVH builder's list); an ordinary range overlapping a withheld one
    /// is withheld with it, since its copy-back would take the other's bytes
    /// along.
    #[test]
    fn only_programs_using_title_memory_are_withheld() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let pools = [crate::watch::tests::pool(1), crate::watch::tests::pool(1)];
        for &p in &pools {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
            crate::shadow::register(p, p as u64, PAGE as u64);
        }
        let copies: Vec<_> = pools.iter().map(|&p| crate::shadow::find(p as u64, PAGE as u64).unwrap()).collect();
        let list = host.new_list().unwrap();
        // (copy, start, bytes, the writer uses title memory, byte written at +0 and +16)
        let writes = [(0, 0, 32, false, 0x22), (1, 16, 16, true, 0x33), (1, 0, 32, false, 0x44)];
        for &(k, at, len, memory, value) in &writes {
            let _writer = list.writer_uses_memory(memory);
            let (copy, address) = (copies[k].clone(), (pools[k] + at) as u64);
            copy.refresh(&list, address, len);
            copy.gpu_writes(&list, address, len, false);
            // The shader's write, after the step that keeps the range's bytes.
            list.defer_host((0, 0), (0, 0), move || copy.write_host(address, &[value]));
        }
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: bytes of our pools.
        let title = |k: usize, at: usize| unsafe { ((pools[k] + at) as *const u8).read_volatile() };
        let copy = |k: usize, at: usize| {
            copies[k].update(pools[k], PAGE);
            copies[k].read_host((pools[k] + at) as u64, 1)[0] & 0xff
        };
        assert_eq!((title(0, 0), copy(0, 0)), (0x22, 0x22), "an ordinary program's write goes back and stays");
        assert_eq!((title(1, 16), copy(1, 16)), (0x11, 0x11), "withheld, then dropped from the copy");
        assert_eq!((title(1, 0), copy(1, 0)), (0, 0), "overlapping a withheld range: withheld with it");
        for (&p, copy) in pools.iter().zip(copies) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }
}

#[cfg(test)]
mod submissions {
    use std::sync::Mutex;

    use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, ID3D12Fence};

    use crate::queue::Written;

    /// Serializes tests that temporarily gate the shared host queue so two
    /// gates never overlap.
    static GATE_LOCK: Mutex<()> = Mutex::new(());

    /// Private fence armed via `Host::execute_gated`; always signalled on drop
    /// so even a test panic cannot leave the shared host queue stalled.
    struct Gate(ID3D12Fence);

    impl Gate {
        fn new(host: &super::Host) -> Self {
            // SAFETY: private fence on the live host device.
            let fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
            Gate(fence)
        }

        fn release(&self) {
            // SAFETY: our private fence.
            unsafe {
                let _ = self.0.Signal(1);
            }
        }
    }

    impl Drop for Gate {
        fn drop(&mut self) {
            self.release();
        }
    }

    #[test]
    fn a_list_reset_while_it_runs_records_into_fresh_memory() {
        let Some(host) = super::host() else { return }; // no host GPU here
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let list = host.new_list().unwrap();
        let (first, second) = (Box::new(0u64), Box::new(0u64));
        list.defer_write((&raw const *first).cast_mut().cast(), Written::Value(7), 8);
        list.close();
        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);
        // Gate guarantees the first submission is still in flight when reset runs.
        assert!(list.running());
        // Recorded again at once, as titles do: the running submission
        // keeps what it uses, including its bottom-of-pipe write.
        list.reset();
        list.defer_write((&raw const *second).cast_mut().cast(), Written::Value(9), 8);
        list.close();
        gate.release();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: our words; the finishing thread is done with them.
        let (a, b) = unsafe {
            (
                std::ptr::read_volatile(&*first),
                std::ptr::read_volatile(&*second),
            )
        };
        assert_eq!((a, b), (7, 9));
        assert!(!list.running());
    }

    /// A present waits for the previous frame's submissions, not its own:
    /// the frame it presents is still running when it returns, and is
    /// finished (bottom-of-pipe write included) once the next one begins.
    #[test]
    fn a_present_waits_for_the_previous_frame_not_its_own() {
        let Some(host) = super::host() else { return }; // no host GPU here
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let word = Box::new(0u64);
        let list = host.new_list().unwrap();
        list.defer_write((&raw const *word).cast_mut().cast(), Written::Value(5), 8);
        list.close();
        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);
        // This frame is presented while its GPU work is held back.
        let presented = host.frame_done();
        assert!(presented.is_some());
        assert!(list.running());
        // SAFETY: our word; nothing has written it yet.
        assert_eq!(unsafe { std::ptr::read_volatile(&*word) }, 0);
        gate.release();
        // The next frame's present finishes it first.
        host.previous_frame(presented);
        assert!(!list.running());
        // SAFETY: our word; the finishing thread is done with it.
        assert_eq!(unsafe { std::ptr::read_volatile(&*word) }, 5);
    }

    /// A submission naming one list twice: the repeat's prelude records into
    /// its own allocator and list while the first is still pending, so both
    /// run with their own barriers, and nothing waits on the CPU.
    #[test]
    fn a_list_named_twice_in_a_submission_has_two_preludes() {
        use std::sync::atomic::Ordering;
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let null = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        // A 16x16 RGBA8 render target, starting in COMMON.
        let desc: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, (1 << 16) | 1, 28, 1, 0, 0, 1, 0];
        let mut tex = null;
        let made = crate::resource::create_at_address(
            null,
            0x5_0000,
            desc.as_ptr().cast(),
            0,
            null.cast(),
            &iid,
            &mut tex,
        );
        assert_eq!(made, crate::object::S_OK);
        let list = host.new_list().unwrap();
        // The list starts at RENDER_TARGET and ends at COPY_SOURCE.
        assert!(list.transition(tex, 0x4).is_none());
        let (texture, before) = list.transition(tex, 0x800).unwrap();
        assert_eq!(before, 0x4);
        // SAFETY: an open list and a live texture.
        unsafe {
            list.list.ResourceBarrier(&[super::transition(
                &texture,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES(0x4),
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        list.close();
        let gate = Gate::new(host);
        // COMMON -> RENDER_TARGET before the first run, COPY_SOURCE ->
        // RENDER_TARGET before the second: two different barriers.
        host.execute_gated(&[&list, &list], &gate.0);
        assert!(list.running());
        let mut spares = 0;
        host.timeline.each_work(|w| {
            if w.lists.iter().any(|l| std::ptr::eq(l.0, &list)) {
                spares += w.preludes.len();
            }
        });
        assert_eq!(spares, 1, "the repeat records into its own prelude");
        gate.release();
        host.drain();
        let copy = crate::resource::host_copy(tex).unwrap();
        assert_eq!(copy.state.load(Ordering::Relaxed), 0x800);
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        type Release = extern "system" fn(crate::object::This) -> u32;
        crate::object::slot::<Release>(tex, 0x10)(tex);
    }

    #[test]
    fn a_repeated_list_reads_its_own_upload_memory() {
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (word, later) = (Box::new(0x1111_1111u32), Box::new(0x2222_2222u32));
        let (out, seen) = host.shared_buffer(256).unwrap();
        // SAFETY: our mapped buffer of 256 bytes.
        unsafe { std::ptr::write_bytes(seen, 0, 8) };
        let list = host.new_list().unwrap();
        let (_, to) = list.scratch(4).unwrap();
        list.defer_fill(to, (&raw const *word).cast(), 4);
        let (scratch, offset) = {
            let s = list.scratch.lock().unwrap();
            (s.chunks[0].0.clone(), (to as usize - s.chunks[0].1) as u64)
        };
        // SAFETY: an open list and live buffers.
        unsafe { list.list.CopyBufferRegion(&out, 4, &scratch, offset, 4) };
        list.close();
        // Between the runs the title word changes (a step), and the GPU
        // reads the list's upload memory as the first run left it.
        let between = host.new_list().unwrap();
        between.defer_copy(
            (&raw const *word).cast_mut().cast(),
            (&raw const *later).cast(),
            4,
        );
        // SAFETY: as above.
        unsafe { between.list.CopyBufferRegion(&out, 0, &scratch, offset, 4) };
        between.close();
        host.execute(&[&list, &between, &list]);
        host.drain();
        // SAFETY: our mapped buffer; the GPU is done with it.
        let read = unsafe {
            [
                (seen as *const u32).read_volatile(),
                (seen as *const u32).add(1).read_volatile(),
            ]
        };
        assert_eq!(
            read,
            [0x1111_1111, 0x2222_2222],
            "between the runs, then after the repeat"
        );
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn signals_follow_the_writes_of_earlier_submissions() {
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        let word: &'static mut u64 = Box::leak(Box::new(0));
        let at = (&raw mut *word) as usize;
        list.defer_write(at as *mut u8, Written::Value(3), 8);
        list.close();
        host.execute(&[&list]);
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let s = seen.clone();
        host.after(move || {
            // SAFETY: the leaked word lives on.
            let value = unsafe { std::ptr::read_volatile(at as *const u64) };
            *s.lock().unwrap() = Some(value);
        });
        host.drain();
        assert_eq!(*seen.lock().unwrap(), Some(3));
    }

    #[test]
    fn a_step_reading_what_an_earlier_submission_writes_waits_for_it() {
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (first, second) = (host.new_list().unwrap(), host.new_list().unwrap());
        let word = Box::new(0u64);
        let at = (&raw const *word).cast_mut().cast::<u8>();
        first.defer_write(at, Written::Value(0x1234_5678_9abc), 8);
        first.close();
        // The second submission copies the word at submission: after the
        // first's write, as the GPU would order them.
        let copy = Box::new(0u64);
        second.defer_fill((&raw const *copy).cast_mut().cast(), at, 8);
        second.close();
        let gate = Gate::new(host);
        host.execute_gated(&[&first], &gate.0);
        // Signal the gate from another thread so `second` genuinely waits in
        // `settle` without ever deadlocking parallel host tests.
        let gate_fence = gate.0.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            // SAFETY: our private fence.
            unsafe {
                let _ = gate_fence.Signal(1);
            }
        });
        host.execute(&[&second]);
        releaser.join().unwrap();
        // SAFETY: our word, copied while the second was submitted.
        assert_eq!(unsafe { std::ptr::read_volatile(&*copy) }, 0x1234_5678_9abc);
        second.reset();
        first.reset();
    }

    #[test]
    fn dynamic_descriptor_leases_stay_held_until_async_submission_finishes() {
        use crate::descriptor::{HeapDesc, create_heap, heap};
        use crate::object::{S_OK, This, guid, slot};

        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        #[repr(C)]
        struct ExtendedHeapDesc {
            words: [u32; 4], // kind=0 (views), count=8, flags=1 (shader visible), nodes=0
            addr: u64,
            size: u64,
        }
        let heap_iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let ext = ExtendedHeapDesc {
            words: [0, 8, 1, 0],
            addr: 0,
            size: 0,
        };
        let mut title_heap: This = std::ptr::null();
        assert_eq!(
            create_heap(
                std::ptr::null(),
                std::ptr::from_ref(&ext).cast::<HeapDesc>(),
                &heap_iid,
                &mut title_heap
            ),
            S_OK
        );
        let heap_ref = heap(title_heap);
        let host_heap = heap_ref.host().unwrap();
        // Lease every dynamic slot on `list`, in 2^16 runs (a heap has 2^16
        // or 2^18 of them, depending on what the host takes).
        let list = host.new_list().unwrap();
        while let Some(first) = host_heap.dynamic(1 << 16) {
            list.keep_dynamic(heap_ref.id(), first, 1 << 16);
        }
        assert_eq!(host_heap.dynamic(1), None);
        list.close();

        // Hold the host queue behind a private fence armed atomically inside
        // `execute_gated` so parallel tests cannot deadlock before `release()`.
        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);
        list.reset();
        // Reset while in flight must NOT free the leased dynamic range yet.
        assert_eq!(
            host_heap.dynamic(1),
            None,
            "dynamic descriptor slots must remain leased until the async fence completes"
        );
        gate.release();
        host.drain();
        let after = host_heap.dynamic(1);
        assert!(
            after.is_some(),
            "dynamic descriptor slots must be released once the submission finishes"
        );
        host_heap.release(after.unwrap(), 1);
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(title_heap, 0x10)(title_heap),
            0
        );
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod timing_report {
    use super::{cost_kind, cost_label, group_members, parse_draws, BUSY_END, BUSY_START, MAX_TIMED_COUNT, STAMPS};

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn stamps_leave_room_for_the_busy_pair() {
        assert_eq!(STAMPS, 16384);
        assert_eq!(BUSY_START, STAMPS - 2);
        assert_eq!(BUSY_END, STAMPS - 1);
        assert_eq!(MAX_TIMED_COUNT, STAMPS - 3);
        assert!(MAX_TIMED_COUNT < BUSY_START);
    }

    #[test]
    fn cost_rows_split_kind_then_pipeline() {
        assert_eq!(cost_kind("draw 36x1 [vs ab, ps cd] | u0=0x1"), "draw");
        assert_eq!(cost_kind("indexed 6x2 [vs ab, ps cd]"), "indexed");
        assert_eq!(cost_kind("dispatch 4x2x2 [cs ab]"), "dispatch");
        assert_eq!(cost_kind("mesh 8 [ms ab]"), "mesh");
        assert_eq!(cost_kind("copy resource 0x1 to 0x2"), "copy");
        assert_eq!(cost_kind("DROPPED draw (no host pipeline) [vs ab]"), "DROPPED");
        assert_eq!(cost_kind("something odd here"), "other");
        assert_eq!(
            cost_label("draw 36x1 [vs ab, ps cd] | u0=0x1234"),
            "draw 36x1 [vs ab, ps cd]"
        );
        assert_eq!(cost_label("copy resource 0x1 to 0x2"), "copy resource");
    }

    #[test]
    fn draws_count_instances_and_groups() {
        assert_eq!(parse_draws("draw 36x1 [vs ab]"), 1);
        assert_eq!(parse_draws("indexed 6x3 [vs ab]"), 3);
        assert_eq!(parse_draws("dispatch 4x2x2 [cs ab]"), 16);
        assert_eq!(parse_draws("mesh 8 [ms ab]"), 8);
        assert_eq!(parse_draws("copy resource 0x1 to 0x2"), 1);
        assert_eq!(parse_draws("resolve 0x1 to 0x2 format 1"), 1);
    }

    #[test]
    fn cuts_groups_hold_their_runs() {
        assert_eq!(group_members(&[0, 2], &[0, 4]), vec![vec![0, 1]]);
        assert_eq!(group_members(&[0, 2, 3], &[0, 2, 4]), vec![vec![0], vec![1, 2]]);
        assert_eq!(group_members(&[], &[0]), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn residual_is_busy_minus_noted_clamped() {
        let residual = |busy: f64, noted: f64| (busy - noted).max(0.0);
        assert_eq!(residual(3.0, 2.0), 1.0);
        assert_eq!(residual(2.0, 2.0), 0.0);
        assert_eq!(residual(1.0, 2.0), 0.0);
        assert_eq!(residual(0.0, 0.0), 0.0);
    }
}
