//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//! Pipelines a draw waits for go ahead of those started early (at the
//! title's creation, because an earlier run drew them).
//!
//! Driver compiles past 256 KiB of containers run one at a time: several at
//! once exhaust the commit charge and contend inside the driver (each takes
//! seconds), stalling every worker behind them.
//!
//! shortcut: the budget (half a second of waiting per title frame) is a
//! guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    /// Whether it went into the queue of pipelines a draw waits for, and
    /// whether a worker took it (it may be in both queues).
    now: AtomicBool,
    taken: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// Its worker was lost to it: it will not be made.
    lost: AtomicBool,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log("created", &key, || {
            let state = graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
            format!("{shaders:?}\t{state}")
        });
        Pending {
            key,
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            now: AtomicBool::new(false),
            taken: AtomicBool::new(false),
            made: OnceLock::new(),
            lost: AtomicBool::new(false),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        // Not `get_or_init`: a worker lost in it would leave it locked.
        let (started, cpu) = (Instant::now(), super::thread_cpu());
        // Huge pipelines compile one at a time (see `HugeGuard`).
        let _huge = HugeGuard::take_if(self.is_huge());
        let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log_built(&self.key);
        let (user, kernel) = super::thread_cpu_since(cpu);
        tracing::debug!(
            shaders = ?self.shaders,
            bytes = self.bytes(),
            ms = started.elapsed().as_millis() as u64,
            user_ms = user,
            kernel_ms = kernel,
            "pipeline made"
        );
        let _ = self.made.set(made);
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }

    fn is_huge(&self) -> bool {
        huge(self.shaders.iter().map(|s| s.bytes()).sum())
    }

    /// The pipeline's containers in bytes (what the driver compiles).
    fn bytes(&self) -> usize {
        self.shaders.iter().map(|s| s.bytes()).sum()
    }
}

/// Driver compiles past [`HUGE_BYTES`] of containers run one at a time:
/// several at once exhaust the commit charge and contend inside the driver
/// (each takes seconds), stalling every worker behind them. The 256 KiB
/// threshold is xcl's default huge-PSO size.
const HUGE_BYTES: usize = 256 * 1024;

/// Whether a byte count is a huge driver compile ([`HUGE_BYTES`]).
fn huge(bytes: usize) -> bool {
    bytes >= HUGE_BYTES
}

/// At most one huge driver compile at a time (the others wait here).
static HUGE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// The huge-compile slot, held while a huge pipeline is made.
struct HugeGuard {
    held: bool,
}

impl HugeGuard {
    /// Takes the slot for a huge pipeline now (waiting for it first);
    /// nothing for an ordinary one.
    fn take_if(huge: bool) -> Self {
        if !huge {
            return HugeGuard { held: false };
        }
        let (lock, gate) = &HUGE;
        let mut busy = lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut waited_ms = 0;
        while *busy {
            let started = Instant::now();
            busy = gate.wait(busy).unwrap_or_else(|e| e.into_inner());
            waited_ms += started.elapsed().as_millis() as u64;
        }
        *busy = true;
        if waited_ms > 0 {
            tracing::info!(waited_ms, "huge pipeline waited for the compile slot");
        }
        HugeGuard { held: true }
    }
}

impl Drop for HugeGuard {
    fn drop(&mut self) {
        if self.held {
            let (lock, gate) = &HUGE;
            *lock.lock().unwrap_or_else(|e| e.into_inner()) = false;
            gate.notify_one();
        }
    }
}

/// What identifies a title pipeline across runs: its shader containers
/// and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Logs the library name of the host pipeline this thread just made or
/// loaded for `key` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_built(key: &str) {
    if let Some((name, loaded)) = super::library::LAST.take() {
        log("built", key, || format!("{name}\t{}", if loaded { "loaded" } else { "made" }));
    }
}

/// `VICTORY_PIPELINE_LOG=<file>` (diagnostics): a line per title pipeline
/// key created (`created`, its shaders, its state as described) and first
/// drawn or dispatched with (`used`), and its host pipeline's library name
/// once made or loaded (`built`, also from the pre-build), to compare what
/// a run creates with what the pre-build makes.
#[cfg(feature = "diagnostics")]
fn log(what: &str, key: &str, rest: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::io::Write;
    static LOG: Mutex<Option<(std::fs::File, HashSet<String>)>> = Mutex::new(None);
    static PATH: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_LOG")) else {
        return;
    };
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.is_none() {
        *log = std::fs::File::create(path).ok().map(|f| (f, HashSet::new()));
    }
    let Some((file, seen)) = log.as_mut() else { return };
    if seen.insert(format!("{what} {key}")) {
        let _ = writeln!(file, "{what}\t{key}\t{}", rest());
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(500);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // Listed for an early start in later runs from its first use, made or
    // not: one a run never got made is the one it should start early.
    if !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        #[cfg(feature = "diagnostics")]
        log("used", &p.key, String::new);
    }
    if p.made.get().is_none() {
        start(p, true);
        if p.lost.load(Ordering::Relaxed) {
            return None;
        }
        // With a world's worth queued there is no use in waiting.
        if waiting() < 64 {
            wait(p, crate::frame::current());
        }
    }
    p.made.get()?.as_ref()
}

/// Queues `p` for the workers: once, and once more ahead of the early
/// starts if a draw waits for it (`now`) while it is still queued.
fn start(p: &Arc<Pending>, now: bool) {
    if let Some(now) = enqueue(p, now) {
        queue(p.clone(), now);
    }
}

/// Which queue `p` goes into now (true: the one draws wait for), if any.
fn enqueue(p: &Pending, now: bool) -> Option<bool> {
    let first = !p.queued.swap(true, Ordering::Relaxed);
    if first {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let ahead = now && !p.now.swap(true, Ordering::Relaxed);
    (first || ahead && !p.taken.load(Ordering::Relaxed)).then_some(ahead)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    super::library::note_created(&p.key);
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let left = BUDGET.saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Pipelines for the workers: those draws wait for, then those started
/// early.
static QUEUES: (Mutex<[VecDeque<Arc<Pending>>; 2]>, Condvar) =
    (Mutex::new([VecDeque::new(), VecDeque::new()]), Condvar::new());

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>, now: bool) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        (0..n).filter(|i| worker(*i)).count()
    });
    if workers == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            p.make();
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    let (lock, work) = &QUEUES;
    lock.lock().unwrap_or_else(|e| e.into_inner())[usize::from(!now)].push_back(p);
    work.notify_one();
}

/// What the workers (by thread) are making.
static AT_WORK: Mutex<Vec<(u32, Arc<Pending>)>> = Mutex::new(Vec::new());

/// Starts worker `i`; true if it runs.
fn worker(i: usize) -> bool {
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{i}"))
        .spawn(work);
    if let Err(e) = &spawned {
        tracing::warn!(error = %e, "no pipeline worker");
    }
    spawned.is_ok()
}

/// A worker is lost to a compile that crashed or never ends: another takes
/// its place (its pipeline stays unmade). With all of them lost, the rest of
/// a world's pipelines waited forever.
#[allow(dead_code)] // No caller yet: the library.rs half is excluded.
pub(super) fn worker_lost(id: u32) {
    static LOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
    let mut lost = false;
    AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| {
        if w.0 == id {
            w.1.lost.store(true, Ordering::Relaxed);
            lost = true;
        }
        w.0 != id
    });
    if lost {
        // A lost worker never comes back from its compile (it sleeps in the
        // crash handler or stays stopped), so it never reaches its own
        // count below: it is accounted for here instead.
        WAITING.fetch_sub(1, Ordering::Relaxed);
        worker(LOST.fetch_add(1, Ordering::Relaxed));
    }
}

/// A worker: makes queued pipelines, those draws wait for first, and saves
/// the cache when idle (first uses of pipelines made before).
fn work() {
    // Below the title's own threads (loading, recording).
    // SAFETY: plain calls on our own thread.
    let id = unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        GetCurrentThreadId()
    };
    let (lock, work) = &QUEUES;
    loop {
        let p = {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(p) = queues.iter_mut().find_map(VecDeque::pop_front) {
                    break p;
                }
                let (held, waited) = work
                    .wait_timeout(queues, Duration::from_secs(20))
                    .unwrap_or_else(|e| e.into_inner());
                queues = held;
                if waited.timed_out() {
                    drop(queues);
                    super::library::save(false);
                    queues = lock.lock().unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        if p.taken.swap(true, Ordering::Relaxed) {
            continue; // its other queue entry
        }
        AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).push((id, p.clone()));
        p.make();
        AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != id);
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An early start is queued once; a draw waiting for it puts it ahead,
    /// once, unless a worker has taken it.
    #[test]
    fn draws_go_ahead_of_early_starts() {
        let p = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&p, false), Some(false));
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue(&p, true), Some(true));
        assert_eq!(enqueue(&p, true), None);
        let q = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&q, false), Some(false));
        q.taken.store(true, Ordering::Relaxed);
        assert_eq!(enqueue(&q, true), None);
        let r = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&r, true), Some(true));
        assert_eq!(enqueue(&r, false), None);
        // Counted as waiting by `enqueue`, never handed to a worker.
        WAITING.fetch_sub(3, Ordering::Relaxed);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// Huge driver compiles wait for the slot; ordinary ones never do.
    #[test]
    fn huge_compiles_wait_for_the_slot() {
        assert!(!huge(0));
        assert!(!huge(HUGE_BYTES - 1));
        assert!(huge(HUGE_BYTES));
        let _first = HugeGuard::take_if(true);
        let waited = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let probe = waited.clone();
        let other = std::thread::spawn(move || {
            let started = Instant::now();
            let _second = HugeGuard::take_if(true);
            probe.store(started.elapsed().as_millis() as u64, Ordering::Relaxed);
        });
        std::thread::sleep(Duration::from_millis(150));
        drop(_first);
        other.join().unwrap();
        assert!(
            waited.load(Ordering::Relaxed) >= 100,
            "second huge compile did not wait"
        );
        let started = Instant::now();
        let _plain = HugeGuard::take_if(false);
        assert!(started.elapsed() < Duration::from_millis(100));
    }
}
