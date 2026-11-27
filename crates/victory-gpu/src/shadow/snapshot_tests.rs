use super::*;

/// Parts kept of a copy, against a byte-by-byte model.
#[test]
fn snapshots_keep_each_declared_byte_once() {
    let _gpu = crate::host::gpu_test_lock();
    const LEN: usize = 300;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for _ in 0..2000 {
        let (mut parts, mut bytes, mut model) = (Vec::<Part>::new(), Vec::new(), [false; LEN]);
        for _ in 0..random(8) {
            let from = random(LEN);
            let to = from + random(LEN - from + 1);
            for (from, to) in uncovered(&parts, from, to) {
                let at = bytes.len();
                bytes.extend((from..to).map(|i| i as u8));
                parts.push(Part { from, to, at });
            }
            parts.sort_unstable_by_key(|p| p.from);
            model[from..to].fill(true);
        }
        assert_eq!(bytes.len(), model.iter().filter(|&&m| m).count());
        let mut room = Vec::new();
        for _ in 0..20 {
            let from = random(LEN);
            let to = from + 1 + random(LEN - from);
            let want: Option<Vec<u8>> = (from..to).map(|i| model[i].then_some(i as u8)).collect();
            assert_eq!(kept(&parts, &bytes, (from, to), &mut room), want.as_deref());
        }
    }
}

/// A keep of megabytes fills its slots on threads: what the GPU wrote
/// still goes back exactly, and the block is empty with the last range.
#[test]
fn big_snapshot_fills_in_parallel() {
    let _gpu = crate::host::gpu_test_lock();
    let len: usize = 32 * PAGE;
    let mut title = vec![0x44u8; len + PAGE];
    let start = (title.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let Some((buffer, data)) = host().and_then(|h| h.shared_buffer(len as u64)) else {
        return; // no host GPU here
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, start as u64, len as u64, None);
    assert_eq!(shadow.update(start, len), len);
    assert!(len >= 1 << 20, "the keep takes the threaded path");
    shadow.writing(start, len, false);
    // The shaders write across the whole keep, and the title meanwhile.
    // SAFETY: inside the mapping and the title's memory.
    unsafe {
        data.add(100).write(0x77);
        data.add(len - 1).write(0x78);
        ((start + 200) as *mut u8).write(0x79);
    }
    shadow.copy_back(start, len);
    // SAFETY: the title's memory.
    unsafe {
        assert_eq!((start as *const u8).read_volatile(), 0x44);
        assert_eq!(((start + 100) as *const u8).read_volatile(), 0x77);
        assert_eq!(((start + len - 1) as *const u8).read_volatile(), 0x78);
        assert_eq!(((start + 200) as *const u8).read_volatile(), 0x79);
    }
    assert_eq!(all_kept(), (0, 0));
}

/// Throughput of the snapshot fill patterns on this machine (serial
/// extend vs reserve + parallel raw fill), to tell a slow mechanism from
/// a heavy scene. Run with
/// `cargo test --release -p victory-gpu -- --ignored --nocapture snapshot_fill_throughput`.
#[test]
#[ignore = "benchmark"]
fn snapshot_fill_throughput() {
    const N: usize = 512 << 20;
    const CHUNK: usize = 4 << 20;
    let src = vec![0x5Au8; N];
    let gbs = |n: usize, t: std::time::Instant| n as f64 / t.elapsed().as_secs_f64() / 1e9;
    // The old path: extend per gap under the lock.
    let mut dst = Vec::new();
    let t = std::time::Instant::now();
    for c in src.chunks(CHUNK) {
        dst.extend_from_slice(c);
    }
    println!("serial extend: {:.1} GB/s", gbs(N, t));
    std::hint::black_box(&dst);
    // The new path: reserve once, fill disjoint slots on threads (the
    // same pattern production uses, lint allowances and all).
    let mut dst: Vec<u8> = Vec::with_capacity(N);
    // SAFETY: every byte is filled below before any read.
    #[allow(clippy::uninit_vec)]
    unsafe {
        dst.set_len(N)
    };
    let base = dst.as_ptr() as usize;
    let t = std::time::Instant::now();
    let threads = snapshot_threads(N);
    println!("threads: {threads}");
    std::thread::scope(|s| {
        let quota = N.div_ceil(threads);
        let mut handles = Vec::new();
        for i in 1..threads {
            let (from, n) = (i * quota, quota.min(N.saturating_sub(i * quota)));
            if n == 0 {
                continue;
            }
            let src = &src;
            handles.push(s.spawn(move || {
                // SAFETY: disjoint slice of the reservation.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr().add(from), (base + from) as *mut u8, n);
                }
            }));
        }
        let n = quota.min(N);
        // SAFETY: the first chunk of the reservation, filled here while
        // workers fill the rest.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), base as *mut u8, n);
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    println!("parallel fill: {:.1} GB/s", gbs(N, t));
    assert!(dst.iter().all(|&b| b == 0x5Au8));
}
/// changed go back, the title's own concurrent writes stay, and what
/// changed outside every declared range does not go back.
#[test]
fn copies_back_what_changed_in_declared_ranges() {
    let _gpu = crate::host::gpu_test_lock();
    const LEN: usize = 8 * PAGE;
    let mut title = vec![0x11u8; LEN + PAGE];
    let start = (title.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let Some((buffer, data)) = host().and_then(|h| h.shared_buffer(LEN as u64)) else {
        return; // no host GPU here
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, start as u64, LEN as u64, None);
    let ranges = [(100, 2 * PAGE), (PAGE, 3 * PAGE), (6 * PAGE + 8, 64)];
    assert_eq!(shadow.update(start, LEN), LEN);
    for &(at, len) in &ranges {
        shadow.writing(start + at, len, false);
    }
    let written = [150, PAGE + 5, 2 * PAGE, 4 * PAGE - 1, 6 * PAGE + 8];
    // SAFETY: inside the mapping and the title's memory.
    unsafe {
        for at in written.into_iter().chain([5 * PAGE]) {
            data.add(at).write(0x22);
        }
        ((start + 151) as *mut u8).write(0x33);
    }
    for &(at, len) in &ranges {
        shadow.copy_back(start + at, len);
    }
    for i in 0..LEN {
        // SAFETY: the title's memory.
        let byte = unsafe { ((start + i) as *const u8).read_volatile() };
        let want = match i {
            151 => 0x33,
            i if written.contains(&i) => 0x22,
            _ => 0x11,
        };
        assert_eq!(byte, want, "at {i:#x}");
    }
    assert!(shadow.before.lock().unwrap().0.is_empty());
    assert!(shadow.pending.lock().unwrap().is_empty());
    drop(title);
}

/// Two pages of title memory and a copy of it.
fn copy_of(title: &[u8]) -> Option<Arc<Shadow>> {
    let (buffer, data) = host()?.shared_buffer(title.len() as u64)?;
    Some(Shadow::new(
        buffer,
        data,
        0,
        title.as_ptr() as u64,
        title.len() as u64,
        None,
    ))
}

/// How many parts and bytes are kept.
fn all_kept() -> (usize, usize) {
    let s = snapshots();
    (s.parts, s.bytes.len())
}

/// A copy dropped with ranges never copied back gives its snapshots up.
#[test]
fn a_dropped_copy_leaves_no_snapshot() {
    let _gpu = crate::host::gpu_test_lock();
    let title = vec![0x11u8; 2 * PAGE];
    let Some(a) = copy_of(&title) else {
        return; // no host GPU here
    };
    a.writing(title.as_ptr() as usize, 100, false);
    a.writing(title.as_ptr() as usize + PAGE, 300, false);
    assert_eq!(all_kept(), (2, 400));
    drop(a);
    assert_eq!(all_kept(), (0, 0));
}

/// Two copies' snapshots in one submission, taken in turns: each copies
/// back its own bytes, also after the other is gone.
#[test]
fn two_copies_keep_their_snapshots_apart() {
    let _gpu = crate::host::gpu_test_lock();
    let (first, second) = (vec![0x11u8; 2 * PAGE], vec![0x22u8; 2 * PAGE]);
    let (Some(a), Some(b)) = (copy_of(&first), copy_of(&second)) else {
        return; // no host GPU here
    };
    let (at, bt) = (first.as_ptr() as usize, second.as_ptr() as usize);
    assert_eq!(a.update(at, 2 * PAGE) + b.update(bt, 2 * PAGE), 4 * PAGE);
    a.writing(at, 100, false);
    b.writing(bt, 200, false);
    a.writing(at + PAGE, 300, false);
    b.writing(bt + PAGE, 400, false);
    assert_eq!(all_kept(), (4, 1000));
    // SAFETY: inside both mappings: what shaders wrote, one byte undeclared.
    unsafe {
        a.data.add(50).write(7);
        a.data.add(101).write(7);
        a.data.add(PAGE + 299).write(7);
        b.data.add(199).write(8);
        b.data.add(PAGE).write(8);
    }
    a.copy_back(at, 100);
    b.copy_back(bt, 200);
    a.copy_back(at + PAGE, 300);
    // `a` is done; `b`'s second range still has its bytes.
    assert_eq!(all_kept().0, 2);
    drop(a);
    assert_eq!(all_kept().0, 2);
    b.copy_back(bt + PAGE, 400);
    assert_eq!(all_kept(), (0, 0));
    let changed = |title: &[u8], was: u8| -> Vec<(usize, u8)> {
        let now = title.iter().enumerate();
        now.filter(|(_, v)| **v != was)
            .map(|(i, v)| (i, *v))
            .collect()
    };
    assert_eq!(
        changed(std::hint::black_box(&first), 0x11),
        [(50, 7), (PAGE + 299, 7)]
    );
    assert_eq!(
        changed(std::hint::black_box(&second), 0x22),
        [(199, 8), (PAGE, 8)]
    );
}
