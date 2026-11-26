//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, ResourceType, RootType, UsedResource};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in a folder of `dir` from now on: the folder
/// of this translator, `None` if it cannot be opened.
pub fn use_shader_cache(dir: PathBuf) -> Option<PathBuf> {
    let dir = dir.join(&blake3::Hash::from(*build()).to_hex()[..16]);
    match Cache::open(&dir) {
        Ok(cache) => {
            tracing::info!(dir = %dir.display(), "shader cache");
            let _ = CACHE.set(cache);
            Some(dir)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no shader cache");
            None
        }
    }
}

/// Shaders found in the cache and translated since the start.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// translation settings (the loop limit, which a diagnostics build can
/// change at run time).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        key(victory_shader::SOURCE_HASH, ENCODING, limit)
    })
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(translator: &str, encoding: &[u8], limit: u32) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    *h.finalize().as_bytes()
}

/// A console shader container's translation: the host shader (or why it
/// has none), the console resources its code reads, and whether the
/// translation came from the cache (so no native decode ran for it).
pub struct Cached {
    pub shader: Result<HostShader, String>,
    pub used: Vec<UsedResource>,
    pub hit: bool,
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub fn host_shader(container: &[u8]) -> Cached {
    // Writes through views mark their pages: copy-backs compare only those
    // (`crate::shadow::copy_back_marks`).
    let options = victory_shader::emit::Options { view_marks: true };
    let translate = |c: &[u8]| -> Result<(HostShader, Vec<UsedResource>), String> {
        victory_shader::emit::translate_with_used(c, options).map_err(|e| e.to_string())
    };
    match CACHE.get() {
        Some(cache) => cached(cache, build(), container, translate),
        None => match translate(container) {
            Ok((shader, used)) => Cached { shader: Ok(shader), used, hit: false },
            Err(reason) => Cached { shader: Err(reason), used: Vec::new(), hit: false },
        },
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) with its used resources out. `ran`
/// tells a miss from a hit; `made` keeps a miss's translation, so the
/// entry just written is not read back (the first read of a new file
/// costs about as much as a small translation where a virus scanner
/// looks at it: 0.5-3 ms here).
/// A translation the cache holds for a miss: the host shader (or why it
/// has none) with the console resources its code reads.
type Made = Result<(HostShader, Vec<UsedResource>), String>;

struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Made>>,
}

impl<F: Fn(&[u8]) -> Result<(HostShader, Vec<UsedResource>), String> + Send + Sync> Converter
    for Translate<F>
{
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let (translation, used) = match (self.translate)(&input[32..]) {
            Ok((shader, used)) => (Ok(shader), used),
            Err(reason) => (Err(reason), Vec::new()),
        };
        let bytes = encode(&translation, &used);
        // A miss hands back its translation, not the entry read back.
        let made = match translation {
            Ok(shader) => Ok((shader, used)),
            Err(reason) => Err(reason),
        };
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<(HostShader, Vec<UsedResource>), String> + Send + Sync,
) -> Cached {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(made) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader translated and cached");
            return match made {
                Ok((shader, used)) => Cached { shader: Ok(shader), used, hit: false },
                Err(reason) => Cached { shader: Err(reason), used: Vec::new(), hit: false },
            };
        }
        if let Some((shader, used)) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Cached { shader, used, hit };
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation whose entry could not be written is not made again.
    match made().unwrap_or_else(|| (converter.translate)(container)) {
        Ok((shader, used)) => Cached { shader: Ok(shader), used, hit: false },
        Err(reason) => Cached { shader: Err(reason), used: Vec::new(), hit: false },
    }
}

const MAGIC: &[u8; 4] = b"VSH1";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

fn kind_code(kind: ResourceType) -> u32 {
    match kind {
        ResourceType::Srv => 0,
        ResourceType::Sampler => 1,
        ResourceType::Cbv => 2,
        ResourceType::Uav => 3,
        ResourceType::Other(x) => 4 + u32::from(x),
    }
}

fn kind_type(code: u32) -> Option<ResourceType> {
    match code {
        0 => Some(ResourceType::Srv),
        1 => Some(ResourceType::Sampler),
        2 => Some(ResourceType::Cbv),
        3 => Some(ResourceType::Uav),
        x => Some(ResourceType::Other(u8::try_from(x - 4).ok()?)),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text. The used resources follow the flags word: entries
/// written by an older translator end there and do not decode.
fn encode(translation: &Result<HostShader, String>, used: &[UsedResource]) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    word(&mut out, used.len() as u32);
    for u in used {
        for v in [
            kind_code(u.kind),
            root_code(u.root),
            u.table_offset_dwords,
            u.lower_bound,
            u.stride,
            u32::from(u.dwords),
            u32::from(u.user_data),
            u32::from(u.slot),
        ] {
            word(&mut out, v);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<(Result<HostShader, String>, Vec<UsedResource>)> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some((Err(String::from_utf8(reason.to_vec()).ok()?), Vec::new()));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let memory = r.word()?;
    let n = r.word()? as usize;
    let mut used = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        used.push(UsedResource {
            kind: kind_type(r.word()?)?,
            root: root_type(r.word()?),
            table_offset_dwords: r.word()?,
            lower_bound: r.word()?,
            stride: r.word()?,
            dwords: u8::try_from(r.word()?).ok()?,
            user_data: u8::try_from(r.word()?).ok()?,
            slot: u8::try_from(r.word()?).ok()?,
        });
    }
    r.0.is_empty().then_some((
        Ok(HostShader {
            stage,
            dxil,
            user_data,
            samplers,
            static_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            memory: memory & 1 != 0,
            memory_writes: memory & 2 != 0,
            view_addresses: memory & 4 != 0,
            heap: memory & 8 != 0,
            heap_writes: memory & 16 != 0,
            view_marks: memory & 32 != 0,
        }),
        used,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            view_marks: true,
        }
    }

    fn used() -> Vec<UsedResource> {
        vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 2,
                lower_bound: 5,
                stride: 4,
                dwords: 8,
                user_data: 9,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Other(7),
                root: RootType::Other(11),
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ]
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok((shader(), used()))
        };
        let hit = |cached: Cached| (cached.shader, cached.used, cached.hit);
        assert_eq!(
            hit(cached(&cache, &build, container, translate)),
            (Ok(shader()), used(), false)
        );
        assert_eq!(
            hit(cached(&cache, &build, container, translate)),
            (Ok(shader()), used(), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            hit(cached(&cache, &[8; 32], container, translate)).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            hit(cached(&cache, &build, container, translate)),
            (Ok(shader()), used(), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            hit(cached(&cache, &build, container, translate)),
            (Ok(shader()), used(), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<(HostShader, Vec<UsedResource>), _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), Vec::new(), false);
        assert_eq!(hit(cached(&cache, &build, b"other", fail)), refused);
        assert_eq!(
            hit(cached(&cache, &build, b"other", fail)),
            (refused.0, refused.1, true)
        );
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok((shader(), Vec::new()))
        };
        let got = cached(&cache, &[7; 32], b"a container", translate);
        assert_eq!((got.shader, got.used, got.hit), (Ok(shader()), Vec::new(), false));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit) makes another key, so no entry
    /// written before it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        let k = key("0123456789abcdef", b"encoding", 1 << 20);
        assert_eq!(k, key("0123456789abcdef", b"encoding", 1 << 20));
        assert_ne!(k, key("0123456789abcdee", b"encoding", 1 << 20));
        assert_ne!(k, key("0123456789abcdef", b"encoding.", 1 << 20));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 19));
        assert_eq!(
            *build(),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit()
            )
        );
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>, used: &[UsedResource])")
        );
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()), &used());
        assert_eq!(decode(&bytes), Some((Ok(shader()), used())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        // An entry written before the used resources were kept ends where
        // the flags word does, and is translated again.
        let legacy = encode_legacy(&Ok(shader()));
        assert_eq!(decode(&legacy), None);
        // A used entry of unknown kind is damage, not a silent default.
        let mut bad_kind = bytes.clone();
        let at = bad_kind.len() - used().len() * 32;
        bad_kind[at..at + 4].copy_from_slice(&300u32.to_le_bytes());
        assert_eq!(decode(&bad_kind), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal, &[])), Some((refusal, Vec::new())));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }

    /// The entry layout before the used resources were kept (for the
    /// upgrade test above): the flags word ends the entry.
    fn encode_legacy(translation: &Result<HostShader, String>) -> Vec<u8> {
        let bytes = encode(translation, &used());
        let used_words = used().len() * 8 * 4 + 4;
        bytes[..bytes.len() - used_words].to_vec()
    }
}
