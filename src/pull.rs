//! Model download from Hugging Face: resumable, priority-ordered, hash-verified, and usable
//! while still downloading (lazy mode).
//!
//! The file is written in place under its final name with a sidecar
//! `<file>.banach-incomplete` that lists the byte ranges already on disk, one "start end"
//! pair per line. Order: GGUF header, then every non-expert tensor (the dense path must be
//! complete before the engine starts), then routed experts, hottest first when a usage file is
//! given, then whatever is left. `ensure()` moves a range to the front of the queue and
//! blocks until it is written; the engine's lazy mode calls it for experts that are not on
//! disk yet. When everything is present the SHA-256 is checked against the LFS object id and
//! the sidecar is removed.

use crate::gguf::{self, Gguf};
use crate::source::{ByteSource, HttpRange};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

const CHUNK: u64 = 16 << 20;

/// Byte ranges [start, end) kept merged and sorted.
#[derive(Debug, Default, Clone)]
pub struct Ranges {
    r: BTreeMap<u64, u64>,
}

impl Ranges {
    pub fn add(&mut self, mut s: u64, mut e: u64) {
        if e <= s {
            return;
        }
        // merge with every overlapping or touching range
        let overlapping: Vec<(u64, u64)> = self
            .r
            .range(..=e)
            .filter(|(&a, &b)| b >= s && a <= e)
            .map(|(&a, &b)| (a, b))
            .collect();
        for (a, b) in overlapping {
            self.r.remove(&a);
            s = s.min(a);
            e = e.max(b);
        }
        self.r.insert(s, e);
    }
    pub fn contains(&self, s: u64, e: u64) -> bool {
        if e <= s {
            return true;
        }
        match self.r.range(..=s).next_back() {
            Some((_, &b)) => b >= e,
            None => false,
        }
    }
    /// Parts of [s, e) not yet covered.
    pub fn missing(&self, s: u64, e: u64) -> Vec<(u64, u64)> {
        let mut out = vec![];
        let mut cur = s;
        for (&a, &b) in self.r.range(..e) {
            if b <= cur {
                continue;
            }
            if a > cur {
                out.push((cur, a.min(e)));
            }
            cur = cur.max(b);
            if cur >= e {
                break;
            }
        }
        if cur < e {
            out.push((cur, e));
        }
        out
    }
    pub fn total(&self) -> u64 {
        self.r.iter().map(|(a, b)| b - a).sum()
    }
}

#[derive(Debug, Clone)]
pub struct PartInfo {
    pub url: String,
    pub path: PathBuf,
    pub size: u64,
    pub sha256: Option<String>,
    pub header: Gguf,
}

struct Shared {
    present: Vec<Ranges>,
    urgent: VecDeque<(usize, u64, u64)>,
    queue: VecDeque<(usize, u64, u64)>,
    in_flight: Vec<(usize, u64, u64)>,
    error: Option<String>,
    done: bool,
}

pub struct Puller {
    pub parts: Vec<PartInfo>,
    state: Mutex<Shared>,
    cv: Condvar,
    sidecars: Vec<Mutex<File>>,
}

fn sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".banach-incomplete");
    PathBuf::from(s)
}

/// Is a local model file complete (present and no sidecar)?
pub fn is_complete(path: &Path) -> bool {
    path.exists() && !sidecar(path).exists()
}

/// Owner and repo names are single plain path segments.
fn valid_name(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// A file path inside a repo, or a revision such as `refs/pr/1`: `/`-separated segments, none
/// empty, `.` or `..`, and no characters that could leave the download directory.
fn valid_subpath(s: &str) -> bool {
    !s.is_empty()
        && s.split('/').all(|seg| {
            !seg.is_empty()
                && seg != "."
                && seg != ".."
                && seg.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'+' | b'=' | b'~' | b'@' | b' '))
        })
}

fn hf_parts(spec: &str) -> Result<Vec<(String, String, String, String)>, String> {
    // -> (owner, repo, rev, file path in repo) per split part
    let mut out = vec![];
    for part in crate::source::split_parts(spec) {
        let rest = part.strip_prefix("hf:").ok_or("pull needs an hf:owner/repo/file.gguf model")?;
        let (path, rev) = match rest.rsplit_once('@') {
            Some((p, r)) => (p, r.to_string()),
            None => (rest, "main".to_string()),
        };
        let mut it = path.splitn(3, '/');
        let (o, r, f) = (it.next(), it.next(), it.next());
        match (o, r, f) {
            (Some(o), Some(r), Some(f)) if valid_name(o) && valid_name(r) && valid_subpath(f) && valid_subpath(&rev) => {
                out.push((o.to_string(), r.to_string(), rev, f.to_string()))
            }
            _ => return Err(format!("bad hf spec {part:?}: owner, repo, file and revision must be plain names without `..`, backslashes or absolute paths")),
        }
    }
    Ok(out)
}

fn lfs_sha256(owner: &str, repo: &str, rev: &str, file: &str) -> Option<String> {
    let dir = file.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    let url = format!("https://huggingface.co/api/models/{owner}/{repo}/tree/{rev}/{dir}");
    if !crate::source::host_allowed(&url) {
        return None;
    }
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
    let body = agent.get(&url).call().ok()?.into_string().ok()?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    v.as_array()?
        .iter()
        .find(|e| e["path"].as_str() == Some(file))
        .and_then(|e| e["lfs"]["oid"].as_str().map(|s| s.to_string()))
}

fn is_routed(name: &str) -> bool {
    name.contains("_exps.") && !name.contains("_shexp")
}

impl Puller {
    /// Prepare a download of `spec` into `out_dir`. Reads only the headers.
    pub fn new(spec: &str, out_dir: &Path, usage: Option<&[Vec<f64>]>) -> Result<Arc<Self>, String> {
        let mut parts = vec![];
        for (i, (o, r, rev, f)) in hf_parts(spec)?.into_iter().enumerate() {
            let url = format!("https://huggingface.co/{o}/{r}/resolve/{rev}/{f}");
            let mut src = HttpRange::new(&url)?;
            let header = gguf::parse(&mut src, i).map_err(|e| format!("{url}: {e}"))?;
            let size = src.size()?;
            let path = out_dir.join(&o).join(&r).join(&f);
            parts.push(PartInfo { url, path, size, sha256: lfs_sha256(&o, &r, &rev, &f), header });
        }
        let mut present = vec![];
        let mut sidecars = vec![];
        for p in &parts {
            std::fs::create_dir_all(p.path.parent().unwrap()).map_err(|e| e.to_string())?;
            let complete = is_complete(&p.path) && std::fs::metadata(&p.path).map(|m| m.len()).unwrap_or(0) == p.size;
            let mut rg = Ranges::default();
            if complete {
                rg.add(0, p.size);
            } else {
                let f = OpenOptions::new().create(true).write(true).truncate(false).open(&p.path).map_err(|e| e.to_string())?;
                f.set_len(p.size).map_err(|e| e.to_string())?;
                if let Ok(text) = std::fs::read_to_string(sidecar(&p.path)) {
                    for line in text.lines() {
                        let mut it = line.split_whitespace();
                        if let (Some(a), Some(b)) = (it.next(), it.next()) {
                            if let (Ok(a), Ok(b)) = (a.parse::<u64>(), b.parse::<u64>()) {
                                rg.add(a, b.min(p.size));
                            }
                        }
                    }
                }
            }
            let sc = OpenOptions::new().create(true).append(true).open(sidecar(&p.path)).map_err(|e| e.to_string())?;
            sidecars.push(Mutex::new(sc));
            present.push(rg);
        }
        // priority order
        let mut order: Vec<(usize, u64, u64)> = vec![];
        for (si, p) in parts.iter().enumerate() {
            order.push((si, 0, p.header.data_offset.min(p.size)));
        }
        for (si, p) in parts.iter().enumerate() {
            for t in p.header.tensors.iter().filter(|t| !is_routed(&t.name)) {
                let s = p.header.data_offset + t.offset;
                order.push((si, s, s + t.nbytes));
            }
        }
        // routed experts: per (layer, expert) slices, hottest first if usage is known
        let mut experts: Vec<(f64, usize, u64, u64)> = vec![];
        for (si, p) in parts.iter().enumerate() {
            for t in p.header.tensors.iter().filter(|t| is_routed(&t.name)) {
                let n_e = *t.dims.last().unwrap_or(&1);
                let per = t.nbytes / n_e.max(1);
                let layer: usize = t.name.split('.').nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
                for e in 0..n_e {
                    let heat = usage
                        .and_then(|u| u.get(layer))
                        .and_then(|row| row.get(e as usize))
                        .copied()
                        .unwrap_or(0.0);
                    let s = p.header.data_offset + t.offset + e * per;
                    experts.push((heat, si, s, s + per));
                }
            }
        }
        experts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        order.extend(experts.into_iter().map(|(_, si, s, e)| (si, s, e)));
        for (si, p) in parts.iter().enumerate() {
            order.push((si, 0, p.size));
        }
        let mut queue = VecDeque::new();
        for (si, s, e) in order {
            let mut a = s;
            while a < e {
                let b = (a + CHUNK).min(e);
                queue.push_back((si, a, b));
                a = b;
            }
        }
        Ok(Arc::new(Self {
            parts,
            state: Mutex::new(Shared { present, urgent: VecDeque::new(), queue, in_flight: vec![], error: None, done: false }),
            cv: Condvar::new(),
            sidecars,
        }))
    }

    pub fn total_bytes(&self) -> u64 {
        self.parts.iter().map(|p| p.size).sum()
    }

    pub fn present_bytes(&self) -> u64 {
        self.state.lock().unwrap().present.iter().map(|r| r.total()).sum()
    }

    /// Bytes needed before the engine can start: headers and every non-expert tensor.
    pub fn dense_ready(&self) -> bool {
        let st = self.state.lock().unwrap();
        self.parts.iter().enumerate().all(|(si, p)| {
            st.present[si].contains(0, p.header.data_offset)
                && p.header.tensors.iter().filter(|t| !is_routed(&t.name)).all(|t| {
                    let s = p.header.data_offset + t.offset;
                    st.present[si].contains(s, s + t.nbytes)
                })
        })
    }

    pub fn complete(&self) -> bool {
        let st = self.state.lock().unwrap();
        self.parts.iter().enumerate().all(|(si, p)| st.present[si].contains(0, p.size))
    }

    fn next_job(&self) -> Option<(usize, u64, u64)> {
        let mut st = self.state.lock().unwrap();
        let mut busy_seen = 0usize;
        loop {
            if st.error.is_some() || st.done {
                return None;
            }
            let (job, urgent) = match st.urgent.pop_front() {
                Some(j) => (Some(j), true),
                None => (st.queue.pop_front(), false),
            };
            let Some((si, s, e)) = job else {
                if st.in_flight.is_empty() {
                    st.done = true;
                    self.cv.notify_all();
                    return None;
                }
                st = self.cv.wait(st).unwrap();
                continue;
            };
            let missing = st.present[si].missing(s, e);
            let Some(&(a, b)) = missing.first() else { continue };
            if st.in_flight.iter().any(|&(fs, x, y)| fs == si && x < b && a < y) {
                // an overlapping range is being fetched: put it back where it came from
                if urgent { st.urgent.push_back((si, s, e)) } else { st.queue.push_back((si, s, e)) }
                busy_seen += 1;
                if busy_seen > st.urgent.len() + st.queue.len() {
                    st = self.cv.wait(st).unwrap();
                    busy_seen = 0;
                }
                continue;
            }
            st.in_flight.push((si, a, b));
            if b < e {
                if urgent { st.urgent.push_front((si, b, e)) } else { st.queue.push_front((si, b, e)) }
            }
            return Some((si, a, b));
        }
    }

    fn fetch(&self, src: &mut HttpRange, file: &File, si: usize, s: u64, e: u64) -> Result<(), String> {
        let mut last = String::new();
        for attempt in 0..4 {
            match src.read_at(s, (e - s) as usize) {
                Ok(buf) if buf.len() as u64 == e - s => {
                    write_all_at(file, &buf, s).map_err(|er| er.to_string())?;
                    let mut sc = self.sidecars[si].lock().unwrap();
                    writeln!(sc, "{s} {e}").map_err(|er| er.to_string())?;
                    let mut st = self.state.lock().unwrap();
                    st.present[si].add(s, e);
                    st.in_flight.retain(|&j| j != (si, s, e));
                    self.cv.notify_all();
                    return Ok(());
                }
                Ok(buf) => last = format!("short read: {} of {} bytes", buf.len(), e - s),
                Err(er) => last = er,
            }
            std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
        }
        Err(format!("{} [{s}, {e}): {last}", self.parts[si].url))
    }

    /// Run `workers` download threads until everything is present (or an error).
    pub fn spawn(self: &Arc<Self>, workers: usize) -> Vec<std::thread::JoinHandle<()>> {
        (0..workers.max(1))
            .map(|_| {
                let me = Arc::clone(self);
                std::thread::spawn(move || {
                    let mut srcs: Vec<Option<HttpRange>> = me.parts.iter().map(|_| None).collect();
                    let files: Vec<File> = match me
                        .parts
                        .iter()
                        .map(|p| OpenOptions::new().write(true).open(&p.path))
                        .collect::<Result<Vec<_>, _>>()
                    {
                        Ok(f) => f,
                        Err(e) => {
                            me.fail(e.to_string());
                            return;
                        }
                    };
                    while let Some((si, s, e)) = me.next_job() {
                        if srcs[si].is_none() {
                            match HttpRange::new(&me.parts[si].url) {
                                Ok(h) => srcs[si] = Some(h),
                                Err(er) => {
                                    me.fail(er);
                                    return;
                                }
                            }
                        }
                        if let Err(er) = me.fetch(srcs[si].as_mut().unwrap(), &files[si], si, s, e) {
                            me.fail(er);
                            return;
                        }
                    }
                })
            })
            .collect()
    }

    fn fail(&self, e: String) {
        let mut st = self.state.lock().unwrap();
        if st.error.is_none() {
            st.error = Some(e);
        }
        self.cv.notify_all();
    }

    pub fn error(&self) -> Option<String> {
        self.state.lock().unwrap().error.clone()
    }

    /// Block until [s, e) of split `si` is on disk, fetching it ahead of everything else.
    pub fn ensure(&self, si: usize, s: u64, e: u64) -> Result<(), String> {
        let mut st = self.state.lock().unwrap();
        if si >= st.present.len() {
            return Err(format!("no split {si}"));
        }
        if st.present[si].contains(s, e) {
            return Ok(());
        }
        let mut a = s;
        while a < e {
            let b = (a + CHUNK).min(e);
            st.urgent.push_back((si, a, b));
            a = b;
        }
        self.cv.notify_all();
        while !st.present[si].contains(s, e) {
            if let Some(er) = &st.error {
                return Err(er.clone());
            }
            st = self.cv.wait(st).unwrap();
        }
        Ok(())
    }

    /// When complete: check SHA-256 against the LFS object id and drop the sidecars.
    pub fn finish(&self) -> Result<Vec<PathBuf>, String> {
        if !self.complete() {
            return Err("download incomplete".into());
        }
        for p in &self.parts {
            if let Some(want) = &p.sha256 {
                let got = sha256_file(&p.path)?;
                if &got != want {
                    return Err(format!("{}: SHA-256 {got} does not match the LFS object id {want}", p.path.display()));
                }
            }
            let _ = std::fs::remove_file(sidecar(&p.path));
        }
        Ok(self.parts.iter().map(|p| p.path.clone()).collect())
    }
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_and_query() {
        let mut r = Ranges::default();
        r.add(10, 20);
        r.add(30, 40);
        assert!(r.contains(12, 18));
        assert!(!r.contains(15, 35));
        assert_eq!(r.missing(0, 50), vec![(0, 10), (20, 30), (40, 50)]);
        r.add(20, 30);
        assert!(r.contains(10, 40));
        assert_eq!(r.total(), 30);
        r.add(5, 12);
        assert_eq!(r.missing(0, 50), vec![(0, 5), (40, 50)]);
        assert_eq!(r.missing(12, 38), vec![]);
    }

    #[test]
    fn sha256_known_value() {
        let dir = std::env::temp_dir().join(format!("banach-sha-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hf_specs_cannot_leave_the_download_directory() {
        assert!(hf_parts("hf:unsloth/Kimi-K2-Instruct-GGUF/UD-Q2_K_XL/Kimi-K2-Instruct-UD-Q2_K_XL-00001-of-00008.gguf").is_ok());
        assert!(hf_parts("hf:o/r/dir/file.gguf@refs/pr/1").is_ok());
        for bad in [
            "hf:o/r/../../../etc/x.gguf",
            "hf:../r/f.gguf",
            "hf:o/../f.gguf",
            "hf:o/r//f.gguf",
            "hf:o/r/./f.gguf",
            "hf:o/r/a\\b.gguf",
            "hf:o/r/f.gguf@../../x",
            "hf:o/r/f.gguf@",
            "hf:/r/f.gguf",
            "hf:o/r/",
        ] {
            assert!(hf_parts(bad).is_err(), "{bad} should be rejected");
        }
    }
}

/// Positional write that leaves the file cursor alone, so worker threads can share one handle.
#[cfg(unix)]
fn write_all_at(f: &File, buf: &[u8], off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(buf, off)
}

#[cfg(windows)]
fn write_all_at(f: &File, mut buf: &[u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_write(buf, off)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "short write"));
        }
        buf = &buf[n..];
        off += n as u64;
    }
    Ok(())
}
