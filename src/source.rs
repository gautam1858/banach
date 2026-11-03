//! Byte sources for model files: local paths and Hugging Face HTTP range reads.
//!
//! Model files are untrusted input. Sources only hand out bytes; all parsing and bounds
//! checks live in `gguf`.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, String>;

pub trait ByteSource {
    /// Read exactly `len` bytes at `offset`, or fewer only at end of file.
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>>;
    /// Total size in bytes, if known.
    fn size(&mut self) -> Result<u64>;
    fn describe(&self) -> String;
    /// Number of network requests made so far (0 for local files).
    fn requests(&self) -> u64 {
        0
    }
}

pub struct LocalFile {
    path: PathBuf,
    file: File,
}

impl LocalFile {
    pub fn open(path: &Path) -> Result<Self> {
        let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!("{}: refusing to follow a symlink", path.display()));
        }
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { path: path.to_path_buf(), file })
    }
}

impl ByteSource for LocalFile {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; len];
        let mut got = 0;
        while got < len {
            let n = self.file.read(&mut buf[got..]).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            got += n;
        }
        buf.truncate(got);
        Ok(buf)
    }
    fn size(&mut self) -> Result<u64> {
        self.file.metadata().map(|m| m.len()).map_err(|e| e.to_string())
    }
    fn describe(&self) -> String {
        self.path.display().to_string()
    }
}

/// HTTP source using Range requests. Only Hugging Face hosts are allowed (non-negotiable 1:
/// the only network use is model download on request).
pub struct HttpRange {
    url: String,
    agent: ureq::Agent,
    size: Option<u64>,
    requests: u64,
}

pub const ALLOWED_HOSTS: &[&str] = &["huggingface.co", "cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"];

fn host_of(url: &str) -> Option<&str> {
    let rest = url.strip_prefix("https://")?;
    Some(rest.split(['/', ':', '?']).next().unwrap_or(""))
}

pub fn host_allowed(url: &str) -> bool {
    match host_of(url) {
        Some(h) => ALLOWED_HOSTS.iter().any(|a| h == *a || h.ends_with(&format!(".{a}"))),
        None => false,
    }
}

impl HttpRange {
    pub fn new(url: &str) -> Result<Self> {
        if !host_allowed(url) {
            return Err(format!("refusing to fetch from {url}: only Hugging Face over https is allowed"));
        }
        let agent = ureq::AgentBuilder::new()
            .try_proxy_from_env(true)
            .redirects(5)
            .timeout(std::time::Duration::from_secs(60))
            .build();
        Ok(Self { url: url.to_string(), agent, size: None, requests: 0 })
    }
}

fn parse_content_range_total(v: &str) -> Option<u64> {
    // "bytes 0-15/528754944"
    v.rsplit('/').next()?.trim().parse().ok()
}

impl ByteSource for HttpRange {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let end = offset
            .checked_add(len as u64 - 1)
            .ok_or_else(|| "range overflow".to_string())?;
        self.requests += 1;
        let resp = self
            .agent
            .get(&self.url)
            .set("Range", &format!("bytes={offset}-{end}"))
            .set("User-Agent", "banach/0.1")
            .call()
            .map_err(|e| format!("GET {}: {e}", self.url))?;
        if resp.status() != 206 {
            return Err(format!("GET {}: expected 206 Partial Content, got {}", self.url, resp.status()));
        }
        if let Some(total) = resp.header("Content-Range").and_then(parse_content_range_total) {
            self.size = Some(total);
        }
        let mut buf = Vec::with_capacity(len);
        resp.into_reader()
            .take(len as u64)
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        Ok(buf)
    }
    fn size(&mut self) -> Result<u64> {
        if self.size.is_none() {
            self.read_at(0, 1)?;
        }
        self.size.ok_or_else(|| "server did not report a size".to_string())
    }
    fn describe(&self) -> String {
        self.url.clone()
    }
    fn requests(&self) -> u64 {
        self.requests
    }
}

/// Resolve "hf:owner/repo/path/file.gguf[@revision]" or an https URL or a local path.
pub fn resolve(spec: &str) -> Result<Box<dyn ByteSource>> {
    if let Some(rest) = spec.strip_prefix("hf:") {
        let (path, rev) = match rest.rsplit_once('@') {
            Some((p, r)) => (p, r),
            None => (rest, "main"),
        };
        let mut parts = path.splitn(3, '/');
        let (owner, repo, file) = (parts.next(), parts.next(), parts.next());
        match (owner, repo, file) {
            (Some(o), Some(r), Some(f)) if !o.is_empty() && !r.is_empty() && !f.is_empty() => {
                let url = format!("https://huggingface.co/{o}/{r}/resolve/{rev}/{f}");
                Ok(Box::new(HttpRange::new(&url)?))
            }
            _ => Err(format!("bad hf spec {spec:?}; expected hf:owner/repo/file.gguf")),
        }
    } else if spec.starts_with("https://") || spec.starts_with("http://") {
        Ok(Box::new(HttpRange::new(spec)?))
    } else {
        Ok(Box::new(LocalFile::open(Path::new(spec))?))
    }
}

/// For a split model "<name>-00001-of-00005.gguf", the specs of all parts in order.
pub fn split_parts(spec: &str) -> Vec<String> {
    let (stem, suffix) = match spec.rsplit_once('@') {
        Some((s, r)) if spec.starts_with("hf:") => (s, format!("@{r}")),
        _ => (spec, String::new()),
    };
    let Some(base) = stem.strip_suffix(".gguf") else { return vec![spec.to_string()] };
    let bytes = base.as_bytes();
    // ...-NNNNN-of-MMMMM
    if bytes.len() < 15 {
        return vec![spec.to_string()];
    }
    let tail = &base[base.len() - 15..];
    let ok = tail.as_bytes()[0] == b'-'
        && &tail[6..10] == "-of-"
        && tail[1..6].bytes().all(|c| c.is_ascii_digit())
        && tail[10..15].bytes().all(|c| c.is_ascii_digit());
    if !ok {
        return vec![spec.to_string()];
    }
    let n: usize = tail[10..15].parse().unwrap_or(1);
    let head = &base[..base.len() - 15];
    (1..=n.max(1)).map(|i| format!("{head}-{i:05}-of-{n:05}.gguf{suffix}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts() {
        assert!(host_allowed("https://huggingface.co/a/b/resolve/main/x.gguf"));
        assert!(host_allowed("https://cdn-lfs.huggingface.co/x"));
        assert!(!host_allowed("http://huggingface.co/x"));
        assert!(!host_allowed("https://evil.example/huggingface.co"));
        assert!(!host_allowed("https://huggingface.co.evil.example/x"));
    }

    #[test]
    fn splits() {
        assert_eq!(split_parts("m.gguf"), vec!["m.gguf"]);
        assert_eq!(
            split_parts("hf:o/r/m-00001-of-00003.gguf@v1"),
            vec![
                "hf:o/r/m-00001-of-00003.gguf@v1",
                "hf:o/r/m-00002-of-00003.gguf@v1",
                "hf:o/r/m-00003-of-00003.gguf@v1"
            ]
        );
    }

    #[test]
    fn content_range() {
        assert_eq!(parse_content_range_total("bytes 0-15/528754944"), Some(528754944));
        assert_eq!(parse_content_range_total("bytes */12"), Some(12));
    }
}
