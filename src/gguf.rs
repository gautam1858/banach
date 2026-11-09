//! GGUF header parser. Reads metadata and tensor infos only, never tensor data.
//!
//! Every length and offset from the file is checked: arithmetic is checked, strings and
//! arrays have hard caps, and tensor extents must lie inside the file when its size is known.

use crate::source::{ByteSource, Result};
use std::collections::BTreeMap;

const MAX_STRING: u64 = 64 << 20; // 64 MiB
const MAX_ARRAY: u64 = 1 << 26; // 67M elements
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u32 = 8;
const MAX_HEADER: u64 = 1 << 30; // header (metadata + tensor infos) must fit in 1 GiB

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// Arrays keep their length; elements are kept only for small arrays of numbers.
    Array { elem_type: u32, len: u64, items: Vec<Value> },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            Value::Bool(b) => Some(b as u64),
            _ => None,
        }
    }
    #[allow(dead_code)]
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // offsets and split index are used by the fetcher (Phase 2)
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset relative to the data section.
    pub offset: u64,
    /// Size in bytes (from the type table, or from offset gaps for unknown types).
    pub nbytes: u64,
    /// Index of the split file the tensor lives in.
    pub split: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Gguf {
    pub version: u32,
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_size: Option<u64>,
    /// Bytes fetched from the source while reading the header.
    pub header_bytes: u64,
}

/// (block size in elements, bytes per block) for ggml types, from ggml-common.h.
pub fn type_layout(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (1, 4),       // F32
        1 => (1, 2),       // F16
        2 => (32, 18),     // Q4_0
        3 => (32, 20),     // Q4_1
        6 => (32, 22),     // Q5_0
        7 => (32, 24),     // Q5_1
        8 => (32, 34),     // Q8_0
        9 => (32, 36),     // Q8_1
        10 => (256, 84),   // Q2_K
        11 => (256, 110),  // Q3_K
        12 => (256, 144),  // Q4_K
        13 => (256, 176),  // Q5_K
        14 => (256, 210),  // Q6_K
        15 => (256, 292),  // Q8_K
        16 => (256, 66),   // IQ2_XXS
        17 => (256, 74),   // IQ2_XS
        18 => (256, 98),   // IQ3_XXS
        19 => (256, 50),   // IQ1_S
        20 => (32, 18),    // IQ4_NL
        21 => (256, 110),  // IQ3_S
        22 => (256, 82),   // IQ2_S
        23 => (256, 136),  // IQ4_XS
        24 => (1, 1),      // I8
        25 => (1, 2),      // I16
        26 => (1, 4),      // I32
        27 => (1, 8),      // I64
        28 => (1, 8),      // F64
        29 => (256, 56),   // IQ1_M
        30 => (1, 2),      // BF16
        34 => (256, 54),   // TQ1_0
        35 => (256, 66),   // TQ2_0
        39 => (32, 17),    // MXFP4
        40 => (64, 36),    // NVFP4
        41 => (128, 18),   // Q1_0
        42 => (64, 18),    // Q2_0
        43 => (1, 1),      // F8_E4M3
        _ => return None,
    })
}

pub fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
        9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K",
        15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S",
        20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S", 23 => "IQ4_XS", 24 => "I8", 25 => "I16",
        26 => "I32", 27 => "I64", 28 => "F64", 29 => "IQ1_M", 30 => "BF16", 34 => "TQ1_0",
        35 => "TQ2_0", 39 => "MXFP4", 40 => "NVFP4", 41 => "Q1_0", 42 => "Q2_0", 43 => "F8_E4M3",
        _ => "unknown",
    }
}

/// Bits per weight for a type, if known.
pub fn type_bpw(t: u32) -> Option<f64> {
    type_layout(t).map(|(b, s)| s as f64 * 8.0 / b as f64)
}

/// Buffered sequential reader over a ByteSource, fetching in growing chunks.
struct Reader<'a> {
    src: &'a mut dyn ByteSource,
    buf: Vec<u8>,
    buf_start: u64,
    pos: u64,
    chunk: usize,
    eof: bool,
    fetched: u64,
}

impl<'a> Reader<'a> {
    fn new(src: &'a mut dyn ByteSource, first_chunk: usize) -> Self {
        Self { src, buf: Vec::new(), buf_start: 0, pos: 0, chunk: first_chunk, eof: false, fetched: 0 }
    }

    fn ensure(&mut self, n: u64) -> Result<()> {
        let end = self.pos.checked_add(n).ok_or("offset overflow")?;
        if end > MAX_HEADER {
            return Err(format!("header larger than {MAX_HEADER} bytes"));
        }
        while self.buf_start + (self.buf.len() as u64) < end {
            if self.eof {
                return Err("unexpected end of file in header".into());
            }
            let have_end = self.buf_start + self.buf.len() as u64;
            let want = (end - have_end).max(self.chunk as u64) as usize;
            let got = self.src.read_at(have_end, want)?;
            self.fetched += got.len() as u64;
            if got.len() < want {
                self.eof = true;
            }
            // drop consumed prefix to bound memory
            if self.pos > self.buf_start {
                let drop = (self.pos - self.buf_start) as usize;
                self.buf.drain(..drop.min(self.buf.len()));
                self.buf_start = self.pos.min(self.buf_start + drop as u64);
            }
            self.buf.extend_from_slice(&got);
            self.chunk = (self.chunk * 2).min(8 << 20);
        }
        Ok(())
    }

    fn bytes(&mut self, n: u64) -> Result<&[u8]> {
        self.ensure(n)?;
        let s = (self.pos - self.buf_start) as usize;
        self.pos += n;
        Ok(&self.buf[s..s + n as usize])
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        // Skipping still needs the bytes for sequential sources; fine for headers.
        let mut left = n;
        while left > 0 {
            let step = left.min(16 << 20);
            self.bytes(step)?;
            left -= step;
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return Err(format!("string length {n} exceeds limit"));
        }
        let b = self.bytes(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    fn scalar_size(t: u32) -> Option<u64> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }

    fn value(&mut self, t: u32, depth: u32) -> Result<Value> {
        if depth > 4 {
            return Err("nested arrays too deep".into());
        }
        Ok(match t {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => {
                let b = self.u8()?;
                if b > 1 {
                    return Err(format!("invalid bool {b}"));
                }
                Value::Bool(b == 1)
            }
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let len = self.u64()?;
                if len > MAX_ARRAY {
                    return Err(format!("array length {len} exceeds limit"));
                }
                let mut items = Vec::new();
                if let Some(sz) = Self::scalar_size(et) {
                    if len <= 4096 {
                        for _ in 0..len {
                            items.push(self.value(et, depth + 1)?);
                        }
                    } else {
                        self.skip(len.checked_mul(sz).ok_or("array size overflow")?)?;
                    }
                } else if et == 8 {
                    for _ in 0..len {
                        let n = self.u64()?;
                        if n > MAX_STRING {
                            return Err("string in array too long".into());
                        }
                        self.skip(n)?;
                    }
                } else if et == 9 {
                    for _ in 0..len {
                        self.value(9, depth + 1)?;
                    }
                } else {
                    return Err(format!("unknown array element type {et}"));
                }
                Value::Array { elem_type: et, len, items }
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(format!("unknown value type {t}")),
        })
    }
}

/// Parse a GGUF header. `split` tags the tensors with the split index.
pub fn parse(src: &mut dyn ByteSource, split: usize) -> Result<Gguf> {
    let file_size = src.size().ok();
    let mut r = Reader::new(src, 256 << 10);
    let magic = r.bytes(4)?;
    if magic != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
        return Err(format!("implausible counts: {n_tensors} tensors, {n_kv} kv pairs"));
    }
    let mut kv = BTreeMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let t = r.u32()?;
        let v = r.value(t, 0)?;
        kv.insert(key, v);
    }
    let alignment = match kv.get("general.alignment").and_then(|v| v.as_u64()) {
        Some(a) if a > 0 && a.is_power_of_two() && a <= 1 << 20 => a,
        Some(a) => return Err(format!("invalid alignment {a}")),
        None => 32,
    };
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if nd == 0 || nd > MAX_DIMS {
            return Err(format!("tensor {name}: bad dimension count {nd}"));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(r.u64()?);
        }
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        if offset % alignment != 0 {
            return Err(format!("tensor {name}: offset {offset} not aligned to {alignment}"));
        }
        let n_el = dims
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d))
            .ok_or_else(|| format!("tensor {name}: element count overflow"))?;
        let nbytes = match type_layout(ggml_type) {
            Some((blk, sz)) => {
                if dims[0] % blk != 0 {
                    return Err(format!("tensor {name}: row of {} not a multiple of block {blk}", dims[0]));
                }
                (n_el / blk).checked_mul(sz).ok_or_else(|| format!("tensor {name}: size overflow"))?
            }
            None => 0, // filled from offset gaps below
        };
        tensors.push(TensorInfo { name, dims, ggml_type, offset, nbytes, split });
    }
    let header_end = r.pos;
    let data_offset = header_end
        .checked_add(alignment - 1)
        .ok_or("offset overflow")?
        / alignment
        * alignment;
    let header_bytes = r.fetched;

    // Unknown types: size from the gap to the next tensor (or to end of file).
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].offset);
    for w in 0..order.len() {
        let i = order[w];
        if tensors[i].nbytes == 0 {
            let next = if w + 1 < order.len() {
                Some(tensors[order[w + 1]].offset)
            } else {
                file_size.and_then(|fs| fs.checked_sub(data_offset))
            };
            match next {
                Some(n) if n >= tensors[i].offset => tensors[i].nbytes = n - tensors[i].offset,
                _ => {
                    return Err(format!(
                        "tensor {}: unknown type {} and no way to infer its size",
                        tensors[i].name, tensors[i].ggml_type
                    ))
                }
            }
        }
    }
    // Extents must not overlap and must lie inside the file.
    let mut prev_end = 0u64;
    for &i in &order {
        let t = &tensors[i];
        if t.offset < prev_end {
            return Err(format!("tensor {} overlaps the previous tensor", t.name));
        }
        let end = t.offset.checked_add(t.nbytes).ok_or("tensor extent overflow")?;
        if let Some(fs) = file_size {
            let abs = data_offset.checked_add(end).ok_or("tensor extent overflow")?;
            if abs > fs {
                return Err(format!("tensor {} extends past end of file", t.name));
            }
        }
        prev_end = end;
    }
    Ok(Gguf { version, kv, tensors, alignment, data_offset, file_size, header_bytes })
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub struct Mem(pub Vec<u8>);
    impl ByteSource for Mem {
        fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
            let s = (offset as usize).min(self.0.len());
            let e = (s + len).min(self.0.len());
            Ok(self.0[s..e].to_vec())
        }
        fn size(&mut self) -> Result<u64> {
            Ok(self.0.len() as u64)
        }
        fn describe(&self) -> String {
            "mem".into()
        }
    }

    /// Minimal GGUF writer for tests and synthetic models.
    pub struct Builder {
        kv: Vec<(String, u32, Vec<u8>)>,
        tensors: Vec<(String, Vec<u64>, u32)>,
    }

    fn s(v: &str) -> Vec<u8> {
        let mut o = (v.len() as u64).to_le_bytes().to_vec();
        o.extend_from_slice(v.as_bytes());
        o
    }

    impl Builder {
        pub fn new() -> Self {
            Self { kv: vec![], tensors: vec![] }
        }
        pub fn kv_str(mut self, k: &str, v: &str) -> Self {
            self.kv.push((k.into(), 8, s(v)));
            self
        }
        pub fn kv_u32(mut self, k: &str, v: u32) -> Self {
            self.kv.push((k.into(), 4, v.to_le_bytes().to_vec()));
            self
        }
        pub fn tensor(mut self, name: &str, dims: &[u64], t: u32) -> Self {
            self.tensors.push((name.into(), dims.to_vec(), t));
            self
        }
        pub fn build(self) -> Vec<u8> {
            let mut o = b"GGUF".to_vec();
            o.extend_from_slice(&3u32.to_le_bytes());
            o.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            o.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
            for (k, t, v) in &self.kv {
                o.extend(s(k));
                o.extend_from_slice(&t.to_le_bytes());
                o.extend_from_slice(v);
            }
            let mut off = 0u64;
            let mut sizes = vec![];
            for (name, dims, t) in &self.tensors {
                o.extend(s(name));
                o.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    o.extend_from_slice(&d.to_le_bytes());
                }
                o.extend_from_slice(&t.to_le_bytes());
                o.extend_from_slice(&off.to_le_bytes());
                let (b, sz) = type_layout(*t).unwrap();
                let n = dims.iter().product::<u64>() / b * sz;
                sizes.push(n);
                off += (n + 31) / 32 * 32;
            }
            while o.len() % 32 != 0 {
                o.push(0);
            }
            o.resize(o.len() + off as usize, 0);
            o
        }
    }

    #[test]
    fn roundtrip() {
        let bytes = Builder::new()
            .kv_str("general.architecture", "toy")
            .kv_u32("toy.block_count", 2)
            .tensor("token_embd.weight", &[64, 100], 1)
            .tensor("blk.0.ffn_up_exps.weight", &[64, 32, 8], 8)
            .build();
        let g = parse(&mut Mem(bytes), 0).unwrap();
        assert_eq!(g.kv["general.architecture"].as_str(), Some("toy"));
        assert_eq!(g.kv["toy.block_count"].as_u64(), Some(2));
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.tensors[0].nbytes, 64 * 100 * 2);
        assert_eq!(g.tensors[1].nbytes, 64 * 32 * 8 / 32 * 34);
    }

    #[test]
    fn rejects_garbage_and_truncation() {
        assert!(parse(&mut Mem(b"NOPE".to_vec()), 0).is_err());
        let bytes = Builder::new().tensor("a", &[32, 4], 8).build();
        for cut in [3, 10, 24, 30, 40] {
            assert!(parse(&mut Mem(bytes[..cut].to_vec()), 0).is_err(), "cut {cut}");
        }
        // file truncated inside tensor data (the tensor is 136 bytes followed by 24 of padding)
        let n = bytes.len();
        assert!(parse(&mut Mem(bytes[..n - 24].to_vec()), 0).is_ok());
        assert!(parse(&mut Mem(bytes[..n - 30].to_vec()), 0).is_err());
    }

    #[test]
    fn rejects_hostile_lengths() {
        let mut o = b"GGUF".to_vec();
        o.extend_from_slice(&3u32.to_le_bytes());
        o.extend_from_slice(&0u64.to_le_bytes());
        o.extend_from_slice(&1u64.to_le_bytes());
        o.extend_from_slice(&u64::MAX.to_le_bytes()); // key length
        assert!(parse(&mut Mem(o), 0).is_err());

        let mut o = b"GGUF".to_vec();
        o.extend_from_slice(&3u32.to_le_bytes());
        o.extend_from_slice(&u64::MAX.to_le_bytes());
        o.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse(&mut Mem(o), 0).is_err());
    }

    #[test]
    fn rejects_dims_overflow() {
        let bytes = Builder::new().tensor("a", &[32, 4], 8).build();
        // patch the dims of tensor "a" to huge values: header layout is fixed here
        let mut b = bytes.clone();
        let pos = 4 + 4 + 8 + 8 + 8 + 1 + 4; // magic, ver, nt, nkv, name len, "a", ndims
        b[pos..pos + 8].copy_from_slice(&(u64::MAX / 2 + 32).to_le_bytes());
        b[pos + 8..pos + 16].copy_from_slice(&4u64.to_le_bytes());
        assert!(parse(&mut Mem(b), 0).is_err());
    }

    #[test]
    fn fuzz_random_mutations_never_panic() {
        let bytes = Builder::new()
            .kv_str("general.architecture", "toy")
            .kv_u32("toy.expert_count", 8)
            .tensor("blk.0.ffn_up_exps.weight", &[64, 32, 8], 12)
            .build();
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..3000 {
            let mut b = bytes.clone();
            for _ in 0..4 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let i = (x as usize) % b.len().min(200);
                b[i] = (x >> 32) as u8;
            }
            let _ = parse(&mut Mem(b), 0);
        }
    }
}
