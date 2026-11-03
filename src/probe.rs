//! Hardware probe: capacities from the OS, bandwidths and compute from short microbenchmarks.
//!
//! The numbers are achieved, not rated: decode speed depends on what a streaming read
//! actually gets, so the planner uses these directly.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpu {
    pub name: String,
    pub vram_bytes: u64,
    /// Rough bandwidth from a lookup on the name; None when unknown.
    pub bandwidth: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hardware {
    pub cpu: String,
    pub cores: usize,
    pub ram_total: u64,
    pub ram_available: u64,
    pub unified_memory: bool,
    /// Achieved multi-threaded read bandwidth, bytes/s.
    pub mem_bw: f64,
    /// Achieved f32 FLOP/s on a vectorisable dot product, all cores.
    pub cpu_flops: f64,
    /// Achieved large-block read bandwidth from the model disk, bytes/s (None if not measured).
    pub disk_bw: Option<f64>,
    pub disk_measured_direct: bool,
    pub gpus: Vec<Gpu>,
    pub probe_seconds: f64,
}

fn read_to_string(p: &str) -> Option<String> {
    std::fs::read_to_string(p).ok()
}

fn meminfo(key: &str) -> Option<u64> {
    let s = read_to_string("/proc/meminfo")?;
    let line = s.lines().find(|l| l.starts_with(key))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn sysctl(key: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl").arg("-n").arg(key).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn cpu_name() -> String {
    if let Some(s) = read_to_string("/proc/cpuinfo") {
        if let Some(l) = s.lines().find(|l| l.starts_with("model name")) {
            return l.split(':').nth(1).unwrap_or("").trim().to_string();
        }
    }
    sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into())
}

fn ram() -> (u64, u64) {
    if let (Some(t), Some(a)) = (meminfo("MemTotal:"), meminfo("MemAvailable:")) {
        return (t, a);
    }
    let t = sysctl("hw.memsize").and_then(|s| s.parse().ok()).unwrap_or(0);
    (t, t * 3 / 4)
}

/// Approximate bandwidth for common GPUs, bytes/s. Unknown cards return None and the planner
/// leaves them out of speed predictions rather than guess.
pub fn gpu_bandwidth_lookup(name: &str) -> Option<f64> {
    let n = name.to_ascii_lowercase();
    let table: &[(&str, f64)] = &[
        ("5090", 1792.0), ("5080", 960.0), ("5070 ti", 896.0), ("5070", 672.0), ("5060 ti", 448.0),
        ("4090", 1008.0), ("4080", 717.0), ("4070 ti", 504.0), ("4070", 504.0), ("4060 ti", 288.0),
        ("3090", 936.0), ("3080", 760.0), ("3070", 448.0), ("3060", 360.0),
        ("a100", 1555.0), ("h100", 3350.0), ("l40", 864.0), ("a6000", 768.0),
    ];
    table.iter().find(|(k, _)| n.contains(k)).map(|(_, gb)| gb * 1e9)
}

fn nvidia_gpus() -> Vec<Gpu> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();
    let Ok(out) = out else { return vec![] };
    if !out.status.success() {
        return vec![];
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.split(',');
            let name = it.next()?.trim().to_string();
            let mib: u64 = it.next()?.trim().parse().ok()?;
            Some(Gpu { bandwidth: gpu_bandwidth_lookup(&name), name, vram_bytes: mib << 20 })
        })
        .collect()
}

/// Multi-threaded streaming read bandwidth over `bytes` of memory (best of `reps`).
pub fn mem_bandwidth(bytes: usize, threads: usize, reps: usize) -> f64 {
    let n = bytes / 8;
    let data: Vec<u64> = (0..n as u64).collect();
    let chunk = n.div_ceil(threads.max(1));
    let mut best = 0.0f64;
    for _ in 0..reps {
        let t0 = Instant::now();
        let total: u64 = std::thread::scope(|s| {
            let hs: Vec<_> = data
                .chunks(chunk)
                .map(|c| {
                    s.spawn(move || {
                        // four accumulators so the loop is bandwidth-bound, not latency-bound
                        let (mut a, mut b, mut cc, mut d) = (0u64, 0u64, 0u64, 0u64);
                        let mut it = c.chunks_exact(4);
                        for q in &mut it {
                            a = a.wrapping_add(q[0]);
                            b = b.wrapping_add(q[1]);
                            cc = cc.wrapping_add(q[2]);
                            d = d.wrapping_add(q[3]);
                        }
                        for x in it.remainder() {
                            a = a.wrapping_add(*x);
                        }
                        a ^ b ^ cc ^ d
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).fold(0, |x, y| x ^ y)
        });
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(total);
        best = best.max(bytes as f64 / dt);
    }
    best
}

/// Peak f32 multiply-add throughput on cache-resident data, all threads, FLOP/s.
///
/// Uses the widest FMA the CPU has (AVX-512, AVX2+FMA) through runtime detection, so the
/// number is comparable to what tuned kernels (ggml) can reach; the planner applies an
/// efficiency factor on top.
pub fn cpu_flops(threads: usize, dur: Duration) -> f64 {
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1)).map(|_| s.spawn(move || fma_rate(dur))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn fma_rate(dur: Duration) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { fma_x86::avx512(dur) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { fma_x86::avx2(dur) };
        }
    }
    fma_portable(dur)
}

fn fma_portable(dur: Duration) -> f64 {
    // 8 independent accumulator vectors of 8 lanes: the compiler vectorises this for the
    // baseline target (SSE2 on x86_64, NEON on aarch64).
    let mut acc = [[1.0f32; 8]; 8];
    let m = [0.999_999f32; 8];
    let a = [1e-7f32; 8];
    let t0 = Instant::now();
    let mut iters = 0u64;
    while t0.elapsed() < dur {
        for _ in 0..4096 {
            for v in acc.iter_mut() {
                for k in 0..8 {
                    v[k] = v[k] * m[k] + a[k];
                }
            }
        }
        iters += 4096;
    }
    std::hint::black_box(acc);
    iters as f64 * 64.0 * 2.0 / t0.elapsed().as_secs_f64()
}

#[cfg(target_arch = "x86_64")]
mod fma_x86 {
    use std::arch::x86_64::*;
    use std::time::{Duration, Instant};

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn avx2(dur: Duration) -> f64 {
        let m = _mm256_set1_ps(0.999_999);
        let a = _mm256_set1_ps(1e-7);
        let mut r = [_mm256_set1_ps(1.0); 10];
        let t0 = Instant::now();
        let mut iters = 0u64;
        while t0.elapsed() < dur {
            for _ in 0..4096 {
                for v in r.iter_mut() {
                    *v = _mm256_fmadd_ps(*v, m, a);
                }
            }
            iters += 4096;
        }
        std::hint::black_box(r);
        iters as f64 * 10.0 * 8.0 * 2.0 / t0.elapsed().as_secs_f64()
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn avx512(dur: Duration) -> f64 {
        let m = _mm512_set1_ps(0.999_999);
        let a = _mm512_set1_ps(1e-7);
        let mut r = [_mm512_set1_ps(1.0); 12];
        let t0 = Instant::now();
        let mut iters = 0u64;
        while t0.elapsed() < dur {
            for _ in 0..4096 {
                for v in r.iter_mut() {
                    *v = _mm512_fmadd_ps(*v, m, a);
                }
            }
            iters += 4096;
        }
        std::hint::black_box(r);
        iters as f64 * 12.0 * 16.0 * 2.0 / t0.elapsed().as_secs_f64()
    }
}

/// Large random reads from `path`. Uses O_DIRECT on Linux when possible so the page cache
/// does not inflate the number; falls back to buffered reads (reported).
pub fn disk_bandwidth(path: &Path, block: usize, dur: Duration) -> Result<(f64, bool), String> {
    let size = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size < block as u64 * 4 {
        return Err(format!("{} is too small for a disk test", path.display()));
    }
    #[cfg(target_os = "linux")]
    let (mut f, direct) = {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new().read(true).custom_flags(0o40000).open(path) {
            Ok(f) => (f, true),
            Err(_) => (File::open(path).map_err(|e| e.to_string())?, false),
        }
    };
    #[cfg(not(target_os = "linux"))]
    let (mut f, direct) = (File::open(path).map_err(|e| e.to_string())?, false);

    // 4 KiB-aligned buffer for O_DIRECT
    let mut raw = vec![0u8; block + 4096];
    let off = raw.as_ptr().align_offset(4096);
    let buf = &mut raw[off..off + block];
    let blocks = size / block as u64;
    let mut x: u64 = 0x2545F4914F6CDD1D;
    let mut read = 0u64;
    let t0 = Instant::now();
    use std::io::{Seek, SeekFrom};
    while t0.elapsed() < dur {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let b = x % blocks;
        f.seek(SeekFrom::Start(b * block as u64)).map_err(|e| e.to_string())?;
        let n = f.read(buf).map_err(|e| format!("read: {e}"))?;
        read += n as u64;
    }
    Ok((read as f64 / t0.elapsed().as_secs_f64(), direct))
}

pub struct ProbeOptions<'a> {
    pub mem_bytes: usize,
    pub disk_file: Option<&'a Path>,
    pub quick: bool,
}

pub fn probe(opts: &ProbeOptions) -> Hardware {
    let t0 = Instant::now();
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (ram_total, ram_available) = ram();
    let cpu = cpu_name();
    let unified = cpu.contains("Apple");
    let reps = if opts.quick { 2 } else { 5 };
    let mem_bw = mem_bandwidth(opts.mem_bytes, cores, reps);
    let cpu_flops = cpu_flops(cores, Duration::from_millis(if opts.quick { 200 } else { 1000 }));
    let (disk_bw, direct) = match opts.disk_file {
        Some(p) => match disk_bandwidth(p, 8 << 20, Duration::from_secs(if opts.quick { 1 } else { 5 })) {
            Ok((bw, d)) => (Some(bw), d),
            Err(e) => {
                eprintln!("disk probe skipped: {e}");
                (None, false)
            }
        },
        None => (None, false),
    };
    Hardware {
        cpu,
        cores,
        ram_total,
        ram_available,
        unified_memory: unified,
        mem_bw,
        cpu_flops,
        disk_bw,
        disk_measured_direct: direct,
        gpus: nvidia_gpus(),
        probe_seconds: t0.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_is_positive_and_sane() {
        let bw = mem_bandwidth(64 << 20, 2, 2);
        assert!(bw > 1e8 && bw < 1e13, "{bw}");
        let f = cpu_flops(1, Duration::from_millis(50));
        assert!(f > 1e7, "{f}");
    }

    #[test]
    fn gpu_lookup() {
        assert_eq!(gpu_bandwidth_lookup("NVIDIA GeForce RTX 5080"), Some(960e9));
        assert_eq!(gpu_bandwidth_lookup("Mystery card"), None);
    }
}
