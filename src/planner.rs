//! Planner v1: roofline cost model, tier placement and the plan options shown on the
//! preflight card (SPEC-SYSTEM section 2, BANACH-SPEC section 6).
//!
//! Decode time per token, per step:
//!   t_mem     = sum over memory tiers of bytes read there / (bandwidth * efficiency)
//!               + miss bytes / RAM bandwidth (missed experts land in RAM first)
//!               + KV bytes read over the context / (bandwidth * attention efficiency)
//!   t_cpu     = 2 * active params / (cpu flops * cpu efficiency)
//!   t_compute = max(t_mem, t_cpu) + (1 - compute_overlap) * min(t_mem, t_cpu)
//!   t_io      = miss bytes / disk bandwidth
//!   t_token   = t_compute + max(0, t_io - overlap * t_compute) + t_sync
//!
//! Hit rate comes from measured expert usage when given (static hot set, which tracked GDSF
//! within 2 points in the Phase 0 pilot), otherwise from the power law h(f) = f^a, where f is
//! the cached share of the expert pool and a depends on how sparse routing is:
//!   a = 0.15 + 0.8 * k/N, band a * [0.75, 1.35]
//! Fitted to colibri issue 905 (V4 Flash, 12% cached, 65% hit), the SPEC-DERISK estimate for
//! V4 Flash (35% cached, 80 to 88% hit) and the Phase 0 Granite pilot (k/N = 0.25). A routing
//! trace of the target model replaces it.

use crate::mckp::{self, Opt};
use crate::probe::Hardware;
use crate::profile::ModelProfile;
use serde::{Deserialize, Serialize};

pub const GB: f64 = 1e9;
pub const GIB: u64 = 1 << 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub mem_eff: f64,
    pub cpu_eff: f64,
    /// How much compute hides under memory traffic: 1 = max(t_mem, t_cpu), 0 = sum.
    pub compute_overlap: f64,
    /// Fraction of compute time NVMe reads can hide under. 0 for demand fetch (the Phase 2
    /// engine blocks a layer until its experts are in); lookahead prefetch raises it.
    pub overlap: f64,
    /// Attention reads the KV cache at this fraction of memory bandwidth (CPU attention is far
    /// from streaming speed).
    #[serde(default = "default_attn_eff")]
    pub attn_eff: f64,
    /// Fixed per-token overhead per layer, seconds.
    pub sync_per_layer: f64,
    /// Multiplier on predicted time from a measured run (1.0 = uncalibrated).
    pub time_scale: f64,
    pub calibrated_on: Option<String>,
}

fn default_attn_eff() -> f64 {
    0.2
}

impl Default for Calibration {
    fn default() -> Self {
        // compute_overlap 0 and cpu_eff 0.18 of peak FMA: fitted on llama.cpp Q8_0 decode of
        // Granite 3.1 1B-A400M on a 4-thread AVX-512 Xeon, validated on Q2_K held out (+24%).
        // See docs/VALIDATION.md. K1 on the target machine replaces these.
        Self {
            mem_eff: 0.85,
            cpu_eff: 0.18,
            compute_overlap: 0.0,
            overlap: 0.0,
            attn_eff: 0.2,
            sync_per_layer: 2.0e-5,
            time_scale: 1.0,
            calibrated_on: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Constraints {
    pub ctx: u64,
    pub min_tok_s: f64,
    /// RAM kept free for the OS and other programs.
    pub reserve_bytes: u64,
    /// Per-layer expert usage frequencies (each row sums to top_k), from a trace.
    pub usage: Option<Vec<Vec<f64>>>,
    /// Assumed disk bandwidth when the probe did not measure one.
    pub default_disk_bw: f64,
}

impl Default for Constraints {
    fn default() -> Self {
        Self { ctx: 8192, min_tok_s: 5.0, reserve_bytes: 2 * GIB, usage: None, default_disk_bw: 5.5 * GB }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Range {
    pub low: f64,
    pub mid: f64,
    pub high: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub name: String,
    pub lossless: bool,
    pub fidelity: String,
    pub tok_s: Range,
    pub hit_rate: Range,
    pub bytes_vram: u64,
    pub bytes_ram: u64,
    pub hot_experts: u64,
    pub expert_bpw: f64,
    pub nvme_bytes_per_token: f64,
    pub download_first: u64,
    pub disk_needed: u64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub model: String,
    pub feasible: bool,
    pub plans: Vec<Plan>,
    pub verdict: String,
    pub warnings: Vec<String>,
}

/// Fraction of routed accesses that hit a cache holding fraction `f` of the expert pool,
/// for exponent `a` (see the module docs).
pub fn hit_model(f: f64, a: f64) -> f64 {
    if f <= 0.0 {
        0.0
    } else if f >= 1.0 {
        1.0
    } else {
        f.powf(a)
    }
}

/// Central exponent for the hit model from the routing sparsity k/N.
pub fn hit_exponent(top_k: u64, n_experts: u64) -> f64 {
    let r = if n_experts > 0 { top_k as f64 / n_experts as f64 } else { 1.0 };
    0.15 + 0.8 * r.clamp(0.0, 1.0)
}

/// Hit rate of a static hot set of `capacity` experts chosen by usage density.
pub fn hit_from_usage(usage: &[Vec<f64>], expert_bytes: &[u64], capacity_bytes: u64) -> (f64, u64) {
    let items: Vec<Vec<Opt>> = usage
        .iter()
        .zip(expert_bytes)
        .flat_map(|(row, &b)| row.iter().map(move |&p| vec![Opt { bytes: 0, cost: p }, Opt { bytes: b, cost: 0.0 }]))
        .collect();
    let total: f64 = usage.iter().flatten().sum();
    if total <= 0.0 {
        return (0.0, 0);
    }
    let s = mckp::solve(&items, capacity_bytes);
    let kept = s.choice.iter().filter(|&&c| c == 1).count() as u64;
    (1.0 - s.cost / total, kept)
}

struct Tiers {
    vram: u64,
    vram_bw: Option<f64>,
    ram: u64,
    ram_bw: f64,
    disk_bw: f64,
    disk_assumed: bool,
}

fn tiers(hw: &Hardware, c: &Constraints, cal: &Calibration) -> Tiers {
    let gpu = hw.gpus.iter().filter(|g| g.bandwidth.is_some()).max_by_key(|g| g.vram_bytes);
    let (vram, vram_bw) = match gpu {
        Some(g) if !hw.unified_memory => (g.vram_bytes.saturating_sub(GIB), g.bandwidth.map(|b| b * cal.mem_eff)),
        _ => (0, None),
    };
    let ram = hw.ram_available.saturating_sub(c.reserve_bytes);
    let (disk_bw, assumed) = match hw.disk_bw {
        Some(b) => (b, false),
        None => (c.default_disk_bw, true),
    };
    Tiers { vram, vram_bw, ram, ram_bw: hw.mem_bw * cal.mem_eff, disk_bw, disk_assumed: assumed }
}

struct Placement {
    dense_vram: u64,
    dense_ram: u64,
    hot_vram: u64,
    hot_ram: u64,
    feasible: bool,
}

/// Dense path and KV first (fastest tier that fits), then experts fill what is left.
fn place(m: &ModelProfile, t: &Tiers, kv: u64, expert_bytes: u64) -> Placement {
    let must = m.dense_read_bytes + m.embedding_bytes + kv;
    let (dense_vram, dense_ram) = if t.vram_bw.is_some() && must <= t.vram { (must, 0) } else { (0, must) };
    let feasible = dense_ram <= t.ram;
    let vram_left = t.vram.saturating_sub(dense_vram);
    let ram_left = t.ram.saturating_sub(dense_ram);
    let n_total = m.n_experts * m.n_moe_layers;
    let per = expert_bytes.max(1);
    let hot_vram = if t.vram_bw.is_some() { (vram_left / per).min(n_total) } else { 0 };
    let hot_ram = (ram_left / per).min(n_total - hot_vram);
    Placement { dense_vram, dense_ram, hot_vram, hot_ram, feasible }
}

struct Speed {
    tok_s: f64,
    nvme_bytes: f64,
}

#[allow(clippy::too_many_arguments)]
fn speed(m: &ModelProfile, t: &Tiers, cal: &Calibration, hw: &Hardware, p: &Placement, hit: f64,
         expert_scale: f64, bw_scale: f64, ctx: u64) -> Speed {
    let routed = m.routed_read_bytes as f64 * expert_scale;
    let hot = (p.hot_vram + p.hot_ram) as f64;
    // share of hits served from VRAM (uniform-usage approximation)
    let vram_share = if hot > 0.0 { p.hot_vram as f64 / hot } else { 0.0 };
    let hit_bytes = routed * hit;
    let miss_bytes = routed * (1.0 - hit);
    let dense_on_gpu = p.dense_vram > 0;
    let mut t_mem = 0.0;
    if let Some(vbw) = t.vram_bw {
        let vram_read = if dense_on_gpu { m.dense_read_bytes as f64 } else { 0.0 } + hit_bytes * vram_share;
        t_mem += vram_read / (vbw * bw_scale);
    }
    let ram_read = if dense_on_gpu { 0.0 } else { m.dense_read_bytes as f64 } + hit_bytes * (1.0 - vram_share) + miss_bytes;
    t_mem += ram_read / (t.ram_bw * bw_scale);
    // attention over a context of `ctx` tokens (KV on the same tier as the dense path)
    let kv_read = (m.kv_bytes_per_token * ctx) as f64;
    let kv_bw = if dense_on_gpu { t.vram_bw.unwrap_or(t.ram_bw) } else { t.ram_bw };
    t_mem += kv_read / (kv_bw * cal.attn_eff * bw_scale);
    // CPU work: the share of active parameters not handled by the GPU
    let cpu_share = if dense_on_gpu { (1.0 - vram_share) * routed / (routed + m.dense_read_bytes as f64) } else { 1.0 };
    let t_cpu = 2.0 * m.active_params as f64 * cpu_share / (hw.cpu_flops * cal.cpu_eff);
    let t_compute = t_mem.max(t_cpu) + (1.0 - cal.compute_overlap) * t_mem.min(t_cpu);
    let t_io = miss_bytes / (t.disk_bw * bw_scale);
    let t_sync = cal.sync_per_layer * m.n_layers as f64;
    let t_tok = (t_compute + (t_io - cal.overlap * t_compute).max(0.0) + t_sync) * cal.time_scale;
    Speed { tok_s: 1.0 / t_tok, nvme_bytes: miss_bytes }
}

fn plan_for(name: &str, m: &ModelProfile, hw: &Hardware, c: &Constraints, cal: &Calibration,
            expert_scale: f64, lossless: bool) -> Option<Plan> {
    let t = tiers(hw, c, cal);
    let kv = m.kv_bytes_per_token * c.ctx;
    let expert_bytes = (m.expert_bytes as f64 * expert_scale) as u64;
    let p = place(m, &t, kv, expert_bytes);
    if !p.feasible {
        return None;
    }
    let n_total = (m.n_experts * m.n_moe_layers).max(1);
    let hot = p.hot_vram + p.hot_ram;
    let frac = hot as f64 / n_total as f64;
    let hits = match &c.usage {
        Some(u) => {
            let eb: Vec<u64> = m.layers.iter().map(|l| (l.expert_bytes as f64 * expert_scale) as u64).collect();
            let cap = hot * expert_bytes;
            let (h, _) = hit_from_usage(u, &eb, cap);
            // usage from a trace is one workload: widen for domain shift
            Range { low: (h - 0.08).max(0.0), mid: h, high: h }
        }
        None => {
            let a = hit_exponent(m.top_k, m.n_experts);
            Range { low: hit_model(frac, a * 1.35), mid: hit_model(frac, a), high: hit_model(frac, a * 0.75) }
        }
    };
    let s_lo = speed(m, &t, cal, hw, &p, hits.low, expert_scale, 0.9, c.ctx);
    let s_mid = speed(m, &t, cal, hw, &p, hits.mid, expert_scale, 1.0, c.ctx);
    let s_hi = speed(m, &t, cal, hw, &p, hits.high, expert_scale, 1.1, c.ctx);
    let mut notes = Vec::new();
    if t.disk_assumed && hot < n_total {
        notes.push(format!("disk bandwidth assumed {:.1} GB/s (run `banach probe --disk <model file>`)", t.disk_bw / GB));
    }
    if c.usage.is_none() && hot < n_total {
        notes.push("hit rate from the generic curve; pass --usage from a routing trace for a tighter range".into());
    }
    if cal.calibrated_on.is_none() {
        notes.push("uncalibrated; `banach calibrate` with one measured run tightens this".into());
    }
    let fidelity = if lossless {
        "100% (byte-exact weights)".to_string()
    } else {
        format!("projected only: experts at {:.2} bpw need the ladder store and fidelity meter (Phase 3) to certify", m.layers.first().map(|l| l.bpw).unwrap_or(0.0) * expert_scale)
    };
    let bpw = m.layers.first().map(|l| l.bpw).unwrap_or(0.0) * expert_scale;
    let hot_bytes = hot * expert_bytes;
    Some(Plan {
        name: name.to_string(),
        lossless,
        fidelity,
        tok_s: Range { low: s_lo.tok_s, mid: s_mid.tok_s, high: s_hi.tok_s },
        hit_rate: hits,
        bytes_vram: p.dense_vram + p.hot_vram * expert_bytes,
        bytes_ram: p.dense_ram + p.hot_ram * expert_bytes,
        hot_experts: hot,
        expert_bpw: bpw,
        nvme_bytes_per_token: s_mid.nvme_bytes,
        download_first: m.dense_read_bytes + m.embedding_bytes + hot_bytes.min(m.routed_bytes),
        disk_needed: m.total_bytes + if lossless { 0 } else { (m.routed_bytes as f64 * expert_scale) as u64 },
        notes,
    })
}

/// Extra RAM needed for the exact plan to reach `target` tok/s, by bisection (None if even
/// a fully resident model would not reach it).
pub fn ram_needed_for(m: &ModelProfile, hw: &Hardware, c: &Constraints, cal: &Calibration, target: f64) -> Option<u64> {
    let mut h = hw.clone();
    let full = m.total_bytes + m.kv_bytes_per_token * c.ctx + c.reserve_bytes + GIB;
    h.ram_available = full;
    let best = plan_for("x", m, &h, c, cal, 1.0, true)?;
    if best.tok_s.mid < target {
        return None;
    }
    let (mut lo, mut hi) = (hw.ram_available, full);
    for _ in 0..40 {
        let mid = lo + (hi - lo) / 2;
        h.ram_available = mid;
        match plan_for("x", m, &h, c, cal, 1.0, true) {
            Some(p) if p.tok_s.mid >= target => hi = mid,
            _ => lo = mid,
        }
    }
    Some(hi.saturating_sub(hw.ram_available))
}

pub fn plan(m: &ModelProfile, hw: &Hardware, c: &Constraints, cal: &Calibration) -> Report {
    let mut plans = Vec::new();
    let mut warnings = Vec::new();
    if m.n_experts == 0 || m.n_moe_layers == 0 {
        warnings.push("dense model: no experts to stream; the plan is resident or nothing".into());
    }
    let exact = plan_for("Exact", m, hw, c, cal, 1.0, true);
    if let Some(p) = &exact {
        plans.push(p.clone());
    }
    // Balanced: hot-set copies at about 3 bpw when the native experts are wider.
    let native_bpw = m.layers.first().map(|l| l.bpw).unwrap_or(0.0);
    let streaming = exact.as_ref().map(|p| p.hot_experts < m.n_experts * m.n_moe_layers).unwrap_or(true);
    if m.n_moe_layers > 0 && native_bpw > 3.4 && streaming {
        if let Some(b) = plan_for("Balanced (ladder)", m, hw, c, cal, 3.06 / native_bpw, false) {
            plans.push(b);
        }
    }
    let feasible = !plans.is_empty();
    let best = plans.iter().map(|p| p.tok_s.mid).fold(0.0, f64::max);
    let verdict = if !feasible {
        let need = m.dense_read_bytes + m.embedding_bytes + m.kv_bytes_per_token * c.ctx + c.reserve_bytes;
        format!(
            "Will not run: the always-resident part alone needs {:.1} GB of RAM and {:.1} GB is available. \
             Try a smaller context, a sibling model, or LAN pooling.",
            need as f64 / GB,
            hw.ram_available as f64 / GB
        )
    } else if best >= c.min_tok_s {
        format!("Runs. Best plan about {best:.1} tok/s against a target of {:.1}.", c.min_tok_s)
    } else {
        match ram_needed_for(m, hw, c, cal, c.min_tok_s) {
            Some(extra) => format!(
                "Runs below target: about {best:.2} tok/s against {:.1}. Reaching the target needs about {:.0} GB more RAM, \
                 or slow mode.",
                c.min_tok_s,
                extra as f64 / GB
            ),
            None => format!(
                "Runs below target: about {best:.2} tok/s against {:.1}, and no amount of RAM on this CPU reaches it \
                 (bandwidth-bound). Use slow mode, a GPU, or a sibling model.",
                c.min_tok_s
            ),
        }
    };
    if m.has_mtp {
        warnings.push("model has an MTP head: speculation can add throughput (not in this estimate)".into());
    }
    Report { model: m.name.clone(), feasible, plans, verdict, warnings }
}

/// Fit time_scale from one measured run of the exact plan.
pub fn calibrate(m: &ModelProfile, hw: &Hardware, c: &Constraints, measured_tok_s: f64, label: &str) -> Calibration {
    let base = Calibration::default();
    let pred = plan_for("x", m, hw, c, &base, 1.0, true).map(|p| p.tok_s.mid).unwrap_or(measured_tok_s);
    Calibration { time_scale: pred / measured_tok_s, calibrated_on: Some(label.to_string()), ..base }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::LayerExperts;

    fn v4_like() -> ModelProfile {
        // DeepSeek V4 Flash shape (SPEC-DERISK section 1) with MXFP4 experts.
        let eb = 13_370_000u64;
        ModelProfile {
            name: "v4-like".into(),
            arch: "deepseek4".into(),
            n_layers: 43,
            n_moe_layers: 43,
            n_experts: 256,
            top_k: 6,
            n_shared: 1,
            total_bytes: 7_870_000_000 + eb * 256 * 43,
            dense_read_bytes: 7_870_000_000,
            embedding_bytes: 0,
            routed_bytes: eb * 256 * 43,
            layers: (0..43).map(|l| LayerExperts { layer: l, expert_bytes: eb, n_experts: 256, type_name: "MXFP4".into(), bpw: 4.25 }).collect(),
            expert_bytes: eb,
            routed_read_bytes: eb * 6 * 43,
            kv_bytes_per_token: 4000,
            context_length: 1 << 20,
            has_mtp: true,
            params: 284_000_000_000,
            active_params: 13_000_000_000,
        }
    }

    fn box32() -> Hardware {
        Hardware {
            cpu: "test".into(),
            cores: 16,
            ram_total: 32 * GIB,
            ram_available: 30 * GIB,
            unified_memory: false,
            mem_bw: 70e9,
            cpu_flops: 2e12,
            disk_bw: Some(6e9),
            disk_measured_direct: true,
            gpus: vec![],
            probe_seconds: 0.0,
        }
    }

    #[test]
    fn hit_model_matches_evidence() {
        assert_eq!(hit_model(0.0, 0.2), 0.0);
        assert_eq!(hit_model(1.0, 0.2), 1.0);
        let a = hit_exponent(6, 256);
        // colibri issue 905: V4 Flash, 12% of the pool cached, 65.3% hit
        assert!((hit_model(0.12, a) - 0.653).abs() < 0.05, "{}", hit_model(0.12, a));
        // SPEC-DERISK estimate: 35% cached, 80 to 88% hit
        let h = hit_model(0.35, a);
        assert!((0.78..=0.9).contains(&h), "{h}");
        // Granite pilot (8 of 32): GDSF 0.53 at 25%, 0.82 at 50%
        let g = hit_exponent(8, 32);
        assert!((hit_model(0.25, g) - 0.53).abs() < 0.12);
        assert!((hit_model(0.5, g) - 0.82).abs() < 0.08);
    }

    #[test]
    fn v4_on_32gb_is_streaming_and_below_10() {
        let r = plan(&v4_like(), &box32(), &Constraints { min_tok_s: 10.0, ..Default::default() }, &Calibration::default());
        assert!(r.feasible);
        let ex = &r.plans[0];
        assert!(ex.hot_experts < 256 * 43);
        assert!(ex.tok_s.mid < 10.0, "{}", ex.tok_s.mid);
        assert!(ex.tok_s.low <= ex.tok_s.mid && ex.tok_s.mid <= ex.tok_s.high);
        assert!(r.verdict.contains("below target"));
    }

    #[test]
    fn resident_ceiling_matches_derisk() {
        // all experts resident: decode is bound by dense 7.87 GB + routed 3.45 GB at ~60 GB/s
        let mut hw = box32();
        hw.ram_available = 300 * GIB;
        let r = plan(&v4_like(), &hw, &Constraints::default(), &Calibration::default());
        let ex = &r.plans[0];
        assert_eq!(ex.hot_experts, 256 * 43);
        // additive compute: 11.3 GB / 59.5 GB/s + 26 GFLOP / 360 GFLOP/s, about 3.8 tok/s
        assert!(ex.tok_s.mid > 3.3 && ex.tok_s.mid < 4.5, "{}", ex.tok_s.mid);
    }

    #[test]
    fn refuses_when_dense_does_not_fit() {
        let mut hw = box32();
        hw.ram_available = 6 * GIB;
        let r = plan(&v4_like(), &hw, &Constraints::default(), &Calibration::default());
        assert!(!r.feasible);
        assert!(r.verdict.contains("Will not run"));
    }

    #[test]
    fn more_ram_never_slower() {
        let mut last = 0.0;
        for gb in [12u64, 16, 24, 32, 64, 128] {
            let mut hw = box32();
            hw.ram_available = gb * GIB;
            let r = plan(&v4_like(), &hw, &Constraints::default(), &Calibration::default());
            if let Some(p) = r.plans.first() {
                assert!(p.tok_s.mid >= last - 1e-9);
                last = p.tok_s.mid;
            }
        }
    }

    #[test]
    fn usage_hit_rate() {
        let usage = vec![vec![3.0, 1.0, 1.0, 1.0], vec![0.5, 0.5, 2.5, 2.5]];
        let (h, kept) = hit_from_usage(&usage, &[10, 10], 20);
        // keeps the 3.0 expert of layer 0 and one 2.5 expert of layer 1: 5.5 of 12 accesses
        assert_eq!(kept, 2);
        assert!((h - 5.5 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn calibration_reproduces_measurement() {
        let m = v4_like();
        let mut hw = box32();
        hw.ram_available = 300 * GIB;
        let c = Constraints::default();
        let cal = calibrate(&m, &hw, &c, 4.0, "test");
        let r = plan(&m, &hw, &c, &cal);
        assert!((r.plans[0].tok_s.mid - 4.0).abs() < 1e-6);
    }
}
