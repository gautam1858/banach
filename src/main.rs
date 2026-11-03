//! banach: weights-free preflight, hardware probe and planner (Phase 1).

mod gguf;
mod mckp;
mod planner;
mod probe;
mod profile;
mod source;

use planner::{Calibration, Constraints, Plan, Report, GB};
use probe::Hardware;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
usage:
  banach probe [--disk FILE] [--quick] [--mem-mb N]
      measure this machine and save the profile
  banach inspect MODEL [--json]
      read the model header only (local path, https URL, or hf:owner/repo/file.gguf)
  banach plan MODEL [--ctx N] [--min-tok-s X] [--usage FILE] [--hw FILE] [--json]
      (--ctx is the context length decode runs at; attention cost grows with it)
      preflight card: predicted speed and fidelity per plan, before downloading weights
      (--hw plans for another machine from its saved hardware.json)
  banach run MODEL (-p TEXT | -f FILE) [-n N] [--ctx N] [--budget-mb N] [--exact-check]
      plan, then decode with banach-engine using the planned RAM budget for routed experts
      (engine binary: $BANACH_ENGINE, next to this binary, or on PATH)
  banach calibrate MODEL --measured TOK_S [--label TEXT]
      fit the cost model to one measured decode run of MODEL on this machine

Split models: pass the first part (...-00001-of-0000N.gguf); the other parts are found.
State lives in $BANACH_HOME (default ~/.cache/banach).";

fn home() -> PathBuf {
    if let Ok(h) = std::env::var("BANACH_HOME") {
        return PathBuf::from(h);
    }
    let base = std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
    base.join(".cache").join("banach")
}

fn save_json<T: serde::Serialize>(path: &Path, v: &T) -> Result<(), String> {
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    let s = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let s = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&s).ok()
}

struct Args {
    pos: Vec<String>,
    flags: std::collections::HashMap<String, String>,
    switches: std::collections::HashSet<String>,
}

fn parse_args(raw: &[String], with_value: &[&str]) -> Result<Args, String> {
    let mut a = Args { pos: vec![], flags: Default::default(), switches: Default::default() };
    let mut i = 0;
    while i < raw.len() {
        let s = &raw[i];
        let name = s.strip_prefix("--").or_else(|| s.strip_prefix('-').filter(|n| with_value.contains(n)));
        if let Some(name) = name {
            if with_value.contains(&name) {
                let v = raw.get(i + 1).ok_or_else(|| format!("--{name} needs a value"))?;
                a.flags.insert(name.to_string(), v.clone());
                i += 1;
            } else {
                a.switches.insert(name.to_string());
            }
        } else {
            a.pos.push(s.clone());
        }
        i += 1;
    }
    Ok(a)
}

fn num<T: std::str::FromStr>(a: &Args, k: &str, d: T) -> Result<T, String> {
    match a.flags.get(k) {
        Some(v) => v.parse().map_err(|_| format!("--{k}: bad number {v:?}")),
        None => Ok(d),
    }
}

fn hardware(path: &Path) -> Result<Hardware, String> {
    match load_json::<Hardware>(path) {
        Some(h) => Ok(h),
        None => {
            eprintln!("no saved hardware profile; running a quick probe (use `banach probe` for a full one)");
            let h = probe::probe(&probe::ProbeOptions { mem_bytes: 512 << 20, disk_file: None, quick: true });
            save_json(path, &h)?;
            Ok(h)
        }
    }
}

struct Loaded {
    profile: profile::ModelProfile,
    requests: u64,
    header_bytes: u64,
    /// Bytes fetched past the start of tensor data (read-ahead of the last header chunk).
    weight_bytes: u64,
    parts: usize,
}

fn load_model(spec: &str) -> Result<Loaded, String> {
    let specs = source::split_parts(spec);
    let mut parts = Vec::new();
    let mut requests = 0;
    let mut header_bytes = 0;
    let mut weight_bytes = 0;
    for (i, s) in specs.iter().enumerate() {
        let mut src = source::resolve(s)?;
        let g = gguf::parse(src.as_mut(), i).map_err(|e| format!("{}: {e}", src.describe()))?;
        requests += src.requests();
        header_bytes += g.header_bytes;
        weight_bytes += g.header_bytes.saturating_sub(g.data_offset);
        parts.push(g);
    }
    let profile = profile::build(&parts)?;
    Ok(Loaded { profile, requests, header_bytes, weight_bytes, parts: parts.len() })
}

fn header_line(l: &Loaded) -> String {
    let req = if l.requests > 0 { format!(" in {} request(s)", l.requests) } else { String::new() };
    format!("{:.2} MB{req}, {:.2} MB past the header", l.header_bytes as f64 / 1e6, l.weight_bytes as f64 / 1e6)
}

fn gb(b: f64) -> String {
    format!("{:.1} GB", b / GB)
}

fn cmd_probe(a: &Args) -> Result<(), String> {
    let disk = a.flags.get("disk").map(PathBuf::from);
    let mem_mb: usize = num(a, "mem-mb", 1024)?;
    let h = probe::probe(&probe::ProbeOptions {
        mem_bytes: mem_mb << 20,
        disk_file: disk.as_deref(),
        quick: a.switches.contains("quick"),
    });
    let path = home().join("hardware.json");
    save_json(&path, &h)?;
    println!("cpu         {} ({} threads)", h.cpu, h.cores);
    println!("ram         {} total, {} available", gb(h.ram_total as f64), gb(h.ram_available as f64));
    println!("memory bw   {:.1} GB/s (measured, streaming read)", h.mem_bw / GB);
    println!("cpu         {:.0} GFLOP/s f32 fma (measured)", h.cpu_flops / 1e9);
    match h.disk_bw {
        Some(bw) => println!(
            "disk        {:.2} GB/s random 8 MiB reads{}",
            bw / GB,
            if h.disk_measured_direct { " (O_DIRECT)" } else { " (buffered: page cache may inflate this)" }
        ),
        None => println!("disk        not measured (pass --disk <model file>)"),
    }
    for g in &h.gpus {
        println!(
            "gpu         {} {}{}",
            g.name,
            gb(g.vram_bytes as f64),
            g.bandwidth.map(|b| format!(", ~{:.0} GB/s", b / GB)).unwrap_or_else(|| ", bandwidth unknown".into())
        );
    }
    println!("probe took  {:.1} s; saved {}", h.probe_seconds, path.display());
    Ok(())
}

fn cmd_inspect(a: &Args) -> Result<(), String> {
    let spec = a.pos.first().ok_or("inspect needs a MODEL")?;
    let l = load_model(spec)?;
    if a.switches.contains("json") {
        println!("{}", serde_json::to_string_pretty(&l.profile).unwrap());
        return Ok(());
    }
    let p = &l.profile;
    println!("model        {} ({})", p.name, p.arch);
    println!("size         {} in {} file(s), {:.1}B params, {:.1}B active", gb(p.total_bytes as f64), l.parts, p.params as f64 / 1e9, p.active_params as f64 / 1e9);
    println!("layers       {} ({} MoE), {} experts, top-{}, {} shared", p.n_layers, p.n_moe_layers, p.n_experts, p.top_k, p.n_shared);
    if let Some(l0) = p.layers.first() {
        println!("experts      {:.2} MB each, {} ({:.2} bpw)", p.expert_bytes as f64 / 1e6, l0.type_name, l0.bpw);
    }
    println!("per token    dense {} + routed {}", gb(p.dense_read_bytes as f64), gb(p.routed_read_bytes as f64));
    println!("kv           {:.1} KB/token (estimate)", p.kv_bytes_per_token as f64 / 1e3);
    println!("header read  {}", header_line(&l));
    Ok(())
}

fn card(r: &Report, l: &Loaded, hw: &Hardware, c: &Constraints) -> String {
    let p = &l.profile;
    let mut o = String::new();
    let line = "+------------------------------------------------------------------------+\n";
    o += line;
    o += &format!("| {:<70} |\n", format!("{}  ({})", p.name, p.arch));
    o += &format!("| {:<70} |\n", format!("{} on disk, {} experts x {} layers, top-{}", gb(p.total_bytes as f64), p.n_experts, p.n_moe_layers, p.top_k));
    o += &format!("| {:<70} |\n", format!("machine: {} RAM free, {:.0} GB/s memory, ctx {}", gb(hw.ram_available as f64), hw.mem_bw / GB, c.ctx));
    o += line;
    for pl in &r.plans {
        o += &plan_lines(pl);
        o += line;
    }
    o += &wrap(&r.verdict, 70).iter().map(|s| format!("| {s:<70} |\n")).collect::<String>();
    for w in &r.warnings {
        o += &wrap(&format!("note: {w}"), 70).iter().map(|s| format!("| {s:<70} |\n")).collect::<String>();
    }
    o += &format!("| {:<70} |\n", format!("header {}", header_line(l)));
    o += line;
    o
}

fn plan_lines(pl: &Plan) -> String {
    let mut o = String::new();
    let t = &pl.tok_s;
    o += &format!("| {:<70} |\n", format!("{:<18} {:>6.2} tok/s  (range {:.2} to {:.2})", pl.name, t.mid, t.low, t.high));
    for s in wrap(&format!("fidelity: {}", pl.fidelity), 68) {
        o += &format!("|   {s:<68} |\n");
    }
    let h = &pl.hit_rate;
    o += &format!(
        "|   {:<68} |\n",
        format!("hot set {} experts at {:.2} bpw, hit {:.0}% ({:.0} to {:.0}%)", pl.hot_experts, pl.expert_bpw, h.mid * 100.0, h.low * 100.0, h.high * 100.0)
    );
    o += &format!(
        "|   {:<68} |\n",
        format!("RAM {}, VRAM {}, NVMe {:.2} GB/token", gb(pl.bytes_ram as f64), gb(pl.bytes_vram as f64), pl.nvme_bytes_per_token / GB)
    );
    o += &format!("|   {:<68} |\n", format!("download now {}, disk needed {}", gb(pl.download_first as f64), gb(pl.disk_needed as f64)));
    for n in &pl.notes {
        for s in wrap(&format!("- {n}"), 68) {
            o += &format!("|   {s:<68} |\n");
        }
    }
    o
}

fn wrap(s: &str, w: usize) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > w {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn constraints(a: &Args) -> Result<Constraints, String> {
    let mut c = Constraints { ctx: num(a, "ctx", 8192)?, min_tok_s: num(a, "min-tok-s", 5.0)?, ..Default::default() };
    if let Some(u) = a.flags.get("usage") {
        let s = std::fs::read_to_string(u).map_err(|e| format!("{u}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| format!("{u}: {e}"))?;
        let rows: Vec<Vec<f64>> = serde_json::from_value(v["usage"].clone()).map_err(|e| format!("{u}: usage: {e}"))?;
        c.usage = Some(rows);
    }
    Ok(c)
}

fn cal_path(model: &profile::ModelProfile) -> PathBuf {
    let safe: String = model.name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    home().join(format!("calibration-{safe}.json"))
}

fn cmd_plan(a: &Args) -> Result<(), String> {
    let spec = a.pos.first().ok_or("plan needs a MODEL")?;
    let hw = match a.flags.get("hw") {
        Some(p) => load_json::<Hardware>(Path::new(p)).ok_or_else(|| format!("{p}: not a hardware profile"))?,
        None => hardware(&home().join("hardware.json"))?,
    };
    let l = load_model(spec)?;
    let c = constraints(a)?;
    if let Some(u) = &c.usage {
        if u.len() as u64 != l.profile.n_moe_layers {
            return Err(format!("usage has {} layers, model has {} MoE layers", u.len(), l.profile.n_moe_layers));
        }
    }
    let cal = load_json::<Calibration>(&cal_path(&l.profile))
        .or_else(|| load_json::<Calibration>(&home().join("calibration.json")))
        .unwrap_or_default();
    let r = planner::plan(&l.profile, &hw, &c, &cal);
    if a.switches.contains("json") {
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
    } else {
        print!("{}", card(&r, &l, &hw, &c));
    }
    Ok(())
}

fn cmd_calibrate(a: &Args) -> Result<(), String> {
    let spec = a.pos.first().ok_or("calibrate needs a MODEL")?;
    let measured: f64 = num(a, "measured", 0.0)?;
    if measured <= 0.0 {
        return Err("--measured TOK_S is required".into());
    }
    let hw = hardware(&home().join("hardware.json"))?;
    let l = load_model(spec)?;
    let c = constraints(a)?;
    let label = a.flags.get("label").cloned().unwrap_or_else(|| format!("{} measured {measured} tok/s", l.profile.name));
    let cal = planner::calibrate(&l.profile, &hw, &c, measured, &label);
    let path = home().join("calibration.json");
    save_json(&path, &cal)?;
    save_json(&cal_path(&l.profile), &cal)?;
    println!("time scale {:.3} (uncalibrated prediction / measured); saved {}", cal.time_scale, path.display());
    Ok(())
}

fn engine_path() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("BANACH_ENGINE") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(me) = std::env::current_exe() {
        let side = me.with_file_name("banach-engine");
        if side.exists() {
            return Ok(side);
        }
    }
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let c = Path::new(dir).join("banach-engine");
        if c.exists() {
            return Ok(c);
        }
    }
    Err("banach-engine not found: build engine/ and set BANACH_ENGINE or put it next to banach".into())
}

fn cmd_run(a: &Args) -> Result<(), String> {
    let spec = a.pos.first().ok_or("run needs a local MODEL path")?;
    if spec.starts_with("hf:") || spec.starts_with("http") {
        return Err("run needs the model on local disk (lazy download arrives in Phase 4)".into());
    }
    let hw = hardware(&home().join("hardware.json"))?;
    let l = load_model(spec)?;
    let c = constraints(a)?;
    let cal = load_json::<Calibration>(&cal_path(&l.profile)).unwrap_or_default();
    let r = planner::plan(&l.profile, &hw, &c, &cal);
    let exact = r.plans.iter().find(|p| p.lossless).ok_or_else(|| r.verdict.clone())?;
    let budget_mb: f64 = match a.flags.get("budget-mb") {
        Some(v) => v.parse().map_err(|_| "bad --budget-mb")?,
        None => {
            let experts = exact.hot_experts as f64 * l.profile.expert_bytes as f64;
            if exact.hot_experts >= l.profile.n_experts * l.profile.n_moe_layers { 0.0 } else { experts / 1e6 }
        }
    };
    eprintln!(
        "plan: Exact, predicted {:.2} tok/s ({:.2} to {:.2}); expert budget {}",
        exact.tok_s.mid,
        exact.tok_s.low,
        exact.tok_s.high,
        if budget_mb > 0.0 { format!("{budget_mb:.0} MB") } else { "unlimited (fits)".into() }
    );
    let mut cmd = std::process::Command::new(engine_path()?);
    cmd.arg("-m").arg(spec).arg("-c").arg(c.ctx.to_string());
    if let Some(p) = a.flags.get("p") {
        cmd.arg("-p").arg(p);
    }
    if let Some(f) = a.flags.get("f") {
        cmd.arg("-f").arg(f);
    }
    if let Some(n) = a.flags.get("n") {
        cmd.arg("-n").arg(n);
    }
    if budget_mb > 0.0 {
        cmd.arg("--budget-mb").arg(format!("{budget_mb:.0}"));
    }
    if a.switches.contains("exact-check") {
        cmd.arg("--exact-check");
    }
    let heat = home().join(format!("heat-{}.txt", l.profile.name.replace(|c: char| !c.is_ascii_alphanumeric(), "_")));
    if heat.exists() {
        cmd.arg("--heat-in").arg(&heat);
    }
    cmd.arg("--heat-out").arg(&heat);
    let status = cmd.status().map_err(|e| format!("starting banach-engine: {e}"))?;
    if !status.success() {
        return Err(format!("banach-engine exited with {status}"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first().cloned() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &raw[1..];
    let res = match cmd.as_str() {
        "probe" => parse_args(rest, &["disk", "mem-mb"]).and_then(|a| cmd_probe(&a)),
        "inspect" => parse_args(rest, &[]).and_then(|a| cmd_inspect(&a)),
        "plan" => parse_args(rest, &["ctx", "min-tok-s", "usage", "hw"]).and_then(|a| cmd_plan(&a)),
        "run" => parse_args(rest, &["ctx", "min-tok-s", "usage", "p", "f", "n", "budget-mb"]).and_then(|a| cmd_run(&a)),
        "calibrate" => parse_args(rest, &["ctx", "measured", "label", "usage"]).and_then(|a| cmd_calibrate(&a)),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("banach: {e}");
            ExitCode::FAILURE
        }
    }
}
