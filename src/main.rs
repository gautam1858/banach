//! banach: weights-free preflight, hardware probe and planner (Phase 1).

mod catalog;
mod fidelity;
mod gguf;
mod mckp;
mod planner;
mod probe;
mod pull;
mod serve;
mod profile;
mod source;
mod tools;

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
  banach pick [MODEL ...] [--hw FILE] [--ctx N] [--min-tok-s X] [--json]
      rank models by the speed predicted for this machine, biggest model that is fast enough first
      (no MODEL: the built-in list of large open MoE models; only headers are read)
  banach compare REFERENCE.gguf CANDIDATE.gguf (-f FILE | -p TEXT) [--min-margin M] [--json]
      how far a candidate model (another quantization, a pruned pack) drifts from the reference on
      the same text: top-1 agreement, KL, and a 95% upper bound on the flip rate
  banach run MODEL (-p TEXT | -f FILE) [-n N] [--ctx N] [--budget-mb N] [--gpu-layers N] [--exact-check]
      plan, then decode with banach-engine using the planned RAM budget for routed experts
      (engine binary: $BANACH_ENGINE, next to this binary, or on PATH)
  banach pull hf:OWNER/REPO/FILE.gguf [--out DIR] [--usage FILE] [--workers N]
      resumable download: header and dense tensors first, hot experts next (with --usage),
      SHA-256 checked against the Hugging Face LFS object id
  banach serve MODEL [--host H] [--port P] [--ctx N] [--budget-mb N] [--gpu-layers N] [--lazy]
               [--ladder LOW.gguf --eps FILE --tau T --target P --min-margin M --period N]
      OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) compatible server on
      127.0.0.1:8080; MODEL may be hf:... (downloaded first, or served while downloading
      with --lazy); with a ladder, shadow replays feed the fidelity meter and tau controller
  banach certify FLIPS [--target P] [--alpha A] [--min-margin M]
      anytime-valid certificate that the top-1 flip rate is below P (default 0.01), from a
      banach-engine --flips-out file (columns: flip, exact margin)
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

fn cmd_pick(a: &Args) -> Result<(), String> {
    let hw = match a.flags.get("hw") {
        Some(p) => load_json::<Hardware>(Path::new(p)).ok_or_else(|| format!("{p}: not a hardware profile"))?,
        None => hardware(&home().join("hardware.json"))?,
    };
    let c = constraints(a)?;
    let candidates: Vec<(String, String)> = if a.pos.is_empty() {
        catalog::MODELS.iter().map(|(l, s)| (l.to_string(), s.to_string())).collect()
    } else {
        a.pos.iter().map(|s| (s.clone(), s.clone())).collect()
    };

    struct Row {
        label: String,
        params: u64,
        size: u64,
        tok_s: f64,
        low: f64,
        high: f64,
    }
    // header reads are network-bound, so read them side by side
    let results: Vec<(String, Result<Row, String>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = candidates
            .iter()
            .map(|(label, spec)| {
                let (hw, c) = (&hw, &c);
                sc.spawn(move || {
                    let row = load_model(spec).and_then(|l| {
                        let cal = load_json::<Calibration>(&cal_path(&l.profile))
                            .or_else(|| load_json::<Calibration>(&home().join("calibration.json")))
                            .unwrap_or_default();
                        let r = planner::plan(&l.profile, hw, c, &cal);
                        let exact = r.plans.iter().find(|p| p.lossless).ok_or_else(|| r.verdict.clone())?;
                        Ok(Row {
                            label: label.clone(),
                            params: l.profile.params,
                            size: l.profile.total_bytes,
                            tok_s: exact.tok_s.mid,
                            low: exact.tok_s.low,
                            high: exact.tok_s.high,
                        })
                    });
                    (label.clone(), row)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| ("?".into(), Err("worker panicked".into())))).collect()
    });

    let mut rows = Vec::new();
    for (label, r) in results {
        match r {
            Ok(row) => rows.push(row),
            Err(e) => eprintln!("skipped {label}: {e}"),
        }
    }
    // models that meet the target, biggest first; then the rest, fastest first
    let (mut fast, mut slow): (Vec<Row>, Vec<Row>) = rows.into_iter().partition(|r| r.tok_s >= c.min_tok_s);
    fast.sort_by(|x, y| y.params.cmp(&x.params));
    slow.sort_by(|x, y| y.tok_s.partial_cmp(&x.tok_s).unwrap_or(std::cmp::Ordering::Equal));

    if a.switches.contains("json") {
        let to_json = |r: &Row, ok: bool| {
            serde_json::json!({"model": r.label, "params": r.params, "bytes": r.size, "tok_s": r.tok_s,
                "tok_s_low": r.low, "tok_s_high": r.high, "meets_target": ok})
        };
        let all: Vec<_> = fast.iter().map(|r| to_json(r, true)).chain(slow.iter().map(|r| to_json(r, false))).collect();
        println!("{}", serde_json::to_string_pretty(&all).unwrap());
        return Ok(());
    }
    println!(
        "machine: {}, {} RAM, {} memory bandwidth, {} disk",
        hw.cpu,
        gb(hw.ram_available as f64),
        format!("{:.0} GB/s", hw.mem_bw / 1e9),
        hw.disk_bw.map(|d| format!("{:.1} GB/s", d / 1e9)).unwrap_or_else(|| "unmeasured".into())
    );
    println!("exact weights only; speeds are predictions from headers (ctx {}), target {:.1} tok/s\n", c.ctx, c.min_tok_s);
    let print = |title: &str, rows: &[Row]| {
        println!("{title}");
        if rows.is_empty() {
            println!("  (none)\n");
            return;
        }
        println!("  {:<34} {:>8} {:>9} {:>11}   {}", "model", "params", "on disk", "tok/s", "range");
        for r in rows {
            println!(
                "  {:<34} {:>7.0}B {:>9} {:>11.1}   {:.1} to {:.1}",
                r.label,
                r.params as f64 / 1e9,
                gb(r.size as f64),
                r.tok_s,
                r.low,
                r.high
            );
        }
        println!();
    };
    print(&format!("Fast enough (at least {:.1} tok/s), biggest first:", c.min_tok_s), &fast);
    print("Runs, but slower than the target, fastest first:", &slow);
    if let Some(best) = fast.first() {
        println!("Pick: {}. Try it: banach serve <its hf: file> --lazy", best.label);
    } else if let Some(best) = slow.first() {
        println!("Nothing reaches the target. Fastest is {} at {:.1} tok/s.", best.label, best.tok_s);
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
    let name = format!("banach-engine{}", std::env::consts::EXE_SUFFIX);
    if let Ok(me) = std::env::current_exe() {
        let side = me.with_file_name(&name);
        if side.exists() {
            return Ok(side);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let c = dir.join(&name);
            if c.exists() {
                return Ok(c);
            }
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
    if let Some(g) = a.flags.get("gpu-layers") {
        cmd.arg("--gpu-layers").arg(g);
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

fn cmd_compare(a: &Args) -> Result<(), String> {
    let (Some(reference), Some(candidate)) = (a.pos.first(), a.pos.get(1)) else {
        return Err("compare needs REFERENCE.gguf CANDIDATE.gguf".into());
    };
    if a.flags.get("f").is_none() && a.flags.get("p").is_none() {
        return Err("compare needs the text to score: -f FILE or -p TEXT".into());
    }
    let tmp = std::env::temp_dir().join(format!("banach-compare-{}", std::process::id()));
    let mut cmd = std::process::Command::new(engine_path()?);
    cmd.arg("-m").arg(reference).arg("--compare").arg(candidate).arg("--flips-out").arg(&tmp);
    cmd.arg("-c").arg(num::<u64>(a, "ctx", 4096)?.to_string());
    for k in ["p", "f", "min-margin"] {
        if let Some(v) = a.flags.get(k) {
            cmd.arg(if k == "min-margin" { "--min-margin".to_string() } else { format!("-{k}") }).arg(v);
        }
    }
    let out = cmd.stderr(std::process::Stdio::inherit()).output().map_err(|e| format!("starting banach-engine: {e}"))?;
    if !out.status.success() {
        return Err(format!("banach-engine exited with {}", out.status));
    }
    let stats: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("engine output: {e}"))?;
    let flips_file = format!("{}.txt", tmp.display());
    let text = std::fs::read_to_string(&flips_file).map_err(|e| format!("{flips_file}: {e}"))?;
    let _ = std::fs::remove_file(&flips_file);
    let rows: Vec<(bool, f64)> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let f = it.next()? == "1";
            Some((f, it.next().and_then(|m| m.parse().ok()).unwrap_or(f64::INFINITY)))
        })
        .collect();
    let alpha: f64 = num(a, "alpha", 0.05)?;
    let mm: f64 = num(a, "min-margin", 0.1)?;
    let all = fidelity::certify(&rows, 0.0, alpha, None);
    let confident = fidelity::certify(&rows, 0.0, alpha, Some(mm));
    let agree = stats["top1_agreement"].as_f64().unwrap_or(0.0);
    let result = serde_json::json!({
        "reference": reference,
        "candidate": candidate,
        "tokens": stats["tokens"],
        "top1_agreement": agree,
        "kl_mean": stats["kl_mean"],
        "flip_rate": all.point(),
        "flip_rate_upper_95": all.ucb(),
        "confident_tokens": stats["confident_tokens"],
        "confident_flip_rate": confident.point(),
        "confident_flip_rate_upper_95": confident.ucb(),
        "min_margin": mm,
        "note": "upper bounds treat tokens as independent; consecutive tokens are correlated, so read them as optimistic",
    });
    if a.switches.contains("json") {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!(
            "{} tokens scored\n  top-1 agreement with the reference: {:.1}%  (answers change on {:.1}% of tokens, at most {:.1}% with 95% confidence)\n  KL(reference || candidate): {:.4} per token\n  tokens the reference was sure of (margin >= {} nats): {}, flips {:.1}% (at most {:.1}%)\n  note: bounds treat tokens as independent; consecutive tokens are correlated",
            stats["tokens"], agree * 100.0, all.point() * 100.0, all.ucb() * 100.0,
            stats["kl_mean"].as_f64().unwrap_or(0.0), mm, stats["confident_tokens"],
            confident.point() * 100.0, confident.ucb() * 100.0
        );
    }
    Ok(())
}

fn cmd_certify(a: &Args) -> Result<(), String> {
    let path = a.pos.first().ok_or("certify needs a FLIPS file")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        let Some(f) = it.next() else { continue };
        let flip = match f {
            "0" => false,
            "1" => true,
            _ => return Err(format!("{path}:{}: expected 0 or 1, got {f:?}", i + 1)),
        };
        let margin: f64 = it.next().map(|m| m.parse().unwrap_or(f64::INFINITY)).unwrap_or(f64::INFINITY);
        rows.push((flip, margin));
    }
    let target: f64 = num(a, "target", 0.01)?;
    let alpha: f64 = num(a, "alpha", 0.05)?;
    let mm = match a.flags.get("min-margin") {
        Some(v) => Some(v.parse::<f64>().map_err(|_| "bad --min-margin")?),
        None => None,
    };
    let m = fidelity::certify(&rows, target, alpha, mm);
    let out = serde_json::json!({
        "tokens": m.tokens,
        "flips": m.flips,
        "point_estimate": m.point(),
        "ucb": m.ucb(),
        "target": target,
        "alpha": alpha,
        "min_margin": mm,
        "certified": m.certified_at().is_some(),
        "certified_at_tokens": m.certified_at(),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}

fn models_dir(a: &Args) -> PathBuf {
    a.flags.get("out").map(PathBuf::from).unwrap_or_else(|| home().join("models"))
}

fn read_usage(a: &Args) -> Result<Option<Vec<Vec<f64>>>, String> {
    match a.flags.get("usage") {
        None => Ok(None),
        Some(u) => {
            let s = std::fs::read_to_string(u).map_err(|e| format!("{u}: {e}"))?;
            let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| format!("{u}: {e}"))?;
            Ok(Some(serde_json::from_value(v["usage"].clone()).map_err(|e| format!("{u}: usage: {e}"))?))
        }
    }
}

fn start_pull(spec: &str, a: &Args) -> Result<std::sync::Arc<pull::Puller>, String> {
    let usage = read_usage(a)?;
    let p = pull::Puller::new(spec, &models_dir(a), usage.as_deref())?;
    let workers: usize = num(a, "workers", 4)?;
    let handles = p.spawn(workers);
    // detach the workers; progress is polled
    std::mem::forget(handles);
    Ok(p)
}

fn progress(p: &pull::Puller) {
    let (have, total) = (p.present_bytes(), p.total_bytes());
    eprint!("\rdownloaded {:.2} / {:.2} GB ({:.0}%)   ", have as f64 / GB, total as f64 / GB, 100.0 * have as f64 / total.max(1) as f64);
}

fn wait_for(p: &pull::Puller, cond: impl Fn(&pull::Puller) -> bool) -> Result<(), String> {
    while !cond(p) {
        if let Some(e) = p.error() {
            eprintln!();
            return Err(e);
        }
        progress(p);
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    progress(p);
    eprintln!();
    Ok(())
}

fn cmd_pull(a: &Args) -> Result<(), String> {
    let spec = a.pos.first().ok_or("pull needs hf:OWNER/REPO/FILE.gguf")?;
    let p = start_pull(spec, a)?;
    for part in &p.parts {
        eprintln!("{} -> {} ({:.2} GB, sha256 {})", part.url, part.path.display(), part.size as f64 / GB,
                  part.sha256.as_deref().unwrap_or("unknown"));
    }
    wait_for(&p, |p| p.complete())?;
    eprintln!("verifying SHA-256 ...");
    let paths = p.finish()?;
    for path in paths {
        println!("{}", path.display());
    }
    Ok(())
}

fn cmd_serve(a: &Args) -> Result<(), String> {
    let spec = a.pos.first().ok_or("serve needs a MODEL")?;
    let mut lazy = None;
    let model: PathBuf = if spec.starts_with("hf:") {
        let p = start_pull(spec, a)?;
        let first = p.parts[0].path.clone();
        if a.switches.contains("lazy") {
            eprintln!("waiting for the header and dense tensors (experts stream in afterwards)");
            wait_for(&p, |p| p.dense_ready())?;
            let bg = std::sync::Arc::clone(&p);
            std::thread::spawn(move || {
                while !bg.complete() && bg.error().is_none() {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
                match bg.finish() {
                    Ok(_) => eprintln!("download complete and verified"),
                    Err(e) => eprintln!("download: {e}"),
                }
            });
            lazy = Some(p);
        } else {
            wait_for(&p, |p| p.complete())?;
            p.finish()?;
        }
        first
    } else {
        let path = PathBuf::from(spec);
        if !pull::is_complete(&path) {
            return Err(format!("{spec} is incomplete (a download is unfinished); use the hf: name with --lazy"));
        }
        path
    };
    let model_id = model.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    let opts = serve::ServeOptions {
        engine: engine_path()?,
        model,
        model_id,
        host: a.flags.get("host").cloned().unwrap_or_else(|| "127.0.0.1".into()),
        port: num(a, "port", 8080)?,
        ctx: num(a, "ctx", 8192)?,
        threads: a.flags.get("threads").and_then(|t| t.parse().ok()),
        gpu_layers: a.flags.get("gpu-layers").and_then(|g| g.parse().ok()),
        budget_mb: a.flags.get("budget-mb").and_then(|b| b.parse().ok()),
        ladder: a.flags.get("ladder").map(PathBuf::from),
        eps: a.flags.get("eps").map(PathBuf::from),
        tau: num(a, "tau", 7.0)?,
        target: num(a, "target", 0.01)?,
        min_margin: num(a, "min-margin", 0.1)?,
        period: num(a, "period", 5000)?,
        shadow_every: num(a, "shadow-every", 1)?,
        lazy,
    };
    serve::serve(opts)
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
        "compare" => parse_args(rest, &["ctx", "p", "f", "min-margin", "alpha"]).and_then(|a| cmd_compare(&a)),
        "pick" => parse_args(rest, &["ctx", "min-tok-s", "hw"]).and_then(|a| cmd_pick(&a)),
        "run" => parse_args(rest, &["ctx", "min-tok-s", "usage", "p", "f", "n", "budget-mb", "gpu-layers"]).and_then(|a| cmd_run(&a)),
        "pull" => parse_args(rest, &["out", "usage", "workers"]).and_then(|a| cmd_pull(&a)),
        "serve" => parse_args(rest, &["host", "port", "ctx", "threads", "gpu-layers", "budget-mb", "ladder", "eps", "tau", "target",
                                      "min-margin", "period", "shadow-every", "out", "usage", "workers"])
            .and_then(|a| cmd_serve(&a)),
        "certify" => parse_args(rest, &["target", "alpha", "min-margin"]).and_then(|a| cmd_certify(&a)),
        "calibrate" => parse_args(rest, &["ctx", "measured", "label", "usage"]).and_then(|a| cmd_calibrate(&a)),
        "-V" | "--version" | "version" => {
            println!("banach {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
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
