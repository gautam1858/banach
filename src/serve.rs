//! Local HTTP server with OpenAI and Anthropic compatible endpoints (Phase 4).
//!
//!   POST /v1/chat/completions   OpenAI chat, stream or not
//!   GET  /v1/models             OpenAI model list
//!   POST /v1/messages           Anthropic messages, stream or not
//!   GET  /banach/status         engine, ladder and fidelity meter state
//!
//! Requests run one at a time on a persistent banach-engine worker (line-delimited JSON over
//! its stdin/stdout). With a ladder installed, every response is followed by a shadow replay
//! on the exact model; flips on confident tokens feed the betting meter, and the PI
//! controller moves tau so the meter's upper bound stays under the budget. Binds to
//! 127.0.0.1 unless told otherwise: the server is local by design.

use crate::fidelity::{Meter, TauController};
use crate::pull::Puller;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ServeOptions {
    pub engine: PathBuf,
    pub model: PathBuf,
    pub model_id: String,
    pub host: String,
    pub port: u16,
    pub ctx: u64,
    pub threads: Option<u32>,
    pub budget_mb: Option<f64>,
    pub ladder: Option<PathBuf>,
    pub eps: Option<PathBuf>,
    pub tau: f64,
    pub target: f64,
    pub min_margin: f64,
    pub period: u64,
    pub shadow_every: u64,
    pub lazy: Option<Arc<Puller>>,
}

// ---------------------------------------------------------------- engine worker

pub struct EngineProc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pub ready: Value,
    lazy: Option<Arc<Puller>>,
}

impl EngineProc {
    pub fn start(o: &ServeOptions) -> Result<Self, String> {
        let mut cmd = Command::new(&o.engine);
        cmd.arg("--serve").arg("-m").arg(&o.model).arg("-c").arg(o.ctx.to_string());
        if let Some(t) = o.threads {
            cmd.arg("-t").arg(t.to_string());
        }
        if let Some(b) = o.budget_mb {
            cmd.arg("--budget-mb").arg(format!("{b:.0}"));
        }
        if let Some(l) = &o.ladder {
            cmd.arg("--ladder").arg(l).arg("--tau").arg(o.tau.to_string());
        }
        if let Some(e) = &o.eps {
            cmd.arg("--eps").arg(e);
        }
        if o.lazy.is_some() {
            cmd.arg("--lazy");
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd.spawn().map_err(|e| format!("starting {}: {e}", o.engine.display()))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut me = Self { child, stdin, stdout, ready: Value::Null, lazy: o.lazy.clone() };
        // discovery in lazy mode reads only the dense path, which is on disk already
        let first = me.next_event()?;
        if first["event"] != "ready" {
            return Err(format!("engine did not start: {first}"));
        }
        me.ready = first;
        Ok(me)
    }

    fn send(&mut self, v: &Value) -> Result<(), String> {
        let mut s = v.to_string();
        s.push('\n');
        self.stdin.write_all(s.as_bytes()).and_then(|_| self.stdin.flush()).map_err(|e| format!("engine stdin: {e}"))
    }

    /// Next event; lazy "need" requests are served here and never surface.
    fn next_event(&mut self) -> Result<Value, String> {
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).map_err(|e| format!("engine stdout: {e}"))?;
            if n == 0 {
                let status = self.child.try_wait().ok().flatten();
                return Err(format!("engine exited ({status:?})"));
            }
            let v: Value = serde_json::from_str(line.trim()).map_err(|e| format!("bad engine output {line:?}: {e}"))?;
            if v["event"] == "need" {
                let p = self.lazy.clone().ok_or("engine asked for data but no download is running")?;
                for r in v["ranges"].as_array().cloned().unwrap_or_default() {
                    let si = r[0].as_u64().unwrap_or(0) as usize;
                    let off = r[1].as_u64().unwrap_or(0);
                    let len = r[2].as_u64().unwrap_or(0);
                    p.ensure(si, off, off + len)?;
                }
                self.send(&json!({"op": "have"}))?;
                continue;
            }
            return Ok(v);
        }
    }

    /// Send a request and pass every event for it to `on_event` until a terminal one.
    pub fn call(&mut self, req: &Value, mut on_event: impl FnMut(&Value)) -> Result<Value, String> {
        self.send(req)?;
        loop {
            let ev = self.next_event()?;
            match ev["event"].as_str() {
                Some("token") => on_event(&ev),
                Some("error") => return Err(ev["message"].as_str().unwrap_or("engine error").to_string()),
                _ => return Ok(ev),
            }
        }
    }
}

impl Drop for EngineProc {
    fn drop(&mut self) {
        let _ = self.send(&json!({"op": "quit"}));
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------- fidelity loop state

pub struct MeterState {
    pub meter: Meter,
    pub lifetime: Meter,
    pub controller: Option<TauController>,
    pub tau: f64,
    pub min_margin: f64,
    pub responses: u64,
    pub shadowed: u64,
    pub shadow_every: u64,
    pub tau_history: Vec<(u64, f64)>,
}

struct App {
    engine: Mutex<EngineProc>,
    meter: Mutex<MeterState>,
    opts: ServeOptions,
    ladder: bool,
    counter: AtomicU64,
}

/// Messages after API normalisation.
#[derive(Debug)]
struct Gen {
    messages: Vec<(String, String)>,
    max_tokens: u64,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    seed: Option<u64>,
    stop: Vec<String>,
    stream: bool,
    model: String,
}

#[derive(Debug)]
struct ApiError {
    status: u16,
    kind: &'static str,
    message: String,
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError { status: 400, kind: "invalid_request_error", message: msg.into() }
}

fn text_of(content: &Value) -> Result<String, ApiError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p["type"].as_str() {
                    Some("text") => out.push_str(p["text"].as_str().unwrap_or("")),
                    Some(t) => return Err(bad(format!("content parts of type {t:?} are not supported"))),
                    None => return Err(bad("content part without a type")),
                }
            }
            Ok(out)
        }
        Value::Null => Ok(String::new()),
        _ => Err(bad("content must be a string or an array of text parts")),
    }
}

fn stops(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
        _ => vec![],
    }
}

fn reject_tools(body: &Value) -> Result<(), ApiError> {
    if body.get("tools").map(|t| !t.is_null() && t.as_array().map(|a| !a.is_empty()).unwrap_or(true)).unwrap_or(false)
        || body.get("functions").is_some()
    {
        return Err(bad("tool use is not supported by banach serve yet"));
    }
    Ok(())
}

fn parse_openai(body: &Value, default_model: &str) -> Result<Gen, ApiError> {
    reject_tools(body)?;
    if body["n"].as_u64().unwrap_or(1) != 1 {
        return Err(bad("only n = 1 is supported"));
    }
    let msgs = body["messages"].as_array().ok_or_else(|| bad("messages is required"))?;
    let mut messages = vec![];
    for m in msgs {
        let role = m["role"].as_str().ok_or_else(|| bad("message without a role"))?;
        let role = match role {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            other => return Err(bad(format!("messages with role {other:?} are not supported"))),
        };
        messages.push((role.to_string(), text_of(&m["content"])?));
    }
    Ok(Gen {
        messages,
        max_tokens: body["max_completion_tokens"].as_u64().or(body["max_tokens"].as_u64()).unwrap_or(512),
        temperature: body["temperature"].as_f64().unwrap_or(1.0),
        top_p: body["top_p"].as_f64().unwrap_or(1.0),
        top_k: 0,
        seed: body["seed"].as_u64(),
        stop: stops(&body["stop"]),
        stream: body["stream"].as_bool().unwrap_or(false),
        model: body["model"].as_str().unwrap_or(default_model).to_string(),
    })
}

fn parse_anthropic(body: &Value, default_model: &str) -> Result<Gen, ApiError> {
    reject_tools(body)?;
    let max_tokens = body["max_tokens"].as_u64().ok_or_else(|| bad("max_tokens is required"))?;
    let mut messages = vec![];
    match &body["system"] {
        Value::Null => {}
        s => messages.push(("system".to_string(), text_of(s)?)),
    }
    let msgs = body["messages"].as_array().ok_or_else(|| bad("messages is required"))?;
    for m in msgs {
        let role = match m["role"].as_str() {
            Some("user") => "user",
            Some("assistant") => "assistant",
            other => return Err(bad(format!("messages with role {other:?} are not supported"))),
        };
        messages.push((role.to_string(), text_of(&m["content"])?));
    }
    Ok(Gen {
        messages,
        max_tokens,
        temperature: body["temperature"].as_f64().unwrap_or(1.0),
        top_p: body["top_p"].as_f64().unwrap_or(1.0),
        top_k: body["top_k"].as_u64().unwrap_or(0),
        seed: None,
        stop: stops(&body["stop_sequences"]),
        stream: body["stream"].as_bool().unwrap_or(false),
        model: body["model"].as_str().unwrap_or(default_model).to_string(),
    })
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------- generation + meter

struct Outcome {
    text: String,
    done: Value,
}

impl App {
    fn engine_request(&self, g: &Gen, id: &str) -> Value {
        let msgs: Vec<Value> = g.messages.iter().map(|(r, c)| json!({"role": r, "content": c})).collect();
        let mut req = json!({
            "op": "generate", "id": id, "messages": msgs, "max_tokens": g.max_tokens,
            "temperature": g.temperature, "top_p": g.top_p, "top_k": g.top_k, "stop": g.stop,
        });
        if let Some(s) = g.seed {
            req["seed"] = json!(s);
        }
        req
    }

    /// Run one generation; `on_text` gets every text delta. Runs the shadow replay after.
    fn generate(&self, g: &Gen, id: &str, mut on_text: impl FnMut(&str)) -> Result<Outcome, String> {
        let mut eng = self.engine.lock().unwrap();
        let mut text = String::new();
        let mut argmax: Vec<i64> = vec![];
        let done = eng.call(&self.engine_request(g, id), |ev| {
            let t = ev["text"].as_str().unwrap_or("");
            if !t.is_empty() {
                text.push_str(t);
                on_text(t);
            }
            if ev["token"].as_i64().unwrap_or(-1) >= 0 {
                argmax.push(ev["argmax"].as_i64().unwrap_or(-1));
            }
        })?;
        if done["event"] != "done" {
            return Err(format!("unexpected engine reply {done}"));
        }
        if self.ladder {
            self.shadow(&mut eng, &done, &argmax);
        }
        Ok(Outcome { text, done })
    }

    fn shadow(&self, eng: &mut EngineProc, done: &Value, argmax: &[i64]) {
        let mut m = self.meter.lock().unwrap();
        m.responses += 1;
        if m.shadow_every == 0 || (m.responses - 1) % m.shadow_every != 0 || argmax.is_empty() {
            return;
        }
        let toks = done["tokens"].as_array().cloned().unwrap_or_default();
        let n_prompt = done["n_prompt"].as_u64().unwrap_or(0) as usize;
        // generated tokens the candidate actually decoded: one exact prediction each
        let n_gen = argmax.len().min(toks.len().saturating_sub(n_prompt));
        if n_gen == 0 {
            return;
        }
        let toks = &toks[..n_prompt + n_gen];
        let reply = match eng.call(&json!({"op": "shadow", "id": "shadow", "tokens": toks, "from": n_prompt}), |_| {}) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("shadow replay failed: {e}");
                return;
            }
        };
        let exact = reply["argmax"].as_array().cloned().unwrap_or_default();
        let margin = reply["margin"].as_array().cloned().unwrap_or_default();
        m.shadowed += 1;
        let mut new_tau = None;
        for i in 0..n_gen.min(exact.len()) {
            let mg = margin.get(i).and_then(|v| v.as_f64()).unwrap_or(f64::INFINITY);
            if mg < m.min_margin {
                continue;
            }
            let flip = exact[i].as_i64() != Some(argmax[i]);
            m.meter.update(flip);
            m.lifetime.update(flip);
            let mut ctl = m.controller.take();
            if let Some(c) = ctl.as_mut() {
                if let Some(t) = c.step(&m.meter) {
                    new_tau = Some(t);
                    // a fresh window per control step keeps the bound about the current tau
                    let (target, alpha) = (m.meter.target(), m.meter.alpha);
                    m.meter = Meter::new(target, alpha);
                }
            }
            m.controller = ctl;
        }
        if let Some(t) = new_tau {
            if (t - m.tau).abs() > 1e-12 {
                let _ = eng.call(&json!({"op": "set_tau", "tau": t}), |_| {});
                m.tau = t;
                let n = m.lifetime.tokens;
                m.tau_history.push((n, t));
                if m.tau_history.len() > 200 {
                    m.tau_history.remove(0);
                }
            }
        }
    }

    fn status(&self) -> Value {
        let m = self.meter.lock().unwrap();
        let eng = self.engine.lock().unwrap();
        json!({
            "model": self.opts.model_id,
            "engine": eng.ready,
            "ladder": self.ladder,
            "tau": m.tau,
            "fidelity": {
                "target": m.lifetime.target(),
                "alpha": m.lifetime.alpha,
                "min_margin": m.min_margin,
                "confident_tokens": m.lifetime.tokens,
                "flips": m.lifetime.flips,
                "flip_rate": m.lifetime.point(),
                "ucb_lifetime": m.lifetime.ucb(),
                "certified_below_target_at": m.lifetime.certified_at(),
                "window_tokens": m.meter.tokens,
                "window_ucb": m.meter.ucb(),
            },
            "responses": m.responses,
            "shadowed_responses": m.shadowed,
            "tau_history": m.tau_history.iter().map(|(n, t)| json!([n, t])).collect::<Vec<_>>(),
            "download": self.opts.lazy.as_ref().map(|p| json!({
                "present_bytes": p.present_bytes(), "total_bytes": p.total_bytes(), "complete": p.complete()})),
        })
    }
}

// ---------------------------------------------------------------- HTTP plumbing

struct ChannelReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(b) => {
                    self.buf = b;
                    self.pos = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let n = out.len().min(self.buf.len() - self.pos);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn header(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
}

fn respond_json(req: tiny_http::Request, status: u16, v: &Value) {
    let body = v.to_string();
    let resp = tiny_http::Response::from_string(body)
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json"));
    let _ = req.respond(resp);
}

fn openai_error(e: &ApiError) -> Value {
    json!({"error": {"message": e.message, "type": e.kind, "code": null}})
}

fn anthropic_error(e: &ApiError) -> Value {
    json!({"type": "error", "error": {"type": e.kind, "message": e.message}})
}

fn stream_response(req: tiny_http::Request, produce: impl FnOnce(Sender<Vec<u8>>) + Send + 'static) {
    let (tx, rx) = channel::<Vec<u8>>();
    std::thread::spawn(move || produce(tx));
    let resp = tiny_http::Response::new(
        tiny_http::StatusCode(200),
        vec![
            header("Content-Type", "text/event-stream"),
            header("Cache-Control", "no-cache"),
            header("Connection", "keep-alive"),
        ],
        ChannelReader { rx, buf: vec![], pos: 0 },
        None,
        None,
    );
    let _ = req.respond(resp);
}

fn finish_openai(reason: &str) -> &'static str {
    if reason == "length" {
        "length"
    } else {
        "stop"
    }
}

fn finish_anthropic(done: &Value) -> (&'static str, Value) {
    if done["finish_reason"] == "length" {
        ("max_tokens", Value::Null)
    } else if done["stop"].is_string() {
        ("stop_sequence", done["stop"].clone())
    } else {
        ("end_turn", Value::Null)
    }
}

fn handle_openai(app: Arc<App>, req: tiny_http::Request, body: Value) {
    let g = match parse_openai(&body, &app.opts.model_id) {
        Ok(g) => g,
        Err(e) => return respond_json(req, e.status, &openai_error(&e)),
    };
    let id = format!("chatcmpl-{:x}", app.counter.fetch_add(1, Ordering::Relaxed) + now() * 1000);
    let created = now();
    if !g.stream {
        match app.generate(&g, &id, |_| {}) {
            Ok(o) => {
                let pt = o.done["prompt_tokens"].as_u64().unwrap_or(0);
                let ct = o.done["completion_tokens"].as_u64().unwrap_or(0);
                respond_json(req, 200, &json!({
                    "id": id, "object": "chat.completion", "created": created, "model": g.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": o.text},
                                 "finish_reason": finish_openai(o.done["finish_reason"].as_str().unwrap_or("stop"))}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                              "prompt_tokens_details": {"cached_tokens": o.done["cached_tokens"]}},
                }));
            }
            Err(e) => respond_json(req, 500, &json!({"error": {"message": e, "type": "server_error", "code": null}})),
        }
        return;
    }
    let include_usage = body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    stream_response(req, move |tx| {
        let chunk = |delta: Value, finish: Value| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": g.model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        };
        let send = |v: Value| tx.send(format!("data: {v}\n\n").into_bytes()).is_ok();
        send(chunk(json!({"role": "assistant", "content": ""}), Value::Null));
        let r = app.generate(&g, &id, |t| {
            send(chunk(json!({"content": t}), Value::Null));
        });
        match r {
            Ok(o) => {
                send(chunk(json!({}), json!(finish_openai(o.done["finish_reason"].as_str().unwrap_or("stop")))));
                if include_usage {
                    let pt = o.done["prompt_tokens"].as_u64().unwrap_or(0);
                    let ct = o.done["completion_tokens"].as_u64().unwrap_or(0);
                    send(json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": g.model,
                                "choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}));
                }
            }
            Err(e) => {
                send(json!({"error": {"message": e, "type": "server_error"}}));
            }
        }
        let _ = tx.send(b"data: [DONE]\n\n".to_vec());
    });
}

fn handle_anthropic(app: Arc<App>, req: tiny_http::Request, body: Value) {
    let g = match parse_anthropic(&body, &app.opts.model_id) {
        Ok(g) => g,
        Err(e) => return respond_json(req, e.status, &anthropic_error(&e)),
    };
    let id = format!("msg_{:x}", app.counter.fetch_add(1, Ordering::Relaxed) + now() * 1000);
    if !g.stream {
        match app.generate(&g, &id, |_| {}) {
            Ok(o) => {
                let (reason, seq) = finish_anthropic(&o.done);
                respond_json(req, 200, &json!({
                    "id": id, "type": "message", "role": "assistant", "model": g.model,
                    "content": [{"type": "text", "text": o.text}],
                    "stop_reason": reason, "stop_sequence": seq,
                    "usage": {"input_tokens": o.done["prompt_tokens"], "output_tokens": o.done["completion_tokens"],
                              "cache_read_input_tokens": o.done["cached_tokens"]},
                }));
            }
            Err(e) => respond_json(req, 500, &json!({"type": "error", "error": {"type": "api_error", "message": e}})),
        }
        return;
    }
    stream_response(req, move |tx| {
        let send = |ev: &str, v: Value| tx.send(format!("event: {ev}\ndata: {v}\n\n").into_bytes()).is_ok();
        send("message_start", json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": g.model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0}}}));
        send("content_block_start", json!({"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "text", "text": ""}}));
        let r = app.generate(&g, &id, |t| {
            send("content_block_delta", json!({"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "text_delta", "text": t}}));
        });
        send("content_block_stop", json!({"type": "content_block_stop", "index": 0}));
        match r {
            Ok(o) => {
                let (reason, seq) = finish_anthropic(&o.done);
                send("message_delta", json!({"type": "message_delta",
                    "delta": {"stop_reason": reason, "stop_sequence": seq},
                    "usage": {"input_tokens": o.done["prompt_tokens"], "output_tokens": o.done["completion_tokens"]}}));
                send("message_stop", json!({"type": "message_stop"}));
            }
            Err(e) => {
                send("error", json!({"type": "error", "error": {"type": "api_error", "message": e}}));
            }
        }
    });
}

fn route(app: Arc<App>, mut req: tiny_http::Request) {
    let method = req.method().as_str().to_string();
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let mut raw = String::new();
    if method == "POST" {
        if req.as_reader().take(32 << 20).read_to_string(&mut raw).is_err() {
            return respond_json(req, 400, &json!({"error": {"message": "body is not UTF-8 text", "type": "invalid_request_error"}}));
        }
    }
    let body: Value = if raw.is_empty() { Value::Null } else {
        match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => return respond_json(req, 400, &json!({"error": {"message": format!("invalid JSON: {e}"), "type": "invalid_request_error"}})),
        }
    };
    match (method.as_str(), path.as_str()) {
        ("POST", "/v1/chat/completions") => handle_openai(app, req, body),
        ("POST", "/v1/messages") => handle_anthropic(app, req, body),
        ("GET", "/v1/models") => {
            let id = app.opts.model_id.clone();
            respond_json(req, 200, &json!({"object": "list", "data": [
                {"id": id, "object": "model", "created": 0, "owned_by": "banach"}]}))
        }
        ("GET", "/banach/status") | ("GET", "/health") => {
            let st = app.status();
            respond_json(req, 200, &st)
        }
        _ => respond_json(req, 404, &json!({"error": {"message": format!("no route for {method} {path}"), "type": "not_found"}})),
    }
}

pub fn serve(opts: ServeOptions) -> Result<(), String> {
    if opts.host != "127.0.0.1" && opts.host != "localhost" && opts.host != "::1" {
        eprintln!("warning: listening on {}: anyone who can reach this address can use the model", opts.host);
    }
    let engine = EngineProc::start(&opts)?;
    let ladder = engine.ready["ladder"].as_bool().unwrap_or(false);
    eprintln!("banach serve: {} ready ({})", opts.model_id, engine.ready);
    let target = opts.target;
    let meter = MeterState {
        meter: Meter::new(target, 0.05),
        lifetime: Meter::new(target, 0.05),
        controller: if ladder { Some(TauController::new(opts.tau.max(1e-6), 1e-4, 1e4, opts.period)) } else { None },
        tau: opts.tau,
        min_margin: opts.min_margin,
        responses: 0,
        shadowed: 0,
        shadow_every: opts.shadow_every,
        tau_history: vec![],
    };
    let addr = format!("{}:{}", opts.host, opts.port);
    let server = tiny_http::Server::http(&addr).map_err(|e| format!("listen on {addr}: {e}"))?;
    eprintln!(
        "banach serve: http://{addr}  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, status: /banach/status)"
    );
    let app = Arc::new(App { engine: Mutex::new(engine), meter: Mutex::new(meter), opts, ladder, counter: AtomicU64::new(1) });
    for req in server.incoming_requests() {
        let app = Arc::clone(&app);
        std::thread::spawn(move || route(app, req));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_request_normalisation() {
        let body = json!({"model": "m", "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]}],
            "max_tokens": 7, "temperature": 0.2, "stop": "\n"});
        let g = parse_openai(&body, "d").unwrap();
        assert_eq!(g.messages, vec![("system".into(), "be brief".into()), ("user".into(), "hi there".into())]);
        assert_eq!(g.max_tokens, 7);
        assert_eq!(g.stop, vec!["\n".to_string()]);
        assert!(!g.stream);
    }

    #[test]
    fn anthropic_request_normalisation() {
        let body = json!({"model": "m", "max_tokens": 9, "system": [{"type": "text", "text": "sys"}],
                          "messages": [{"role": "user", "content": "q"}], "stop_sequences": ["END"], "stream": true});
        let g = parse_anthropic(&body, "d").unwrap();
        assert_eq!(g.messages[0], ("system".into(), "sys".into()));
        assert_eq!(g.stop, vec!["END".to_string()]);
        assert!(g.stream);
        assert!(parse_anthropic(&json!({"messages": []}), "d").is_err()); // max_tokens required
    }

    #[test]
    fn rejects_what_is_not_supported() {
        let tools = json!({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]});
        assert!(parse_openai(&tools, "d").unwrap_err().message.contains("tool use"));
        let n2 = json!({"messages": [{"role": "user", "content": "x"}], "n": 2});
        assert!(parse_openai(&n2, "d").is_err());
        let img = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]});
        assert!(parse_openai(&img, "d").is_err());
        let tool_role = json!({"messages": [{"role": "tool", "content": "x"}]});
        assert!(parse_openai(&tool_role, "d").is_err());
    }

    #[test]
    fn anthropic_stop_reasons() {
        assert_eq!(finish_anthropic(&json!({"finish_reason": "length"})).0, "max_tokens");
        assert_eq!(finish_anthropic(&json!({"finish_reason": "stop", "stop": "END"})).0, "stop_sequence");
        assert_eq!(finish_anthropic(&json!({"finish_reason": "stop", "stop": null})).0, "end_turn");
    }
}
