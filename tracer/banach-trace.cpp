// banach-trace: teacher-forced routing tracer for MoE models (Phase 0.1).
//
// Runs prefill over a JSONL corpus with llama.cpp and records, per token and per MoE layer:
//   routed expert ids (ffn_moe_topk), final gate weights (last of ffn_moe_weights*),
//   ||x|| of the MoE input (src1 of the expert mat-mul), ||f_i(x)|| per routed expert
//   (last of ffn_moe_down* / ffn_moe_post_exp_norm, before gate weighting),
//   and the top-N next-token log-probabilities.
//
// Uses only the scheduler eval callback, so it needs no patch to llama.cpp.
//
// Output: <out>.bin (fixed-size little-endian records, one per token) and <out>.json
// (layout, layer ids, workload names, model and run metadata). Convert with
//   python -m banach_phase0 import-trace <out>.json
//
// Input JSONL, one session per line:
//   {"session": "s1", "workload": "coding", "text": "..."}
//   {"session": "s2", "workload": "chat", "messages": [{"role": "user", "content": "..."}]}

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------- minimal JSON reader
// Enough for the corpus format: objects, arrays, strings, numbers, true/false/null.

namespace mini_json {

struct Value {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    const Value * get(const std::string & k) const {
        for (auto & kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct Parser {
    const std::string & s;
    size_t i = 0;
    explicit Parser(const std::string & src) : s(src) {}

    [[noreturn]] void fail(const char * what) {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i));
    }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++; }
    bool lit(const char * w) {
        size_t n = strlen(w);
        if (s.compare(i, n, w) == 0) { i += n; return true; }
        return false;
    }
    static void put_utf8(std::string & o, uint32_t c) {
        if (c < 0x80) o += char(c);
        else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
        else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string string() {
        if (s[i] != '"') fail("expected string");
        i++;
        std::string o;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') { o += c; continue; }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t c1 = hex4();
                    if (c1 >= 0xD800 && c1 < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t c2 = hex4();
                        c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                    }
                    put_utf8(o, c1);
                } break;
                default: fail("bad escape");
            }
        }
        return o;
    }
    Value value(int depth = 0) {
        if (depth > 64) fail("nesting too deep");
        ws();
        if (i >= s.size()) fail("unexpected end");
        Value v;
        char c = s[i];
        if (c == '{') {
            v.kind = Value::OBJ; i++; ws();
            if (i < s.size() && s[i] == '}') { i++; return v; }
            while (true) {
                ws(); std::string k = string(); ws();
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                i++;
                v.obj.emplace_back(k, value(depth + 1)); ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.kind = Value::ARR; i++; ws();
            if (i < s.size() && s[i] == ']') { i++; return v; }
            while (true) {
                v.arr.push_back(value(depth + 1)); ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.kind = Value::STR; v.str = string();
        } else if (lit("true")) { v.kind = Value::BOOL; v.b = true; }
        else if (lit("false")) { v.kind = Value::BOOL; v.b = false; }
        else if (lit("null")) { v.kind = Value::NUL; }
        else {
            size_t j = i;
            while (j < s.size() && (isdigit((unsigned char) s[j]) || strchr("+-.eE", s[j]))) j++;
            if (j == i) fail("unexpected character");
            v.kind = Value::NUM; v.num = std::stod(s.substr(i, j - i)); i = j;
        }
        return v;
    }
};

Value parse(const std::string & s) {
    Parser p(s);
    Value v = p.value();
    p.ws();
    if (p.i != s.size()) p.fail("trailing characters");
    return v;
}

std::string escape(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += char(c);
        }
    }
    return o + "\"";
}

} // namespace mini_json

// ---------------------------------------------------------------- tensor reading

// Read any F32/F16/BF16/I32 tensor (possibly a strided view) into a flat float vector in
// ggml element order (ne0 fastest).
static std::vector<float> read_tensor(const ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t) n);
    std::vector<uint8_t> buf;
    const uint8_t * base;
    // Copy the whole byte extent once, then gather with strides.
    size_t extent = 0;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (t->ne[d] > 0) extent += (size_t) (t->ne[d] - 1) * t->nb[d];
    }
    extent += ggml_type_size(t->type);
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
        buf.resize(extent);
        ggml_backend_tensor_get(t, buf.data(), 0, extent);
        base = buf.data();
    } else {
        base = (const uint8_t *) t->data;
    }
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
        const uint8_t * p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        float v;
        switch (t->type) {
            case GGML_TYPE_F32:  memcpy(&v, p, 4); break;
            case GGML_TYPE_F16:  v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p); break;
            case GGML_TYPE_BF16: v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) p); break;
            case GGML_TYPE_I32:  { int32_t x; memcpy(&x, p, 4); v = (float) x; } break;
            default:
                throw std::runtime_error(std::string("unsupported tensor type ") + ggml_type_name(t->type) +
                                         " for " + t->name);
        }
        out[k++] = v;
    }
    return out;
}

// ---------------------------------------------------------------- trace state

struct LayerBuf {
    std::vector<int32_t> topk;   // [T][k]
    std::vector<float>   gates;  // [T][k]
    std::vector<float>   onorm;  // [T][k]
    std::vector<float>   xnorm;  // [T]
    std::vector<float>   xvec;   // [T][n_embd], only while --dump-x wants more tokens
    bool have_topk = false, have_gates = false, have_onorm = false, have_xnorm = false;
};

struct Tracer {
    bool recording = false;
    bool want_x = false;  // keep the raw MoE input vectors of this ubatch
    int64_t n_embd = 0;
    int k = -1;
    int64_t n_tok = 0;                    // tokens in the current ubatch
    std::map<int, LayerBuf> layers;       // by llama layer index, for the current ubatch
    std::vector<int> moe_layers;          // fixed after the first recorded ubatch
    std::set<std::string> unknown_types;  // diagnostics

    void begin(int64_t n) { n_tok = n; layers.clear(); }
};

static bool parse_name(const char * name, std::string & base, int & il) {
    const char * dash = strrchr(name, '-');
    if (!dash || !dash[1]) return false;
    char * end = nullptr;
    long v = strtol(dash + 1, &end, 10);
    if (*end != '\0') return false;
    base.assign(name, dash - name);
    il = (int) v;
    return true;
}

static bool is_gate_name(const std::string & b) {
    return b == "ffn_moe_weights" || b == "ffn_moe_weights_softmax" || b == "ffn_moe_weights_norm" ||
           b == "ffn_moe_weights_scaled";
}
static bool is_down_name(const std::string & b) {
    return b == "ffn_moe_down" || b == "ffn_moe_down_scaled" || b == "ffn_moe_down_biased" ||
           b == "ffn_moe_post_exp_norm";
}
static bool is_input_name(const std::string & b) {
    return b == "ffn_moe_gate_up" || b == "ffn_moe_up" || b == "ffn_moe_gate";
}

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * tr = (Tracer *) ud;
    std::string base;
    int il;
    if (!parse_name(t->name, base, il)) return false;
    const bool wanted = base == "ffn_moe_topk" || is_gate_name(base) || is_down_name(base) ||
                        (is_input_name(base) && t->op == GGML_OP_MUL_MAT_ID);
    if (ask) return tr->recording && wanted;
    if (!tr->recording || !wanted) return true;

    LayerBuf & L = tr->layers[il];
    const int64_t T = tr->n_tok;

    if (base == "ffn_moe_topk") {
        const int kk = (int) t->ne[0];
        if (t->ne[1] != T) throw std::runtime_error("ffn_moe_topk token count mismatch");
        if (tr->k < 0) tr->k = kk;
        if (kk != tr->k) throw std::runtime_error("varying top-k across layers is not supported");
        auto v = read_tensor(t);
        L.topk.assign(v.begin(), v.end());
        L.have_topk = true;
    } else if (is_gate_name(base)) {
        auto v = read_tensor(t);
        if ((int64_t) v.size() != (int64_t) tr->k * T) throw std::runtime_error("gate weight size mismatch");
        L.gates = std::move(v);  // later names overwrite earlier ones: the final weights win
        L.have_gates = true;
    } else if (is_down_name(base)) {
        // [n_embd, k, T]
        const int64_t E = t->ne[0];
        auto v = read_tensor(t);
        if ((int64_t) v.size() != E * tr->k * T) throw std::runtime_error("expert output size mismatch");
        L.onorm.assign((size_t) (tr->k * T), 0.0f);
        for (int64_t s = 0; s < (int64_t) tr->k * T; s++) {
            double acc = 0;
            for (int64_t e = 0; e < E; e++) { double x = v[s * E + e]; acc += x * x; }
            L.onorm[s] = (float) std::sqrt(acc);
        }
        L.have_onorm = true;
    } else if (!L.have_xnorm) {
        // src1 of the expert mat-mul is the MoE input: [n_embd, 1 or k, T]
        const ggml_tensor * x = t->src[1];
        auto v = read_tensor(x);
        const int64_t E = x->ne[0], R = x->ne[1];
        if (x->ne[2] != T) throw std::runtime_error("moe input token count mismatch");
        L.xnorm.assign((size_t) T, 0.0f);
        if (tr->want_x) { L.xvec.assign((size_t) (T * E), 0.0f); tr->n_embd = E; }
        for (int64_t tt = 0; tt < T; tt++) {
            double acc = 0;
            const float * p = v.data() + (size_t) (tt * R * E);  // first row of the token
            for (int64_t e = 0; e < E; e++) acc += (double) p[e] * p[e];
            L.xnorm[tt] = (float) std::sqrt(acc);
            if (tr->want_x) std::copy(p, p + E, L.xvec.begin() + tt * E);
        }
        L.have_xnorm = true;
    }
    return true;
}

// ---------------------------------------------------------------- record layout

struct Layout {
    int L = 0, k = 0, ntop = 0;
    size_t off_token = 0, off_pos = 4, off_session = 8, off_workload = 12;
    size_t off_topk = 16, off_gates = 0, off_xnorm = 0, off_onorm = 0, off_top_ids = 0, off_top_lp = 0, size = 0;
    void compute() {
        off_gates = off_topk + (size_t) L * k * 2;
        off_gates = (off_gates + 3) & ~(size_t) 3;
        off_xnorm = off_gates + (size_t) L * k * 4;
        off_onorm = off_xnorm + (size_t) L * 4;
        off_top_ids = off_onorm + (size_t) L * k * 4;
        off_top_lp = off_top_ids + (size_t) ntop * 4;
        size = off_top_lp + (size_t) ntop * 4;
    }
};

// ---------------------------------------------------------------- main

struct Args {
    std::string model, corpus, out;
    int ctx = 6144, ubatch = 512, threads = 0, ntop = 20, warmup = 1;
    long max_tokens = 0;
    long dump_x = 0;
    bool no_chat_template = false;
};

static void usage() {
    fprintf(stderr,
        "usage: banach-trace -m MODEL.gguf -i CORPUS.jsonl -o OUT_PREFIX [options]\n"
        "  -c N        context per session (default 6144; protocol wants >= 5632)\n"
        "  -ub N       tokens per decode call / ubatch (default 512)\n"
        "  -t N        threads (default: all)\n"
        "  --top N     next-token log-probs kept per token (default 20)\n"
        "  --max-tokens N  stop after N recorded tokens\n"
        "  --warmup N  unrecorded warm-up decodes before tracing (default 1)\n"
        "  --raw       never apply the chat template to \"messages\"\n"
        "  --dump-x N  also write the MoE input vectors of the first N tokens to OUT.x.bin\n"
        "              (float32 [N][moe layer][n_embd], for the eps profiler)\n");
}

static Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (s == "-m") a.model = next();
        else if (s == "-i") a.corpus = next();
        else if (s == "-o") a.out = next();
        else if (s == "-c") a.ctx = std::stoi(next());
        else if (s == "-ub") a.ubatch = std::stoi(next());
        else if (s == "-t") a.threads = std::stoi(next());
        else if (s == "--top") a.ntop = std::stoi(next());
        else if (s == "--max-tokens") a.max_tokens = std::stol(next());
        else if (s == "--warmup") a.warmup = std::stoi(next());
        else if (s == "--dump-x") a.dump_x = std::stol(next());
        else if (s == "--raw") a.no_chat_template = true;
        else if (s == "-h" || s == "--help") { usage(); exit(0); }
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); usage(); exit(2); }
    }
    if (a.model.empty() || a.corpus.empty() || a.out.empty()) { usage(); exit(2); }
    if (a.ctx <= 0 || a.ubatch <= 0 || a.ntop < 0) { fprintf(stderr, "bad numeric argument\n"); exit(2); }
    return a;
}

struct Session {
    std::string id, workload, text;
    std::vector<std::pair<std::string, std::string>> messages;
};

static std::vector<Session> read_corpus(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open corpus " + path);
    std::vector<Session> out;
    std::string line;
    int ln = 0;
    while (std::getline(f, line)) {
        ln++;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        mini_json::Value v;
        try { v = mini_json::parse(line); }
        catch (std::exception & e) { throw std::runtime_error(path + ":" + std::to_string(ln) + ": " + e.what()); }
        Session s;
        const auto * id = v.get("session");
        s.id = id ? (id->kind == mini_json::Value::STR ? id->str : std::to_string((long long) id->num))
                  : std::to_string(ln);
        const auto * w = v.get("workload");
        s.workload = (w && w->kind == mini_json::Value::STR) ? w->str : "default";
        if (const auto * t = v.get("text"); t && t->kind == mini_json::Value::STR) s.text = t->str;
        if (const auto * m = v.get("messages"); m && m->kind == mini_json::Value::ARR) {
            for (auto & msg : m->arr) {
                const auto * r = msg.get("role");
                const auto * c = msg.get("content");
                if (!r || !c || r->kind != mini_json::Value::STR || c->kind != mini_json::Value::STR)
                    throw std::runtime_error(path + ":" + std::to_string(ln) + ": bad message");
                s.messages.emplace_back(r->str, c->str);
            }
        }
        if (s.text.empty() && s.messages.empty())
            throw std::runtime_error(path + ":" + std::to_string(ln) + ": need \"text\" or \"messages\"");
        out.push_back(std::move(s));
    }
    return out;
}

static std::string render(const llama_model * model, const Session & s, bool raw) {
    if (s.messages.empty()) return s.text;
    if (raw) {
        std::string o;
        for (auto & m : s.messages) o += m.first + ": " + m.second + "\n";
        return o;
    }
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("model has no chat template; pre-render the text or pass --raw");
    std::vector<llama_chat_message> msgs;
    size_t total = 0;
    for (auto & m : s.messages) { msgs.push_back({m.first.c_str(), m.second.c_str()}); total += m.second.size(); }
    std::vector<char> buf(2 * total + 1024);
    int32_t n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), false, buf.data(), (int32_t) buf.size());
    if (n < 0) throw std::runtime_error("chat template not supported by llama_chat_apply_template; pre-render the text");
    if ((size_t) n > buf.size()) {
        buf.resize((size_t) n + 1);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), false, buf.data(), (int32_t) buf.size());
    }
    return std::string(buf.data(), (size_t) n);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool templated) {
    int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, !templated, true);
    if (n == INT32_MIN || n < 0) throw std::runtime_error("tokenization failed");
    std::vector<llama_token> toks((size_t) n);
    int32_t got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), n, !templated, true);
    if (got < 0) throw std::runtime_error("tokenization failed");
    toks.resize((size_t) got);
    return toks;
}

static void put(std::vector<uint8_t> & rec, size_t off, const void * src, size_t n) {
    memcpy(rec.data() + off, src, n);
}

int main(int argc, char ** argv) {
    Args a = parse_args(argc, argv);
    try {
        auto sessions = read_corpus(a.corpus);
        if (sessions.empty()) throw std::runtime_error("corpus is empty");

        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) throw std::runtime_error("failed to load model " + a.model);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        Tracer tr;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) a.ctx;
        cp.n_batch = (uint32_t) a.ubatch;
        cp.n_ubatch = (uint32_t) a.ubatch;
        if (a.threads > 0) { cp.n_threads = a.threads; cp.n_threads_batch = a.threads; }
        cp.cb_eval = eval_cb;
        cp.cb_eval_user_data = &tr;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("failed to create context");
        llama_memory_t mem = llama_get_memory(ctx);

        llama_batch batch = llama_batch_init(a.ubatch, 0, 1);

        auto decode = [&](const llama_token * toks, int n, int pos0) {
            batch.n_tokens = n;
            for (int i = 0; i < n; i++) {
                batch.token[i] = toks[i];
                batch.pos[i] = pos0 + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            tr.begin(n);
            int rc = llama_decode(ctx, batch);
            if (rc != 0) throw std::runtime_error("llama_decode failed with " + std::to_string(rc));
        };

        // Warm-up: the first pass of a fresh context is discarded (protocol P0.1).
        for (int w = 0; w < a.warmup; w++) {
            auto toks = tokenize(vocab, render(model, sessions[0], a.no_chat_template), !sessions[0].messages.empty());
            int n = std::min<int>((int) toks.size(), a.ubatch);
            if (n > 0) decode(toks.data(), n, 0);
            llama_memory_clear(mem, true);
        }

        FILE * fbin = fopen((a.out + ".bin").c_str(), "wb");
        if (!fbin) throw std::runtime_error("cannot write " + a.out + ".bin");
        FILE * fx = nullptr;
        long x_written = 0;
        if (a.dump_x > 0) {
            fx = fopen((a.out + ".x.bin").c_str(), "wb");
            if (!fx) throw std::runtime_error("cannot write " + a.out + ".x.bin");
        }

        std::map<std::string, int> workload_ids;
        std::vector<std::string> workload_names;
        Layout lay;
        long recorded = 0;
        int session_index = 0;
        std::vector<uint8_t> rec;
        std::vector<std::pair<float, int>> scratch((size_t) n_vocab);
        tr.recording = true;

        for (auto & s : sessions) {
            if (a.max_tokens && recorded >= a.max_tokens) break;
            const bool templated = !s.messages.empty() && !a.no_chat_template;
            auto toks = tokenize(vocab, render(model, s, a.no_chat_template), templated);
            if ((int) toks.size() > a.ctx) toks.resize((size_t) a.ctx);
            auto wit = workload_ids.find(s.workload);
            int wid;
            if (wit == workload_ids.end()) {
                wid = (int) workload_names.size();
                workload_ids[s.workload] = wid;
                workload_names.push_back(s.workload);
            } else wid = wit->second;

            llama_memory_clear(mem, true);
            for (int p = 0; p < (int) toks.size(); p += a.ubatch) {
                int n = std::min<int>(a.ubatch, (int) toks.size() - p);
                if (a.max_tokens) n = (int) std::min<long>(n, a.max_tokens - recorded);
                if (n <= 0) break;
                tr.want_x = fx && x_written < a.dump_x;
                decode(toks.data() + p, n, p);

                if (tr.moe_layers.empty()) {
                    for (auto & kv : tr.layers) tr.moe_layers.push_back(kv.first);
                    if (tr.moe_layers.empty()) throw std::runtime_error("no ffn_moe_topk tensors seen: not an MoE model?");
                    lay.L = (int) tr.moe_layers.size();
                    lay.k = tr.k;
                    lay.ntop = std::min(a.ntop, n_vocab);
                    lay.compute();
                    rec.assign(lay.size, 0);
                    fprintf(stderr, "banach-trace: %d MoE layers, top-%d, record %zu bytes\n", lay.L, lay.k, lay.size);
                }
                for (int li = 0; li < lay.L; li++) {
                    auto it = tr.layers.find(tr.moe_layers[li]);
                    if (it == tr.layers.end()) throw std::runtime_error("layer missing from ubatch");
                    LayerBuf & B = it->second;
                    if (!B.have_topk || !B.have_gates)
                        throw std::runtime_error("layer " + std::to_string(tr.moe_layers[li]) + " missing topk or gates");
                    if (!B.have_onorm) B.onorm.assign((size_t) (lay.k * n), 0.0f);
                    if (!B.have_xnorm) B.xnorm.assign((size_t) n, 0.0f);
                }
                if ((int) tr.layers.size() != lay.L)
                    throw std::runtime_error("set of MoE layers changed between ubatches");

                for (int i = 0; i < n; i++) {
                    std::fill(rec.begin(), rec.end(), 0);
                    int32_t tok = toks[(size_t) (p + i)], pos = p + i, sid = session_index, w32 = wid;
                    put(rec, lay.off_token, &tok, 4);
                    put(rec, lay.off_pos, &pos, 4);
                    put(rec, lay.off_session, &sid, 4);
                    put(rec, lay.off_workload, &w32, 4);
                    for (int li = 0; li < lay.L; li++) {
                        LayerBuf & B = tr.layers[tr.moe_layers[li]];
                        for (int j = 0; j < lay.k; j++) {
                            int16_t e = (int16_t) B.topk[(size_t) (i * lay.k + j)];
                            float g = B.gates[(size_t) (i * lay.k + j)];
                            float o = B.onorm[(size_t) (i * lay.k + j)];
                            size_t slot = (size_t) (li * lay.k + j);
                            put(rec, lay.off_topk + slot * 2, &e, 2);
                            put(rec, lay.off_gates + slot * 4, &g, 4);
                            put(rec, lay.off_onorm + slot * 4, &o, 4);
                        }
                        float xn = B.xnorm[(size_t) i];
                        put(rec, lay.off_xnorm + (size_t) li * 4, &xn, 4);
                    }
                    if (lay.ntop > 0) {
                        const float * lg = llama_get_logits_ith(ctx, i);
                        if (!lg) throw std::runtime_error("no logits for token");
                        float mx = -INFINITY;
                        for (int v = 0; v < n_vocab; v++) mx = std::max(mx, lg[v]);
                        double z = 0;
                        for (int v = 0; v < n_vocab; v++) z += std::exp((double) lg[v] - mx);
                        const float lz = mx + (float) std::log(z);
                        for (int v = 0; v < n_vocab; v++) scratch[(size_t) v] = {lg[v], v};
                        std::partial_sort(scratch.begin(), scratch.begin() + lay.ntop, scratch.end(),
                                          [](auto & x, auto & y) { return x.first > y.first; });
                        for (int j = 0; j < lay.ntop; j++) {
                            int32_t id = scratch[(size_t) j].second;
                            float lp = scratch[(size_t) j].first - lz;
                            put(rec, lay.off_top_ids + (size_t) j * 4, &id, 4);
                            put(rec, lay.off_top_lp + (size_t) j * 4, &lp, 4);
                        }
                    }
                    if (fwrite(rec.data(), 1, rec.size(), fbin) != rec.size()) throw std::runtime_error("write failed");
                    if (tr.want_x && x_written < a.dump_x) {
                        for (int li = 0; li < lay.L; li++) {
                            LayerBuf & B = tr.layers[tr.moe_layers[li]];
                            if (B.xvec.empty()) throw std::runtime_error("no MoE input captured for --dump-x");
                            fwrite(B.xvec.data() + (size_t) i * tr.n_embd, sizeof(float), (size_t) tr.n_embd, fx);
                        }
                        x_written++;
                    }
                }
                recorded += n;
            }
            session_index++;
            fprintf(stderr, "banach-trace: session %d (%s) done, %ld tokens recorded\n", session_index,
                    s.workload.c_str(), recorded);
        }
        fclose(fbin);
        if (fx) fclose(fx);

        char desc[256];
        llama_model_desc(model, desc, sizeof desc);
        // Routed expert count from GGUF metadata (<arch>.expert_count).
        long n_expert = -1;
        for (int32_t mi = 0; mi < llama_model_meta_count(model); mi++) {
            char key[256], val[256];
            if (llama_model_meta_key_by_index(model, mi, key, sizeof key) < 0) continue;
            const std::string ks = key;
            const std::string suffix = ".expert_count";
            if (ks.size() > suffix.size() && ks.compare(ks.size() - suffix.size(), suffix.size(), suffix) == 0 &&
                llama_model_meta_val_str_by_index(model, mi, val, sizeof val) >= 0) {
                n_expert = strtol(val, nullptr, 10);
                break;
            }
        }
        std::ofstream fj(a.out + ".json");
        fj << "{\n";
        fj << "  \"format\": \"banach-trace-v1\",\n";
        fj << "  \"model_path\": " << mini_json::escape(a.model) << ",\n";
        fj << "  \"model_desc\": " << mini_json::escape(desc) << ",\n";
        fj << "  \"n_tokens\": " << recorded << ",\n";
        fj << "  \"n_sessions\": " << session_index << ",\n";
        fj << "  \"n_layers_total\": " << llama_model_n_layer(model) << ",\n";
        fj << "  \"moe_layers\": [";
        for (size_t i = 0; i < tr.moe_layers.size(); i++) fj << (i ? ", " : "") << tr.moe_layers[i];
        fj << "],\n";
        fj << "  \"top_k\": " << lay.k << ",\n";
        fj << "  \"n_experts\": " << n_expert << ",\n";
        fj << "  \"x_tokens\": " << x_written << ",\n  \"n_embd\": " << tr.n_embd << ",\n";
        fj << "  \"n_vocab\": " << n_vocab << ",\n";
        fj << "  \"ctx\": " << a.ctx << ",\n  \"ubatch\": " << a.ubatch << ",\n";
        fj << "  \"workload_names\": [";
        for (size_t i = 0; i < workload_names.size(); i++) fj << (i ? ", " : "") << mini_json::escape(workload_names[i]);
        fj << "],\n";
        fj << "  \"record\": {\"size\": " << lay.size << ", \"ntop\": " << lay.ntop
           << ", \"token\": " << lay.off_token << ", \"pos\": " << lay.off_pos << ", \"session\": " << lay.off_session
           << ", \"workload\": " << lay.off_workload << ", \"topk\": " << lay.off_topk << ", \"gates\": " << lay.off_gates
           << ", \"xnorm\": " << lay.off_xnorm << ", \"onorm\": " << lay.off_onorm << ", \"top_ids\": " << lay.off_top_ids
           << ", \"top_logprobs\": " << lay.off_top_lp << "}\n";
        fj << "}\n";
        fj.close();

        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        fprintf(stderr, "banach-trace: wrote %s.bin and %s.json (%ld tokens)\n", a.out.c_str(), a.out.c_str(), recorded);
    } catch (std::exception & e) {
        fprintf(stderr, "banach-trace: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
