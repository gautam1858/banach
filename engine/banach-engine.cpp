// banach-engine: lossless expert streaming on llama.cpp (Phase 2).
//
// The model stays memory-mapped and byte-identical; the OS page cache is the RAM tier and
// banach-engine decides which routed experts live in it:
//
//   * a byte budget for routed experts, kept by LRU eviction (MADV_PAGEOUT)
//   * demand fetch: when the router of layer l has chosen its experts (ffn_moe_topk eval
//     callback), every missing expert is read whole and in parallel with readahead(2), then
//     mapped in one go with MADV_POPULATE_READ, before the expert mat-muls run. Without this
//     the mat-mul page-faults the expert in 4 KiB pieces (llama.cpp advises random access).
//   * warm start: per-expert heat saved at exit and used to preload the hot set
//   * --exact-check: a reference run without any of this, compared bit for bit
//
// Dense tensors are left to the page cache. Weights and kernels are untouched, so the output
// is byte-identical to stock llama.cpp by construction; --exact-check verifies it.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static const size_t PAGE = (size_t) sysconf(_SC_PAGESIZE);

// ---------------------------------------------------------------- worker pool

class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; i++) workers.emplace_back([this] { loop(); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> g(m); stop = true; }
        cv.notify_all();
        for (auto & t : workers) t.join();
    }
    // Queue a job without waiting for it.
    void submit(std::function<void()> j) {
        { std::lock_guard<std::mutex> g(m); q.push_back(std::move(j)); pending++; }
        cv.notify_one();
    }
    // Wait until every queued job has finished.
    void drain() {
        std::unique_lock<std::mutex> l(m);
        done_cv.wait(l, [this] { return pending == 0; });
    }
    // Run all jobs and wait for them.
    void run(std::vector<std::function<void()>> & jobs) {
        if (jobs.empty()) return;
        {
            std::lock_guard<std::mutex> g(m);
            for (auto & j : jobs) q.push_back(std::move(j));
            pending += jobs.size();
        }
        cv.notify_all();
        std::unique_lock<std::mutex> l(m);
        done_cv.wait(l, [this] { return pending == 0; });
        jobs.clear();
    }
private:
    void loop() {
        while (true) {
            std::function<void()> j;
            {
                std::unique_lock<std::mutex> l(m);
                cv.wait(l, [this] { return stop || !q.empty(); });
                if (stop && q.empty()) return;
                j = std::move(q.front());
                q.pop_front();
            }
            j();
            {
                std::lock_guard<std::mutex> g(m);
                if (--pending == 0) done_cv.notify_all();
            }
        }
    }
    std::vector<std::thread> workers;
    std::list<std::function<void()>> q;
    std::mutex m;
    std::condition_variable cv, done_cv;
    size_t pending = 0;
    bool stop = false;
};

// ---------------------------------------------------------------- expert map

struct Slice {
    const uint8_t * ptr = nullptr;  // in llama.cpp's mapping
    size_t bytes = 0;
    int fd = -1;                    // our own descriptor of the same file
    uint64_t file_off = 0;
};

struct TensorLoc {
    int split = 0;
    uint64_t file_off = 0;  // absolute offset of the tensor data in its file
};

struct Expert {
    std::vector<Slice> slices;  // gate, up, down (or gate_up, down)
    size_t bytes = 0;
};

enum class Mode { Off, Naive, Banach };
enum class Policy { Lru, Admit };

struct Stats {
    uint64_t accesses = 0, hits = 0, misses = 0, evictions = 0, rejected = 0;
    uint64_t fetched_bytes = 0;
    uint64_t resident_checked = 0, resident_true = 0;  // mincore ground truth at access time
    double fetch_seconds = 0, evict_seconds = 0, callback_seconds = 0;
    void reset() { *this = Stats(); }
};

struct LadderCount {
    uint64_t slots = 0, low = 0;
};

struct Engine {
    LadderCount * ladder_count = nullptr;
    Mode mode = Mode::Off;
    Policy policy = Policy::Admit;
    double half_life = 256;  // in decode steps (one step = one ubatch through all layers)
    uint64_t step = 0;
    int last_layer = -1;
    // decayed usage frequency: value and the step it was last updated
    std::unordered_map<int64_t, std::pair<double, uint64_t>> freq;
    // fetched for the current layer but not admitted: paged out at the next callback
    std::vector<int64_t> transient;
    size_t budget = 0;
    bool discovering = true;
    bool check_residency = true;
    int n_expert = 0;
    std::map<int, std::vector<const ggml_tensor *>> layer_weights;  // il -> expert weight tensors
    std::map<std::string, TensorLoc> locs;                          // tensor name -> file location
    std::vector<int> fds;
    std::unordered_map<int64_t, Expert> experts;                    // key il * n_expert + e
    // LRU of resident routed experts
    std::list<int64_t> lru;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> where;
    size_t resident_bytes = 0;
    std::unordered_map<int64_t, double> heat;
    Stats st;
    Pool * pool = nullptr;
    Pool * evictor = nullptr;  // background page-out, off the critical path
    bool pageout = false;      // MADV_PAGEOUT (synchronous reclaim) instead of DONTNEED + fadvise
    std::string error;

    Expert * expert(int il, int e) {
        const int64_t key = (int64_t) il * n_expert + e;
        auto it = experts.find(key);
        if (it != experts.end()) return &it->second;
        auto lw = layer_weights.find(il);
        if (lw == layer_weights.end()) return nullptr;
        Expert x;
        for (const ggml_tensor * w : lw->second) {
            auto loc = locs.find(w->name);
            if (loc == locs.end()) return nullptr;
            Slice s;
            s.bytes = w->nb[2];
            s.ptr = (const uint8_t *) w->data + (size_t) e * w->nb[2];
            s.fd = fds[(size_t) loc->second.split];
            s.file_off = loc->second.file_off + (uint64_t) e * w->nb[2];
            x.bytes += s.bytes;
            x.slices.push_back(s);
        }
        return &experts.emplace(key, std::move(x)).first->second;
    }

    static void aligned_outer(const uint8_t * p, size_t n, uint8_t *& a, size_t & len) {
        uintptr_t s = (uintptr_t) p & ~(uintptr_t) (PAGE - 1);
        uintptr_t e = ((uintptr_t) p + n + PAGE - 1) & ~(uintptr_t) (PAGE - 1);
        a = (uint8_t *) s;
        len = e - s;
    }
    static bool aligned_inner(const uint8_t * p, size_t n, uint8_t *& a, size_t & len) {
        uintptr_t s = ((uintptr_t) p + PAGE - 1) & ~(uintptr_t) (PAGE - 1);
        uintptr_t e = ((uintptr_t) p + n) & ~(uintptr_t) (PAGE - 1);
        if (e <= s) return false;
        a = (uint8_t *) s;
        len = e - s;
        return true;
    }

    // Fraction of the expert's pages present in the page cache and mapped.
    double residency(const Expert & x) {
        size_t present = 0, total = 0;
        std::vector<unsigned char> vec;
        for (const Slice & s : x.slices) {
            uint8_t * a; size_t len;
            aligned_outer(s.ptr, s.bytes, a, len);
            vec.resize(len / PAGE);
            if (mincore(a, len, vec.data()) != 0) return -1;
            for (unsigned char c : vec) present += c & 1;
            total += vec.size();
        }
        return total ? (double) present / (double) total : 1.0;
    }

    static void drop(const Slice & s, bool pageout) {
        uint8_t * a; size_t len;
        if (aligned_inner(s.ptr, s.bytes, a, len)) madvise(a, len, pageout ? MADV_PAGEOUT : MADV_DONTNEED);
        posix_fadvise(s.fd, (off_t) s.file_off, (off_t) s.bytes, POSIX_FADV_DONTNEED);
    }

    void evict(const Expert & x) {
        auto t0 = clk::now();
        if (evictor) {
            const bool po = pageout;
            for (const Slice & s : x.slices) evictor->submit([s, po] { drop(s, po); });
        } else {
            for (const Slice & s : x.slices) drop(s, pageout);
        }
        st.evict_seconds += secs(t0, clk::now());
    }

    static void fetch_one(const Slice & s) {
        // Whole-slice read into the page cache (no copy), then map it in one call.
        if (readahead(s.fd, (off64_t) s.file_off, s.bytes) != 0) {
            posix_fadvise(s.fd, (off_t) s.file_off, (off_t) s.bytes, POSIX_FADV_WILLNEED);
        }
        uint8_t * a; size_t len;
        aligned_outer(s.ptr, s.bytes, a, len);
        madvise(a, len, MADV_POPULATE_READ);
    }

    void touch(int64_t key) {
        auto it = where.find(key);
        lru.splice(lru.end(), lru, it->second);
    }

    double frequency(int64_t key) {
        auto it = freq.find(key);
        if (it == freq.end()) return 0.0;
        return it->second.first * std::exp2(-(double) (step - it->second.second) / half_life);
    }
    void bump(int64_t key, double w) {
        double f = frequency(key) + w;
        freq[key] = {f, step};
    }

    // TinyLFU-style admission: a missed expert stays only if it is used more often than the
    // expert it would evict.
    bool should_admit(int64_t key, const Expert & x, const std::set<int64_t> & protect) {
        if (policy == Policy::Lru || resident_bytes + x.bytes <= budget) return true;
        double need = 0, victims = 0;
        for (auto it = lru.begin(); it != lru.end() && resident_bytes + x.bytes - need > budget; ++it) {
            if (protect.count(*it)) continue;
            victims = std::max(victims, frequency(*it));
            need += experts.at(*it).bytes;
        }
        return frequency(key) > victims;
    }

    void admit(int64_t key, const Expert & x, const std::set<int64_t> & protect) {
        while (resident_bytes + x.bytes > budget && !lru.empty()) {
            // evict the least recently used expert that this layer does not need right now
            auto victim = lru.begin();
            while (victim != lru.end() && protect.count(*victim)) ++victim;
            if (victim == lru.end()) break;
            const int64_t vk = *victim;
            evict(experts.at(vk));
            resident_bytes -= experts.at(vk).bytes;
            where.erase(vk);
            lru.erase(victim);
            st.evictions++;
        }
        lru.push_back(key);
        where[key] = std::prev(lru.end());
        resident_bytes += x.bytes;
    }

    // Called with the experts layer il selected for this ubatch (n_tokens tokens).
    void on_selected(int il, const std::vector<int> & ids, int64_t n_tokens) {
        if (il <= last_layer) step++;  // a new ubatch started
        last_layer = il;
        std::set<int64_t> keys;
        std::map<int64_t, int> occ;
        for (int e : ids) {
            const int64_t k = (int64_t) il * n_expert + e;
            keys.insert(k);
            occ[k]++;
        }
        // page out experts that were only needed by the previous layer
        for (int64_t k : transient) {
            if (!where.count(k) && !keys.count(k)) evict(experts.at(k));
        }
        transient.clear();
        // per-token usage, so a long prefill counts like one token per expert-use rate
        for (auto & kv : occ) bump(kv.first, (double) kv.second / (double) std::max<int64_t>(1, n_tokens));
        std::vector<std::function<void()>> jobs;
        for (int64_t key : keys) {
            const int e = (int) (key % n_expert);
            Expert * x = expert(il, e);
            if (!x) return;  // layer not discovered yet
            st.accesses++;
            heat[key] += (double) occ[key] / (double) std::max<int64_t>(1, n_tokens);
            if (check_residency) {
                st.resident_checked++;
                if (residency(*x) >= 0.99) st.resident_true++;
            }
            if (where.count(key)) {
                st.hits++;
                touch(key);
                continue;
            }
            st.misses++;
            if (should_admit(key, *x, keys)) {
                admit(key, *x, keys);
            } else {
                st.rejected++;
                transient.push_back(key);
            }
            if (mode == Mode::Banach) {
                for (const Slice & s : x->slices) jobs.push_back([s] { fetch_one(s); });
                st.fetched_bytes += x->bytes;
            }
        }
        if (!jobs.empty()) {
            auto t0 = clk::now();
            pool->run(jobs);
            st.fetch_seconds += secs(t0, clk::now());
        }
    }

    void evict_all() {
        for (auto & kv : layer_weights) {
            for (int e = 0; e < n_expert; e++) {
                Expert * x = expert(kv.first, e);
                if (x) evict(*x);
            }
        }
        lru.clear();
        where.clear();
        resident_bytes = 0;
    }

    // Preload the hottest experts until the budget is full.
    size_t warm_start() {
        std::vector<std::pair<double, int64_t>> order;
        for (auto & kv : heat) order.emplace_back(kv.second, kv.first);
        std::sort(order.begin(), order.end(), [](auto & a, auto & b) { return a.first > b.first; });
        std::vector<std::function<void()>> jobs;
        size_t n = 0;
        for (auto & o : order) {
            const int il = (int) (o.second / n_expert), e = (int) (o.second % n_expert);
            Expert * x = expert(il, e);
            if (!x || where.count(o.second)) continue;
            if (resident_bytes + x->bytes > budget) break;
            lru.push_front(o.second);  // coldest first, so recency takes over quickly
            where[o.second] = lru.begin();
            resident_bytes += x->bytes;
            for (const Slice & s : x->slices) jobs.push_back([s] { fetch_one(s); });
            n++;
        }
        pool->run(jobs);
        return n;
    }
};

static std::vector<int> read_ids(const ggml_tensor * t) {
    std::vector<int> out;
    const int64_t k = t->ne[0], T = t->ne[1];
    std::vector<uint8_t> buf;
    const uint8_t * base = (const uint8_t *) t->data;
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
        buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        base = buf.data();
    }
    for (int64_t j = 0; j < T; j++)
        for (int64_t i = 0; i < k; i++) {
            int32_t v;
            memcpy(&v, base + i * t->nb[0] + j * t->nb[1], 4);
            out.push_back(v);
        }
    return out;
}

static bool layer_of(const char * name, const char * prefix, int & il) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return false;
    char * end;
    long v = strtol(name + n + 1, &end, 10);
    if (*end) return false;
    il = (int) v;
    return true;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * E = (Engine *) ud;
    int il;
    if (E->ladder_count && layer_of(t->name, "banach_ids_lo", il)) {
        if (ask) return true;
        for (int v : read_ids(t)) {
            E->ladder_count->slots++;
            if (v >= 0) E->ladder_count->low++;
        }
        return true;
    }
    if (E->mode == Mode::Off) return ask ? false : true;
    const bool is_topk = layer_of(t->name, "ffn_moe_topk", il);
    const bool is_mm = t->op == GGML_OP_MUL_MAT_ID;
    if (ask) return is_topk || (E->discovering && is_mm);
    try {
        if (is_mm && E->discovering) {
            const ggml_tensor * w = t->src[0];
            int lw;
            // weight names look like blk.<il>.ffn_up_exps.weight
            if (w && sscanf(w->name, "blk.%d.", &lw) == 1) {
                auto & v = E->layer_weights[lw];
                if (std::find(v.begin(), v.end(), w) == v.end()) v.push_back(w);
                if (E->n_expert == 0) E->n_expert = (int) w->ne[2];
            }
        } else if (is_topk && !E->discovering) {
            auto t0 = clk::now();
            E->on_selected(il, read_ids(t), t->ne[1]);
            E->st.callback_seconds += secs(t0, clk::now());
        }
    } catch (std::exception & e) {
        E->error = e.what();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- run

struct Args {
    std::string model, prompt, prompt_file, heat_in, heat_out, json_out;
    Policy policy = Policy::Admit;
    bool async_evict = true, pageout = false;
    std::string ladder, eps_file, flips_out;
    double tau = INFINITY;
    double min_margin = 0.1;  // confident tokens: exact top-1 beats top-2 by this many nats
    std::vector<double> taus;  // --fidelity sweep
    bool fidelity = false;
    double half_life = 256;
    int n_gen = 64, threads = 0, io_threads = 8, ctx = 4096, ubatch = 512;
    double budget_mb = 0;
    Mode mode = Mode::Banach;
    bool cold = true, exact_check = false, residency = true;
};

static void usage() {
    fprintf(stderr,
        "usage: banach-engine -m MODEL.gguf (-p TEXT | -f FILE) [options]\n"
        "  --mode off|naive|banach   off: stock llama.cpp; naive: budget only, page-fault reads;\n"
        "                            banach: budget + parallel whole-expert fetch (default)\n"
        "  --budget-mb N             RAM for routed experts (default: unlimited)\n"
        "  -n N                      tokens to generate greedily (default 64)\n"
        "  -t N / --io-threads N     compute threads / fetch threads (default all / 8)\n"
        "  -c N / -ub N              context / ubatch (default 4096 / 512)\n"
        "  --policy lru|admit        eviction LRU; admit adds frequency-gated admission (default admit)\n"
        "  --half-life N             decay of usage frequency in decode steps (default 256)\n"
        "  --sync-evict / --pageout  evict on the critical path / with MADV_PAGEOUT (for comparison)\n"
        "  --ladder LOW.gguf         precision ladder: serve slots from LOW unless gate*eps*|x| > tau\n"
        "  --eps FILE / --tau T      per-expert eps table (text: one float per line, layer-major) / threshold\n"
        "  --fidelity T1,T2,...      teacher-forced sweep over tau: top-1 flips and KL against exact\n"
        "  --flips-out PREFIX        with --fidelity: write PREFIX.<tau>.txt: per token, flip and exact margin\n"
        "  --min-margin M            confident tokens have an exact top-1/top-2 margin >= M nats (default 0.1)\n"
        "  --warm                    keep whatever is in the page cache (default: start cold)\n"
        "  --heat-in F / --heat-out F  warm-start heat file\n"
        "  --exact-check             also run stock llama.cpp and compare tokens and logits bit for bit\n"
        "  --no-residency            skip mincore() ground-truth checks\n"
        "  --json F                  write stats as JSON\n");
}

static Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (s == "-m") a.model = next();
        else if (s == "-p") a.prompt = next();
        else if (s == "-f") a.prompt_file = next();
        else if (s == "-n") a.n_gen = std::stoi(next());
        else if (s == "-t") a.threads = std::stoi(next());
        else if (s == "--io-threads") a.io_threads = std::stoi(next());
        else if (s == "-c") a.ctx = std::stoi(next());
        else if (s == "-ub") a.ubatch = std::stoi(next());
        else if (s == "--budget-mb") a.budget_mb = std::stod(next());
        else if (s == "--mode") {
            std::string m = next();
            if (m == "off") a.mode = Mode::Off;
            else if (m == "naive") a.mode = Mode::Naive;
            else if (m == "banach") a.mode = Mode::Banach;
            else { fprintf(stderr, "bad --mode %s\n", m.c_str()); exit(2); }
        }
        else if (s == "--warm") a.cold = false;
        else if (s == "--policy") {
            std::string p = next();
            if (p == "lru") a.policy = Policy::Lru;
            else if (p == "admit") a.policy = Policy::Admit;
            else { fprintf(stderr, "bad --policy %s\n", p.c_str()); exit(2); }
        }
        else if (s == "--half-life") a.half_life = std::stod(next());
        else if (s == "--sync-evict") a.async_evict = false;
        else if (s == "--ladder") a.ladder = next();
        else if (s == "--eps") a.eps_file = next();
        else if (s == "--tau") a.tau = std::stod(next());
        else if (s == "--fidelity") {
            a.fidelity = true;
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) if (!item.empty()) a.taus.push_back(std::stod(item));
        }
        else if (s == "--flips-out") a.flips_out = next();
        else if (s == "--min-margin") a.min_margin = std::stod(next());
        else if (s == "--pageout") a.pageout = true;
        else if (s == "--heat-in") a.heat_in = next();
        else if (s == "--heat-out") a.heat_out = next();
        else if (s == "--exact-check") a.exact_check = true;
        else if (s == "--no-residency") a.residency = false;
        else if (s == "--json") a.json_out = next();
        else if (s == "-h" || s == "--help") { usage(); exit(0); }
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); usage(); exit(2); }
    }
    if (a.model.empty() || (a.prompt.empty() && a.prompt_file.empty())) { usage(); exit(2); }
    return a;
}

static std::vector<std::string> split_paths(const std::string & first) {
    // "<prefix>-00001-of-0000N.gguf" -> all parts; otherwise just the file
    std::vector<std::string> out{first};
    const std::string suf = ".gguf";
    if (first.size() < 15 + suf.size() || first.compare(first.size() - suf.size(), suf.size(), suf) != 0) return out;
    std::string stem = first.substr(0, first.size() - suf.size());
    std::string tail = stem.substr(stem.size() - 15);
    int no, cnt;
    if (sscanf(tail.c_str(), "-%5d-of-%5d", &no, &cnt) != 2 || no != 1 || cnt < 2) return out;
    std::string prefix = stem.substr(0, stem.size() - 15);
    out.clear();
    for (int i = 1; i <= cnt; i++) {
        char buf[4096];
        llama_split_path(buf, sizeof buf, prefix.c_str(), i - 1, cnt);
        out.push_back(buf);
    }
    return out;
}

struct ProcIo { uint64_t read_bytes = 0; long majflt = 0, minflt = 0; };
static ProcIo proc_io() {
    ProcIo p;
    std::ifstream f("/proc/self/io");
    std::string k;
    uint64_t v;
    while (f >> k >> v) if (k == "read_bytes:") p.read_bytes = v;
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    p.majflt = ru.ru_majflt;
    p.minflt = ru.ru_minflt;
    return p;
}

struct RunResult {
    std::vector<llama_token> tokens;
    std::vector<uint64_t> logit_hash;
    double prefill_s = 0, decode_s = 0;
    int n_prompt = 0;
    Stats prefill, decode;
    uint64_t read_prefill = 0, read_decode = 0;
    long majflt_decode = 0;
};

static uint64_t fnv(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n * sizeof(float); i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static RunResult run(llama_model * model, const std::vector<llama_token> & prompt, const Args & a, Engine * E) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) a.ctx;
    cp.n_batch = (uint32_t) a.ubatch;
    cp.n_ubatch = (uint32_t) a.ubatch;
    if (a.threads > 0) { cp.n_threads = a.threads; cp.n_threads_batch = a.threads; }
    cp.cb_eval = E ? eval_cb : nullptr;
    cp.cb_eval_user_data = E;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("failed to create context");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(a.ubatch, 0, 1);
    auto decode = [&](const llama_token * t, int n, int pos0, bool last_logits) {
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = t[i]; batch.pos[i] = pos0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = last_logits && i == n - 1;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        if (E && !E->error.empty()) throw std::runtime_error(E->error);
    };

    RunResult r;
    r.n_prompt = (int) prompt.size();
    if (E) E->st.reset();
    ProcIo io0 = proc_io();
    auto t0 = clk::now();
    for (int p = 0; p < (int) prompt.size(); p += a.ubatch) {
        int n = std::min<int>(a.ubatch, (int) prompt.size() - p);
        decode(prompt.data() + p, n, p, p + n == (int) prompt.size());
    }
    auto t1 = clk::now();
    ProcIo io1 = proc_io();
    if (E) { r.prefill = E->st; E->st.reset(); }
    int pos = (int) prompt.size();
    for (int i = 0; i < a.n_gen; i++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        r.logit_hash.push_back(fnv(lg, (size_t) n_vocab));
        llama_token best = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        r.tokens.push_back(best);
        if (llama_vocab_is_eog(vocab, best)) break;
        decode(&best, 1, pos++, true);
    }
    auto t2 = clk::now();
    ProcIo io2 = proc_io();
    if (E) r.decode = E->st;
    r.prefill_s = secs(t0, t1);
    r.decode_s = secs(t1, t2);
    r.read_prefill = io1.read_bytes - io0.read_bytes;
    r.read_decode = io2.read_bytes - io1.read_bytes;
    r.majflt_decode = io2.majflt - io1.majflt;
    llama_batch_free(batch);
    llama_free(ctx);
    return r;
}

static std::string stats_json(const char * name, const Stats & s, double seconds, int tokens, uint64_t disk) {
    std::ostringstream o;
    o << "\"" << name << "\": {\"tokens\": " << tokens << ", \"seconds\": " << seconds
      << ", \"tok_s\": " << (seconds > 0 ? tokens / seconds : 0) << ", \"accesses\": " << s.accesses
      << ", \"hits\": " << s.hits << ", \"hit_rate\": " << (s.accesses ? (double) s.hits / s.accesses : 0)
      << ", \"resident_true_rate\": " << (s.resident_checked ? (double) s.resident_true / s.resident_checked : 0)
      << ", \"evictions\": " << s.evictions << ", \"rejected\": " << s.rejected << ", \"fetched_bytes\": " << s.fetched_bytes
      << ", \"fetch_seconds\": " << s.fetch_seconds << ", \"evict_seconds\": " << s.evict_seconds
      << ", \"callback_seconds\": " << s.callback_seconds << ", \"disk_read_bytes\": " << disk
      << ", \"disk_bytes_per_token\": " << (tokens ? (double) disk / tokens : 0) << "}";
    return o.str();
}

// ---------------------------------------------------------------- fidelity (Phase 3)

struct TopN {
    std::vector<int32_t> ids;  // [T * n]
    std::vector<float> lp;     // [T * n]
};

// Teacher-forced pass over the prompt with logits at every position; keeps the top-n log-probs.
static TopN forced(llama_context * ctx, const std::vector<llama_token> & toks, int ubatch, int n_vocab, int n) {
    TopN out;
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::vector<std::pair<float, int>> sc((size_t) n_vocab);
    for (int p = 0; p < (int) toks.size(); p += ubatch) {
        int m = std::min<int>(ubatch, (int) toks.size() - p);
        batch.n_tokens = m;
        for (int i = 0; i < m; i++) {
            batch.token[i] = toks[(size_t) (p + i)]; batch.pos[i] = p + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        for (int i = 0; i < m; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            float mx = *std::max_element(lg, lg + n_vocab);
            double z = 0;
            for (int v = 0; v < n_vocab; v++) z += std::exp((double) lg[v] - mx);
            const float lz = mx + (float) std::log(z);
            for (int v = 0; v < n_vocab; v++) sc[(size_t) v] = {lg[v], v};
            std::partial_sort(sc.begin(), sc.begin() + n, sc.end(), [](auto & x, auto & y) { return x.first > y.first; });
            for (int j = 0; j < n; j++) { out.ids.push_back(sc[(size_t) j].second); out.lp.push_back(sc[(size_t) j].first - lz); }
        }
    }
    llama_batch_free(batch);
    return out;
}

// KL(ref || cand) over the reference top-n, both renormalised; ids missing from the candidate's
// top-n get its smallest stored log-prob (same estimator as banach_phase0.fidelity).
static double topn_kl(const TopN & r, const TopN & c, size_t t, int n) {
    std::unordered_map<int32_t, float> look;
    float floor = 0;
    for (int j = 0; j < n; j++) { look[c.ids[t * n + j]] = c.lp[t * n + j]; floor = std::min(floor, c.lp[t * n + j]); }
    std::vector<double> rp(n), qp(n);
    double rz = -INFINITY, qz = -INFINITY;
    for (int j = 0; j < n; j++) {
        rp[j] = r.lp[t * n + j];
        auto it = look.find(r.ids[t * n + j]);
        qp[j] = it == look.end() ? floor : it->second;
    }
    auto lse = [](const std::vector<double> & v) { double m = *std::max_element(v.begin(), v.end()); double s = 0; for (double x : v) s += std::exp(x - m); return m + std::log(s); };
    rz = lse(rp); qz = lse(qp);
    double kl = 0;
    for (int j = 0; j < n; j++) { double a = rp[j] - rz, b = qp[j] - qz; kl += std::exp(a) * (a - b); }
    return std::min(std::max(kl, 0.0), 10.0);
}

static std::vector<float> read_eps(const std::string & path, size_t n) {
    std::vector<float> e;
    if (path.empty()) return std::vector<float>(n, 1.0f);
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    float v;
    while (f >> v) e.push_back(v);
    if (e.size() != n) throw std::runtime_error(path + ": expected " + std::to_string(n) + " values, got " + std::to_string(e.size()));
    return e;
}

static int run_fidelity(llama_model * model, llama_model * low, const std::vector<llama_token> & toks, const Args & a,
                        int n_expert) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_layer = llama_model_n_layer(model);
    const int n = 20;
    auto eps = read_eps(a.eps_file, (size_t) n_layer * (size_t) n_expert);
    Engine counter;  // only counts ladder slots
    LadderCount lc;
    counter.ladder_count = &lc;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) a.ctx; cp.n_batch = (uint32_t) a.ubatch; cp.n_ubatch = (uint32_t) a.ubatch;
    if (a.threads > 0) { cp.n_threads = a.threads; cp.n_threads_batch = a.threads; }
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = &counter;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("failed to create context");
    auto t0 = clk::now();
    TopN ref = forced(ctx, toks, a.ubatch, n_vocab, n);
    fprintf(stderr, "fidelity: exact reference over %zu tokens in %.1f s\n", toks.size(), secs(t0, clk::now()));
    if (llama_set_ladder(ctx, low, eps.data(), 0.0f) != 0) throw std::runtime_error("llama_set_ladder failed");
    printf("[\n");
    for (size_t ti = 0; ti < a.taus.size(); ti++) {
        const double tau = a.taus[ti];
        llama_set_ladder_tau(ctx, (float) tau);
        lc = LadderCount();
        auto t1 = clk::now();
        TopN cand = forced(ctx, toks, a.ubatch, n_vocab, n);
        const size_t T = toks.size();
        size_t flips = 0, confident = 0, confident_flips = 0;
        double kl_sum = 0;
        std::vector<int> fl(T);
        for (size_t t = 0; t < T; t++) {
            fl[t] = ref.ids[t * n] != cand.ids[t * n];
            flips += (size_t) fl[t];
            kl_sum += topn_kl(ref, cand, t, n);
            if (ref.lp[t * n] - ref.lp[t * n + 1] >= a.min_margin) {
                confident++;
                confident_flips += (size_t) fl[t];
            }
        }
        const double low_share = lc.slots ? (double) lc.low / (double) lc.slots : 0.0;
        printf("  {\"tau\": %g, \"tokens\": %zu, \"flips\": %zu, \"flip_rate\": %.6f, \"kl_mean\": %.6f, "
               "\"min_margin\": %g, \"confident_tokens\": %zu, \"confident_flips\": %zu, \"confident_flip_rate\": %.6f, "
               "\"low_slot_share\": %.6f, \"seconds\": %.2f}%s\n",
               tau, T, flips, (double) flips / (double) T, kl_sum / (double) T, a.min_margin, confident, confident_flips,
               confident ? (double) confident_flips / (double) confident : 0.0, low_share, secs(t1, clk::now()),
               ti + 1 < a.taus.size() ? "," : "");
        fflush(stdout);
        if (!a.flips_out.empty()) {
            // one line per token: flip (0/1) and the exact model's top-1/top-2 log-prob margin
            std::ofstream f(a.flips_out + "." + std::to_string(tau) + ".txt");
            for (size_t t = 0; t < T; t++) f << fl[t] << " " << (ref.lp[t * n] - ref.lp[t * n + 1]) << "\n";
        }
    }
    printf("]\n");
    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);
    try {
        std::string text = a.prompt;
        if (!a.prompt_file.empty()) {
            std::ifstream f(a.prompt_file);
            if (!f) throw std::runtime_error("cannot read " + a.prompt_file);
            std::stringstream ss; ss << f.rdbuf(); text = ss.str();
        }
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) throw std::runtime_error("failed to load " + a.model);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        int n = -llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, true, true);
        std::vector<llama_token> prompt((size_t) n);
        n = llama_tokenize(vocab, text.data(), (int) text.size(), prompt.data(), n, true, true);
        if (n < 0) throw std::runtime_error("tokenize failed");
        prompt.resize((size_t) n);
        if ((int) prompt.size() + a.n_gen > a.ctx) prompt.resize((size_t) std::max(1, a.ctx - a.n_gen));

        llama_model * low = nullptr;
        if (!a.ladder.empty()) {
            low = llama_model_load_from_file(a.ladder.c_str(), mp);
            if (!low) throw std::runtime_error("failed to load ladder model " + a.ladder);
        }
        if (a.fidelity) {
            if (!low) throw std::runtime_error("--fidelity needs --ladder");
            char buf[64] = {0};
            int n_expert = 0;
            for (int32_t mi = 0; mi < llama_model_meta_count(model); mi++) {
                char key[256];
                if (llama_model_meta_key_by_index(model, mi, key, sizeof key) < 0) continue;
                std::string ks = key;
                if (ks.size() > 13 && ks.compare(ks.size() - 13, 13, ".expert_count") == 0 &&
                    llama_model_meta_val_str_by_index(model, mi, buf, sizeof buf) >= 0) { n_expert = atoi(buf); break; }
            }
            if (n_expert <= 0) throw std::runtime_error("model has no expert_count");
            int rc = run_fidelity(model, low, prompt, a, n_expert);
            llama_model_free(low);
            llama_model_free(model);
            llama_backend_free();
            return rc;
        }

        std::string json = "{";
        RunResult ref;
        if (a.exact_check || a.mode == Mode::Off) {
            ref = run(model, prompt, a, nullptr);
            fprintf(stderr, "reference (stock llama.cpp): prefill %d tok %.2f s, decode %zu tok %.2f tok/s\n",
                    ref.n_prompt, ref.prefill_s, ref.tokens.size(), ref.tokens.size() / ref.decode_s);
            json += stats_json("reference_decode", Stats(), ref.decode_s, (int) ref.tokens.size(), ref.read_decode) + ", ";
        }

        if (a.mode != Mode::Off) {
            Engine E;
            E.mode = a.mode;
            E.policy = a.policy;
            E.half_life = a.half_life;
            E.budget = a.budget_mb > 0 ? (size_t) (a.budget_mb * 1e6) : SIZE_MAX / 2;
            E.check_residency = a.residency;
            Pool pool(std::max(1, a.io_threads));
            E.pool = &pool;
            Pool evictor(1);
            E.evictor = a.async_evict ? &evictor : nullptr;
            E.pageout = a.pageout;
            // file locations of every tensor, for whole-slice reads with our own descriptors
            auto paths = split_paths(a.model);
            for (size_t si = 0; si < paths.size(); si++) {
                int fd = open(paths[si].c_str(), O_RDONLY | O_NOFOLLOW);
                if (fd < 0) throw std::runtime_error("cannot open " + paths[si]);
                E.fds.push_back(fd);
                gguf_init_params gp{true, nullptr};
                gguf_context * g = gguf_init_from_file(paths[si].c_str(), gp);
                if (!g) throw std::runtime_error("cannot read GGUF header of " + paths[si]);
                const uint64_t data = gguf_get_data_offset(g);
                for (int64_t ti = 0; ti < gguf_get_n_tensors(g); ti++)
                    E.locs[gguf_get_tensor_name(g, ti)] = TensorLoc{(int) si, data + gguf_get_tensor_offset(g, ti)};
                gguf_free(g);
            }
            // discovery: one short decode records the expert weight tensors of every layer
            {
                Args d = a; d.n_gen = 1;
                std::vector<llama_token> one(prompt.begin(), prompt.begin() + std::min<size_t>(prompt.size(), 8));
                E.discovering = true;
                run(model, one, d, &E);
                E.discovering = false;
                if (E.layer_weights.empty() || E.n_expert == 0) throw std::runtime_error("no routed expert tensors found");
            }
            size_t pool_bytes = 0;
            for (auto & kv : E.layer_weights) {
                Expert * x = E.expert(kv.first, 0);
                if (x) pool_bytes += x->bytes * (size_t) E.n_expert;
            }
            if (a.cold) E.evict_all();
            evictor.drain();
            size_t warmed = 0;
            if (!a.heat_in.empty()) {
                std::ifstream f(a.heat_in);
                int64_t k; double h;
                double total = 0;
                while (f >> k >> h) { E.heat[k] = h; total += h; }
                // seed admission frequencies: heat is per-token use counts from earlier runs
                size_t n_keys = std::max<size_t>(1, E.heat.size());
                for (auto & kv : E.heat) E.freq[kv.first] = {kv.second / total * (double) n_keys, 0};
                warmed = E.warm_start();
            }
            fprintf(stderr, "engine: %zu MoE layers x %d experts, pool %.2f GB, budget %s, %s start%s\n",
                    E.layer_weights.size(), E.n_expert, pool_bytes / 1e9,
                    a.budget_mb > 0 ? (std::to_string((int) a.budget_mb) + " MB").c_str() : "unlimited",
                    a.cold ? "cold" : "warm",
                    warmed ? (", warm-started " + std::to_string(warmed) + " experts").c_str() : "");
            RunResult r = run(model, prompt, a, &E);
            const double tps = r.tokens.size() / r.decode_s;
            fprintf(stderr, "%s: prefill %d tok %.2f s; decode %zu tok %.2f tok/s, hit %.1f%% (page cache %.1f%%), "
                            "disk %.1f MB/token, fetch %.2f s, evict %.2f s, callback %.2f s\n",
                    a.mode == Mode::Banach ? "banach" : "naive", r.n_prompt, r.prefill_s, r.tokens.size(), tps,
                    100.0 * r.decode.hits / std::max<uint64_t>(1, r.decode.accesses),
                    100.0 * r.decode.resident_true / std::max<uint64_t>(1, r.decode.resident_checked),
                    r.read_decode / 1e6 / std::max<size_t>(1, r.tokens.size()), r.decode.fetch_seconds,
                    r.decode.evict_seconds, r.decode.callback_seconds);
            json += stats_json("prefill", r.prefill, r.prefill_s, r.n_prompt, r.read_prefill) + ", ";
            json += stats_json("decode", r.decode, r.decode_s, (int) r.tokens.size(), r.read_decode) + ", ";
            json += "\"mode\": \"" + std::string(a.mode == Mode::Banach ? "banach" : "naive") + "\", ";
            json += "\"budget_bytes\": " + std::to_string(a.budget_mb > 0 ? (uint64_t) (a.budget_mb * 1e6) : 0) + ", ";
            json += "\"expert_pool_bytes\": " + std::to_string(pool_bytes) + ", ";
            if (a.exact_check) {
                bool same = r.tokens == ref.tokens && r.logit_hash == ref.logit_hash;
                size_t first_diff = 0;
                while (first_diff < std::min(r.logit_hash.size(), ref.logit_hash.size()) &&
                       r.logit_hash[first_diff] == ref.logit_hash[first_diff]) first_diff++;
                fprintf(stderr, "exact-check: %s (%zu tokens, logits %s)\n", same ? "PASS byte-identical" : "FAIL",
                        r.tokens.size(), same ? "bit-equal at every step" : ("differ from step " + std::to_string(first_diff)).c_str());
                json += std::string("\"exact_check\": ") + (same ? "true" : "false") + ", ";
                if (!same) {
                    json += "\"tokens\": " + std::to_string(r.tokens.size()) + "}";
                    if (!a.json_out.empty()) std::ofstream(a.json_out) << json << "\n";
                    return 3;
                }
            }
            if (!a.heat_out.empty()) {
                std::ofstream f(a.heat_out);
                for (auto & kv : E.heat) f << kv.first << " " << kv.second << "\n";
            }
            for (int fd : E.fds) close(fd);
        }
        json += "\"generated\": " + std::to_string(ref.tokens.size()) + "}";
        if (!a.json_out.empty()) std::ofstream(a.json_out) << json << "\n";
        llama_model_free(model);
        llama_backend_free();
    } catch (std::exception & e) {
        fprintf(stderr, "banach-engine: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
