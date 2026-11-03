import math

import numpy as np
import pytest

from banach_phase0 import cachesim, entropy, fidelity, ladder, perfmodel
from banach_phase0.model import DEEPSEEK_V4_FLASH, GB, GIB
from banach_phase0.trace import Trace, synthetic_trace


@pytest.fixture(scope="module")
def trace():
    return synthetic_trace(n_tokens=3000, n_layers=6, n_experts=48, top_k=4, seed=1)


# ----------------------------------------------------------------- model constants


def test_v4_flash_expert_sizes_match_spec_table():
    m = DEEPSEEK_V4_FLASH
    assert m.params_per_expert == 25_165_824
    assert m.expert_bytes(4.25) / 1e6 == pytest.approx(13.37, abs=0.01)
    assert m.pool_bytes(4.25) / 1e9 == pytest.approx(147.2, abs=0.2)
    assert m.routed_bytes_per_token(4.25) / 1e9 == pytest.approx(3.45, abs=0.01)
    assert m.experts_that_fit(12 * GB, 4.25) == 897  # spec table says 898 (rounding)
    # spec table divides by the rounded 6.49 MB; exact size gives a few more
    assert m.experts_that_fit(40 * GB, 2.06) == pytest.approx(6163, rel=0.002)


# ----------------------------------------------------------------- trace


def test_trace_roundtrip(tmp_path, trace):
    p = tmp_path / "t.npz"
    trace.save(p)
    t2 = Trace.load(p)
    assert np.array_equal(t2.topk, trace.topk)
    assert np.allclose(t2.gates, trace.gates)
    assert t2.hash_layers == trace.hash_layers
    assert t2.n_experts == trace.n_experts


def test_synthetic_sets_are_unique_and_gates_sorted(trace):
    s = np.sort(trace.topk, axis=-1)
    assert (np.diff(s, axis=-1) > 0).all()
    assert (np.diff(trace.gates, axis=-1) <= 0).all()
    assert np.allclose(trace.gates.sum(-1), 1.0, atol=1e-5)


def test_session_split_keeps_sessions_whole(trace):
    warm, score = trace.split_by_session(0.6)
    assert not (warm & score).any() and (warm | score).all()
    for s in np.unique(trace.session):
        m = trace.session == s
        assert warm[m].all() or score[m].all()
    for w in np.unique(trace.workload):
        assert (warm & (trace.workload == w)).any() and (score & (trace.workload == w)).any()


# ----------------------------------------------------------------- cache simulator


def _brute_belady(keys, cap):
    cache, hits = set(), 0
    for i, k in enumerate(keys):
        if k in cache:
            hits += 1
            continue

        def nxt(x):
            for j in range(i + 1, len(keys)):
                if keys[j] == x:
                    return j
            return math.inf

        if len(cache) < cap:
            cache.add(k)
            continue
        far = max(cache, key=nxt)
        if nxt(k) < nxt(far):
            cache.remove(far)
            cache.add(k)
    return hits


def _brute_lru(keys, cap):
    cache, hits = [], 0
    for k in keys:
        if k in cache:
            hits += 1
            cache.remove(k)
        elif len(cache) >= cap:
            cache.pop(0)
        cache.append(k)
    return hits


@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_policies_match_brute_force(cap):
    rng = np.random.default_rng(cap)
    keys = rng.integers(0, 8, size=300).tolist()
    nu = cachesim.next_use_indices(np.array(keys))
    for name, c, ref in (
        ("belady", cachesim.Belady(cap, nu), _brute_belady(keys, cap)),
        ("lru", cachesim.LRU(cap), _brute_lru(keys, cap)),
    ):
        hits = sum(c.access(k, i, i) for i, k in enumerate(keys))
        assert hits == ref, name


def test_belady_upper_bounds_online_policies(trace):
    cap = trace.n_layers * 12
    res = {p: cachesim.simulate(trace, p, cap, "global").hit_rate
           for p in ("lru", "lfu:256", "gdsf", "static", "belady")}
    for p, h in res.items():
        assert h <= res["belady"] + 1e-12, (p, res)
    assert res["belady"] > 0


def test_hit_rate_monotone_in_capacity(trace):
    caps = [trace.n_layers * c for c in (4, 8, 16, 32)]
    for pol in ("lru", "belady", "static"):
        hs = [cachesim.simulate(trace, pol, c).hit_rate for c in caps]
        assert all(b >= a - 1e-12 for a, b in zip(hs, hs[1:])), (pol, hs)


def test_full_capacity_only_compulsory_misses(trace):
    cap = trace.n_layers * trace.n_experts
    r = cachesim.simulate(trace, "lru", cap, "per_layer")
    warm, score = trace.split_by_session(0.6)
    seen = set()
    for t in np.nonzero(warm)[0]:
        for l in range(trace.n_layers):
            seen.update((l, int(e)) for e in trace.topk[t, l])
    compulsory = set()
    for t in np.nonzero(score)[0]:
        for l in range(trace.n_layers):
            for e in trace.topk[t, l]:
                if (l, int(e)) not in seen:
                    compulsory.add((l, int(e)))
                    seen.add((l, int(e)))
    assert r.accesses - r.hits == len(compulsory)


def test_zero_capacity(trace):
    r = cachesim.simulate(trace, "lru", 0)
    assert r.hits == 0 and r.accesses > 0


def test_transfer_off_domain_not_better_than_in_domain_for_static(trace):
    off, inn, ratio = cachesim.transfer(trace, "static", trace.n_layers * 8, 0, 1)
    assert inn.hit_rate > 0
    assert ratio < 1.0


# ----------------------------------------------------------------- entropy


def test_fano_floor_basic():
    assert entropy.fano_list_floor(0.5, 10, 2) == 0.0
    assert entropy.fano_list_floor(100, 10, 2) == 1.0
    n = entropy.log2_binom(256, 6)
    assert n == pytest.approx(38.4, abs=0.1)  # matches SPEC-MATH section 3


def test_entropy_deterministic_layer_is_zero():
    T, L, N, k = 400, 2, 16, 2
    rng = np.random.default_rng(0)
    topk = np.zeros((T, L, k), dtype=np.int64)
    topk[:, 0, 0] = rng.integers(0, N // 2, T) * 2
    topk[:, 0, 1] = topk[:, 0, 0] + 1
    topk[:, 1, 0] = (topk[:, 0, 0] + 2) % N  # layer 1 fully determined by layer 0 top-1
    topk[:, 1, 1] = (topk[:, 0, 0] + 3) % N
    tr = Trace(topk=topk, gates=np.full((T, L, k), 0.5, np.float32),
               xnorm=np.ones((T, L), np.float32), session=np.zeros(T, np.int32),
               workload=np.zeros(T, np.int16), n_experts=N)
    assert entropy.conditional_set_entropy(tr, 1) == pytest.approx(0.0, abs=1e-6)
    assert entropy.marginal_set_entropy(tr, 1) > 1.0


# ----------------------------------------------------------------- fidelity


def test_certifies_low_rate_and_not_high_rate():
    rng = np.random.default_rng(0)
    low = (rng.random(20_000) < 0.003).astype(float)
    c = fidelity.certify(low, p0=0.01, alpha=0.05)
    assert c.certified and c.certified_at < 5000
    high = (rng.random(20_000) < 0.02).astype(float)
    assert not fidelity.certify(high, p0=0.01).certified


def test_type_one_error_controlled_at_boundary():
    # true p exactly p0: rejection rate must stay <= alpha (Ville), with slack for MC noise
    rng = np.random.default_rng(1)
    trials, rej = 300, 0
    for _ in range(trials):
        d = (rng.random(5000) < 0.01).astype(float)
        rej += fidelity.certify(d, 0.01, 0.05, with_ucb=False).certified
    assert rej / trials <= 0.08


def test_tokens_to_certify_in_spec_range():
    # SPEC-DERISK P0.5: expect 2-4k tokens at <=0.5% true rate
    n = fidelity.tokens_to_certify(0.005, trials=30, n_max=30_000, seed=3)
    med = float(np.median([x if x is not None else np.inf for x in n]))
    assert 1000 < med < 6000


def test_ucb_is_valid_and_shrinks():
    rng = np.random.default_rng(2)
    d = (rng.random(8000) < 0.004).astype(float)
    u = fidelity.upper_confidence_sequence(d)
    assert (np.diff(u) <= 1e-12).all()  # nested
    assert u[-1] >= 0.004 * 0.8 and u[-1] < 0.01


def test_clipped_kl_zero_for_identical():
    rng = np.random.default_rng(0)
    lp = rng.normal(size=(5, 50))
    lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    assert np.allclose(fidelity.clipped_topk_kl(lp, lp), 0.0)
    assert (fidelity.clipped_topk_kl(lp, lp[:, ::-1]) > 0).all()


# ----------------------------------------------------------------- performance model


def test_expert_union_matches_spec():
    # 256 * (1 - (250/256)^4) = 23.17, i.e. ~3.86x one token. SPEC-MATH quotes 22.9 (~3.8x).
    assert perfmodel.expert_union(4, 6, 256) == pytest.approx(23.17, abs=0.01)
    assert perfmodel.expert_union(4, 6, 256) / 6 == pytest.approx(3.8, abs=0.1)
    assert perfmodel.expected_accepted(0.7, 3) == pytest.approx(2.53, abs=0.01)


def test_cpu_only_ceiling_matches_derisk():
    # SPEC-DERISK: dual-channel DDR5 caps decode near 7.5-8 tok/s at 100% hit, no speculation
    m = perfmodel.Machine("ref", dram_bw=70 * GB, nvme_bw=6 * GB, t_sync=0.0)
    p = perfmodel.predict(DEEPSEEK_V4_FLASH, m, 2.06, 1.0)
    assert 6.5 < p.tok_s < 8.5
    assert DEEPSEEK_V4_FLASH.dense_bytes_per_token == pytest.approx(7.33 * GIB)


def test_prediction_monotone_in_hit_rate():
    m = perfmodel.REFERENCE_32GB
    tps = [perfmodel.predict(DEEPSEEK_V4_FLASH, m, 2.06, h).tok_s for h in (0.5, 0.7, 0.9, 1.0)]
    assert all(b >= a for a, b in zip(tps, tps[1:]))


def test_required_hit_rate_unreachable_without_mtp():
    m = perfmodel.REFERENCE_32GB
    assert perfmodel.required_hit_rate(DEEPSEEK_V4_FLASH, m, 2.06, 10.0) is None


def test_fit_recovers_parameters():
    true = perfmodel.Machine("t", dram_bw=70 * GB, nvme_bw=6 * GB, overlap=0.6, t_sync=0.01)
    runs = [{"bpw": 2.06, "hit_rate": h,
             "tok_s": perfmodel.predict(DEEPSEEK_V4_FLASH, true, 2.06, h).tok_s}
            for h in (0.3, 0.6, 0.8, 0.95)]
    ov, ts, err = perfmodel.fit_overlap_sync(DEEPSEEK_V4_FLASH, perfmodel.REFERENCE_32GB, runs)
    assert err < 0.01 and ov == pytest.approx(0.6, abs=0.06) and ts == pytest.approx(0.01, abs=0.002)


# ----------------------------------------------------------------- ladder


def test_ladder_monotone_in_tau(trace):
    rng = np.random.default_rng(0)
    eps = np.sort(rng.uniform(0.001, 0.05, size=(trace.n_layers, trace.n_experts, 3)), axis=-1)[..., ::-1]
    res = ladder.sweep(trace, eps, (2.06, 2.5, 3.06), np.geomspace(0.01, 3, 8), 4.25,
                       DEEPSEEK_V4_FLASH.params_per_expert)
    esc = [r.escalations_per_token for r in res]
    bits = [r.mean_bits for r in res]
    assert all(b <= a + 1e-12 for a, b in zip(esc, esc[1:]))
    assert all(b <= a + 1e-12 for a, b in zip(bits, bits[1:]))
    assert all(abs(r.rung_fraction.sum() - 1) < 1e-9 for r in res)


def test_skip_never_drops_top1():
    g = np.array([[[0.9, 0.06, 0.03, 0.01]]])
    m = ladder.skip_mask(g, m_norm=1.0, tau=10.0)
    assert not m[..., 0].any() and m[..., 1:].all()
    m = ladder.skip_mask(g, m_norm=1.0, tau=0.05)
    assert m.tolist() == [[[False, False, False, True]]]


def test_spearman():
    a = np.arange(10)
    assert ladder.spearman(a, a**2) == pytest.approx(1.0)
    assert ladder.spearman(a, -a) == pytest.approx(-1.0)


# ----------------------------------------------------------------- real-trace import


def _write_banach_trace(tmp_path, T=7, L=3, k=2, n=4, n_experts=5):
    import json as _json
    off = {"token": 0, "pos": 4, "session": 8, "workload": 12, "topk": 16}
    off["gates"] = (off["topk"] + L * k * 2 + 3) & ~3
    off["xnorm"] = off["gates"] + L * k * 4
    off["onorm"] = off["xnorm"] + L * 4
    off["top_ids"] = off["onorm"] + L * k * 4
    off["top_logprobs"] = off["top_ids"] + n * 4
    size = off["top_logprobs"] + n * 4
    rng = np.random.default_rng(0)
    buf = bytearray(T * size)
    topk = np.stack([rng.permutation(n_experts)[:k] for _ in range(T * L)]).reshape(T, L, k)
    for t in range(T):
        b = t * size
        def put(o, arr):
            raw = np.asarray(arr).tobytes()
            buf[b + o : b + o + len(raw)] = raw
        put(off["token"], np.int32(100 + t)); put(off["pos"], np.int32(t))
        put(off["session"], np.int32(t // 4)); put(off["workload"], np.int32(t // 4))
        put(off["topk"], topk[t].astype("<i2"))
        put(off["gates"], np.full((L, k), 0.5, "<f4")); put(off["xnorm"], np.full(L, 2.0, "<f4"))
        put(off["onorm"], np.full((L, k), 3.0, "<f4"))
        put(off["top_ids"], np.arange(n, dtype="<i4") + t)
        put(off["top_logprobs"], np.log(np.full(n, 1 / n, "<f4")))
    (tmp_path / "r.bin").write_bytes(bytes(buf))
    meta = {"format": "banach-trace-v1", "n_tokens": T, "moe_layers": list(range(1, L + 1)),
            "top_k": k, "n_experts": n_experts, "workload_names": ["a", "b"],
            "record": {"size": size, "ntop": n, **off}}
    (tmp_path / "r.json").write_text(_json.dumps(meta))
    return topk


def test_import_banach_trace(tmp_path):
    topk = _write_banach_trace(tmp_path)
    t = Trace.from_banach_trace(tmp_path / "r.json", hash_layers=(0,))
    assert np.array_equal(t.topk, topk)
    assert t.n_experts == 5 and t.hash_layers == (0,)
    assert t.tokens.tolist() == list(range(100, 107))
    assert t.session.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert np.allclose(t.xnorm, 2.0) and np.allclose(t.onorm, 3.0)
    assert t.top_ids.shape == (7, 4)
    t.save(tmp_path / "r.npz")
    assert np.array_equal(Trace.load(tmp_path / "r.npz").top_ids, t.top_ids)


def test_import_rejects_truncated_file(tmp_path):
    _write_banach_trace(tmp_path)
    p = tmp_path / "r.bin"
    p.write_bytes(p.read_bytes()[:-3])
    with pytest.raises(ValueError):
        Trace.from_banach_trace(tmp_path / "r.json")


def test_compare_top_n():
    ids = np.array([[1, 2, 3], [4, 5, 6]])
    lp = np.log(np.array([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]))
    flips, kl = fidelity.compare_top_n(ids, lp, ids, lp)
    assert flips.tolist() == [0, 0] and np.allclose(kl, 0)
    cand = np.array([[2, 1, 3], [4, 9, 6]])
    flips, kl = fidelity.compare_top_n(ids, lp, cand, lp)
    assert flips.tolist() == [1, 0] and (kl > 0).all()
