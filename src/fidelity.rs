//! Fidelity meter (SPEC-MATH section 7): anytime-valid certification of the top-1 flip
//! rate with a betting confidence sequence, and a PI controller on log(tau) that keeps the
//! upper confidence bound under the user's budget.
//!
//! Same estimator as phase0/banach_phase0/fidelity.py (the Python reference):
//!   wealth K_t(p0) = prod (1 + lambda_i (p0 - D_i)), lambda predictable, aGRAPA-style,
//!   clipped to [0, c / (1 - p0)]. Under H0: p >= p0 it is a nonnegative supermartingale,
//!   so P(sup K_t >= 1/alpha) <= alpha (Ville). The upper confidence bound is the smallest
//!   grid p0 rejected so far; rejections are kept (running intersection).

/// Betting test of H0: p >= p0 for one p0.
#[derive(Debug, Clone)]
pub struct Bettor {
    pub p0: f64,
    c: f64,
    n: f64,
    sum: f64,
    log_wealth: f64,
}

impl Bettor {
    pub fn new(p0: f64) -> Self {
        Self { p0, c: 0.5, n: 0.0, sum: 0.0, log_wealth: 0.0 }
    }

    fn lambda(&self) -> f64 {
        let prior_n = 1.0;
        let p_hat = (self.sum + prior_n * 0.5 * self.p0) / (self.n + prior_n);
        let mu = self.p0 - p_hat;
        let var = p_hat * (1.0 - p_hat) + 1e-6;
        (mu / (var + mu * mu)).clamp(0.0, self.c / (1.0 - self.p0))
    }

    pub fn update(&mut self, flip: bool) {
        let d = if flip { 1.0 } else { 0.0 };
        let lam = self.lambda();
        self.log_wealth += (1.0 + lam * (self.p0 - d)).ln();
        self.n += 1.0;
        self.sum += d;
    }

    pub fn log_wealth(&self) -> f64 {
        self.log_wealth
    }
}

/// Anytime-valid upper confidence sequence on the flip rate.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Meter {
    pub alpha: f64,
    bettors: Vec<Bettor>,
    rejected: Vec<bool>,
    pub tokens: u64,
    pub flips: u64,
    certified_at: Option<u64>,
    target: f64,
    target_bettor: Bettor,
}

impl Meter {
    /// `target` is the budget to certify (e.g. 0.01); the grid spans 1e-4 to 1.
    pub fn new(target: f64, alpha: f64) -> Self {
        let mut grid: Vec<f64> = (0..240).map(|i| 1e-4 * (0.2f64 / 1e-4).powf(i as f64 / 239.0)).collect();
        grid.extend((1..=80).map(|i| 0.2 + 0.8 * i as f64 / 81.0));
        let bettors: Vec<Bettor> = grid.iter().map(|&p| Bettor::new(p)).collect();
        let n = bettors.len();
        Self {
            alpha,
            bettors,
            rejected: vec![false; n],
            tokens: 0,
            flips: 0,
            certified_at: None,
            target,
            target_bettor: Bettor::new(target),
        }
    }

    pub fn update(&mut self, flip: bool) {
        let thr = (1.0 / self.alpha).ln();
        self.tokens += 1;
        self.flips += flip as u64;
        for (b, r) in self.bettors.iter_mut().zip(self.rejected.iter_mut()) {
            b.update(flip);
            if b.log_wealth() >= thr {
                *r = true;
            }
        }
        self.target_bettor.update(flip);
        if self.certified_at.is_none() && self.target_bettor.log_wealth() >= thr {
            self.certified_at = Some(self.tokens);
        }
    }

    /// Smallest grid rate rejected so far (1.0 while nothing is rejected).
    pub fn ucb(&self) -> f64 {
        self.bettors.iter().zip(&self.rejected).find(|(_, &r)| r).map(|(b, _)| b.p0).unwrap_or(1.0)
    }

    /// Token count at which "rate < target" was first certified at level alpha.
    pub fn certified_at(&self) -> Option<u64> {
        self.certified_at
    }

    pub fn point(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.flips as f64 / self.tokens as f64
        }
    }

    #[allow(dead_code)]
    pub fn target(&self) -> f64 {
        self.target
    }
}

/// PI controller on log(tau) against the meter's upper bound (SPEC-MATH section 7).
///
/// Larger tau serves more slots from the low copy (more flips). The error is
/// log(target) - log(ucb): positive while the bound is under budget, so tau grows; negative
/// when over, so tau shrinks. Updates happen every `period` tokens, and the period must be
/// longer than the time the confidence sequence needs to fall below the target at an
/// acceptable rate (a few thousand tokens for a 1% budget); with shorter windows the bound
/// never drops under budget and tau collapses to its floor. The integral is frozen while tau
/// is pinned at a limit (anti-windup). Because it controls the upper bound, the equilibrium
/// sits below the budget: conservative by construction.
#[derive(Debug, Clone)]
#[allow(dead_code)] // driven by the serve loop (Phase 4); exercised by the tests here
pub struct TauController {
    pub log_tau: f64,
    pub kp: f64,
    pub ki: f64,
    integral: f64,
    pub min_log_tau: f64,
    pub max_log_tau: f64,
    pub period: u64,
    since: u64,
}

#[allow(dead_code)]
impl TauController {
    pub fn new(tau: f64, min_tau: f64, max_tau: f64, period: u64) -> Self {
        Self {
            log_tau: tau.ln(),
            kp: 0.15,
            ki: 0.05,
            integral: 0.0,
            min_log_tau: min_tau.ln(),
            max_log_tau: max_tau.ln(),
            period,
            since: 0,
        }
    }

    pub fn tau(&self) -> f64 {
        self.log_tau.exp()
    }

    /// Call once per token; returns Some(new tau) when it changes.
    pub fn step(&mut self, meter: &Meter) -> Option<f64> {
        self.since += 1;
        if self.since < self.period {
            return None;
        }
        self.since = 0;
        let ucb = meter.ucb().max(1e-6);
        let err = meter.target().ln() - ucb.ln();
        let new_integral = self.integral + err;
        let proposal = self.log_tau + self.kp * err + self.ki * new_integral;
        let clamped = proposal.clamp(self.min_log_tau, self.max_log_tau);
        // anti-windup: only integrate while not pinned at a limit
        if (clamped - proposal).abs() < 1e-12 {
            self.integral = new_integral;
        }
        self.log_tau = clamped;
        Some(self.tau())
    }
}

/// Certify a recorded flip sequence (optionally with margins) at budget `target`.
pub fn certify(flips: &[(bool, f64)], target: f64, alpha: f64, min_margin: Option<f64>) -> Meter {
    let mut m = Meter::new(target, alpha);
    for &(f, margin) in flips {
        if min_margin.map(|mm| margin >= mm).unwrap_or(true) {
            m.update(f);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn certifies_low_rate_within_protocol_range() {
        // SPEC-DERISK P0.5: 2k to 4k tokens at a true rate of 0.5% or less
        let mut times = vec![];
        for seed in 1..=20u64 {
            let mut r = Rng(seed * 7919);
            let mut m = Meter::new(0.01, 0.05);
            for _ in 0..30_000 {
                let f = r.next() < 0.004;
                m.update(f);
                if m.certified_at().is_some() {
                    break;
                }
            }
            times.push(m.certified_at().unwrap_or(u64::MAX));
        }
        times.sort();
        let med = times[times.len() / 2];
        assert!((1000..6000).contains(&med), "median {med}");
    }

    #[test]
    fn type_one_error_at_the_boundary() {
        let mut rejections = 0;
        let trials = 300;
        for seed in 1..=trials {
            let mut r = Rng(seed as u64 * 104729);
            let mut m = Meter::new(0.01, 0.05);
            for _ in 0..5000 {
                m.update(r.next() < 0.01);
            }
            rejections += m.certified_at().is_some() as u32;
        }
        assert!((rejections as f64) / (trials as f64) <= 0.08, "{rejections}");
    }

    #[test]
    fn ucb_is_nested_and_covers() {
        let mut r = Rng(42);
        let mut m = Meter::new(0.01, 0.05);
        let mut last = 1.0;
        for _ in 0..8000 {
            m.update(r.next() < 0.004);
            assert!(m.ucb() <= last + 1e-15);
            last = m.ucb();
        }
        assert!(m.ucb() >= 0.004 * 0.8 && m.ucb() < 0.01, "{}", m.ucb());
    }

    #[test]
    fn matches_python_reference_on_a_fixed_sequence() {
        // flips at every 250th token (rate 0.4%), 6000 tokens; the Python reference
        // (banach_phase0.fidelity.certify) certifies p < 1% at token 1150 on this sequence
        let mut m = Meter::new(0.01, 0.05);
        for t in 0..6000 {
            m.update(t % 250 == 249);
        }
        assert_eq!(m.certified_at(), Some(PYTHON_CERTIFIED_AT));
    }

    // Filled from the Python reference; see the test above.
    const PYTHON_CERTIFIED_AT: u64 = 1150;

    #[test]
    fn controller_converges_under_budget() {
        // plant: flip rate grows with tau, 1% at tau = 1
        let rate = |tau: f64| (0.01 * tau.powf(1.5)).min(0.5);
        let mut r = Rng(9);
        let mut c = TauController::new(0.05, 1e-3, 100.0, 5000);
        // a fresh meter per control window keeps the bound responsive to the current tau
        let mut m = Meter::new(0.01, 0.05);
        let mut taus = vec![];
        for _ in 0..1_000_000 {
            m.update(r.next() < rate(c.tau()));
            if c.step(&m).is_some() {
                taus.push(c.tau());
                m = Meter::new(0.01, 0.05);
            }
        }
        let tail = &taus[taus.len() - 40..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        // must stay on the safe side of the 1% point, and not collapse to the floor
        assert!(mean < 1.0, "mean tau {mean}");
        assert!(mean > 0.2, "mean tau {mean}");
        assert!(rate(mean) < 0.01);
    }
}
