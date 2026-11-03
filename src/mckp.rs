//! Multiple-choice knapsack by greedy over lower convex hulls (SPEC-MATH section 4).
//!
//! Each item picks exactly one option (bytes, cost). Option 0 should be the zero-byte floor
//! (e.g. "stream from NVMe"). For each item we keep the lower convex hull of its options in
//! (bytes, cost), turn consecutive hull points into increments sorted by cost saved per byte,
//! and take increments greedily while they fit. That is the LP optimum up to at most one
//! fractional item, and exact when all increments have equal size.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opt {
    pub bytes: u64,
    pub cost: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Solution {
    /// Chosen option index per item.
    pub choice: Vec<usize>,
    pub bytes: u64,
    pub cost: f64,
}

fn hull(opts: &[Opt]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..opts.len()).collect();
    idx.sort_by(|&a, &b| {
        opts[a]
            .bytes
            .cmp(&opts[b].bytes)
            .then(opts[a].cost.partial_cmp(&opts[b].cost).unwrap_or(std::cmp::Ordering::Equal))
    });
    // keep only options that strictly reduce cost as bytes grow
    let mut mono: Vec<usize> = Vec::new();
    for i in idx {
        if let Some(&last) = mono.last() {
            if opts[i].bytes == opts[last].bytes || opts[i].cost >= opts[last].cost {
                continue;
            }
        }
        mono.push(i);
    }
    // lower convex hull
    let mut h: Vec<usize> = Vec::new();
    for i in mono {
        while h.len() >= 2 {
            let (a, b) = (opts[h[h.len() - 2]], opts[h[h.len() - 1]]);
            let c = opts[i];
            // keep b only if it lies strictly below the segment a-c (counter-clockwise turn)
            let cross = (b.bytes - a.bytes) as f64 * (c.cost - a.cost) - (c.bytes - a.bytes) as f64 * (b.cost - a.cost);
            if cross <= 0.0 {
                h.pop();
            } else {
                break;
            }
        }
        h.push(i);
    }
    h
}

pub fn solve(items: &[Vec<Opt>], capacity: u64) -> Solution {
    struct Inc {
        item: usize,
        to: usize,
        dbytes: u64,
        dcost: f64,
        eff: f64,
        step: usize,
    }
    let mut choice = vec![0usize; items.len()];
    let mut bytes = 0u64;
    let mut incs = Vec::new();
    let mut hulls = Vec::with_capacity(items.len());
    for (it, opts) in items.iter().enumerate() {
        let h = hull(opts);
        // start at the cheapest-in-bytes hull point
        choice[it] = h[0];
        bytes += opts[h[0]].bytes;
        for s in 1..h.len() {
            let (a, b) = (opts[h[s - 1]], opts[h[s]]);
            let db = b.bytes - a.bytes;
            let dc = a.cost - b.cost;
            incs.push(Inc { item: it, to: h[s], dbytes: db, dcost: dc, eff: dc / db as f64, step: s });
        }
        hulls.push(h);
    }
    incs.sort_by(|a, b| b.eff.partial_cmp(&a.eff).unwrap_or(std::cmp::Ordering::Equal));
    let mut next_step = vec![1usize; items.len()];
    let mut blocked = vec![false; items.len()];
    for inc in &incs {
        if blocked[inc.item] || inc.step != next_step[inc.item] {
            continue;
        }
        if bytes + inc.dbytes > capacity {
            blocked[inc.item] = true;
            continue;
        }
        bytes += inc.dbytes;
        choice[inc.item] = inc.to;
        next_step[inc.item] += 1;
        let _ = inc.dcost;
    }
    let cost = items.iter().zip(&choice).map(|(o, &c)| o[c].cost).sum();
    Solution { choice, bytes, cost }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(items: &[Vec<Opt>], cap: u64) -> f64 {
        fn rec(items: &[Vec<Opt>], i: usize, cap: u64, used: u64, cost: f64, best: &mut f64) {
            if i == items.len() {
                if cost < *best {
                    *best = cost;
                }
                return;
            }
            for o in &items[i] {
                if used + o.bytes <= cap {
                    rec(items, i + 1, cap, used + o.bytes, cost + o.cost, best);
                }
            }
        }
        let mut best = f64::INFINITY;
        rec(items, 0, cap, 0, 0.0, &mut best);
        best
    }

    #[test]
    fn exact_for_uniform_two_option_items() {
        // "stream" (0 bytes, cost p*miss) vs "resident" (1 unit, cost 0): greedy = top-p
        let p = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2];
        let items: Vec<Vec<Opt>> =
            p.iter().map(|&x| vec![Opt { bytes: 0, cost: x }, Opt { bytes: 10, cost: 0.0 }]).collect();
        for cap in [0, 10, 25, 30, 60] {
            let s = solve(&items, cap);
            assert!(s.bytes <= cap);
            assert!((s.cost - brute(&items, cap)).abs() < 1e-12, "cap {cap}");
        }
    }

    #[test]
    fn feasible_and_near_optimal_on_random_instances() {
        let mut x: u64 = 7;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..200 {
            let n = 5;
            let items: Vec<Vec<Opt>> = (0..n)
                .map(|_| {
                    let p = (rnd() % 1000) as f64 / 1000.0;
                    // floor, 2-bit, 3-bit, 4-bit copies: bytes up, cost (miss time + distortion) down
                    vec![
                        Opt { bytes: 0, cost: p * 10.0 },
                        Opt { bytes: 4, cost: p * 1.0 + 0.3 },
                        Opt { bytes: 6, cost: p * 0.5 + 0.1 },
                        Opt { bytes: 8, cost: p * 0.4 },
                    ]
                })
                .collect();
            let cap = rnd() % 40;
            let s = solve(&items, cap);
            assert!(s.bytes <= cap);
            let opt = brute(&items, cap);
            // greedy is within the largest single-increment cost of the optimum
            let max_inc = items.iter().map(|o| o[0].cost).fold(0.0, f64::max);
            assert!(s.cost <= opt + max_inc + 1e-9, "greedy {} vs opt {}", s.cost, opt);
            assert!(s.cost >= opt - 1e-9);
        }
    }

    #[test]
    fn hull_drops_points_above_the_chord() {
        let o = [
            Opt { bytes: 0, cost: 10.0 },
            Opt { bytes: 4, cost: 9.0 }, // above the chord from (0,10) to (6,0.6): not on the hull
            Opt { bytes: 6, cost: 0.6 },
        ];
        assert_eq!(hull(&o), vec![0, 2]);
        let o2 = [Opt { bytes: 0, cost: 10.0 }, Opt { bytes: 4, cost: 1.3 }, Opt { bytes: 6, cost: 0.6 }];
        assert_eq!(hull(&o2), vec![0, 1, 2]);
    }

    #[test]
    fn dominated_options_are_ignored() {
        let items = vec![vec![
            Opt { bytes: 0, cost: 5.0 },
            Opt { bytes: 5, cost: 6.0 }, // more bytes and more cost: never chosen
            Opt { bytes: 8, cost: 1.0 },
        ]];
        let s = solve(&items, 100);
        assert_eq!(s.choice[0], 2);
    }
}
