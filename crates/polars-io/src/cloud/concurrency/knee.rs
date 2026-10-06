//! Knee-based in-flight byte budget: the default model (`POLARS_INFLIGHT_BYTE_BUDGET_MODEL`).
//!
//! RampUp doubles the budget per round while it binds and the delivered bandwidth of the round
//! grows by at least 25% over the best round so far. The knee is the smallest budget at which
//! the delivered bandwidth reached its plateau. In Stable the target is
//! `gain x knee x (bw_max / bw_at_knee)`: it only grows with delivered bandwidth. A brake steps
//! the budget down when in-flight bytes rise without delivered bandwidth, or when delivered
//! bandwidth collapses at unchanged in-flight. RampUp also stops when the implied request
//! lifetime (bytes in use / delivered bandwidth) reaches `ramp_lifetime_ratio` times its minimum
//! in this RampUp: bandwidth that grows for reasons other than the budget (scan warm-up)
//! otherwise keeps the doubling going. Only rounds that use the budget and deliver bandwidth
//! count (not scan start, not stalls), and a rise must hold for a second round. The knee comes
//! from rounds where the budget bound only; without one, RampUp holds. While the requests
//! admitted by the HTTP rate limiter reach its rate, bandwidth follows the limiter, not the
//! budget: RampUp and Probe hold, and the brake and `bw_max` skip the round.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct KneeConfig {
    pub init_budget: u64,
    pub max_budget: u64,
    pub gain: f64,
    pub round_ticks: u32,
    /// Control tick length; a round lasts `round_ticks` ticks.
    pub tick: Duration,
    pub probe_interval: Duration,
    pub idle_grace: Duration,
    /// Idle longer than this since the last traffic: the next traffic starts over from init.
    pub idle_reset: Duration,
    /// Stop RampUp when the implied request lifetime reaches this multiple of its minimum.
    /// 0 disables.
    pub ramp_lifetime_ratio: f64,
}

/// Rounds a bandwidth level must hold to count as sustained: `bw_max` tracks the minimum of the
/// last `SUSTAINED_ROUNDS` rounds.
const SUSTAINED_ROUNDS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct KneeTick {
    pub now: Instant,
    /// Bytes of requests completed in this tick.
    pub bytes_done: u64,
    pub bytes_in_use: u64,
    /// Acquires that waited on the byte budget in this tick.
    pub bytes_parked: u64,
    /// Acquires parked on the byte budget at this tick.
    pub bytes_waiting: u64,
    pub bytes_sat: f64,
    /// Byte budget applied by the admission.
    pub bytes_budget: u64,
    /// Rate of the HTTP rate limiter (requests/s), if on.
    pub limiter_rate: Option<f64>,
    /// Requests admitted by the HTTP rate limiter in this tick (data, metadata and retries).
    pub limiter_admitted: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KneePhase {
    Init,
    RampUp,
    Stable,
    Probe,
}

impl KneePhase {
    pub fn label(&self) -> &'static str {
        match self {
            KneePhase::Init => "init",
            KneePhase::RampUp => "ramp_up",
            KneePhase::Stable => "stable",
            KneePhase::Probe => "probe",
        }
    }
}

/// Ticks in the brake windows (1 s each at a 100 ms tick).
const BRAKE_WINDOW_TICKS: usize = 10;

#[derive(Debug)]
pub struct KneeController {
    cfg: KneeConfig,
    phase: KneePhase,
    budget: u64,

    // Current round.
    round_ticks: u32,
    round_bytes: u64,
    round_parked: u64,
    round_sat_max: f64,
    round_budget: u64,
    round_in_use: u64,
    round_applied_max: u64,
    round_limiter_admitted: u64,
    // Requests the HTTP rate limiter's rate allowed over the round (sum of rate x tick length).
    round_limiter_tokens: Option<f64>,
    last_tick: Option<Instant>,
    last_round_end: Option<Instant>,

    // RampUp.
    best_bw: f64,
    min_lifetime: f64,
    lifetime_held: bool,
    no_growth_rounds: u32,
    not_binding_rounds: u32,
    ramp_hist: Vec<(u64, f64)>,

    // Stable.
    knee: Option<u64>,
    bw_at_knee: f64,
    bw_max: f64,
    last_bw_round: f64,
    recent_bw: VecDeque<f64>,
    stable_since: Option<Instant>,
    probe_rounds: u32,

    // Brake.
    in_use_hist: VecDeque<u64>,
    bytes_hist: VecDeque<u64>,
    brake_hits: u32,
    braking: bool,
    calm_rounds: u32,

    last_io: Option<Instant>,
    // Last traffic before going idle; `None` while not idle.
    idle_since: Option<Instant>,
}

impl KneeController {
    pub fn new(cfg: KneeConfig) -> Self {
        let budget = cfg.init_budget.min(cfg.max_budget);
        Self {
            cfg,
            phase: KneePhase::Init,
            budget,
            round_ticks: 0,
            round_bytes: 0,
            round_parked: 0,
            round_sat_max: 0.0,
            round_budget: budget,
            round_in_use: 0,
            round_applied_max: 0,
            round_limiter_admitted: 0,
            round_limiter_tokens: None,
            last_tick: None,
            last_round_end: None,
            best_bw: 0.0,
            min_lifetime: f64::INFINITY,
            lifetime_held: false,
            no_growth_rounds: 0,
            not_binding_rounds: 0,
            ramp_hist: Vec::new(),
            knee: None,
            bw_at_knee: 0.0,
            bw_max: 0.0,
            last_bw_round: 0.0,
            recent_bw: VecDeque::with_capacity(SUSTAINED_ROUNDS),
            stable_since: None,
            probe_rounds: 0,
            in_use_hist: VecDeque::with_capacity(2 * BRAKE_WINDOW_TICKS),
            bytes_hist: VecDeque::with_capacity(2 * BRAKE_WINDOW_TICKS),
            brake_hits: 0,
            braking: false,
            calm_rounds: 0,
            last_io: None,
            idle_since: None,
        }
    }

    pub fn phase(&self) -> KneePhase {
        self.phase
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    pub fn knee(&self) -> Option<u64> {
        self.knee
    }

    pub fn last_bw_round(&self) -> f64 {
        self.last_bw_round
    }

    pub fn braking(&self) -> bool {
        self.braking
    }

    /// Advances one control tick and returns the byte budget to apply.
    pub fn step(&mut self, t: KneeTick) -> u64 {
        if t.bytes_done > 0 || t.bytes_in_use > 0 {
            if let Some(since) = self.idle_since.take() {
                if t.now.duration_since(since) > self.cfg.idle_reset {
                    *self = Self::new(self.cfg.clone());
                } else if self.knee.is_some() {
                    self.resume_stable(t.now);
                } else {
                    // No knee yet: the RampUp state is from the traffic before the idle period.
                    *self = Self::new(self.cfg.clone());
                }
            }
            self.last_io = Some(t.now);
        } else if self.idle_since.is_some() {
            return self.budget;
        }
        if let Some(last) = self.last_io
            && self.phase != KneePhase::Init
            && t.now.duration_since(last) > self.cfg.idle_grace
        {
            self.enter_idle();
            return self.budget;
        }

        self.push_brake_window(t);

        self.round_ticks += 1;
        self.round_bytes += t.bytes_done;
        self.round_parked += t.bytes_parked + t.bytes_waiting;
        self.round_sat_max = self.round_sat_max.max(t.bytes_sat);
        self.round_in_use += t.bytes_in_use;
        self.round_applied_max = self.round_applied_max.max(t.bytes_budget);
        let tick_secs = self.last_tick.map_or_else(
            || self.cfg.tick.as_secs_f64(),
            |last| t.now.duration_since(last).as_secs_f64(),
        );
        self.last_tick = Some(t.now);
        if let Some(rate) = t.limiter_rate {
            self.round_limiter_admitted += t.limiter_admitted;
            *self.round_limiter_tokens.get_or_insert(0.0) += rate * tick_secs;
        }
        if self.round_ticks >= self.cfg.round_ticks {
            self.end_round(t.now);
        }
        self.budget
    }

    fn push_brake_window(&mut self, t: KneeTick) {
        if self.in_use_hist.len() == 2 * BRAKE_WINDOW_TICKS {
            self.in_use_hist.pop_front();
            self.bytes_hist.pop_front();
        }
        self.in_use_hist.push_back(t.bytes_in_use);
        self.bytes_hist.push_back(t.bytes_done);
    }

    /// Over the last 1 s vs the 1 s before: in-flight bytes up >= 25% while delivered bytes are
    /// up < 5%, or delivered bytes down >= 50% while in-flight bytes are not down.
    fn brake_signal(&self) -> bool {
        if self.in_use_hist.len() < 2 * BRAKE_WINDOW_TICKS {
            return false;
        }
        let sum = |v: &VecDeque<u64>, r: std::ops::Range<usize>| -> f64 {
            v.range(r).map(|&x| x as f64).sum()
        };
        let (prev, now) = (
            0..BRAKE_WINDOW_TICKS,
            BRAKE_WINDOW_TICKS..2 * BRAKE_WINDOW_TICKS,
        );
        let (in_prev, in_now) = (
            sum(&self.in_use_hist, prev.clone()),
            sum(&self.in_use_hist, now.clone()),
        );
        let (bw_prev, bw_now) = (sum(&self.bytes_hist, prev), sum(&self.bytes_hist, now));
        let creep = in_now >= 1.25 * in_prev && bw_now < 1.05 * bw_prev;
        let collapse = bw_now < 0.5 * bw_prev && in_now >= 0.9 * in_prev;
        in_prev > 0.0 && bw_prev > 0.0 && (creep || collapse)
    }

    fn end_round(&mut self, now: Instant) {
        // Measured: the control loop skips missed ticks, so a round can outlast its tick count.
        let secs = self.last_round_end.map_or_else(
            || self.cfg.tick.as_secs_f64() * self.round_ticks as f64,
            |t| now.duration_since(t).as_secs_f64().max(f64::EPSILON),
        );
        self.last_round_end = Some(now);
        let bw_round = self.round_bytes as f64 / secs;
        let binding = self.round_parked > 0 || self.round_sat_max >= 0.9;
        // Admitted requests reach the HTTP rate limiter's rate: bandwidth follows the limiter.
        let limiter_bound = self
            .round_limiter_tokens
            .is_some_and(|tokens| self.round_limiter_admitted as f64 >= 0.9 * tokens);
        let round_budget = self.round_budget;
        self.last_bw_round = bw_round;
        let mean_in_use = self.round_in_use as f64 / self.round_ticks as f64;
        let lifetime = if self.round_bytes > 0 {
            mean_in_use / bw_round
        } else {
            0.0
        };
        if self.recent_bw.len() == SUSTAINED_ROUNDS {
            self.recent_bw.pop_front();
        }
        self.recent_bw.push_back(bw_round);
        // 0 until the window is full (start, after idle).
        let sustained_bw = if self.recent_bw.len() == SUSTAINED_ROUNDS {
            self.recent_bw.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };

        // The round binds and uses at least half of the applied budget (not scan start).
        let lifetime_measured =
            binding && lifetime > 0.0 && mean_in_use >= 0.5 * self.round_applied_max as f64;

        if !limiter_bound && self.brake_signal() {
            self.brake_hits += 1;
            self.calm_rounds = 0;
        } else {
            self.brake_hits = 0;
            self.calm_rounds += 1;
        }
        if self.brake_hits >= 2 {
            self.braking = true;
        } else if self.braking && self.calm_rounds >= 5 {
            self.braking = false;
        }

        match self.phase {
            KneePhase::Init => {
                if self.round_bytes > 0 {
                    self.phase = KneePhase::RampUp;
                    self.best_bw = bw_round;
                    self.min_lifetime = f64::INFINITY;
                    self.lifetime_held = false;
                    self.ramp_hist.clear();
                    // Scan start: not a knee sample.
                    if binding && !limiter_bound {
                        self.set_budget(self.budget.saturating_mul(2));
                    }
                }
            },
            // Bandwidth follows the HTTP rate limiter, not the budget: hold.
            KneePhase::RampUp if limiter_bound => {
                // Growth is judged from the best bandwidth so far, including the hold.
                self.best_bw = self.best_bw.max(bw_round);
                self.no_growth_rounds = 0;
                self.not_binding_rounds = 0;
                self.lifetime_held = false;
            },
            KneePhase::RampUp => {
                let ratio = self.cfg.ramp_lifetime_ratio;
                let lifetime_valid = lifetime_measured && bw_round >= 0.5 * self.best_bw;
                if lifetime_valid {
                    self.min_lifetime = self.min_lifetime.min(lifetime);
                }
                let lifetime_rose =
                    ratio > 0.0 && lifetime_valid && lifetime >= ratio * self.min_lifetime;
                let grew = bw_round >= 1.25 * self.best_bw;
                self.best_bw = self.best_bw.max(bw_round);
                if lifetime_rose && self.lifetime_held {
                    // The knee comes from the rounds before the lifetime rose.
                    self.enter_stable(now);
                } else if lifetime_rose {
                    // Right after a doubling, bytes in use can double before bandwidth does
                    // (requests that outlive a round): confirm in the next round.
                    self.lifetime_held = true;
                } else {
                    self.lifetime_held = false;
                    // The knee comes from rounds that bind and use at least half of the budget.
                    if lifetime_measured {
                        self.ramp_hist.push((round_budget, bw_round));
                    }
                    let has_knee = !self.ramp_hist.is_empty();
                    if self.braking && has_knee {
                        self.enter_stable(now);
                    } else if binding && grew {
                        self.no_growth_rounds = 0;
                        self.not_binding_rounds = 0;
                        self.set_budget(self.budget.saturating_mul(2));
                    } else if binding {
                        self.no_growth_rounds += 1;
                        if self.no_growth_rounds >= 2 && has_knee {
                            self.enter_stable(now);
                        }
                    } else {
                        // Another limit binds (requests, pipeline, decode): hold. Without a
                        // binding round there is no knee yet: stay in RampUp.
                        self.not_binding_rounds += 1;
                        if self.not_binding_rounds >= 5 && has_knee {
                            self.enter_stable(now);
                        }
                    }
                }
            },
            KneePhase::Stable => {
                if binding && !limiter_bound {
                    self.bw_max = self.bw_max.max(sustained_bw);
                }
                let target = self.stable_target();
                if self.braking {
                    // `set_budget` keeps it at or above the initial budget.
                    let floor = self.knee.map_or(self.cfg.init_budget, |k| k / 2);
                    self.set_budget(((self.budget as f64 * 0.8) as u64).max(floor));
                    // The probe interval restarts when the brake releases.
                    self.stable_since = Some(now);
                } else {
                    self.set_budget(target);
                    if self
                        .stable_since
                        .is_some_and(|s| now.duration_since(s) >= self.cfg.probe_interval)
                    {
                        self.phase = KneePhase::Probe;
                        self.probe_rounds = 0;
                        self.set_budget((target as f64 * 1.25) as u64);
                    }
                }
            },
            KneePhase::Probe => {
                self.probe_rounds += 1;
                if self.braking {
                    self.resume_stable(now);
                } else if binding && !limiter_bound && sustained_bw >= 1.1 * self.bw_max {
                    // More bandwidth appeared while the budget binds: ramp again from here.
                    self.phase = KneePhase::RampUp;
                    self.best_bw = bw_round;
                    self.min_lifetime = f64::INFINITY;
                    self.lifetime_held = false;
                    self.no_growth_rounds = 0;
                    self.not_binding_rounds = 0;
                    self.ramp_hist.clear();
                    self.ramp_hist.push((round_budget, bw_round));
                    self.set_budget(self.budget.saturating_mul(2));
                } else if self.probe_rounds >= 5 {
                    self.resume_stable(now);
                }
            },
        }

        self.reset_round();
    }

    fn reset_round(&mut self) {
        self.round_ticks = 0;
        self.round_bytes = 0;
        self.round_parked = 0;
        self.round_sat_max = 0.0;
        self.round_in_use = 0;
        self.round_applied_max = 0;
        self.round_limiter_admitted = 0;
        self.round_limiter_tokens = None;
        self.round_budget = self.budget;
    }

    /// Nothing completed or in flight for `idle_grace` (between queries, or a scan stalled
    /// downstream): keep the knee, `bw_max` and the budget, drop the round, bandwidth and brake
    /// windows. Frozen until traffic resumes; then Stable with a fresh probe timer, or from init
    /// after `idle_reset`.
    fn enter_idle(&mut self) {
        self.idle_since = self.last_io;
        self.last_io = None;
        self.reset_round();
        self.last_round_end = None;
        self.last_tick = None;
        self.recent_bw.clear();
        self.in_use_hist.clear();
        self.bytes_hist.clear();
        self.brake_hits = 0;
        self.braking = false;
        self.calm_rounds = 0;
    }

    /// Back to Stable with the current knee and `bw_max`.
    fn resume_stable(&mut self, now: Instant) {
        self.phase = KneePhase::Stable;
        self.stable_since = Some(now);
        self.set_budget(self.stable_target());
    }

    /// Fixes the knee from the RampUp history and enters Stable.
    fn enter_stable(&mut self, now: Instant) {
        let plateau = self.ramp_hist.iter().map(|&(_, bw)| bw).fold(0.0, f64::max);
        let knee = self
            .ramp_hist
            .iter()
            .filter(|&&(_, bw)| bw >= 0.9 * plateau)
            .map(|&(budget, _)| budget)
            .min()
            .unwrap_or(self.budget);
        self.knee = Some(knee);
        self.bw_at_knee = plateau.max(1.0);
        self.bw_max = self.bw_at_knee;
        self.phase = KneePhase::Stable;
        self.stable_since = Some(now);
        self.set_budget(self.stable_target());
    }

    fn stable_target(&self) -> u64 {
        let knee = self.knee.unwrap_or(self.cfg.init_budget) as f64;
        (self.cfg.gain * knee * (self.bw_max / self.bw_at_knee).max(1.0)) as u64
    }

    fn set_budget(&mut self, budget: u64) {
        let budget = budget.max(self.cfg.init_budget.min(self.budget).max(1));
        self.budget = budget.min(self.cfg.max_budget);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KneeConfig {
        KneeConfig {
            init_budget: 100,
            max_budget: u64::MAX,
            gain: 1.0,
            round_ticks: 2,
            tick: Duration::from_millis(100),
            probe_interval: Duration::from_secs(3),
            idle_grace: Duration::from_secs(5),
            idle_reset: Duration::from_secs(30),
            ramp_lifetime_ratio: 2.0,
        }
    }

    /// One 100 ms tick `i` on the clock `t0`; `parked` acquires waited on the budget.
    fn tick(
        c: &mut KneeController,
        t0: Instant,
        i: u64,
        bytes_done: u64,
        in_use: u64,
        parked: u64,
    ) {
        let budget = c.budget();
        c.step(KneeTick {
            now: t0 + Duration::from_millis(100 * (i + 1)),
            bytes_done,
            bytes_in_use: in_use,
            bytes_parked: parked,
            bytes_waiting: 0,
            bytes_sat: in_use as f64 / budget as f64,
            bytes_budget: budget,
            limiter_rate: None,
            limiter_admitted: 0,
        });
    }

    /// Ticks `from..from + n`: delivered bytes per tick = min(budget, cap) (a link that saturates
    /// at `cap` in flight).
    fn run(c: &mut KneeController, t0: Instant, from: u64, cap: u64, n: u64) {
        for i in from..from + n {
            let budget = c.budget();
            let in_use = budget.min(cap);
            tick(c, t0, i, in_use, in_use, u64::from(budget <= cap));
        }
    }

    #[test]
    fn ramps_to_the_knee_and_holds() {
        let mut c = KneeController::new(cfg());
        run(&mut c, Instant::now(), 0, 1600, 40);
        assert_eq!(c.phase(), KneePhase::Stable);
        // Doubling 100 -> 1600 reaches the plateau; the target is gain x knee.
        assert_eq!(c.knee(), Some(1600));
        assert_eq!(c.budget(), 1600);
    }

    /// Delivered bandwidth grows 30% per round regardless of the budget (warm-up) while the whole
    /// budget stays in use.
    #[test]
    fn warm_up_growth_stops_ramp_up_on_lifetime() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        let mut bytes_per_tick = 100.0;
        for i in 0..16 {
            if i > 0 && i % 2 == 0 {
                bytes_per_tick *= 1.3;
            }
            let budget = c.budget();
            tick(&mut c, t0, i, bytes_per_tick as u64, budget, 1);
        }
        // The lifetime rise at 1600 is confirmed in the next round; the knee is the last budget
        // before it.
        assert_eq!(c.knee(), Some(800));
        assert_eq!(c.phase(), KneePhase::Stable);
    }

    /// Traffic that never fills the budget (metadata) does not fix a knee: RampUp holds until
    /// the budget binds.
    #[test]
    fn no_knee_without_binding() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        for i in 0..20 {
            tick(&mut c, t0, i, 5, 5, 0);
        }
        assert_eq!(c.phase(), KneePhase::RampUp);
        assert_eq!(c.knee(), None);
        run(&mut c, t0, 20, 1600, 40);
        assert_eq!(c.knee(), Some(1600));
        assert_eq!(c.budget(), 1600);
    }

    /// Idle longer than `idle_grace` (e.g. a scan stalled downstream) keeps the knee; idle longer
    /// than `idle_reset` (e.g. the next query) starts over from init.
    #[test]
    fn idle_keeps_the_knee_until_reset() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        run(&mut c, t0, 0, 1600, 40);
        let stable = c.budget();
        for i in 40..120 {
            tick(&mut c, t0, i, 0, 0, 0);
        }
        assert_eq!(c.knee(), Some(1600));
        run(&mut c, t0, 120, 1600, 10);
        assert_eq!(c.phase(), KneePhase::Stable);
        assert_eq!(c.budget(), stable);
        // 40 s without traffic.
        for i in 130..530 {
            tick(&mut c, t0, i, 0, 0, 0);
        }
        run(&mut c, t0, 530, 1600, 1);
        assert_eq!(c.knee(), None);
        assert_eq!(c.budget(), 100);
    }

    /// Scan start (fills the budget, delivers little); then the HTTP rate limiter admits requests
    /// at its rate, half of them metadata, and doubles it every round: data bandwidth doubles too,
    /// but RampUp holds. Once the limiter releases, demand (150) stays below the budget: still no
    /// knee, so no Stable target scaled from scan-start bandwidth.
    #[test]
    fn ramp_up_holds_while_the_limiter_binds() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        let mut step = |i: u64, bytes_done: u64, demand: u64, rate: f64, admitted: u64| {
            let budget = c.budget();
            let in_use = budget.min(demand);
            c.step(KneeTick {
                now: t0 + Duration::from_millis(100 * (i + 1)),
                bytes_done,
                bytes_in_use: in_use,
                bytes_parked: u64::from(budget <= demand),
                bytes_waiting: 0,
                bytes_sat: in_use as f64 / budget as f64,
                bytes_budget: budget,
                limiter_rate: Some(rate),
                limiter_admitted: admitted,
            });
            c.budget()
        };
        step(0, 1, u64::MAX, 100.0, 1);
        step(1, 1, u64::MAX, 100.0, 1);
        let mut rate = 100.0;
        for i in 2..22 {
            if i % 2 == 0 {
                rate *= 2.0;
            }
            let admitted = (rate * 0.1) as u64;
            step(i, 10 * (admitted / 2), u64::MAX, rate, admitted);
        }
        for i in 22..42 {
            step(i, 5000, 150, 1e6, 50);
        }
        assert_eq!(c.phase(), KneePhase::RampUp);
        assert_eq!(c.knee(), None);
        assert_eq!(c.budget(), 200);
    }

    /// Slow link: requests outlive several rounds, so binding rounds deliver nothing. No knee
    /// from those; once bytes arrive the knee is the budget they were delivered at.
    #[test]
    fn no_knee_from_rounds_without_delivery() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        let full = |c: &mut KneeController, i: u64, bytes_done: u64| {
            let budget = c.budget();
            tick(c, t0, i, bytes_done, budget, 1);
        };
        full(&mut c, 0, 1);
        full(&mut c, 1, 1);
        for i in 2..12 {
            full(&mut c, i, 0);
        }
        assert_eq!(c.phase(), KneePhase::RampUp);
        assert_eq!(c.knee(), None);
        for i in 12..24 {
            full(&mut c, i, 50);
        }
        assert_eq!(c.knee(), Some(200));
        assert_eq!(c.budget(), 200);
    }
}
