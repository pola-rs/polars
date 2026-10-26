//! The knee is the smallest in-flight byte budget at which the delivered bandwidth stops growing
//! with the budget. This model, the default (`POLARS_INFLIGHT_BYTE_BUDGET_MODEL`), estimates the
//! knee and sets the budget from it.
//!
//! Successor of the TTFB-based BDP model (`model.rs`, `regime.rs`, after BBR), with its phase
//! structure; it estimates the knee instead of computing the budget as bandwidth x TTFB.
//!
//! Core interactions:
//! - Observation (`Stats`): delivered bandwidth (response body bytes as they arrive), bytes in use
//!   and what bound each round of up to `ROUND_TICKS` ticks. Bandwidth counts the ticks after the
//!   first, assuming a budget change settles within one tick.
//! - Estimation (`Ramp`, `Knee`): the knee, the best guess from the (budget, bandwidth) samples
//!   of RampUp; an accepted probe replaces it.
//! - Control (`Phase`): sets the budget from the knee and what to learn next: RampUp explores
//!   upward, Stable holds the knee, Probe explores above it, the brake backs off.
//!
//! Phases:
//! - Init: waits for a round with arrived bytes (scan start, not a sample); doubles the budget
//!   once if it bound and the limiter did not.
//! - RampUp: doubles the budget per round while it binds and the bandwidth grows by
//!   `RAMP_BW_GROWTH` over the best round (a round ends after two ticks when the growth shows).
//!   A round where the budget bound, bandwidth arrived and a settled tick used half of the budget
//!   is a sample. Once there is a sample, RampUp ends at the knee, the smallest sampled budget with
//!   `KNEE_BW_FRACTION` of the best sampled bandwidth, on: no growth; the budget not binding
//!   (another limit binds: requests, pipeline, decode); the request lifetime (bytes in use /
//!   bandwidth) at `RAMP_LIFETIME_RATIO` x its minimum in two rounds in a row (growth from scan
//!   warm-up, not from the budget; the first such round holds and is no sample); or the brake.
//! - Stable: holds the knee; after `PROBE_INTERVAL` without the brake, Probe.
//! - Probe: tries `PROBE_BUDGET_STEP` x the knee for at most `PROBE_ROUNDS` rounds. If, in a round
//!   where the budget binds, the sustained bandwidth (minimum of the last `SUSTAINED_ROUNDS`
//!   rounds) reaches `PROBE_BW_GROWTH` x the probe reference, that budget becomes the knee. Back
//!   to Stable either way.
//! - Idle: no traffic for `IDLE_GRACE`. A knee is kept until `IDLE_RESET`, then back to Init;
//!   without one, back to Init at once.
//!
//! Bounds:
//! - The budget stays within [initial budget, max budget].
//! - While the HTTP rate limiter binds, bandwidth follows the limiter. Such a round holds RampUp,
//!   cannot make a probe the knee, does not raise the probe reference and is no brake signal; its
//!   bandwidth still counts in the recent rounds that the sustained bandwidth is taken from.
//! - The brake (in-flight bytes up without bandwidth, or a bandwidth collapse) ends RampUp (once
//!   there is a sample) and Probe; in Stable it steps the budget down by `BRAKE_BUDGET_STEP` per
//!   round, not below half the knee.

use std::collections::VecDeque;
use std::mem;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct KneeConfig {
    /// Initial byte budget, also the lowest (capped by `max_budget`).
    pub init_budget: u64,
    /// Highest byte budget.
    pub max_budget: u64,
    /// Control tick length.
    pub tick: Duration,
}

/// Round length in ticks; a growing RampUp round ends after two.
const ROUND_TICKS: u32 = 3;
/// A budget (bytes or requests) binds a tick when this fraction of it is in use; the HTTP rate
/// limiter, when the admitted requests reach this fraction of its rate.
pub(super) const BOUND_FRACTION: f64 = 0.9;
/// RampUp doubles the budget while a round's bandwidth grows by this factor over the best round.
const RAMP_BW_GROWTH: f64 = 1.25;
/// Once there is a sample, RampUp ends after this many bound rounds without growth.
const NO_GROWTH_ROUNDS: u32 = 2;
/// Once there is a sample, RampUp ends after this many rounds in which the budget did not bind.
const NOT_BINDING_ROUNDS: u32 = 5;
/// RampUp ends when the implied request lifetime reaches this multiple of its minimum in two
/// rounds in a row.
const RAMP_LIFETIME_RATIO: f64 = 2.0;
/// The knee is the smallest sampled budget with this fraction of the best sampled bandwidth.
const KNEE_BW_FRACTION: f64 = 0.9;
/// Time in Stable between probes.
const PROBE_INTERVAL: Duration = Duration::from_secs(3);
/// Probe raises the budget by this factor.
const PROBE_BUDGET_STEP: f64 = 1.25;
/// Probe keeps its budget if the sustained bandwidth reaches this multiple of the reference.
const PROBE_BW_GROWTH: f64 = 1.1;
/// Probe ends after at most this many rounds.
const PROBE_ROUNDS: u32 = 5;
/// In Stable, the brake multiplies the budget by this factor per round.
const BRAKE_BUDGET_STEP: f64 = 0.8;
/// No traffic for this long: idle.
const IDLE_GRACE: Duration = Duration::from_secs(5);
/// Idle longer than this since the last traffic: the next traffic starts over from init.
const IDLE_RESET: Duration = Duration::from_secs(30);
/// Rounds a bandwidth level must hold to count as sustained.
const SUSTAINED_ROUNDS: usize = 3;
/// Ticks in each of the two brake windows (1 s each at a 100 ms tick).
const BRAKE_WINDOW_TICKS: usize = 10;

#[derive(Clone, Copy, Debug)]
pub struct KneeTick {
    pub now: Instant,
    /// Response body bytes of data requests that arrived in this tick.
    pub bytes_arrived: u64,
    /// Bytes in use of the byte budget.
    pub bytes_in_use: u64,
    /// Byte acquires that waited in this tick: admitted after parking, or parked at the tick.
    pub byte_acquires_waited: u64,
    /// Byte budget applied by the admission (> 0: at least its floor).
    pub bytes_budget: u64,
    /// The HTTP rate limiter, if on.
    pub limiter: Option<LimiterTick>,
}

impl KneeTick {
    /// The byte budget held requests back: an acquire waited, or `BOUND_FRACTION` of it in use.
    pub fn bytes_bound(&self) -> bool {
        self.byte_acquires_waited > 0
            || (self.bytes_budget > 0
                && self.bytes_in_use as f64 / self.bytes_budget as f64 >= BOUND_FRACTION)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LimiterTick {
    /// Rate in requests/s.
    pub rate: f64,
    /// Requests admitted in this tick (data, metadata and retries).
    pub admitted: u64,
}

impl LimiterTick {
    /// The admitted requests reached `BOUND_FRACTION` of the rate over `tick_secs`.
    pub fn bound(&self, tick_secs: f64) -> bool {
        self.admitted as f64 >= BOUND_FRACTION * self.rate * tick_secs
    }
}

/// What set the knee.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KneeSource {
    NoGrowth,
    NotBinding,
    Lifetime,
    Brake,
    Probe,
}

impl KneeSource {
    pub fn label(&self) -> &'static str {
        match self {
            KneeSource::NoGrowth => "no_growth",
            KneeSource::NotBinding => "not_binding",
            KneeSource::Lifetime => "lifetime",
            KneeSource::Brake => "brake",
            KneeSource::Probe => "probe",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Knee {
    budget: u64,
    /// Reference for Probe, in bytes/s: the best sampled RampUp bandwidth (at least 1), raised to
    /// the sustained bandwidth of Stable rounds where the budget binds and the limiter does not;
    /// an accepted probe sets it to its sustained bandwidth.
    probe_ref_bw: f64,
    source: KneeSource,
}

// TODO: no Drain phase (BBR): leaving RampUp drops the budget to the knee at once, and the
// requests admitted under the RampUp budget drain as they complete. A brake that ended RampUp
// stays on in Stable and steps the budget below the knee.
#[derive(Debug, Default)]
enum Phase {
    #[default]
    Init,
    RampUp(Ramp),
    /// RampUp found the knee.
    Knee(Knee, KneeMode),
}

#[derive(Clone, Copy, Debug)]
enum KneeMode {
    /// `since`: start of the probe interval.
    Stable {
        since: Instant,
    },
    Probe {
        rounds: u32,
    },
    /// `since`: the last traffic.
    Idle {
        since: Instant,
    },
}

#[derive(Debug)]
struct Ramp {
    /// Best round bandwidth so far.
    best_bw: f64,
    /// Shortest request lifetime measured so far.
    min_lifetime: f64,
    /// The last round's lifetime reached `RAMP_LIFETIME_RATIO` times the minimum.
    lifetime_rose: bool,
    /// Bound rounds without growth since the last doubling or limiter-bound round.
    no_growth_rounds: u32,
    /// Rounds in which the budget did not bind since the last doubling or limiter-bound round.
    not_binding_rounds: u32,
    /// (budget, bandwidth) of the sample rounds.
    samples: Vec<(u64, f64)>,
}

enum RampStep {
    Hold,
    Double,
    Stop(KneeSource),
}

impl Ramp {
    fn new(best_bw: f64) -> Self {
        Self {
            best_bw,
            min_lifetime: f64::INFINITY,
            lifetime_rose: false,
            no_growth_rounds: 0,
            not_binding_rounds: 0,
            samples: Vec::new(),
        }
    }

    fn grows(&self, bw: f64) -> bool {
        bw > 0.0 && bw >= RAMP_BW_GROWTH * self.best_bw
    }

    /// Learns from a round. `braking`: the brake after this round; `recent_max`: the best
    /// bandwidth of the last rounds.
    fn round(&mut self, r: &Round, braking: bool, recent_max: f64) -> RampStep {
        if r.limiter_bound {
            // The reference stays at the last level measured without the limiter.
            self.no_growth_rounds = 0;
            self.not_binding_rounds = 0;
            self.lifetime_rose = false;
            return RampStep::Hold;
        }

        let bw = r.bw();
        let lifetime = r.lifetime().filter(|_| bw >= 0.5 * self.best_bw);
        if let Some(lifetime) = lifetime {
            self.min_lifetime = self.min_lifetime.min(lifetime);
        }
        let lifetime_rose =
            lifetime.is_some_and(|lifetime| lifetime >= RAMP_LIFETIME_RATIO * self.min_lifetime);
        let grew = self.grows(bw);
        self.best_bw = self.best_bw.max(bw);

        if lifetime_rose {
            // Right after a doubling, bytes in use can double before bandwidth does (requests
            // that outlive a round): stop on the second rise in a row. The knee comes from the
            // rounds before.
            if self.lifetime_rose {
                return RampStep::Stop(KneeSource::Lifetime);
            }
            self.lifetime_rose = true;
            return RampStep::Hold;
        }
        self.lifetime_rose = false;

        if r.knee_sample() {
            self.record_sample(r.budget, bw);
        }
        let has_samples = !self.samples.is_empty();
        if braking && has_samples {
            return RampStep::Stop(KneeSource::Brake);
        }
        if r.bytes_bound && grew {
            self.no_growth_rounds = 0;
            self.not_binding_rounds = 0;
            return RampStep::Double;
        }
        if r.bytes_bound {
            self.no_growth_rounds += 1;
            if self.no_growth_rounds >= NO_GROWTH_ROUNDS && has_samples {
                return RampStep::Stop(KneeSource::NoGrowth);
            }
        } else {
            // Another limit binds (requests, pipeline, decode). Without a sample there is no knee
            // yet: stay in RampUp.
            self.not_binding_rounds += 1;
            if self.not_binding_rounds >= NOT_BINDING_ROUNDS && has_samples {
                // The budget reached what another limit allows: a knee candidate.
                self.record_sample(r.budget, recent_max);
                return RampStep::Stop(KneeSource::NotBinding);
            }
        }
        RampStep::Hold
    }

    /// RampUp budgets never decrease: a sample at the last budget keeps the higher bandwidth.
    fn record_sample(&mut self, budget: u64, bw: f64) {
        if let Some((last_budget, last_bw)) = self.samples.last_mut()
            && *last_budget == budget
        {
            *last_bw = last_bw.max(bw);
        } else {
            self.samples.push((budget, bw));
        }
    }

    /// The smallest sampled budget with `KNEE_BW_FRACTION` of the best sampled bandwidth. Every
    /// RampUp exit has a sample; `budget` is the fallback without one.
    fn knee(&self, budget: u64, source: KneeSource) -> Knee {
        let plateau = self.samples.iter().map(|&(_, bw)| bw).fold(0.0, f64::max);
        let budget = self
            .samples
            .iter()
            .filter(|&&(_, bw)| bw >= KNEE_BW_FRACTION * plateau)
            .map(|&(budget, _)| budget)
            .min()
            .unwrap_or(budget);
        Knee {
            budget,
            probe_ref_bw: plateau.max(1.0),
            source,
        }
    }
}

/// The ticks of a round. The settled ticks are the ticks after the first.
#[derive(Debug, Default)]
struct Round {
    /// Controller budget at the start of the round: the key of a sample.
    budget: u64,
    ticks: u32,
    /// Arrived bytes over all ticks.
    arrived: u64,
    /// Bytes in use, summed over all ticks.
    in_use_sum: u64,
    /// Highest byte budget applied by the admission: the base of the half-budget checks.
    applied_budget_max: u64,
    /// The byte budget bound in a tick (`KneeTick::bytes_bound`).
    bytes_bound: bool,
    /// The HTTP rate limiter bound in a tick (`LimiterTick::bound`).
    limiter_bound: bool,
    settled_arrived: u64,
    settled_secs: f64,
    settled_in_use_sum: u64,
    settled_in_use_max: u64,
}

impl Round {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            ..Default::default()
        }
    }

    fn push(&mut self, t: &KneeTick, tick_secs: f64) {
        self.ticks += 1;
        self.arrived += t.bytes_arrived;
        self.in_use_sum += t.bytes_in_use;
        self.applied_budget_max = self.applied_budget_max.max(t.bytes_budget);
        self.bytes_bound |= t.bytes_bound();
        self.limiter_bound |= t.limiter.is_some_and(|l| l.bound(tick_secs));
        if self.ticks > 1 {
            self.settled_arrived += t.bytes_arrived;
            self.settled_secs += tick_secs;
            self.settled_in_use_sum += t.bytes_in_use;
            self.settled_in_use_max = self.settled_in_use_max.max(t.bytes_in_use);
        }
    }

    /// Delivered bandwidth over the settled ticks, in bytes/s.
    fn bw(&self) -> f64 {
        if self.settled_secs > 0.0 {
            self.settled_arrived as f64 / self.settled_secs
        } else {
            0.0
        }
    }

    fn half_budget(&self) -> f64 {
        0.5 * self.applied_budget_max as f64
    }

    /// The byte budget bound, bandwidth > 0, and a settled tick used at least half of the budget.
    fn knee_sample(&self) -> bool {
        self.bytes_bound && self.bw() > 0.0 && self.settled_in_use_max as f64 >= self.half_budget()
    }

    /// Mean bytes in use / bandwidth, in seconds, if the byte budget bound and the settled ticks
    /// used at least half of it on the mean. The round has at least two ticks.
    fn lifetime(&self) -> Option<f64> {
        let bw = self.bw();
        let settled_in_use = self.settled_in_use_sum as f64 / (self.ticks - 1) as f64;
        let mean_in_use = self.in_use_sum as f64 / self.ticks as f64;
        (self.bytes_bound && bw > 0.0 && settled_in_use >= self.half_budget())
            .then(|| mean_in_use / bw)
            .filter(|&lifetime| lifetime > 0.0)
    }
}

/// What the controller measures, in every phase.
#[derive(Debug)]
struct Stats {
    round: Round,
    recent_bw: RecentBw,
    brake: Brake,
    last_tick: Option<Instant>,
    /// Bandwidth of the last finished round.
    last_bw_round: f64,
}

impl Stats {
    fn new(budget: u64) -> Self {
        Self {
            round: Round::new(budget),
            recent_bw: RecentBw::new(),
            brake: Brake::new(),
            last_tick: None,
            last_bw_round: 0.0,
        }
    }

    /// Drops the round, bandwidth and brake history (idle); the next round starts at `budget`.
    fn clear(&mut self, budget: u64) {
        self.round = Round::new(budget);
        self.recent_bw.clear();
        self.brake.reset();
        self.last_tick = None;
    }

    /// `nominal_tick`: the length of the first tick, and of the first after `clear`.
    fn push(&mut self, t: &KneeTick, nominal_tick: Duration) {
        self.brake.push(t.bytes_in_use, t.bytes_arrived);
        let tick_secs = self.last_tick.map_or_else(
            || nominal_tick.as_secs_f64(),
            |last| t.now.duration_since(last).as_secs_f64(),
        );
        self.last_tick = Some(t.now);
        self.round.push(t, tick_secs);
    }

    /// Ends the round: records its bandwidth and updates the brake. The next round starts with
    /// `Round::new`.
    fn finish_round(&mut self) -> Round {
        let r = mem::take(&mut self.round);
        self.last_bw_round = r.bw();
        self.recent_bw.push(self.last_bw_round);
        self.brake.end_round(r.limiter_bound);
        r
    }

    fn braking(&self) -> bool {
        self.brake.on
    }
}

/// Congestion detector. Signal, over the last `BRAKE_WINDOW_TICKS` ticks vs the ones before:
/// in-flight bytes up >= 25% while delivered bytes are up < 5%, or delivered bytes down > 50%
/// while in-flight bytes are down <= 10%. On after two rounds with the signal, off after five
/// rounds without; rounds where the HTTP rate limiter binds count as without.
#[derive(Debug)]
struct Brake {
    /// (bytes in use, bytes arrived) of the last `2 * BRAKE_WINDOW_TICKS` ticks.
    window: VecDeque<(u64, u64)>,
    hits: u32,
    calm_rounds: u32,
    on: bool,
}

impl Brake {
    fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(2 * BRAKE_WINDOW_TICKS),
            hits: 0,
            calm_rounds: 0,
            on: false,
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.hits = 0;
        self.calm_rounds = 0;
        self.on = false;
    }

    fn push(&mut self, bytes_in_use: u64, bytes_arrived: u64) {
        if self.window.len() == 2 * BRAKE_WINDOW_TICKS {
            self.window.pop_front();
        }
        self.window.push_back((bytes_in_use, bytes_arrived));
    }

    fn signal(&self) -> bool {
        if self.window.len() < 2 * BRAKE_WINDOW_TICKS {
            return false;
        }
        let sum = |range: std::ops::Range<usize>| {
            self.window
                .range(range)
                .fold((0.0, 0.0), |(in_use, arrived), &(i, a)| {
                    (in_use + i as f64, arrived + a as f64)
                })
        };
        let (in_prev, bw_prev) = sum(0..BRAKE_WINDOW_TICKS);
        let (in_now, bw_now) = sum(BRAKE_WINDOW_TICKS..2 * BRAKE_WINDOW_TICKS);
        let creep = in_now >= 1.25 * in_prev && bw_now < 1.05 * bw_prev;
        let collapse = bw_now < 0.5 * bw_prev && in_now >= 0.9 * in_prev;
        in_prev > 0.0 && bw_prev > 0.0 && (creep || collapse)
    }

    fn end_round(&mut self, limiter_bound: bool) {
        if !limiter_bound && self.signal() {
            self.hits += 1;
            self.calm_rounds = 0;
        } else {
            self.hits = 0;
            self.calm_rounds += 1;
        }
        if self.hits >= 2 {
            self.on = true;
        } else if self.on && self.calm_rounds >= 5 {
            self.on = false;
        }
    }
}

/// Bandwidth of the last `SUSTAINED_ROUNDS` rounds.
#[derive(Debug)]
struct RecentBw(VecDeque<f64>);

impl RecentBw {
    fn new() -> Self {
        Self(VecDeque::with_capacity(SUSTAINED_ROUNDS))
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn push(&mut self, bw: f64) {
        if self.0.len() == SUSTAINED_ROUNDS {
            self.0.pop_front();
        }
        self.0.push_back(bw);
    }

    /// The minimum over the last `SUSTAINED_ROUNDS` rounds; 0 until there are that many (start,
    /// after idle).
    fn sustained(&self) -> f64 {
        if self.0.len() == SUSTAINED_ROUNDS {
            self.0.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            0.0
        }
    }

    fn max(&self) -> f64 {
        self.0.iter().copied().fold(0.0, f64::max)
    }
}

#[derive(Debug)]
pub struct KneeController {
    cfg: KneeConfig,
    budget: u64,
    phase: Phase,
    stats: Stats,
    /// The last tick with traffic; `None` while idle.
    last_io: Option<Instant>,
}

impl KneeController {
    pub fn new(cfg: KneeConfig) -> Self {
        let budget = cfg.init_budget.min(cfg.max_budget);
        Self {
            cfg,
            budget,
            phase: Phase::Init,
            stats: Stats::new(budget),
            last_io: None,
        }
    }

    pub fn phase_label(&self) -> &'static str {
        match self.phase {
            Phase::Init => "init",
            Phase::RampUp(_) => "ramp_up",
            Phase::Knee(_, KneeMode::Stable { .. }) => "stable",
            Phase::Knee(_, KneeMode::Probe { .. }) => "probe",
            Phase::Knee(_, KneeMode::Idle { .. }) => "idle",
        }
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    fn current_knee(&self) -> Option<&Knee> {
        match &self.phase {
            Phase::Knee(knee, _) => Some(knee),
            Phase::Init | Phase::RampUp(_) => None,
        }
    }

    pub fn knee(&self) -> Option<u64> {
        self.current_knee().map(|knee| knee.budget)
    }

    pub fn knee_source(&self) -> Option<KneeSource> {
        self.current_knee().map(|knee| knee.source)
    }

    pub fn last_bw_round(&self) -> f64 {
        self.stats.last_bw_round
    }

    pub fn braking(&self) -> bool {
        self.stats.braking()
    }

    /// Advances one control tick and returns the byte budget to apply.
    pub fn step(&mut self, t: KneeTick) -> u64 {
        let traffic = t.bytes_arrived > 0 || t.bytes_in_use > 0;
        if let Phase::Knee(knee, KneeMode::Idle { since }) = self.phase {
            let expired = t.now.duration_since(since) > IDLE_RESET;
            if expired {
                // Reset while idle, so that the next traffic is admitted under the initial budget.
                *self = Self::new(self.cfg.clone());
            }
            if !traffic {
                return self.budget;
            }
            if !expired {
                self.phase = self.stable(knee, t.now);
            }
        }
        if traffic {
            self.last_io = Some(t.now);
        } else if let Some(last_io) = self.last_io
            && !matches!(self.phase, Phase::Init)
            && t.now.duration_since(last_io) > IDLE_GRACE
        {
            self.enter_idle(last_io);
            return self.budget;
        }

        self.stats.push(&t, self.cfg.tick);

        // A RampUp round whose second tick already shows the growth ends there.
        let round = &self.stats.round;
        let grew_early = match &self.phase {
            Phase::RampUp(ramp) => {
                round.ticks == 2
                    && !round.limiter_bound
                    && round.bytes_bound
                    && ramp.grows(round.bw())
            },
            _ => false,
        };
        if round.ticks >= ROUND_TICKS || grew_early {
            self.end_round(t.now);
        }
        self.budget
    }

    fn end_round(&mut self, now: Instant) {
        let r = self.stats.finish_round();

        self.phase = match mem::take(&mut self.phase) {
            Phase::Init if r.arrived == 0 => Phase::Init,
            Phase::Init => {
                // Scan start: not a sample.
                if r.bytes_bound && !r.limiter_bound {
                    self.set_budget(self.budget.saturating_mul(2));
                }
                Phase::RampUp(Ramp::new(r.bw()))
            },
            Phase::RampUp(mut ramp) => {
                match ramp.round(&r, self.stats.braking(), self.stats.recent_bw.max()) {
                    RampStep::Hold => Phase::RampUp(ramp),
                    RampStep::Double => {
                        self.set_budget(self.budget.saturating_mul(2));
                        Phase::RampUp(ramp)
                    },
                    RampStep::Stop(source) => self.stable(ramp.knee(self.budget, source), now),
                }
            },
            Phase::Knee(mut knee, KneeMode::Stable { since }) => {
                if r.bytes_bound && !r.limiter_bound {
                    knee.probe_ref_bw = knee.probe_ref_bw.max(self.stats.recent_bw.sustained());
                }
                if self.stats.braking() {
                    // `set_budget` keeps it at or above the initial budget.
                    let stepped = (self.budget as f64 * BRAKE_BUDGET_STEP) as u64;
                    self.set_budget(stepped.max(knee.budget / 2));
                    // The probe interval restarts when the brake releases.
                    Phase::Knee(knee, KneeMode::Stable { since: now })
                } else if now.duration_since(since) >= PROBE_INTERVAL {
                    self.set_budget((knee.budget as f64 * PROBE_BUDGET_STEP) as u64);
                    Phase::Knee(knee, KneeMode::Probe { rounds: 0 })
                } else {
                    self.set_budget(knee.budget);
                    Phase::Knee(knee, KneeMode::Stable { since })
                }
            },
            Phase::Knee(mut knee, KneeMode::Probe { rounds }) => {
                let rounds = rounds + 1;
                let sustained = self.stats.recent_bw.sustained();
                let improved = !self.stats.braking()
                    && r.bytes_bound
                    && !r.limiter_bound
                    && sustained >= PROBE_BW_GROWTH * knee.probe_ref_bw;
                if improved {
                    // More bandwidth at the probe's budget: it becomes the knee.
                    knee = Knee {
                        budget: self.budget,
                        probe_ref_bw: sustained,
                        source: KneeSource::Probe,
                    };
                }
                if self.stats.braking() || improved || rounds >= PROBE_ROUNDS {
                    self.stable(knee, now)
                } else {
                    Phase::Knee(knee, KneeMode::Probe { rounds })
                }
            },
            // Not reached: no rounds while idle.
            phase @ Phase::Knee(_, KneeMode::Idle { .. }) => phase,
        };
        self.stats.round = Round::new(self.budget);
    }

    /// No traffic for `IDLE_GRACE` (between queries, or a scan stalled downstream). With a knee,
    /// keep it and the budget, drop the round, bandwidth and brake history, and wait in Idle;
    /// without one, start over from init.
    fn enter_idle(&mut self, last_io: Instant) {
        let Phase::Knee(knee, _) = self.phase else {
            *self = Self::new(self.cfg.clone());
            return;
        };
        self.phase = Phase::Knee(knee, KneeMode::Idle { since: last_io });
        self.last_io = None;
        self.stats.clear(self.budget);
    }

    /// Stable at `knee`, with a fresh probe interval.
    fn stable(&mut self, knee: Knee, now: Instant) -> Phase {
        self.set_budget(knee.budget);
        Phase::Knee(knee, KneeMode::Stable { since: now })
    }

    fn set_budget(&mut self, budget: u64) {
        self.budget = budget.max(self.cfg.init_budget).min(self.cfg.max_budget);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KneeConfig {
        KneeConfig {
            init_budget: 100,
            max_budget: u64::MAX,
            tick: Duration::from_millis(100),
        }
    }

    /// One 100 ms tick `i` on the clock `t0`; `waited` byte acquires waited.
    fn tick(
        c: &mut KneeController,
        t0: Instant,
        i: u64,
        bytes_arrived: u64,
        in_use: u64,
        waited: u64,
        limiter: Option<LimiterTick>,
    ) {
        let budget = c.budget();
        c.step(KneeTick {
            now: t0 + Duration::from_millis(100 * (i + 1)),
            bytes_arrived,
            bytes_in_use: in_use,
            byte_acquires_waited: waited,
            bytes_budget: budget,
            limiter,
        });
    }

    /// Bytes in flight and delivered per tick = min(budget, `inflight_byte_cap`) (another limit
    /// caps bytes in flight).
    fn run_capped_inflight(
        c: &mut KneeController,
        t0: Instant,
        ticks: std::ops::Range<u64>,
        inflight_byte_cap: u64,
    ) {
        for i in ticks {
            let budget = c.budget();
            let bytes = budget.min(inflight_byte_cap);
            tick(
                c,
                t0,
                i,
                bytes,
                bytes,
                u64::from(budget <= inflight_byte_cap),
                None,
            );
        }
    }

    #[test]
    fn ramps_to_the_knee_and_holds() {
        let mut c = KneeController::new(cfg());
        run_capped_inflight(&mut c, Instant::now(), 0..40, 1600);
        assert_eq!(c.phase_label(), "stable");
        // Within one doubling of the cap; the target is the knee.
        let knee = c.knee().unwrap();
        assert!((1600..=3200).contains(&knee), "{knee}");
        assert_eq!(c.budget(), knee);
    }

    /// Delivered bandwidth grows 30% every three ticks regardless of the budget (warm-up) while the
    /// whole budget stays in use.
    #[test]
    fn warm_up_growth_stops_ramp_up_on_lifetime() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        let mut bytes_per_tick = 100.0;
        for i in 0..24 {
            if i > 0 && i % 3 == 0 {
                bytes_per_tick *= 1.3;
            }
            let budget = c.budget();
            tick(&mut c, t0, i, bytes_per_tick as u64, budget, 1, None);
        }
        // Without the lifetime stop, RampUp would double to 12800.
        assert_eq!(c.phase_label(), "stable");
        assert_eq!(c.knee_source(), Some(KneeSource::Lifetime));
        assert!(c.knee().unwrap() <= 1600);
    }

    /// Idle longer than `IDLE_GRACE` (e.g. a scan stalled downstream) keeps the knee; idle longer
    /// than `IDLE_RESET` (e.g. the next query) starts over from init, also when the first tick past
    /// it has traffic.
    #[test]
    fn idle_keeps_the_knee_until_reset() {
        let mut c = KneeController::new(cfg());
        let t0 = Instant::now();
        run_capped_inflight(&mut c, t0, 0..40, 1600);
        let (knee, stable) = (c.knee(), c.budget());
        assert!(knee.is_some());
        for i in 40..120 {
            tick(&mut c, t0, i, 0, 0, 0, None);
        }
        assert_eq!(c.phase_label(), "idle");
        assert_eq!(c.knee(), knee);
        run_capped_inflight(&mut c, t0, 120..130, 1600);
        assert_eq!(c.phase_label(), "stable");
        assert_eq!(c.budget(), stable);
        // Traffic again in the first tick more than `IDLE_RESET` after the last.
        for i in 130..430 {
            tick(&mut c, t0, i, 0, 0, 0, None);
        }
        assert_eq!(c.phase_label(), "idle");
        tick(&mut c, t0, 430, 100, 100, 1, None);
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
        let mut step = |i: u64, bytes_arrived: u64, demand: u64, rate: f64, admitted: u64| {
            let budget = c.budget();
            let limiter = Some(LimiterTick { rate, admitted });
            tick(
                &mut c,
                t0,
                i,
                bytes_arrived,
                budget.min(demand),
                u64::from(budget <= demand),
                limiter,
            );
        };
        for i in 0..3 {
            step(i, 1, u64::MAX, 100.0, 1);
        }
        let mut rate = 100.0;
        for i in 3..24 {
            if i % 3 == 0 {
                rate *= 2.0;
            }
            let admitted = (rate * 0.1) as u64;
            step(i, 10 * (admitted / 2), u64::MAX, rate, admitted);
        }
        for i in 24..45 {
            step(i, 5000, 150, 1e6, 50);
        }
        assert_eq!(c.phase_label(), "ramp_up");
        assert_eq!(c.knee(), None);
        assert_eq!(c.budget(), 200);
    }
}
