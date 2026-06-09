//! Regime: state machine driving the admission control behavior and parameters.

use std::time::{Duration, Instant};

use crate::cloud::concurrency::model::{Model, SignalStats};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegimeState {
    Init,
    RampUp {
        consecutive_no_growth: u32,
        last_bw_observation: f64,
    },
    Stable,
    ProbeUp {
        started_at: Instant,
    },
}

impl RegimeState {
    pub fn label(&self) -> &'static str {
        match *self {
            RegimeState::Init => "init",
            RegimeState::RampUp { .. } => "ramp_up",
            RegimeState::Stable => "stable",
            RegimeState::ProbeUp { .. } => "probe_up",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Regime {
    state: RegimeState,
    last_transition: Instant,

    rampup_growth_threshold: f64,
    rampup_exit_rounds: u32,
    probe_interval: Duration,
    probe_duration: Duration,
}

impl Regime {
    pub fn new(now: Instant) -> Self {
        //kdn TODO consolidate into main Config
        Self {
            state: RegimeState::Init,
            last_transition: now,
            rampup_growth_threshold: 1.05, //kdn ESTIMATE
            rampup_exit_rounds: 3,
            // Interval between ProbeUp spikes
            probe_interval: Duration::from_millis(3000), //kdn TODO TEST & TUNE
            probe_duration: Duration::from_millis(1000), //kdn TODO TEST & TUNE
        }
    }

    // kdn TODO TUNE: add app-limited state (see BBR paper)
    // kdn TODO: Add fall-back to Init after long quiet window.
    pub fn step(&mut self, signal: Option<SignalStats>, now: Instant) -> RegimeState {
        let Some(sig) = signal else {
            return match self.state {
                RegimeState::Init => RegimeState::Init,
                _ => self.transition_to(RegimeState::Init, now),
            };
        };

        match self.state {
            RegimeState::Init => self.transition_to(
                RegimeState::RampUp {
                    consecutive_no_growth: 0,
                    last_bw_observation: sig.bw_avg_bps,
                },
                now,
            ),

            RegimeState::RampUp {
                consecutive_no_growth,
                last_bw_observation,
            } => {
                let growing = sig.bw_avg_bps > last_bw_observation * self.rampup_growth_threshold;
                let consecutive_no_growth = if growing {
                    0
                } else {
                    consecutive_no_growth + 1
                };

                if consecutive_no_growth >= self.rampup_exit_rounds {
                    self.transition_to(RegimeState::Stable, now)
                } else {
                    self.state = RegimeState::RampUp {
                        consecutive_no_growth,
                        last_bw_observation: sig.bw_avg_bps,
                    };
                    self.state
                }
            },

            RegimeState::Stable => {
                if now.duration_since(self.last_transition) > self.probe_interval {
                    self.transition_to(RegimeState::ProbeUp { started_at: now }, now)
                } else {
                    self.state
                }
            },

            RegimeState::ProbeUp { started_at } => {
                if now.duration_since(started_at) > self.probe_duration {
                    self.transition_to(RegimeState::Stable, now)
                } else {
                    self.state
                }
            },
        }
    }

    pub fn state(&self) -> &RegimeState {
        &self.state
    }

    fn transition_to(&mut self, new: RegimeState, now: Instant) -> RegimeState {
        let old_label = self.state.label();
        let new_label = new.label();

        if polars_config::config().verbose() && old_label != new_label {
            eprintln!(
                "[InFlightConcurrency] regime change from {} to {}, after {:.2}s",
                old_label,
                new_label,
                now.duration_since(self.last_transition).as_secs_f64(),
            );
        }

        self.state = new;
        self.last_transition = now;
        new
    }
}
