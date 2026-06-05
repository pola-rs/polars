//! Regime: state machine driving the admission control behavior and parameters.

use std::time::{Duration, Instant};

use crate::cloud::concurrency::model::Model;

#[derive(Clone, Copy, Debug)]
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

#[derive(Debug, Clone)]
pub struct Regime {
    state: RegimeState,
    last_transition: Instant,

    startup_growth_threshold: f64,
    startup_exit_rounds: u32,
    probe_interval: Duration,
    probe_duration: Duration,
}

impl Regime {
    pub fn new(now: Instant) -> Self {
        //kdn TODO consolidate into main Config
        Self {
            state: RegimeState::Init,
            last_transition: now,
            startup_growth_threshold: 1.05, //kdn ESTIMATE
            startup_exit_rounds: 3,
            // Interval between ProbeUp spikes
            probe_interval: Duration::from_millis(2000),
            probe_duration: Duration::from_millis(500),
        }
    }

    // kdn TODO: Add fall-back to Init after long quiet window.
    pub fn step(&mut self, model: &Model, now: Instant) {
        match self.state {
            // No reliable signal yet.
            // kdn TODO - should be event-based?
            // kdn TBD - MultiFileReader may pull a lot of metadata..
            RegimeState::Init => {
                //kdn TODO move to config
                let n_sample_threshold = 5;
                if model.sample_count() > n_sample_threshold {
                    self.transition_to(
                        RegimeState::RampUp {
                            consecutive_no_growth: 0,
                            last_bw_observation: model.bw_max_bps(),
                        },
                        now,
                    );
                }
            },

            RegimeState::RampUp {
                mut consecutive_no_growth,
                last_bw_observation,
            } => {
                //kdn TODO: refactor so we detect steady slow growth
                let current = model.bw_max_bps();
                let growing = current > last_bw_observation * self.startup_growth_threshold;

                if growing {
                    consecutive_no_growth = 0;
                } else {
                    consecutive_no_growth += 1;
                }

                if consecutive_no_growth >= self.startup_exit_rounds {
                    self.transition_to(RegimeState::Stable, now);
                } else {
                    self.state = RegimeState::RampUp {
                        consecutive_no_growth,
                        last_bw_observation: current,
                    };
                }
            },

            RegimeState::Stable => {
                if now.duration_since(self.last_transition) > self.probe_interval {
                    self.transition_to(RegimeState::ProbeUp { started_at: now }, now);
                }
            },

            // kdn TODO: what if BDP goes up?
            RegimeState::ProbeUp { started_at } => {
                if now.duration_since(started_at) > self.probe_duration {
                    self.transition_to(RegimeState::Stable, now);
                }
            },
        }
    }

    pub fn state(&self) -> &RegimeState {
        &self.state
    }

    // Effective inflight_budget will be set at the observed BDP
    // multiplied by a gain factor. This is similar to BBR cwnd_gain.
    pub fn gain_factor(&self) -> f64 {
        match self.state {
            RegimeState::Init => 1.0,
            RegimeState::RampUp { .. } => 2.0,
            RegimeState::Stable => 2.0,
            RegimeState::ProbeUp { .. } => 2.5,
        }
    }

    pub fn state_label(&self) -> &'static str {
        match self.state {
            RegimeState::Init => "init",
            RegimeState::RampUp { .. } => "ramp_up",
            RegimeState::Stable => "stable",
            RegimeState::ProbeUp { .. } => "probe_up",
        }
    }

    fn transition_to(&mut self, new: RegimeState, now: Instant) {
        let old_label = self.state_label();
        self.state = new;
        let new_label = self.state_label();
        //kdn TODO VERBOSE - tune
        if polars_config::config().verbose()
            && matches!(
                (&self.state, &new),
                (RegimeState::RampUp { .. }, RegimeState::Stable)
            )
        {
            eprintln!(
                "[InFlightConcurrency] regime: {} -> {} (after {:?})",
                old_label,
                new_label,
                now.duration_since(self.last_transition),
            );
        }
        self.last_transition = now;
    }
}
