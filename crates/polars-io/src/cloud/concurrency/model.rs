//! Model: model the network parameters using IOSamples
//!
//! Parameters:
//! - BW_max: maximum bandwidth
//! - RTT_min: minimum round-trip time, based on TTFB (time-to-first-byte)
//!
//! Derived parameter:
//! - BDP: bandwidth-delay product

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::cloud::concurrency::IoSample;

/// Snapshot of the network signal, recomputed each `update()`.
/// If this exists, all fields are populated and mutually consistent.
#[derive(Clone, Copy, Debug)]
pub struct SignalStats {
    /// Windowed average bandwidth.
    pub bw_avg_bps: f64,
    /// All-time high-watermark bandwidth (snapshotted; the live value
    /// persists in Model across signal loss).
    pub bw_hwm_bps: f64,
    /// Windowed TTFB statistics.
    pub ttfb_min: Duration,
    pub ttfb_avg: Duration,
}

impl SignalStats {
    /// RTT used to compute BDP.
    ///
    /// Average TTFB, capped at `QUEUING_RATIO_CAP × ttfb_min`. The cap
    /// prevents the BDP estimate from spiraling on self-induced queuing
    /// while still letting the budget grow with bw_hwm.
    pub fn rtt_for_bdp(&self) -> Duration {
        const QUEUING_RATIO_CAP: f64 = 10.0;
        self.ttfb_avg.min(self.ttfb_min.mul_f64(QUEUING_RATIO_CAP))
    }

    pub fn bdp_bytes(&self) -> u64 {
        (self.bw_hwm_bps * self.rtt_for_bdp().as_secs_f64()) as u64
    }
}

#[derive(Debug)]
pub struct Model {
    // Collect and retain samples
    // kdn TODO PERF: lockless queue (consider crossbeam_queue::ArrayQueue)
    samples: VecDeque<IoSample>,
    first_sample_time: Option<Instant>,

    // Model parameters for estimation and lifecycle management.
    window: Duration,

    // Statistics
    bw_hwm_bps: Option<f64>,
    bw_hwm_last_updated: Option<Instant>,
    signal: Option<SignalStats>,
}

impl Model {
    pub fn new(window: Duration) -> Self {
        Self {
            samples: VecDeque::with_capacity(1024),
            first_sample_time: None,
            window,
            bw_hwm_bps: None,
            bw_hwm_last_updated: None,
            signal: None,
        }
    }

    /// Record a completed IO. Hot path.
    pub fn record(&mut self, sample: IoSample) {
        if self.first_sample_time.is_none() {
            self.first_sample_time = Some(sample.completion_time);
            if polars_config::config().verbose() {
                eprintln!(
                    "[InFlightConcurrency]: observed first RTT sample: {}ms, for {} bytes",
                    sample.ttfb.as_millis(),
                    sample.n_bytes
                )
            }
        }

        self.samples.push_back(sample);
    }

    pub fn signal(&self) -> Option<SignalStats> {
        self.signal // Copy
    }

    /// Recompute statistics from current samples.
    /// Returns true if the model contains a valid signal.
    // @TODO. The regime should account for an 'app-limited' state, to
    // account for slow inbound rate not caused by the upstream network,
    // but caused by the downstream processing (see also BBR paper).
    pub fn update(&mut self, now: Instant) {
        const N_SAMPLE_THRESHOLD: usize = 5;

        self.evict_old(Instant::now());

        if self.samples.len() < N_SAMPLE_THRESHOLD {
            self.signal = None;
            return;
        }

        // Single pass over the window (eviction is lazy, so still filter).
        let window_start = now - self.window;
        let mut n_bytes: u64 = 0;
        let mut n_samples: usize = 0;
        let mut ttfb_sum = Duration::ZERO;
        let mut ttfb_min: Option<Duration> = None;

        for s in &self.samples {
            if s.completion_time >= window_start && s.completion_time <= now {
                n_bytes += s.n_bytes;
                n_samples += 1;
                ttfb_sum += s.ttfb;
                ttfb_min = Some(ttfb_min.map_or(s.ttfb, |m| m.min(s.ttfb)));
            }
        }

        let (Some(ttfb_min), true) = (ttfb_min, n_bytes > 0) else {
            self.signal = None;
            return;
        };
        let ttfb_avg = ttfb_sum.div_f64(n_samples as f64);

        // Rate against fixed window duration; bursty traffic may under-report,
        // but the HWM tracks the peak.
        let bw_avg_bps = n_bytes as f64 / self.window.as_secs_f64();

        // All-time high-water-mark (HWM).
        // kdn TODO: Some form of HWM expiration or decay.
        if self.bw_hwm_bps.is_none_or(|hwm| bw_avg_bps > hwm) {
            self.bw_hwm_bps = Some(bw_avg_bps);
            self.bw_hwm_last_updated = Some(now);
        }
        let bw_hwm_bps = self.bw_hwm_bps.unwrap();

        self.signal = Some(SignalStats {
            bw_avg_bps,
            bw_hwm_bps,
            ttfb_min,
            ttfb_avg,
        });
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn evict_old(&mut self, now: Instant) {
        while let Some(front) = self.samples.front() {
            if now.duration_since(front.completion_time) > self.window {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}
