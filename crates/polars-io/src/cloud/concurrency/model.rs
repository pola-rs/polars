//! Model: model the network parameters using IOSamples
//! Parameters:
//! - BW_max: maximum bandwidth
//! - RTT_min: minimum round-trip time, based on TTFB (time-to-first-byte)
//! Derived parameter:
//! - BDP: bandwidth-delay product

//kdn TODO: investigate consolidation with IO_metrics

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
    //kdn TODO RM
    // /// RTT used for BDP: ttfb_avg capped at QUEUING_RATIO_CAP * ttfb_min.
    // pub rtt_for_bdp: Duration,
    // /// bw_hwm_bps * rtt_for_bdp.
    // pub bdp_bytes: u64,
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
    // kdn TODO: re-assess O(n) on update.. probably fine for short windows.
    samples: VecDeque<IoSample>,
    first_sample_time: Option<Instant>,

    // Model parameters for estimation and lifecycle management.
    window: Duration,

    // Statistics
    bw_hwm_bps: Option<f64>,
    bw_hwm_last_updated: Option<Instant>,
    signal: Option<SignalStats>,
    // // Statistics.
    // // Invariant: if the model 'has signal', then all statistics are Some().
    // has_signal: bool,
    // // All min/max/avg statistics are 'in window'. The hwm statistics are 'all time'.
    // bw_avg_bps: Option<f64>,
    // // All-time high-watermark of bw_max.
    // bw_hwm_bps: Option<f64>,
    // bw_hwm_last_updated: Option<Instant>,
    // // Time To First Byte (TTFB) minimum observed.
    // ttfb_min: Option<Duration>,
    // // TTFB average.
    // ttfb_avg: Option<Duration>,

    // // Used in the absence of IOSamples.
    // ttfb_min_default: Duration,
}

impl Model {
    pub fn new(window: Duration, ttfb_default: Duration) -> Self {
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

    // kdn TODO RM
    // /// Record a completed IO. Used as a one-off method.
    // // kdn TODO consolidate with normal IOSample.
    // pub fn record_ttfb(&mut self, ttfb: Duration) {
    //     if let Some(ref mut ttfb_min) = self.ttfb_min {
    //         *ttfb_min = ttfb
    //     } else {
    //         self.ttfb_min = Some(ttfb)
    //     }
    // }

    /// Recompute statistics from current samples.
    /// Returns true if the model contains a valid signal.
    // @TODO. The regime should account for an 'app-limited' state, to
    // account for slow inbound rate not caused by the upstream network,
    // but caused by the downstream processing (see also BBR paper).
    pub fn update(&mut self, now: Instant) {
        const N_SAMPLE_THRESHOLD: usize = 5;
        const QUEUING_RATIO_CAP: f64 = 10.0;

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

        // All-time HWM ratchet.
        // TODO: some form of HWM expiration or decay.
        if self.bw_hwm_bps.is_none_or(|hwm| bw_avg_bps > hwm) {
            self.bw_hwm_bps = Some(bw_avg_bps);
            self.bw_hwm_last_updated = Some(now);
        }
        let bw_hwm_bps = self.bw_hwm_bps.unwrap();

        //kdn TODO RM
        // // Cap prevents BDP spiraling on self-induced queuing.
        // let rtt_for_bdp = ttfb_avg.min(ttfb_min.mul_f64(QUEUING_RATIO_CAP));
        // let bdp_bytes = (bw_hwm_bps * rtt_for_bdp.as_secs_f64()) as u64;

        self.signal = Some(SignalStats {
            bw_avg_bps,
            bw_hwm_bps,
            ttfb_min,
            ttfb_avg,
            //kdn TODO RM
            // rtt_for_bdp,
            // bdp_bytes,
        });

        //kdn TODO RM

        // if self.sample_count() > N_SAMPLE_THRESHOLD {
        //     // Aggregate bytes within the measurement window
        //     let window_start = now - self.window;
        //     let n_bytes = self.sum_bytes_in_window(window_start, now);
        //     let ttfb_min = self.ttfb_min_in_window(window_start, now);
        //     let ttfb_avg = self.ttfb_avg_in_window(window_start, now);

        //     if n_bytes == 0 {
        //         self.has_signal = false;
        //         return self.has_signal;
        //     };

        //     self.has_signal = true;

        //     // Rate computed against fixed window duration.
        //     // Bursty traffic may under-report, but BW_max tracks the peak so that is ok.
        //     let rate = n_bytes as f64 / self.window.as_secs_f64();
        //     self.bw_avg_bps = Some(rate);

        //     // @TODO. Some form of expiration for bw_hwm is needed.
        //     // Currently the bw_hwm never drops, which may not be correct for long-living downloads.
        //     let bw_hwm_changed = self.bw_hwm_bps.map_or(true, |bw_hwm_bps| rate > bw_hwm_bps);
        //     if bw_hwm_changed {
        //         self.bw_hwm_bps = Some(rate);
        //         self.bw_hwm_last_updated = Some(now);
        //     }

        //     self.ttfb_min = ttfb_min;
        //     self.ttfb_avg = ttfb_avg;
        // } else {
        //     self.has_signal = false
        // }
        // self.has_signal
    }

    // pub fn bw_avg_bps(&self) -> Option<f64> {
    //     self.bw_avg_bps
    // }

    // pub fn bw_hwm_bps(&self) -> Option<f64> {
    //     self.bw_hwm_bps
    // }

    // pub fn ttfb_min(&self) -> Option<Duration> {
    //     self.ttfb_min
    // }

    // pub fn ttfb_avg(&self) -> Option<Duration> {
    //     self.ttfb_avg
    // }

    // /// RTT used to compute BDP.
    // ///
    // /// Uses average TTFB but capped at `QUEUING_RATIO_CAP × rtt_min`.
    // /// Capping prevents the BDP estimate from spiraling with self-induced queuing
    // /// while still allowing budget to grow if bw_max grows.
    // pub fn rtt_for_bdp(&self) -> Option<Duration> {
    //     const QUEUING_RATIO_CAP: f64 = 10.0;

    //     match (self.ttfb_min(), self.ttfb_avg()) {
    //         (Some(rtt_min), Some(rtt_avg)) => {
    //             let cap = rtt_min.mul_f64(QUEUING_RATIO_CAP);
    //             Some(rtt_avg.min(cap))
    //         },
    //         _ => None,
    //     }
    //     //kdn TODO RM
    //     // let rtt_min = self.ttfb_min();
    //     // let rtt_avg = self.ttfb_avg();
    //     // let cap = rtt_min.mul_f64(QUEUING_RATIO_CAP);
    //     // rtt_avg.min(cap)
    // }

    // pub fn bdp_bytes(&self) -> Option<u64> {
    //     match (self.bw_hwm_bps, self.rtt_for_bdp()) {
    //         (Some(bw_max), Some(rtt)) => Some((bw_max * rtt.as_secs_f64()) as u64),
    //         _ => None,
    //     }
    //     // kdn TODO RM
    //     // //kdn TODO handle None
    //     // let bw_max_bps = self.bw_hwm_bps.expect("no bandwidth estimates available");
    //     // let rtt = self.rtt_for_bdp().as_secs_f64();
    //     // (bw_max_bps * rtt) as u64
    // }

    // pub fn has_signal(&self) -> bool {
    //     self.has_signal
    //     // //kdn TODO move threshold value to config
    //     // const N_SAMPLE_THRESHOLD: usize = 5;
    //     // self.evict_old(Instant::now());
    //     // self.sample_count() > N_SAMPLE_THRESHOLD
    // }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn sum_bytes_in_window(&self, start: Instant, end: Instant) -> u64 {
        // #kdn TODO PERF: linear scan; see VecDeque comment above.
        self.samples
            .iter()
            .filter(|s| s.completion_time >= start && s.completion_time <= end)
            .map(|s| s.n_bytes)
            .sum()
    }

    fn ttfb_min_in_window(&self, start: Instant, end: Instant) -> Option<Duration> {
        // #kdn TODO PERF: linear scan; see VecDeque comment above.
        self.samples
            .iter()
            .filter(|s| s.completion_time >= start && s.completion_time <= end)
            .map(|s| s.ttfb)
            .min()
    }

    fn ttfb_avg_in_window(&self, start: Instant, end: Instant) -> Option<Duration> {
        let mut total_duration = Duration::default();
        let mut n_samples = 0usize;

        if self.samples.is_empty() {
            return None;
        }

        for s in &self.samples {
            if s.completion_time >= start && s.completion_time <= end {
                n_samples += 1;
                total_duration += s.ttfb;
            }
        }

        if n_samples == 0 {
            return None;
        }

        Some(Duration::from_secs_f64(
            total_duration.as_secs_f64() / n_samples as f64,
        ))
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
