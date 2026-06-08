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

#[derive(Debug)]
pub struct Model {
    // Collect and retain samples
    // #kdn TODO PERF: VecDeque scan on recompute() is O(n). For large retention
    // or very high rates, consider: ring buffer with pre-aggregated sub-buckets,
    // or incrementally maintained sum/count.
    samples: VecDeque<IoSample>,
    retention: Duration,
    first_sample_time: Option<Instant>,

    // Estimation parameters (how)
    window: Duration,

    // Current estimates
    bw_max_bps: f64,
    bw_max_last_updated: Option<Instant>,
    // Time To First Byte (TTFB) minimum observed
    // kdn TODO TUNE: Use get_opts to properly instrument ttfb for every get_range() call.
    ttfb_min: Option<Duration>,
    // kdn TOFO
    ttfb_avg: Option<Duration>,
    // kdn TODO - exponentionally weighted moving average
    // seconds, asymmetric EMA of min(ttfb_window)
    ttfb_ema: Option<f64>,

    // Used in the absence of IOSamples.
    ttfb_min_default: Duration,
}

impl Model {
    pub fn new(retention: Duration, window: Duration, ttfb_default: Duration) -> Self {
        Self {
            samples: VecDeque::with_capacity(1024),
            retention,
            first_sample_time: None,
            window,
            bw_max_bps: 0.0,
            bw_max_last_updated: None,
            ttfb_min: None,
            ttfb_avg: None,
            ttfb_ema: None,
            ttfb_min_default: ttfb_default,
        }
    }

    /// Record a completed IO. Hot path.
    pub fn record(&mut self, sample: IoSample) {
        if self.first_sample_time.is_none() {
            //kdn TODO verbose: report first rtt
            self.first_sample_time = Some(sample.completion_time);
        }
        self.evict_old(sample.completion_time);
        self.samples.push_back(sample);
    }

    /// Record a completed IO. Used as a one-off method.
    // kdn TODO consolidate with normal IOSample.
    pub fn record_ttfb(&mut self, ttfb: Duration) {
        if let Some(ref mut ttfb_min) = self.ttfb_min {
            *ttfb_min = ttfb
        } else {
            self.ttfb_min = Some(ttfb)
        }
    }

    /// Recompute estimates from current samples.
    /// Returns true if the BW_max estimate changed upward.
    // kdn TODO TUNE: add app-limited state (see BBR paper)
    pub fn recompute(
        &mut self,
        now: Instant,
    ) -> (bool, Option<f64>, Option<Duration>, Option<Duration>) {
        // Aggregate bytes within the measurement window
        let window_start = now - self.window;
        let bytes = self.sum_bytes_in_window(window_start, now);
        let ttfb_min_last = self.ttfb_min_in_window(window_start, now);
        let ttfb_avg_last = self.ttfb_avg_in_window(window_start, now);

        // Asymmetric EMA on window min: fast up (load increasing), slow down (stable/recovering)
        if let Some(min_last) = ttfb_min_last {
            let obs = min_last.as_secs_f64();
            self.ttfb_ema = Some(match self.ttfb_ema {
                None => obs,
                Some(ema) => {
                    let alpha = if obs > ema { 0.5 } else { 0.1 };
                    ema * (1.0 - alpha) + obs * alpha
                },
            });
        }

        //kdn TODO RM
        // if polars_config::config().verbose() {
        //     dbg!(&ttfb_avg_last.unwrap_or_default());
        // }

        if bytes == 0 {
            return (false, None, None, None);
        }

        // Rate computed against fixed window duration.
        // Bursty traffic may under-report, but BW_max tracks the peak so that's OK.
        let rate = bytes as f64 / self.window.as_secs_f64();

        // kdn TODO - we need some form of expiration
        self.bw_max_last_updated = Some(now);

        let bw_changed = if rate > self.bw_max_bps {
            self.bw_max_bps = rate;
            true
        } else {
            false
        };

        // kdn HACK => use avg instead if min
        // CAREFUL => this can spin out of control (because of positive feedback loop)
        self.ttfb_min = ttfb_min_last;
        let ttfb_changed = true;

        // Update ttfb_avg (window mean)
        if let Some(avg) = ttfb_avg_last {
            self.ttfb_avg = Some(avg);
        }

        // let ttfb_min_changed = match (ttfb_min_last, self.ttfb_min) {
        //     (Some(new), Some(old)) if new < old => {
        //         self.ttfb_min = Some(new);
        //         true
        //     },
        //     (Some(new), None) => {
        //         self.ttfb_min = Some(new);
        //         true
        //     },
        //     _ => false,
        // };

        let changed = bw_changed || ttfb_changed;
        (changed, Some(rate), ttfb_min_last, ttfb_avg_last)
    }

    pub fn bw_max_bps(&self) -> f64 {
        self.bw_max_bps
    }

    pub fn ttfb_min(&self) -> Duration {
        match self.ttfb_min {
            Some(ttfb_min) => ttfb_min,
            None => self.ttfb_min_default,
        }
    }

    pub fn ttfb_avg(&self) -> Duration {
        match self.ttfb_avg {
            Some(ttfb_avg) => ttfb_avg,
            //kdn TODO CLEANUP
            None => self.ttfb_min_default,
        }
    }

    pub fn ttfb_ema_as_millis(&self) -> f64 {
        match self.ttfb_ema {
            Some(ttfb_ema) => ttfb_ema * 1000.0,
            //kdn TODO CLEANUP
            None => self.ttfb_min_default.as_secs_f64() * 1000.0,
        }
    }

    // pub fn rtt_for_bdp(&self) -> Duration {
    //     self.ttfb_ema
    //         .map(Duration::from_secs_f64)
    //         .unwrap_or(self.ttfb_min_default)
    // }

    pub fn rtt_for_bdp(&self) -> Duration {
        //kdn TOGGLE HERE
        self.ttfb_avg.unwrap_or(self.ttfb_min_default)
    }

    pub fn bdp_bytes(&self) -> u64 {
        //kdn TODO HACK
        (self.bw_max_bps * self.rtt_for_bdp().as_secs_f64()) as u64
    }

    // /// Observed BDP (bw_max × rtt_min). Zero if no samples yet.
    // pub fn bdp_bytes(&self) -> u64 {
    //     (self.bw_max_bps * self.ttfb_min().as_secs_f64()) as u64
    // }

    /// True if we have a usable measurement.
    pub fn has_estimate(&self) -> bool {
        //kdn TODO TUNE
        self.sample_count() >= 5
        // self.bw_max_bps > 0.0
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    // --- internals ---

    fn sum_bytes_in_window(&self, start: Instant, end: Instant) -> u64 {
        // #kdn TODO PERF: linear scan; see VecDeque comment above.
        self.samples
            .iter()
            .filter(|s| s.completion_time >= start && s.completion_time <= end)
            .map(|s| s.nbytes)
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
            if now.duration_since(front.completion_time) > self.retention {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}
