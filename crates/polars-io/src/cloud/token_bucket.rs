// TokenBucket: refill-on-read token bucket with just-in-time (JIT) pricing.
//
// Design notes:
// - NO timer anywhere: pacing precision comes from arithmetic.
// - Fast path: one load + one CAS.
// - The failure path is READ-ONLY: fractional accrual is preserved implicitly
//   because `last_us` is untouched. Zero contention when starved.
// - JIT pricing: the refill uses the rate AT READ TIME. A rate change applies
//   to the entire un-settled elapsed interval at the new rate — one interval
//   of mispricing per change, bounded by burst_cap, control-loop tolerance.
// - Burst cap B = min(rate * BURST_WINDOW, B_ABS).

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use crate::cloud::http_rate_limit::RateCell;

/// Sizing window for accumulating burst tokens, proportional to the rate.
const BURST_WINDOW_SECS: f64 = 10.0 / 1000.0;
/// Absolute burst ceiling. The proportional term (rate * BURST_WINDOW)
/// governs everywhere plausible; this only backstops implausible rates
/// (config error, runaway probe). At a 10ms window it binds above 51,200/s.
/// Lower bound to respect: the wake tick can grant at most `cap` per tick,
/// so cap must exceed rate * WAKE_TICK (50 at 50k/s and 1ms tick) or the
/// slow path silently throttles below the learned rate.
const BURST_ABS_CAP: f64 = 512.0;
/// Q16.16 fixed point.
const FP_ONE: u64 = 1 << 16;

// State packing (one AtomicU64 => refill+take is a single CAS):
//   high 32 bits: tokens, Q16.16 fixed point (max 65535 tokens — plenty; cap
//     is <= B_ABS anyway)
//   low 32 bits: last_refill timestamp, microseconds since epoch, WRAPPING.
//     Wrap analysis: u32 µs wraps every ~71.6 min. elapsed uses wrapping_sub,
//     so an idle gap that aliases (real elapsed ≡ small value mod 2^32 µs)
//     under-refills once — conservative direction, self-healing on the next
//     touch, and only reachable after 71+ minutes of NO traffic on the bucket.
//     Any alias >= burst/rate (~10ms) still fully fills the cap. Accepted.
#[derive(Debug)]
pub struct TokenBucket {
    epoch: Instant,
    // Packed {tokens_q16: u32, last_us: u32}. See state packing analysis.
    state: AtomicU64,
    // Actual rate in requests/s ('rps'), represented as f64 bits.
    // Shared cell: Written exclusively by the AIMD learner via set_rate().
    // Read by the concurrency controller via RateSignal and optionally persisted
    // by the InitPolicy.
    rate_bits: RateCell,
}

#[inline]
fn pack(tokens_q16: u32, last_us: u32) -> u64 {
    ((tokens_q16 as u64) << 32) | last_us as u64
}
#[inline]
fn unpack(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, v as u32)
}

pub enum TryAcquireError {
    NoTokens,
}

impl TokenBucket {
    pub fn new(rate_cell: RateCell) -> Self {
        Self {
            epoch: Instant::now(),
            // Start with 1 token, not a full burst: a fresh bucket must not
            // grant an instant burst-cohort before any pacing has occurred.
            state: AtomicU64::new(pack(FP_ONE as u32, 0)),
            rate_bits: rate_cell,
        }
    }

    #[inline]
    fn now_us(&self) -> u32 {
        // Wrapping by construction (as u32 truncates).
        self.epoch.elapsed().as_micros() as u32
    }

    #[inline]
    fn burst_cap_q16(rate: f64) -> f64 {
        let burst = (rate * BURST_WINDOW_SECS).clamp(1.0, BURST_ABS_CAP);
        burst * (FP_ONE as f64)
    }

    pub fn rate(&self) -> f64 {
        f64::from_bits(self.rate_bits.load(Relaxed))
    }

    /// Fast path. Ok(()) = token taken, proceed immediately. Failure is read-only.
    pub fn try_acquire(&self) -> Result<(), TryAcquireError> {
        let rate = self.rate();
        let burst_cap_q16 = TokenBucket::burst_cap_q16(rate);

        loop {
            let cur = self.state.load(Relaxed);
            let (token_q16, last_us) = unpack(cur);
            let now = self.now_us();
            let elapsed_us = now.wrapping_sub(last_us) as f64;

            let refill_q16 = rate * elapsed_us * (FP_ONE as f64) / 1e6;
            let filled = ((token_q16 as u64) as f64 + refill_q16).min(burst_cap_q16) as u64;

            if filled >= FP_ONE {
                let after = (filled - FP_ONE) as u32;
                if self
                    .state
                    .compare_exchange_weak(cur, pack(after, now), Relaxed, Relaxed)
                    .is_ok()
                {
                    return Ok(());
                }
                // Lost the race: someone else took/refilled. Retry with fresh state.
                continue;
            }

            return Err(TryAcquireError::NoTokens);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::http_rate_limit::HTTP_RATE_LIMIT_WAKE_TICK;

    /// Highest rate the design targets. The abs cap must let a single wake tick
    /// grant a full tick's worth of accrual at this rate, or the parked path
    /// silently throttles below the learned rate.
    const MAX_DESIGN_RATE_RPS: f64 = 50_000.0;

    /// The wake tick can grant at most `burst_cap` tokens per tick (it drains
    /// what the bucket holds). If the cap is below one tick's accrual at the
    /// design rate, the parked path silently throttles below the learned rate
    /// — throughput loss with no error, no log, no signal.
    #[test]
    fn burst_cap_clears_wake_tick_floor() {
        let per_tick_accrual = MAX_DESIGN_RATE_RPS * HTTP_RATE_LIMIT_WAKE_TICK.as_secs_f64();
        assert!(
            BURST_ABS_CAP >= per_tick_accrual,
            "BURST_ABS_CAP ({BURST_ABS_CAP}) < rate * WAKE_TICK ({per_tick_accrual}) \
             at {MAX_DESIGN_RATE_RPS} rps: the wake tick cannot drain a tick's \
             accrual, throttling the slow path below the learned rate"
        );
    }
}
