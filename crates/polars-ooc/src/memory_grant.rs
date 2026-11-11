use std::sync::atomic::{AtomicU64, Ordering};

use polars_config::config;

/// The bytes currently granted from the memory grant pool, whose size is
/// `ooc_memory_grant_budget_bytes()`.
static GRANTED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Memory that cannot be spilled, granted to an operator from the memory grant pool. The pool
/// is a ledger shared by all operators: it allocates nothing. The bytes return to the pool on
/// drop.
#[must_use]
pub struct MemoryGrant {
    bytes: AtomicU64,
}

impl MemoryGrant {
    /// Grants `min(wanted, pool_fraction * free bytes)` of the pool. Never waits, and may grant
    /// 0 bytes.
    pub fn take(wanted: u64, pool_fraction: f64) -> Self {
        let pool_bytes = config().ooc_memory_grant_budget_bytes();
        let mut granted = GRANTED_BYTES.load(Ordering::Relaxed);
        let bytes = loop {
            let free = pool_bytes.saturating_sub(granted);
            let bytes = wanted.min((free as f64 * pool_fraction).round() as u64);
            if bytes == 0 {
                break 0;
            }
            match GRANTED_BYTES.compare_exchange_weak(
                granted,
                granted + bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break bytes,
                Err(current) => granted = current,
            }
        };
        Self {
            bytes: AtomicU64::new(bytes),
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Returns the bytes above `bytes` to the pool once there are more of them than the drift
    /// threshold, so that small shrinks leave the pool alone. Never grows the grant.
    pub fn shrink_to(&self, bytes: u64) {
        let threshold = polars_config::get_ooc_drift_threshold();
        let shrunk = self
            .bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                (old.saturating_sub(bytes) > threshold).then_some(bytes)
            });
        if let Ok(old) = shrunk {
            GRANTED_BYTES.fetch_sub(old - bytes, Ordering::Relaxed);
        }
    }
}

impl Drop for MemoryGrant {
    fn drop(&mut self) {
        GRANTED_BYTES.fetch_sub(*self.bytes.get_mut(), Ordering::Relaxed);
    }
}
