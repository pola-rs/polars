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
    bytes: u64,
}

impl MemoryGrant {
    /// Grants `min(wanted, pool_fraction * free bytes)` of the pool. Never waits, and may grant
    /// 0 bytes.
    pub fn take(wanted: u64, pool_fraction: f64) -> Self {
        let pool_bytes = config().ooc_memory_grant_budget_bytes();
        let mut granted = GRANTED_BYTES.load(Ordering::Relaxed);
        loop {
            let free = pool_bytes.saturating_sub(granted);
            let bytes = wanted.min((free as f64 * pool_fraction).round() as u64);
            if bytes == 0 {
                return Self { bytes: 0 };
            }
            match GRANTED_BYTES.compare_exchange_weak(
                granted,
                granted + bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Self { bytes },
                Err(current) => granted = current,
            }
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the bytes above `bytes` to the pool. Does nothing if the grant is not larger.
    pub fn shrink_to(&mut self, bytes: u64) {
        if bytes < self.bytes {
            GRANTED_BYTES.fetch_sub(self.bytes - bytes, Ordering::Relaxed);
            self.bytes = bytes;
        }
    }
}

impl Drop for MemoryGrant {
    fn drop(&mut self) {
        self.shrink_to(0);
    }
}
