use std::sync::{LazyLock, Mutex};

use sysinfo::{MemoryRefreshKind, System};

use crate::config::verbose;

/// Return the total system memory in bytes.
pub fn total_memory() -> u64 {
    return *TOTAL_MEMORY;

    static TOTAL_MEMORY: LazyLock<u64> = LazyLock::new(|| {
        let mut sys = System::new();

        sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());

        let mut v: u64 = match sys.cgroup_limits() {
            Some(limits) => limits.total_memory,
            None => sys.total_memory(),
        };

        if let Ok(s) = std::env::var("POLARS_OVERRIDE_TOTAL_MEMORY") {
            v = s
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_OVERRIDE_TOTAL_MEMORY: {s}"))
        }

        if verbose() {
            let gib = (v as f64) / (1024.0 * 1024.0 * 1024.0);
            eprintln!("total memory: {gib:.3} GiB ({v} bytes)")
        }

        v
    });
}

/// Startup system is expensive, so we do it once
pub struct MemInfo {
    sys: Mutex<System>,
}

impl MemInfo {
    /// This call is quite expensive, cache the results.
    pub fn free(&self) -> u64 {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        match sys.cgroup_limits() {
            Some(limits) => limits.free_memory,
            None => sys.available_memory(),
        }
    }
}

pub static MEMINFO: LazyLock<MemInfo> = LazyLock::new(|| MemInfo {
    sys: Mutex::new(System::new()),
});
