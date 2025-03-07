use std::sync::{LazyLock, Mutex};

use sysinfo::System;

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
