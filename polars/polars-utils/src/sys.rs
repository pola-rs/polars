use std::sync::Mutex;

use once_cell::sync::Lazy;
use sysinfo::{System, SystemExt};

/// Startup system is expensive, so we do it once
pub struct MemInfo {
    sys: Mutex<System>,
}

impl MemInfo {
    /// This call is quite expensive, cache the results.
    pub fn free(&self) -> u64 {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        sys.free_memory()
    }
}

pub static MEMINFO: Lazy<MemInfo> = Lazy::new(|| MemInfo {
    sys: Mutex::new(System::new()),
});
