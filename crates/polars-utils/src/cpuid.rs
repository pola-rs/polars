// So much conditional stuff going on here...
#![allow(dead_code, unreachable_code, unused)]

use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
use raw_cpuid::CpuId;

/// A sloppy OnceLock<bool> that can call the function multiple times, but most
/// likely won't except in exceptional race conditions.
struct BoolCache {
    state: AtomicU8,
}

impl BoolCache {
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(2),
        }
    }

    pub fn get_or_init<F: FnOnce() -> bool>(&self, f: F) -> bool {
        let state = self.state.load(Ordering::Relaxed);
        if state <= 1 {
            return state == 1;
        }

        let ans = f();
        self.state.store(ans as u8, Ordering::Relaxed);
        ans
    }
}

#[cfg(target_feature = "bmi2")]
#[inline(never)]
#[cold]
fn detect_fast_bmi2() -> bool {
    let cpu_id = CpuId::new();
    let vendor = cpu_id.get_vendor_info().expect("could not read cpu vendor");
    if vendor.as_str() == "AuthenticAMD" || vendor.as_str() == "HygonGenuine" {
        let features = cpu_id
            .get_feature_info()
            .expect("could not read cpu feature info");
        let family_id = features.family_id();

        // Hardcoded blacklist of known-bad AMD families.
        // We'll assume any future releases that support BMI2 have a
        // proper implementation.
        !(family_id >= 0x15 && family_id <= 0x18)
    } else {
        true
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[cold]
fn detect_fast_clmul() -> bool {
    let cpu_id = CpuId::new();
    let features = cpu_id
        .get_feature_info()
        .expect("could not read cpu feature info");
    features.has_pclmulqdq()
}

#[cfg(target_feature = "bmi2")]
#[inline]
pub fn has_fast_bmi2() -> bool {
    #[cfg(target_feature = "bmi2")]
    {
        static CACHE: BoolCache = BoolCache::new();
        return CACHE.get_or_init(detect_fast_bmi2);
    }

    false
}

#[inline]
pub fn has_fast_clmul() -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes"))]
    {
        return true;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    {
        static CACHE: BoolCache = BoolCache::new();
        return CACHE.get_or_init(detect_fast_clmul);
    }

    false
}
