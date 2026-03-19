// So much conditional stuff going on here...
#![allow(dead_code, unreachable_code, unused)]

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use raw_cpuid::CpuId;

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
        !(0x15..=0x18).contains(&family_id)
    } else {
        true
    }
}

#[inline(always)]
pub fn has_fast_bmi2() -> bool {
    #[cfg(target_feature = "bmi2")]
    {
        static CACHE: OnceLock<bool> = OnceLock::new();
        return *CACHE.get_or_init(detect_fast_bmi2);
    }

    false
}

#[inline]
pub fn is_avx512_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static CACHE: OnceLock<bool> = OnceLock::new();
        return *CACHE.get_or_init(|| {
            if !std::arch::is_x86_feature_detected!("avx512f") {
                return false;
            }

            if std::env::var("POLARS_DISABLE_AVX512")
                .map(|var| var == "1")
                .unwrap_or(false)
            {
                return false;
            }

            true
        });
    }

    false
}
