#[cfg(target_feature="bmi2")]
use std::sync::OnceLock;
#[cfg(target_feature="bmi2")]
use raw_cpuid::CpuId;

#[cfg(target_feature="bmi2")]
static HAS_FAST_BMI2: OnceLock<bool> = OnceLock::new();

pub fn has_fast_bmi2() -> bool {
    #[cfg(not(target_feature="bmi2"))]
    { false }

    #[cfg(target_feature="bmi2")]
    {
        *HAS_FAST_BMI2.get_or_init(|| {
            let cpu_id = CpuId::new();
            let vendor = cpu_id.get_vendor_info().expect("could not read cpu vendor");
            if vendor.as_str() == "AuthenticAMD" || vendor.as_str() == "HygonGenuine" {
                let features = cpu_id.get_feature_info().expect("could not read cpu feature info");
                let family_id = features.family_id();
                
                // Hardcoded blacklist of known-bad AMD families.
                // We'll assume any future releases that support BMI2 have a
                // proper implementation.
                !(family_id >= 0x15 && family_id <= 0x18)
            } else {
                true
            }
        })
    }
}