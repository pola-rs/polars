// TODO: This exists because `POLARS_TEMP_DIR_BASE_PATH` in polars-io
// uses `std::env::temp_dir()` which may be tmpfs on Linux. If that is
// fixed upstream to use `/var/tmp` on Linux, this function can be
// replaced by `POLARS_TEMP_DIR_BASE_PATH.join("spill")`.

use std::path::PathBuf;

/// Platform-specific default for the OOC spill directory.
///
/// - Linux: `/var/tmp/polars-{USER}/spill` (always real disk, never tmpfs)
/// - macOS: `{temp_dir}/polars-{USER}/spill` (already disk-backed)
/// - Windows: `{temp_dir}/polars-{USERNAME}/spill` (already disk-backed)
/// - Other (FreeBSD, Raspberry Pi OS, etc.): `{temp_dir}/polars-{USER}/spill`
pub fn default_ooc_spill_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    let (base, user_var) = (PathBuf::from("/var/tmp"), "USER");

    #[cfg(target_os = "macos")]
    let (base, user_var) = (std::env::temp_dir(), "USER");

    #[cfg(target_os = "windows")]
    let (base, user_var) = (std::env::temp_dir(), "USERNAME");

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let (base, user_var) = (std::env::temp_dir(), "USER");

    let user = std::env::var(user_var).unwrap_or_else(|_| "polars".to_string());
    base.join(format!("polars-{user}/spill"))
}
