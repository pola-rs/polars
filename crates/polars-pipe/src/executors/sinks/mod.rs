#[cfg(any(
    feature = "parquet",
    feature = "ipc",
    feature = "csv",
    feature = "json"
))]
mod file_sink;
pub(crate) mod group_by;
mod io;
mod joins;
mod memory;
mod ordered;
mod reproject;
mod slice;
mod sort;
mod utils;

#[cfg(any(
    feature = "parquet",
    feature = "ipc",
    feature = "csv",
    feature = "json"
))]
pub(crate) use file_sink::*;
pub(crate) use joins::*;
pub(crate) use ordered::*;
pub(crate) use reproject::*;
pub(crate) use slice::*;
pub(crate) use sort::*;

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
const HASHMAP_INIT_SIZE: usize = 64;

pub(crate) static POLARS_SPILL_DIR: &str = "POLARS_SPILL_DIR";

pub(crate) fn get_base_spill_dir() -> std::path::PathBuf {
    let base_dir = std::env::var(POLARS_SPILL_DIR)
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());

    if polars_core::config::verbose() {
        eprintln!("Using base directory for spilling: {}", base_dir);
    }

    let mut base_path = std::path::PathBuf::from(base_dir);
    base_path.push("polars");
    base_path
}
