pub(crate) mod group_by;
mod io;
mod joins;
mod memory;
mod ordered;
mod output;
mod reproject;
mod slice;
mod sort;
mod utils;

pub(crate) use joins::*;
pub(crate) use ordered::*;
#[cfg(any(
    feature = "parquet",
    feature = "ipc",
    feature = "csv",
    feature = "json"
))]
pub(crate) use output::*;
pub(crate) use reproject::*;
pub(crate) use slice::*;
pub(crate) use sort::*;

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
const HASHMAP_INIT_SIZE: usize = 64;

pub(crate) static POLARS_TEMP_DIR: &str = "POLARS_TEMP_DIR";

pub(crate) fn get_base_temp_dir() -> String {
    let base_dir = std::env::var(POLARS_TEMP_DIR)
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());

    if polars_core::config::verbose() {
        eprintln!("Temporary directory path in use: {}", base_dir);
    }

    base_dir
}
