//! Spiller: serializes DataFrames to disk when memory is tight and reads
//! them back on demand.
//!
//! When a thread's local memory exceeds the budget, the [`MemoryManager`]
//! calls [`Spiller::spill`] to write the DataFrame to a temporary file
//! (one file per entry, keyed by slotmap key). Later, any async MM method
//! (`df`, `take`, `with_df_mut`) that encounters `is_spilled = true`
//! calls [`Spiller::load`] to bring the data back.
//!
//! Format: IPC (Apache Arrow IPC / Feather v2) preserves Arrow column
//! layout exactly, so deserialization is essentially a load into memory +
//! pointer fixup with no decoding overhead.
//!
//! Currently stubbed, both `spill` and `load` are `unimplemented!()`.
//! The next step is to use `polars-io` IPC writer/reader here.
//!
//! TODO: support additional formats and compression for cases where compute
//! is plentiful and IO is the bottleneck. May also extend to support spilling
//! more than just DataFrames.

use polars_core::error::PolarsResult;
use polars_core::prelude::DataFrame;

/// Serialization format for spilled DataFrames.
#[derive(Debug, Clone, Copy)]
pub enum Format {
    Ipc,
}

/// Handles serializing DataFrames to disk (spilling) and reading them back.
pub struct Spiller {
    #[allow(dead_code)]
    format: Format,
}

impl Spiller {
    pub fn new(format: Format) -> Self {
        Self { format }
    }

    /// Spill a DataFrame to disk
    ///
    /// Called by `MemoryManager::coordinate_spill`. The MM sets
    /// `entry.is_spilled = true` and replaces the in-memory DataFrame
    /// after this returns.
    /// TODO add spill_sync
    #[allow(dead_code)]
    pub async fn spill(&self, _id: u64, _df: DataFrame) {
        unimplemented!("spilling to disk")
    }

    /// Load a previously spilled DataFrame from disk.
    ///
    /// Called by the async variants of `df()`, `take()`, and `with_df_mut()`
    /// when they encounter an entry with `is_spilled = true`.
    pub async fn load(&self, _id: u64) -> PolarsResult<DataFrame> {
        unimplemented!("loading spilled data from disk")
    }

    /// Load a previously spilled DataFrame from disk (blocking).
    ///
    /// Called by `take_sync()` when it encounters spilled data.
    pub fn load_sync(&self, _id: u64) -> PolarsResult<DataFrame> {
        unimplemented!("loading spilled data from disk")
    }
}
