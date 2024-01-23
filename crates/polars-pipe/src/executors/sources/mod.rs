#[cfg(feature = "csv")]
mod csv;
mod frame;
mod ipc_one_shot;
#[cfg(feature = "parquet")]
mod parquet;
mod reproject;
mod union;

use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "csv")]
pub(crate) use csv::CsvSource;
pub(crate) use frame::*;
pub(crate) use ipc_one_shot::*;
#[cfg(feature = "parquet")]
pub(crate) use parquet::*;
use polars_utils::IdxSize;
pub(crate) use reproject::*;
pub(crate) use union::*;

#[cfg(feature = "csv")]
use super::*;

static CHUNK_INDEX: AtomicU32 = AtomicU32::new(0);

pub(super) fn get_source_offset() -> IdxSize {
    // New pipelines are ~1M chunks apart before they have the same count.
    // We don't want chunks with the same count from different pipelines are fed
    // into the same sink as we cannot determine the order.
    (CHUNK_INDEX.fetch_add(1, Ordering::Relaxed).wrapping_shl(20)) as IdxSize
}
