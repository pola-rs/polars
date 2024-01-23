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
pub(crate) use reproject::*;
pub(crate) use union::*;

#[cfg(feature = "csv")]
use super::*;

static CHUNK_INDEX: AtomicU32 = AtomicU32::new(0);

pub(super) fn get_source_index(add: u32) -> u32 {
    CHUNK_INDEX.fetch_add(add, Ordering::Relaxed)
}
