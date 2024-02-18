mod eval;
mod global;
mod hash_table;
mod ooc_state;
mod sink;
mod source;
mod thread_local;

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use arrow::array::BinaryArray;
use eval::Eval;
use hash_table::AggHashTable;
use hashbrown::hash_map::{RawEntryMut, RawVacantEntryMut};
use polars_core::frame::row::AnyValueBufferTrusted;
use polars_core::series::SeriesPhysIter;
use polars_core::IdBuildHasher;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
pub(crate) use sink::GenericGroupby2;
use thread_local::ThreadLocalTable;

use super::*;
use crate::executors::sinks::group_by::aggregates::{AggregateFn, AggregateFunction};
use crate::executors::sinks::io::IOThread;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

type PartitionVec<T> = Vec<T>;
type IOThreadRef = Arc<Mutex<Option<IOThread>>>;

#[derive(Clone)]
struct SpillPayload {
    hashes: Vec<u64>,
    chunk_idx: Vec<IdxSize>,
    keys: BinaryArray<i64>,
    aggs: Vec<Series>,
}

static HASH_COL: &str = "__POLARS_h";
static INDEX_COL: &str = "__POLARS_idx";
static KEYS_COL: &str = "__POLARS_keys";

impl SpillPayload {
    fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keys(&self) -> &BinaryArray<i64> {
        &self.keys
    }

    fn cols(&self) -> &[Series] {
        &self.aggs
    }

    fn chunk_index(&self) -> &[IdxSize] {
        &self.chunk_idx
    }

    fn get_schema(&self) -> Schema {
        let mut schema = Schema::with_capacity(self.aggs.len() + 2);
        schema.with_column(HASH_COL.into(), DataType::UInt64);
        schema.with_column(INDEX_COL.into(), IDX_DTYPE);
        schema.with_column(KEYS_COL.into(), DataType::BinaryOffset);
        for s in &self.aggs {
            schema.with_column(s.name().into(), s.dtype().clone());
        }
        schema
    }

    fn into_df(self) -> DataFrame {
        debug_assert_eq!(self.hashes.len(), self.chunk_idx.len());
        debug_assert_eq!(self.hashes.len(), self.keys.len());

        let hashes = UInt64Chunked::from_vec(HASH_COL, self.hashes).into_series();
        let chunk_idx = IdxCa::from_vec(INDEX_COL, self.chunk_idx).into_series();
        let keys = BinaryOffsetChunked::with_chunk(KEYS_COL, self.keys).into_series();

        let mut cols = Vec::with_capacity(self.aggs.len() + 3);
        cols.push(hashes);
        cols.push(chunk_idx);
        cols.push(keys);
        cols.extend(self.aggs);
        unsafe { DataFrame::new_no_checks(cols) }
    }

    fn spilled_to_columns(
        spilled: &DataFrame,
    ) -> (&[u64], &[IdxSize], &BinaryArray<i64>, &[Series]) {
        let cols = spilled.get_columns();
        let hashes = cols[0].u64().unwrap();
        let hashes = hashes.cont_slice().unwrap();
        let chunk_indexes = cols[1].idx().unwrap();
        let chunk_indexes = chunk_indexes.cont_slice().unwrap();
        let keys = cols[2].binary_offset().unwrap();
        let keys = keys.downcast_iter().next().unwrap();
        let aggs = &cols[3..];
        (hashes, chunk_indexes, keys, aggs)
    }
}

// This is the hash and the Index offset in the linear buffer
#[derive(Copy, Clone)]
pub(super) struct Key {
    pub(super) hash: u64,
    pub(super) offset: u32,
    pub(super) len: u32,
}

impl Key {
    #[inline]
    pub(super) fn new(hash: u64, offset: u32, len: u32) -> Self {
        Self { hash, offset, len }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}
