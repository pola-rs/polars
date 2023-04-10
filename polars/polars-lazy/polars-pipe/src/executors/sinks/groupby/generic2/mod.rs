mod eval;
mod global;
mod hash_table;
mod key;
mod ooc_state;
mod sink;
mod thread_local;

use std::any::Any;
use std::slice::SliceIndex;

use eval::Eval;
use hash_table::AggHashTable;
use hashbrown::hash_map::{RawEntryMut, RawVacantEntryMut};
use polars_core::frame::row::{AnyValueBuffer, AnyValueBufferTrusted};
use polars_core::series::SeriesPhysIter;
use polars_core::IdBuildHasher;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
pub(crate) use sink::GenericGroupby2;
use thread_local::ThreadLocalTable;

use super::*;
use crate::executors::sinks::groupby::aggregates::{AggregateFn, AggregateFunction};
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

type PartitionVec<T> = Vec<T>;

struct SpillPayload {
    hashes: Vec<u64>,
    chunk_idx: Vec<IdxSize>,
    keys_and_aggs: Vec<Series>,
    num_keys: usize,
}

static HASH_COL: &str = "__POLARS_h";
static INDEX_COL: &str = "__POLARS_idx";

impl SpillPayload {
    fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keys(&self) -> &[Series] {
        &self.keys_and_aggs[..self.num_keys]
    }

    fn cols(&self) -> &[Series] {
        &self.keys_and_aggs[self.num_keys..]
    }

    fn chunk_index(&self) -> &[IdxSize] {
        &self.chunk_idx
    }

    fn get_schema(&self) -> Schema {
        let mut schema = Schema::with_capacity(self.keys_and_aggs.len() + 2);
        schema.with_column(HASH_COL.into(), DataType::UInt64);
        schema.with_column(INDEX_COL.into(), IDX_DTYPE);
        for s in &self.keys_and_aggs {
            schema.with_column(s.name().into(), s.dtype().clone());
        }
        schema
    }

    fn into_df(self) -> DataFrame {
        debug_assert_eq!(self.hashes.len(), self.chunk_idx.len());
        debug_assert_eq!(self.hashes.len(), self.keys_and_aggs.len());

        let hashes = UInt64Chunked::from_vec(HASH_COL, self.hashes).into_series();
        let chunk_idx = IdxCa::from_vec(INDEX_COL, self.chunk_idx).into_series();
        let mut cols = Vec::with_capacity(self.keys_and_aggs.len() + 2);
        cols.push(hashes);
        cols.push(chunk_idx);
        cols.extend(self.keys_and_aggs);
        DataFrame::new_no_checks(cols)
    }
}
