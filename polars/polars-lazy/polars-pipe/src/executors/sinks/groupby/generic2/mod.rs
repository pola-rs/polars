mod eval;
mod thread_local;
mod key;
mod sink;
mod global;
mod hash_table;

use std::any::Any;
use std::slice::SliceIndex;

use eval::Eval;
use thread_local::ThreadLocalTable;
use hashbrown::hash_map::{RawEntryMut, RawVacantEntryMut};
use polars_core::frame::row::{AnyValueBuffer, AnyValueBufferTrusted};
use polars_core::series::SeriesPhysIter;
use polars_core::IdBuildHasher;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
pub(crate) use sink::GenericGroupby2;

use super::*;
use crate::executors::sinks::groupby::aggregates::{AggregateFn, AggregateFunction};
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

type PartitionVec<T> = Vec<T>;