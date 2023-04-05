mod key;

use std::any::Any;
use std::slice::SliceIndex;

use hashbrown::hash_map::{RawEntryMut, RawVacantEntryMut};
use polars_core::frame::row::{AnyValueBuffer, AnyValueBufferTrusted};
use polars_core::IdBuildHasher;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::executors::sinks::groupby::aggregates::AggregateFunction;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

struct HashTbl {
    inner_map: PlIdHashMap<Key, IdxSize>,
    keys: Vec<AnyValue<'static>>,
    // the aggregation that are in process
    // the index the hashtable points to the start of the aggregations of that key/group
    running_aggregations: Vec<AggregateFunction>,
    // n aggregation function constructors
    agg_constructors: Vec<AggregateFunction>,
}

impl HashTbl {
    fn get_entry(
        &mut self,
        hash: u64,
        tuples: &[AnyValue],
    ) -> RawEntryMut<Key, IdxSize, IdBuildHasher> {
        self.inner_map
            .raw_entry_mut()
            .from_hash(hash, |hash_map_key| {
                hash_map_key.hash == hash && {
                    let idx = hash_map_key.idx as usize;
                    if tuples.len() > 1 {
                        tuples.iter().enumerate().all(|(i, key)| unsafe {
                            self.keys.get_unchecked_release(i + idx) == key
                        })
                    } else {
                        unsafe {
                            self.keys.get_unchecked_release(idx) == tuples.get_unchecked_release(0)
                        }
                    }
                }
            })
    }

    fn insert<'a>(&'a mut self, hash: u64, key_tuples: &[AnyValue<'_>]) {
        let mut entry = self.get_entry(hash, key_tuples);

        let idx = match entry {
            RawEntryMut::Occupied(entry) => *entry.get(),
            RawEntryMut::Vacant(entry) => {
                // bchk shenanigans:
                // it does not allow us to hold a `raw entry` and in the meantime
                // have &self acces to get the length of keys
                // so we work with pointers instead
                let borrow = &entry;
                let borrow = borrow as *const RawVacantEntryMut<_, _, _> as usize;
                // ensure the bck forgets this guy
                std::mem::forget(entry);

                let aggregation_idx = self.running_aggregations.len() as IdxSize;
                let key_idx = self.keys.len() as IdxSize;

                let key = Key::new(hash, key_idx);
                unsafe {
                    // take a hold of the entry again and ensure it gets dropped
                    let borrow =
                        borrow as *const RawVacantEntryMut<'a, Key, IdxSize, IdBuildHasher>;
                    let entry = std::ptr::read(borrow);
                    entry.insert(key, aggregation_idx);
                }

                for agg in &self.agg_constructors {
                    self.running_aggregations.push(agg.split2())
                }

                unsafe {
                    self.keys.extend(
                        key_tuples
                            .iter()
                            .map(|k| k.clone().into_static().unwrap_unchecked_release()),
                    );
                }
                aggregation_idx
            }
        };
    }
}

struct SpillPartitions {
    // number of different aggregations
    n_aggs: u32,
    // outer vec: partitions (factor of 2)
    // inner vec: number of aggregated columns
    partitions: Vec<Vec<AnyValueBufferTrusted<'static>>>,
}

impl SpillPartitions {
    fn insert(&mut self, hash: u64, tuples: &[AnyValue<'_>]) {
        let partition = hash_to_partition(hash, self.partitions.len());
        unsafe {
            let partitions = self.partitions.get_unchecked_release_mut(partition);
            debug_assert_eq!(tuples.len(), partitions.len());

            // amortize the loop counter
            for i in 0..tuples.len() {
                let av = tuples.get_unchecked(i);
                let buf = partitions.get_unchecked_mut(i);
                buf.add_unchecked_owned_physical(av);
            }
        };
    }
}

pub(crate) struct GenericGroupby {
    thread_local_map: HashTbl,
}

impl Sink for GenericGroupby {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        todo!()
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        todo!()
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        todo!()
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        todo!()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        todo!()
    }

    fn fmt(&self) -> &str {
        todo!()
    }
}
