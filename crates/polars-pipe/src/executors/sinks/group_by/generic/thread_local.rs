use arrow::array::MutableBinaryArray;
use once_cell::sync::Lazy;
use polars_core::export::once_cell;
use polars_utils::hashing::hash_to_partition;

use super::*;
use crate::pipeline::PARTITION_SIZE;

const OB_SIZE: usize = 2048;

static SPILL_SIZE: Lazy<usize> = Lazy::new(|| {
    std::env::var("POLARS_STREAMING_GROUPBY_SPILL_SIZE")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(10_000)
});

#[derive(Clone)]
struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    // inner vec: number of keys + number of aggregated columns
    keys_partitioned: PartitionVec<MutableBinaryArray<i64>>,
    aggs_partitioned: PartitionVec<Vec<AnyValueBufferTrusted<'static>>>,
    hash_partitioned: PartitionVec<Vec<u64>>,
    chunk_index_partitioned: PartitionVec<Vec<IdxSize>>,
    spilled: bool,
    // this only fills during the reduce phase IFF
    // there are spilled tuples
    finished_payloads: PartitionVec<Vec<SpillPayload>>,
    keys_dtypes: Arc<[DataType]>,
    agg_dtypes: Arc<[DataType]>,
    output_schema: SchemaRef,
}

impl SpillPartitions {
    fn new(keys: Arc<[DataType]>, aggs: Arc<[DataType]>, output_schema: SchemaRef) -> Self {
        let hash_partitioned = vec![];
        let chunk_index_partitioned = vec![];

        // construct via split so that pre-allocation succeeds
        Self {
            keys_partitioned: vec![],
            aggs_partitioned: vec![],
            hash_partitioned,
            chunk_index_partitioned,
            spilled: false,
            finished_payloads: vec![],
            keys_dtypes: keys,
            agg_dtypes: aggs,
            output_schema,
        }
        .split()
    }

    fn split(&self) -> Self {
        let n_columns = self.agg_dtypes.as_ref().len();

        let aggs_partitioned = (0..PARTITION_SIZE)
            .map(|_| {
                let mut buf = Vec::with_capacity(n_columns);
                for dtype in self.agg_dtypes.as_ref() {
                    let builder = AnyValueBufferTrusted::new(&dtype.to_physical(), OB_SIZE);
                    buf.push(builder);
                }
                buf
            })
            .collect();

        let keys_partitioned = (0..PARTITION_SIZE)
            .map(|_| MutableBinaryArray::with_capacity(OB_SIZE))
            .collect();

        let hash_partitioned = (0..PARTITION_SIZE)
            .map(|_| Vec::with_capacity(OB_SIZE))
            .collect::<Vec<_>>();
        let chunk_index_partitioned = (0..PARTITION_SIZE)
            .map(|_| Vec::with_capacity(OB_SIZE))
            .collect::<Vec<_>>();

        Self {
            keys_partitioned,
            aggs_partitioned,
            hash_partitioned,
            chunk_index_partitioned,
            spilled: false,
            finished_payloads: vec![],
            keys_dtypes: self.keys_dtypes.clone(),
            agg_dtypes: self.agg_dtypes.clone(),
            output_schema: self.output_schema.clone(),
        }
    }
}

impl SpillPartitions {
    /// Returns (partition, overflowing hashes, chunk_indexes, keys and aggs)
    fn insert(
        &mut self,
        hash: u64,
        chunk_idx: IdxSize,
        row: &[u8],
        agg_iters: &mut [SeriesPhysIter],
    ) -> Option<(usize, SpillPayload)> {
        let partition = hash_to_partition(hash, self.aggs_partitioned.len());
        self.spilled = true;
        unsafe {
            let agg_values = self.aggs_partitioned.get_unchecked_release_mut(partition);
            let hashes = self.hash_partitioned.get_unchecked_release_mut(partition);
            let chunk_indexes = self
                .chunk_index_partitioned
                .get_unchecked_release_mut(partition);
            let key_builder = self.keys_partitioned.get_unchecked_mut(partition);

            hashes.push(hash);
            chunk_indexes.push(chunk_idx);

            // amortize the loop counter
            key_builder.push(Some(row));
            for (i, agg) in agg_iters.iter_mut().enumerate() {
                let av = agg.next().unwrap_unchecked_release();
                let buf = agg_values.get_unchecked_mut(i);
                buf.add_unchecked_borrowed_physical(&av);
            }

            if hashes.len() >= OB_SIZE {
                let mut new_hashes = Vec::with_capacity(OB_SIZE);
                let mut new_chunk_indexes = Vec::with_capacity(OB_SIZE);
                let mut new_keys_builder = MutableBinaryArray::with_capacity(OB_SIZE);
                std::mem::swap(&mut new_hashes, hashes);
                std::mem::swap(&mut new_chunk_indexes, chunk_indexes);
                std::mem::swap(&mut new_keys_builder, key_builder);

                Some((
                    partition,
                    SpillPayload {
                        hashes: new_hashes,
                        chunk_idx: new_chunk_indexes,
                        keys: new_keys_builder.into(),
                        aggs: agg_values
                            .iter_mut()
                            .zip(self.output_schema.iter_names())
                            .map(|(b, name)| {
                                let mut s = b.reset(OB_SIZE);
                                s.rename(name);
                                s
                            })
                            .collect(),
                    },
                ))
            } else {
                None
            }
        }
    }

    fn finish(&mut self) {
        if self.spilled {
            let all_spilled = self.get_all_spilled().collect::<Vec<_>>();
            for (partition_i, payload) in all_spilled {
                let buf = if let Some(buf) = self.finished_payloads.get_mut(partition_i) {
                    buf
                } else {
                    self.finished_payloads.push(vec![]);
                    self.finished_payloads.last_mut().unwrap()
                };
                buf.push(payload)
            }
        }
    }

    fn combine(&mut self, other: &mut Self) {
        match (self.spilled, other.spilled) {
            (false, true) => std::mem::swap(self, other),
            (true, false) => {},
            (false, false) => {},
            (true, true) => {
                self.finish();
                other.finish();
                let other_payloads = std::mem::take(&mut other.finished_payloads);

                for (part_self, part_other) in self.finished_payloads.iter_mut().zip(other_payloads)
                {
                    part_self.extend(part_other)
                }
            },
        }
    }

    fn get_all_spilled(&mut self) -> impl Iterator<Item = (usize, SpillPayload)> + '_ {
        // todo! allocate
        let mut flattened = vec![];
        let finished_payloads = std::mem::take(&mut self.finished_payloads);
        for (part, payloads) in finished_payloads.into_iter().enumerate() {
            for payload in payloads {
                flattened.push((part, payload))
            }
        }

        (0..PARTITION_SIZE)
            .map(|partition| unsafe {
                let spilled_aggs = self.aggs_partitioned.get_unchecked_release_mut(partition);
                let hashes = self.hash_partitioned.get_unchecked_release_mut(partition);
                let chunk_indexes = self
                    .chunk_index_partitioned
                    .get_unchecked_release_mut(partition);
                let keys_builder =
                    std::mem::take(self.keys_partitioned.get_unchecked_mut(partition));
                let hashes = std::mem::take(hashes);
                let chunk_idx = std::mem::take(chunk_indexes);

                (
                    partition,
                    SpillPayload {
                        hashes,
                        chunk_idx,
                        keys: keys_builder.into(),
                        aggs: spilled_aggs.iter_mut().map(|b| b.reset(0)).collect(),
                    },
                )
            })
            .chain(flattened)
    }
}

pub(super) struct ThreadLocalTable {
    inner_map: AggHashTable<true>,
    spill_partitions: SpillPartitions,
}

impl ThreadLocalTable {
    pub(super) fn new(
        agg_constructors: Arc<[AggregateFunction]>,
        key_dtypes: Arc<[DataType]>,
        agg_dtypes: Arc<[DataType]>,
        output_schema: SchemaRef,
    ) -> Self {
        let spill_partitions =
            SpillPartitions::new(key_dtypes.clone(), agg_dtypes, output_schema.clone());

        Self {
            inner_map: AggHashTable::new(
                agg_constructors,
                key_dtypes.as_ref(),
                output_schema,
                Some(*SPILL_SIZE),
            ),
            spill_partitions,
        }
    }

    pub(super) fn split(&self) -> Self {
        // should be called before any chunk is processed
        debug_assert!(self.inner_map.is_empty());

        Self {
            inner_map: self.inner_map.split(),
            spill_partitions: self.spill_partitions.clone(),
        }
    }

    pub(super) fn get_inner_map_mut(&mut self) -> &mut AggHashTable<true> {
        &mut self.inner_map
    }

    /// # Safety
    /// Caller must ensure that `keys` and `agg_iters` are not depleted.
    #[inline]
    pub(super) unsafe fn insert(
        &mut self,
        hash: u64,
        keys_row: &[u8],
        agg_iters: &mut [SeriesPhysIter],
        chunk_index: IdxSize,
    ) -> Option<(usize, SpillPayload)> {
        if self
            .inner_map
            .insert(hash, keys_row, agg_iters, chunk_index)
        {
            self.spill_partitions
                .insert(hash, chunk_index, keys_row, agg_iters)
        } else {
            None
        }
    }

    pub(super) fn combine(&mut self, other: &mut Self) {
        self.inner_map.combine(&other.inner_map);
        self.spill_partitions.combine(&mut other.spill_partitions);
    }

    pub(super) fn finalize(&mut self, slice: &mut Option<(i64, usize)>) -> Option<DataFrame> {
        if !self.spill_partitions.spilled {
            Some(self.inner_map.finalize(slice))
        } else {
            None
        }
    }

    pub(super) fn get_all_spilled(&mut self) -> impl Iterator<Item = (usize, SpillPayload)> + '_ {
        self.spill_partitions.get_all_spilled()
    }
}
