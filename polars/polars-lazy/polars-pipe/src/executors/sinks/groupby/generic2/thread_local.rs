use std::cell::UnsafeCell;

use polars_arrow::trusted_len::PushUnchecked;

use super::*;
use crate::pipeline::PARTITION_SIZE;

const SPILL_SIZE: usize = 128;

#[derive(Clone)]
struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    // inner vec: number of keys + number of aggregated columns
    partitions: PartitionVec<Vec<AnyValueBufferTrusted<'static>>>,
    hash_partitioned: PartitionVec<Vec<u64>>,
    chunk_index_partitioned: PartitionVec<Vec<IdxSize>>,
    // outer vec: partitions
    // inner vec: aggregation columns
    finished: PartitionVec<Vec<Series>>,
    spill_count: u16,
}

impl SpillPartitions {
    fn new(keys: &[DataType], aggs: &[DataType]) -> Self {
        let n_spills = keys.len() + aggs.len();

        let mut buf = Vec::with_capacity(n_spills);
        for dtype in keys {
            let builder = AnyValueBufferTrusted::new(dtype, 0);
            buf.push(builder);
        }
        for dtype in aggs {
            let builder = AnyValueBufferTrusted::new(dtype, 0);
            buf.push(builder);
        }

        let partitions = (0..PARTITION_SIZE).map(|partition| buf.clone()).collect();
        let hash_partitioned = vec![vec![]; PARTITION_SIZE];
        let chunk_index_partitioned = vec![vec![]; PARTITION_SIZE];

        Self {
            partitions,
            hash_partitioned,
            chunk_index_partitioned,
            finished: vec![],
            spill_count: 0,
        }
    }
}

impl SpillPartitions {
    fn insert(
        &mut self,
        hash: u64,
        chunk_idx: IdxSize,
        keys: &[AnyValue<'_>],
        aggs: &mut [SeriesPhysIter],
    ) {
        let partition = hash_to_partition(hash, self.partitions.len());
        unsafe {
            let partitions = self.partitions.get_unchecked_release_mut(partition);
            let hashes = self.hash_partitioned.get_unchecked_release_mut(partition);
            let chunk_indexes = self
                .chunk_index_partitioned
                .get_unchecked_release_mut(partition);
            debug_assert_eq!(keys.len(), partitions.len());

            hashes.push(hash);
            chunk_indexes.push(chunk_idx);

            // amortize the loop counter
            for i in 0..keys.len() {
                let av = keys.get_unchecked(i);
                let buf = partitions.get_unchecked_mut(i);
                // safety: we can trust the input types to be of consistent dtype
                buf.add_unchecked_owned_physical(av);
            }
            let mut i = keys.len();
            for agg in aggs {
                let av = agg.next().unwrap_unchecked_release();
                let buf = partitions.get_unchecked_mut(i);
                buf.add_unchecked_owned_physical(&av);
                i += 1;
            }
        };
    }

    fn finish(&mut self) {
        if self.finished.is_empty() {
            let parts = std::mem::take(&mut self.partitions);
            self.finished = parts
                .into_iter()
                .map(|part| part.into_iter().map(|v| v.into_series()).collect())
                .collect();
        }
    }

    fn combine(&mut self, other: &mut Self) {
        self.finish();
        other.finish();

        for (part_self, part_other) in self.finished.iter_mut().zip(other.finished.iter()) {
            for (s, other) in part_self.iter_mut().zip(part_other.iter()) {
                s.append(other).unwrap();
            }
        }
    }

    // round robin a partition that will be spilled
    fn prepare_for_global_spill(&mut self) -> SpillPayload {
        let i = (self.spill_count % (PARTITION_SIZE as u16)) as usize;
        self.spill_count += 1;

        let builder = unsafe { self.partitions.get_unchecked_release_mut(i) };
        todo!()
    }
}

pub(super) struct ThreadLocalTable {
    inner_map: AggHashTable<true>,
    spill_partitions: SpillPartitions,
}

impl ThreadLocalTable {
    pub(super) fn new(
        agg_constructors: Arc<[AggregateFunction]>,
        key_dtypes: &[DataType],
        output_schema: SchemaRef,
    ) -> Self {
        let agg_dtypes = agg_constructors
            .iter()
            .map(|agg| agg.dtype())
            .collect::<Vec<_>>();
        let spill_partitions = SpillPartitions::new(key_dtypes, &agg_dtypes);

        Self {
            inner_map: AggHashTable::new(agg_constructors, key_dtypes, output_schema, Some(128)),
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

    pub(super) fn len(&self) -> usize {
        self.inner_map.len()
    }

    /// # Safety
    /// Caller must ensure that `keys` and `agg_iters` are not depleted.
    pub(super) unsafe fn insert<'a>(
        &'a mut self,
        hash: u64,
        keys: &mut [SeriesPhysIter],
        agg_iters: &mut [SeriesPhysIter],
        chunk_index: IdxSize,
    ) {
        if let Some(keys) = self.inner_map.insert(hash, keys, agg_iters, chunk_index) {
            self.spill_partitions
                .insert(hash, chunk_index, keys, agg_iters);
        }
    }

    pub(super) fn combine(&mut self, other: &mut Self) {
        self.inner_map.combine(&mut other.inner_map);
        self.spill_partitions.combine(&mut other.spill_partitions);
    }

    pub(super) fn finalize(
        &mut self,
        slice: &mut Option<(i64, usize)>,
    ) -> (Option<DataFrame>, Vec<Vec<Series>>) {
        (
            self.inner_map.finalize(slice),
            std::mem::take(&mut self.spill_partitions.finished),
        )
    }
}
