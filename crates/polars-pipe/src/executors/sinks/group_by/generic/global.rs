use std::collections::LinkedList;
use std::sync::atomic::{AtomicU16, Ordering};

use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::POOL;
use rayon::prelude::*;

use super::*;
use crate::pipeline::{FORCE_OOC, PARTITION_SIZE};

struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    partitions: PartitionVec<Mutex<LinkedList<SpillPayload>>>,
}

impl SpillPartitions {
    fn new() -> Self {
        let mut partitions = Vec::with_capacity(PARTITION_SIZE);
        partitions.resize_with(PARTITION_SIZE, Default::default);

        Self { partitions }
    }

    #[inline]
    fn insert(&self, partition: usize, to_spill: SpillPayload) -> usize {
        let partition = &self.partitions[partition];
        let mut partition = partition.lock().unwrap();
        partition.push_back(to_spill);
        partition.len()
    }

    fn drain_partition(
        &self,
        partition: usize,
        min_size: usize,
    ) -> Option<LinkedList<SpillPayload>> {
        let partition = &self.partitions[partition];
        let mut partition = partition.lock().unwrap();
        if partition.len() > min_size {
            Some(std::mem::take(&mut partition))
        } else {
            None
        }
    }

    fn spill_schema(&self) -> Option<Schema> {
        for part in &self.partitions {
            let bucket = part.lock().unwrap();
            if let Some(payload) = bucket.front() {
                return Some(payload.get_schema());
            }
        }
        None
    }
}

pub(super) struct GlobalTable {
    inner_maps: PartitionVec<Mutex<AggHashTable<false>>>,
    spill_partitions: SpillPartitions,
    early_merge_counter: Arc<AtomicU16>,
    // IO is expensive so we only spill if we have `N` payloads to dump.
    spill_partition_ob_size: usize,
}

impl GlobalTable {
    pub(super) fn new(
        agg_constructors: Arc<[AggregateFunction]>,
        key_dtypes: &[DataType],
        output_schema: SchemaRef,
    ) -> Self {
        let spill_partitions = SpillPartitions::new();

        let spill_partition_ob_size = if std::env::var(FORCE_OOC).is_ok() {
            1
        } else {
            64
        };

        let mut inner_maps = Vec::with_capacity(PARTITION_SIZE);
        inner_maps.resize_with(PARTITION_SIZE, || {
            Mutex::new(AggHashTable::new(
                agg_constructors.clone(),
                key_dtypes,
                output_schema.clone(),
                None,
            ))
        });

        Self {
            inner_maps,
            spill_partitions,
            early_merge_counter: Default::default(),
            spill_partition_ob_size,
        }
    }

    #[inline]
    pub(super) fn spill(&self, partition: usize, payload: SpillPayload) {
        self.spill_partitions.insert(partition, payload);
    }

    pub(super) fn early_merge(&self) {
        // round robin a partition to merge early
        let partition =
            self.early_merge_counter.fetch_add(1, Ordering::Relaxed) as usize % PARTITION_SIZE;
        self.process_partition(partition)
    }

    pub(super) fn get_ooc_dump_schema(&self) -> Option<Schema> {
        self.spill_partitions.spill_schema()
    }

    pub(super) fn get_ooc_dump(&self) -> Option<(usize, DataFrame)> {
        // round robin a partition to dump
        let partition =
            self.early_merge_counter.fetch_add(1, Ordering::Relaxed) as usize % PARTITION_SIZE;

        // IO is expensive so we only spill if we have `N` payloads to dump.
        let bucket = self
            .spill_partitions
            .drain_partition(partition, self.spill_partition_ob_size)?;
        Some((
            partition,
            accumulate_dataframes_vertical_unchecked(bucket.into_iter().map(|pl| pl.into_df())),
        ))
    }

    fn process_partition_impl(
        &self,
        hash_map: &mut AggHashTable<false>,
        hashes: &[u64],
        chunk_indexes: &[IdxSize],
        keys: &BinaryArray<i64>,
        agg_cols: &[Series],
    ) {
        debug_assert_eq!(hashes.len(), chunk_indexes.len());
        debug_assert_eq!(hashes.len(), keys.len());

        // let mut keys_iters = keys.iter().map(|s| s.phys_iter()).collect::<Vec<_>>();
        let mut agg_cols_iters = agg_cols.iter().map(|s| s.phys_iter()).collect::<Vec<_>>();

        // amortize loop counter
        for (i, row) in keys.values_iter().enumerate() {
            unsafe {
                let hash = *hashes.get_unchecked(i);
                let chunk_index = *chunk_indexes.get_unchecked(i);

                // SAFETY: keys_iters and cols_iters are not depleted
                let overflow = hash_map.insert(hash, row, &mut agg_cols_iters, chunk_index);
                // should never overflow
                debug_assert!(!overflow);
            }
        }
    }

    pub(super) fn process_partition_from_dumped(&self, partition: usize, spilled: &DataFrame) {
        let mut hash_map = self.inner_maps[partition].lock().unwrap();
        let (hashes, chunk_indexes, keys, aggs) = SpillPayload::spilled_to_columns(spilled);
        self.process_partition_impl(&mut hash_map, hashes, chunk_indexes, keys, aggs);
    }

    fn process_partition(&self, partition: usize) {
        if let Some(bucket) = self.spill_partitions.drain_partition(partition, 0) {
            let mut hash_map = self.inner_maps[partition].lock().unwrap();

            for payload in bucket {
                let hashes = payload.hashes();
                let keys = payload.keys();
                let chunk_indexes = payload.chunk_index();
                let agg_cols = payload.cols();
                self.process_partition_impl(&mut hash_map, hashes, chunk_indexes, keys, agg_cols);
            }
        }
    }

    pub(super) fn merge_local_map(&self, finalized_local_map: &AggHashTable<true>) {
        // TODO! maybe parallelize?
        // needs unsafe, first benchmark.
        for (partition_i, pt_map) in self.inner_maps.iter().enumerate() {
            let mut pt_map = pt_map.lock().unwrap();
            pt_map.combine_on_partition(partition_i, finalized_local_map)
        }
    }

    pub(super) fn finalize_partition(
        &self,
        partition: usize,
        slice: &mut Option<(i64, usize)>,
    ) -> DataFrame {
        // ensure all spilled partitions are processed
        self.process_partition(partition);
        let mut hash_map = self.inner_maps[partition].lock().unwrap();
        hash_map.finalize(slice)
    }

    // only should be called if all state is in-memory
    pub(super) fn finalize(&self, slice: &mut Option<(i64, usize)>) -> Vec<DataFrame> {
        if slice.is_none() {
            POOL.install(|| {
                (0..PARTITION_SIZE)
                    .into_par_iter()
                    .map(|part_i| {
                        self.process_partition(part_i);
                        let mut hash_map = self.inner_maps[part_i].lock().unwrap();
                        hash_map.finalize(&mut None)
                    })
                    .collect()
            })
        } else {
            (0..PARTITION_SIZE)
                .map(|part_i| {
                    self.process_partition(part_i);
                    let mut hash_map = self.inner_maps[part_i].lock().unwrap();
                    hash_map.finalize(slice)
                })
                .collect()
        }
    }
}
