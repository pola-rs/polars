use std::collections::LinkedList;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Mutex;

use super::*;
use crate::pipeline::PARTITION_SIZE;

struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    partitions: PartitionVec<Mutex<LinkedList<SpillPayload>>>,
}

impl SpillPartitions {
    fn new() -> Self {
        let mut partitions = Vec::with_capacity(PARTITION_SIZE);
        partitions.fill_with(Default::default);

        Self { partitions }
    }

    #[inline]
    fn insert(&self, partition: usize, to_spill: SpillPayload) {
        let partition = &self.partitions[partition];
        let mut partition = partition.lock().unwrap();
        partition.push_back(to_spill)
    }

    fn drain_partition(&self, partition: usize) -> LinkedList<SpillPayload> {
        let partition = &self.partitions[partition];
        let mut partition = partition.lock().unwrap();
        std::mem::take(&mut partition)
    }
}

pub(super) struct GlobalTable {
    inner_maps: PartitionVec<Mutex<AggHashTable<false>>>,
    spill_partitions: SpillPartitions,
    early_merge_counter: Arc<AtomicU16>,
}

impl GlobalTable {
    pub(super) fn new(
        agg_constructors: Arc<[AggregateFunction]>,
        key_dtypes: &[DataType],
        output_schema: SchemaRef,
    ) -> Self {
        let spill_partitions = SpillPartitions::new();
        let mut inner_maps = Vec::with_capacity(PARTITION_SIZE);
        inner_maps.fill_with(|| {
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
        }
    }

    #[inline]
    pub(super) fn spill(&self, partition: usize, payload: SpillPayload) {
        self.spill_partitions.insert(partition, payload)
    }

    pub(super) fn early_merge(&self) {
        // round robin a partition to merge early
        let partition =
            self.early_merge_counter.fetch_add(1, Ordering::Relaxed) as usize % PARTITION_SIZE;
        self.process_partition(partition)
    }

    fn process_partition(&self, partition: usize) {
        let bucket = self.spill_partitions.drain_partition(partition);
        let mut hash_map = self.inner_maps[partition].lock().unwrap();

        for payload in bucket {
            let hashes = payload.hashes();
            let keys = payload.keys();
            let chunk_indexes = payload.chunk_index();
            let agg_cols = payload.cols();
            debug_assert_eq!(hashes.len(), agg_cols.len());
            debug_assert_eq!(hashes.len(), keys.len());

            let mut keys_iters = keys.iter().map(|s| s.phys_iter()).collect::<Vec<_>>();
            let mut agg_cols_iters = agg_cols.iter().map(|s| s.phys_iter()).collect::<Vec<_>>();

            // amortize loop counter
            for i in 0..hashes.len() {
                unsafe {
                    let hash = *hashes.get_unchecked(i);
                    let chunk_index = *chunk_indexes.get_unchecked(i);

                    // safety: keys_iters and cols_iters are not depleted
                    hash_map
                        .insert(hash, &mut keys_iters, &mut agg_cols_iters, chunk_index)
                        .unwrap_unchecked();
                }
            }
        }
    }
}
