use std::collections::LinkedList;
use std::sync::Mutex;

use super::*;

struct SpillPayload {
    hashes: Vec<u64>,
    keys_cols: Vec<Series>,
    num_keys: usize,
}

impl SpillPayload {
    fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keys(&self) -> &[Series] {
        &self.keys_cols[..self.num_keys]
    }

    fn cols(&self) -> &[Series] {
        &self.keys_cols[self.num_keys..]
    }
}

struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    partitions: PartitionVec<Mutex<LinkedList<SpillPayload>>>,
}

impl SpillPartitions {
    fn insert(&self, hash: u64, to_spill: SpillPayload) {
        let partition = hash_to_partition(hash, self.partitions.len());
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
    inner_map: PartitionVec<PlIdHashMap<Key, IdxSize>>,
    spill_partitions: SpillPartitions,
    num_keys: usize,
}

impl GlobalTable {
    fn process_partition(&mut self, partition: usize) {
        let bucket = self.spill_partitions.drain_partition(partition);
        let hash_map = &mut self.inner_map[partition];

        for payload in bucket {
            let hashes = payload.hashes();
            let keys = payload.keys();
            let agg_cols = payload.cols();
            debug_assert_eq!(hashes.len(), agg_cols.len());
            debug_assert_eq!(hashes.len(), keys.len());

            for i in 0..hashes.len() {
                unsafe {
                    let hash = hashes.get_unchecked(i);
                    let key = keys.get_unchecked(i);
                    let col = agg_cols.get_unchecked(i);
                }
            }
        }
    }
}
