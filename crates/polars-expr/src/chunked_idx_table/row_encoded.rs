use std::sync::atomic::AtomicU64;

use arrow::bitmap::MutableBitmap;
use polars_row::EncodingField;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::idx_map::bytes_idx_map::{BytesIndexMap, Entry};
use polars_utils::idx_vec::UnitVec;
use polars_utils::itertools::Itertools;
use polars_utils::unitvec;
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::HashKeys;

#[derive(Default)]
pub struct RowEncodedChunkedIdxTable {
    // These AtomicU64s actually are ChunkIds, but we use the top bit of the
    // first chunk in each to mark keys during probing.
    idx_map: BytesIndexMap<UnitVec<AtomicU64>>,
    chunk_ctr: IdxSize,
}

impl RowEncodedChunkedIdxTable {
    pub fn new() -> Self {
        Self {
            idx_map: BytesIndexMap::new(),
            chunk_ctr: 0,
        }
    }
}

impl ChunkedIdxTable for RowEncodedChunkedIdxTable {
    fn new_empty(&self) -> Box<dyn ChunkedIdxTable> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_keys(&self) -> IdxSize {
        self.idx_map.len()
    }

    fn insert_key_chunk(&mut self, hash_keys: HashKeys) {
        let HashKeys::RowEncoded(keys) = hash_keys else {
            unreachable!()
        };
        if keys.keys.len() >= 1 << 31 {
            panic!("overly large chunk in RowEncodedChunkedIdxTable");
        }

        // for in keys.hashes
        // group_idxs.clear();
        // group_idxs.reserve(keys.hashes.len());
        for (i, (hash, key)) in keys.hashes.values_iter().zip(keys.keys.iter()).enumerate_idx() {
            if let Some(key) = key {
                let chunk_id = AtomicU64::new(ChunkId::<_>::store(self.chunk_ctr, i).into_inner());
                match self.idx_map.entry(*hash, key) {
                    Entry::Occupied(o) => { o.into_mut().push(chunk_id); },
                    Entry::Vacant(v) => { v.insert(unitvec![chunk_id]); },
                }

            }
        }
        
        self.chunk_ctr = self.chunk_ctr.checked_add(1).unwrap();
    }
    
    fn probe(&self, keys: &HashKeys, table_match: &mut Vec<ChunkId>, probe_match: &mut Vec<IdxSize>, mark_matches: bool, limit: usize) -> usize {
        todo!()
    }
    
    fn unmarked_keys(&self, out: &mut Vec<ChunkId>) {
        todo!()
    }

}

/*
impl Grouper for RowEncodedChunkedIdxTable {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new(self.key_schema.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_groups(&self) -> IdxSize {
        self.idx_map.len()
    }

    fn insert_keys(&mut self, keys: HashKeys, group_idxs: &mut Vec<IdxSize>) {
        let HashKeys::RowEncoded(keys) = keys else {
            unreachable!()
        };
        group_idxs.clear();
        group_idxs.reserve(keys.hashes.len());
        for (hash, key) in keys.hashes.iter().zip(keys.keys.values_iter()) {
            unsafe {
                group_idxs.push_unchecked(self.insert_key(*hash, key));
            }
        }
    }

    fn combine(&mut self, other: &dyn Grouper, group_idxs: &mut Vec<IdxSize>) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        // TODO: cardinality estimation.
        self.idx_map.reserve(other.idx_map.len() as usize);

        unsafe {
            group_idxs.clear();
            group_idxs.reserve(other.idx_map.len() as usize);
            for (hash, key) in other.idx_map.iter_hash_keys() {
                group_idxs.push_unchecked(self.insert_key(hash, key));
            }
        }
    }

    unsafe fn gather_combine(
        &mut self,
        other: &dyn Grouper,
        subset: &[IdxSize],
        group_idxs: &mut Vec<IdxSize>,
    ) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        // TODO: cardinality estimation.
        self.idx_map.reserve(subset.len());

        unsafe {
            group_idxs.clear();
            group_idxs.reserve(subset.len());
            for i in subset {
                let (hash, key, ()) = other.idx_map.get_index_unchecked(*i);
                group_idxs.push_unchecked(self.insert_key(hash, key));
            }
        }
    }

    fn get_keys_in_group_order(&self) -> DataFrame {
        unsafe {
            let mut key_rows: Vec<&[u8]> = Vec::with_capacity(self.idx_map.len() as usize);
            for (_, key) in self.idx_map.iter_hash_keys() {
                key_rows.push_unchecked(key);
            }
            self.finalize_keys(key_rows)
        }
    }

    fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
    ) {
        let num_partitions = partitioner.num_partitions();
        assert!(partition_idxs.len() == num_partitions);
        assert!(sketches.len() == num_partitions);

        // Two-pass algorithm to prevent reallocations.
        let mut partition_sizes = vec![0; num_partitions];
        unsafe {
            for (hash, _key) in self.idx_map.iter_hash_keys() {
                let p_idx = partitioner.hash_to_partition(hash);
                *partition_sizes.get_unchecked_mut(p_idx) += 1;
                sketches.get_unchecked_mut(p_idx).insert(hash);
            }
        }

        for (partition, sz) in partition_idxs.iter_mut().zip(partition_sizes) {
            partition.clear();
            partition.reserve(sz);
        }

        unsafe {
            for (i, (hash, _key)) in self.idx_map.iter_hash_keys().enumerate() {
                let p_idx = partitioner.hash_to_partition(hash);
                let p = partition_idxs.get_unchecked_mut(p_idx);
                p.push_unchecked(i as IdxSize);
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
*/