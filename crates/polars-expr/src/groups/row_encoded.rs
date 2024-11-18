use arrow::array::Array;
use polars_row::EncodingField;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::idx_map::bytes_idx_map::{BytesIndexMap, Entry};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::HashKeys;

#[derive(Default)]
pub struct RowEncodedHashGrouper {
    key_schema: Arc<Schema>,
    idx_map: BytesIndexMap<()>,
}

impl RowEncodedHashGrouper {
    pub fn new(key_schema: Arc<Schema>) -> Self {
        Self {
            key_schema,
            idx_map: BytesIndexMap::new(),
        }
    }

    fn insert_key(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        match self.idx_map.entry(hash, key) {
            Entry::Occupied(o) => o.index(),
            Entry::Vacant(v) => {
                let index = v.index();
                v.insert(());
                index
            },
        }
    }

    fn finalize_keys(&self, mut key_rows: Vec<&[u8]>) -> DataFrame {
        let key_dtypes = self
            .key_schema
            .iter()
            .map(|(_name, dt)| dt.to_physical().to_arrow(CompatLevel::newest()))
            .collect::<Vec<_>>();
        let fields = vec![EncodingField::new_unsorted(); key_dtypes.len()];
        let key_columns =
            unsafe { polars_row::decode::decode_rows(&mut key_rows, &fields, &key_dtypes) };

        let cols = self
            .key_schema
            .iter()
            .zip(key_columns)
            .map(|((name, dt), col)| {
                let s = Series::try_from((name.clone(), col)).unwrap();
                unsafe { s.to_logical_repr_unchecked(dt) }
                    .unwrap()
                    .into_column()
            })
            .collect();
        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }
}

impl Grouper for RowEncodedHashGrouper {
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
        assert!(!keys.hashes.has_nulls());
        assert!(!keys.keys.has_nulls());

        group_idxs.clear();
        group_idxs.reserve(keys.hashes.len());
        for (hash, key) in keys.hashes.values_iter().zip(keys.keys.values_iter()) {
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
