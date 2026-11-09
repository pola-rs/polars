#![allow(unsafe_op_in_unsafe_fn)]

use hashbrown::hash_table::{Entry, HashTable};
use polars_arrow::array::Array;
use polars_arrow::bitmap::MutableBitmap;
use polars_core::error::constants::LENGTH_LIMIT_MSG;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::{BLOCK_SIZE, HashKeys, MIN_PREFETCH_BUCKETS};

pub struct SingleKeyHashGrouper<T: PolarsDataType> {
    /// Each key with its group.
    table: HashTable<(T::Physical<'static>, IdxSize)>,
    num_groups: IdxSize,
    /// The group of the null key, IdxSize::MAX if there is none.
    null_idx: IdxSize,
    random_state: PlRandomState,
}

impl<K, T: PolarsDataType> SingleKeyHashGrouper<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq + Copy + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            table: HashTable::new(),
            num_groups: 0,
            null_idx: IdxSize::MAX,
            random_state: PlRandomState::default(),
        }
    }

    #[inline(always)]
    fn should_prefetch(&self) -> bool {
        self.table.num_buckets() >= MIN_PREFETCH_BUCKETS
    }

    #[inline(always)]
    fn group_idx(&self, hash: u64, key: &K) -> Option<IdxSize> {
        self.table
            .find(hash, |(k, _)| k.tot_eq(key))
            .map(|(_, group)| *group)
    }

    #[inline(always)]
    fn null_group_idx(&self) -> Option<IdxSize> {
        (self.null_idx < IdxSize::MAX).then_some(self.null_idx)
    }

    /// Inserts the keys in blocks, first hashing (and prefetching) the whole
    /// block so the CPU can work on the lookups of several keys at once.
    ///
    /// # Safety
    /// The subset indices must be in-bounds for `arr`.
    #[inline(always)]
    unsafe fn insert_impl(
        &mut self,
        arr: &T::Array,
        null_is_valid: bool,
        subset: &[IdxSize],
        mut on_group: impl FnMut(IdxSize),
    ) {
        // IdxSize::MAX marks a missing null group, so it can't be a group.
        assert!(
            self.num_groups as usize + subset.len() < IdxSize::MAX as usize,
            "{}",
            LENGTH_LIMIT_MSG
        );

        let validity = arr.validity().filter(|_| arr.has_nulls());
        let random_state = &self.random_state;
        let hasher = |(k, _): &(K, IdxSize)| random_state.tot_hash_one(*k);
        let mut hashes = [0; BLOCK_SIZE];
        for block in subset.chunks(BLOCK_SIZE) {
            // Reserve up front so the table doesn't move after prefetching.
            self.table.reserve(block.len(), hasher);
            let prefetch = self.should_prefetch();
            for (hash, idx) in hashes.iter_mut().zip(block) {
                *hash = random_state.tot_hash_one(arr.value_unchecked(*idx as usize));
                if prefetch {
                    self.table.prefetch_insert(*hash);
                }
            }

            for (hash, idx) in hashes.iter().zip(block) {
                let idx = *idx as usize;
                if validity.is_some_and(|v| !v.get_bit_unchecked(idx)) {
                    if null_is_valid {
                        if self.null_idx == IdxSize::MAX {
                            self.null_idx = self.num_groups;
                            self.num_groups += 1;
                        }
                        on_group(self.null_idx);
                    }
                } else {
                    let key = arr.value_unchecked(idx);
                    match self.table.entry(*hash, |(k, _)| k.tot_eq(&key), hasher) {
                        Entry::Occupied(o) => on_group(o.get().1),
                        Entry::Vacant(v) => {
                            v.insert((key, self.num_groups));
                            on_group(self.num_groups);
                            self.num_groups += 1;
                        },
                    }
                }
            }
        }
    }

    /// Looks up each key in the grouper of its partition, in blocks as when
    /// inserting. `f` gets the index of each key, and the partition and group
    /// of the keys that are found.
    ///
    /// # Safety
    /// All groupers must be a SingleKeyHashGrouper<T>.
    #[inline(always)]
    unsafe fn probe_partitions(
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        mut f: impl FnMut(IdxSize, Option<(usize, IdxSize)>),
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };
        let ca: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        let arr = ca.downcast_as_array();
        let validity = arr.validity().filter(|_| arr.has_nulls());
        assert!(partitioner.num_partitions() == groupers.len());
        let grouper =
            |p: usize| &*(&**groupers.get_unchecked(p) as *const dyn Grouper as *const Self);
        let null_p = partitioner.null_partition();
        let null_group = hash_keys
            .null_is_valid
            .then(|| grouper(null_p).null_group_idx())
            .flatten()
            .map(|g| (null_p, g));

        let mut hashes = [0; BLOCK_SIZE];
        let mut partitions = [0; BLOCK_SIZE];
        for block_start in (0..arr.len()).step_by(BLOCK_SIZE) {
            let block = block_start..arr.len().min(block_start + BLOCK_SIZE);
            for ((hash, p), idx) in hashes.iter_mut().zip(&mut partitions).zip(block.clone()) {
                let key = arr.value_unchecked(idx);
                *p = partitioner.hash_to_partition(hash_keys.random_state.tot_hash_one(key));
                let p_grouper = grouper(*p);
                *hash = p_grouper.random_state.tot_hash_one(key);
                if p_grouper.should_prefetch() {
                    p_grouper.table.prefetch_get(*hash);
                }
            }

            for ((hash, p), idx) in hashes.iter().zip(&partitions).zip(block) {
                let found = if validity.is_some_and(|v| !v.get_bit_unchecked(idx)) {
                    null_group
                } else {
                    let key = arr.value_unchecked(idx);
                    grouper(*p).group_idx(*hash, &key).map(|g| (*p, g))
                };
                f(idx as IdxSize, found);
            }
        }
    }

    fn finalize_keys(&self, schema: &Schema, keys: Vec<K>) -> DataFrame {
        let (name, dtype) = schema.get_at_index(0).unwrap();
        let mut keys =
            T::Array::from_vec(keys, dtype.to_physical().to_arrow(CompatLevel::newest()));
        if self.null_idx < IdxSize::MAX {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(keys.len(), true);
            validity.set(self.null_idx as usize, false);
            keys = keys.with_validity_typed(Some(validity.freeze()));
        }
        unsafe {
            let s =
                Series::from_chunks_and_dtype_unchecked(name.clone(), vec![Box::new(keys)], dtype);
            DataFrame::new_unchecked(s.len(), vec![Column::from(s)])
        }
    }
}

impl<K, T: PolarsDataType> Grouper for SingleKeyHashGrouper<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq + Copy + Send + Sync + 'static,
{
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        let random_state = &self.random_state;
        self.table
            .reserve(additional, |(k, _)| random_state.tot_hash_one(*k));
    }

    fn num_groups(&self) -> IdxSize {
        self.num_groups
    }

    unsafe fn insert_keys_subset(
        &mut self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        group_idxs: Option<&mut Vec<IdxSize>>,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };
        let ca: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        let arr = ca.downcast_as_array();
        let null_is_valid = hash_keys.null_is_valid;
        if let Some(group_idxs) = group_idxs {
            group_idxs.reserve(subset.len());
            self.insert_impl(arr, null_is_valid, subset, |group| {
                group_idxs.push_unchecked(group)
            });
        } else {
            self.insert_impl(arr, null_is_valid, subset, |_| {});
        }
    }

    fn get_keys_in_group_order(&self, schema: &Schema) -> DataFrame {
        // The null group keeps a default key.
        let mut keys = vec![K::default(); self.num_groups as usize];
        for (key, group) in self.table.iter() {
            unsafe { *keys.get_unchecked_mut(*group as usize) = *key };
        }
        self.finalize_keys(schema, keys)
    }

    /// # Safety
    /// All groupers must be a SingleKeyHashGrouper<T>.
    unsafe fn probe_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        probe_matches: &mut Vec<IdxSize>,
    ) {
        Self::probe_partitions(groupers, hash_keys, partitioner, |idx, found| {
            if found.is_some() != invert {
                probe_matches.push(idx);
            }
        });
    }

    /// # Safety
    /// All groupers must be a SingleKeyHashGrouper<T>.
    unsafe fn contains_key_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        contains_key: &mut BitmapBuilder,
    ) {
        Self::probe_partitions(groupers, hash_keys, partitioner, |_idx, found| {
            contains_key.push(found.is_some() != invert);
        });
    }

    /// # Safety
    /// All groupers must be a SingleKeyHashGrouper<T>.
    unsafe fn mark_groups_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        marks: &mut [MutableBitmap],
    ) {
        assert!(marks.len() == groupers.len());
        Self::probe_partitions(groupers, hash_keys, partitioner, |_idx, found| {
            if let Some((p, group_idx)) = found {
                marks
                    .get_unchecked_mut(p)
                    .set_unchecked(group_idx as usize, true);
            }
        });
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
