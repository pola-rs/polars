#![allow(unsafe_op_in_unsafe_fn)]

use std::mem::MaybeUninit;

use polars_arrow::array::Array;
use polars_arrow::bitmap::MutableBitmap;
use polars_utils::mem::prefetch::prefetch_l1;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::flat_table::{BLOCK_SIZE, FlatTable, slots_for};
use crate::hash_keys::HashKeys;

pub struct SingleKeyHashGrouper<T: PolarsDataType> {
    /// The group of each key.
    table: FlatTable<IdxSize>,
    /// The key of each group. The null group has a default key.
    keys: Vec<T::Physical<'static>>,
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
            table: FlatTable::new(),
            keys: Vec::new(),
            null_idx: IdxSize::MAX,
            random_state: PlRandomState::default(),
        }
    }

    #[inline(always)]
    unsafe fn key(&self, group: IdxSize) -> &K {
        self.keys.get_unchecked(group as usize)
    }

    #[inline(always)]
    unsafe fn insert_key(&mut self, hash: u64, key: K) -> IdxSize {
        match self
            .table
            .find_slot(hash, |group| self.key(*group).tot_eq(&key))
        {
            Ok(pos) => *self.table.slot(pos),
            Err(pos) => {
                let group: IdxSize = self.keys.len().try_into().unwrap();
                self.keys.push(key);
                self.table.insert(pos, hash, group);
                group
            },
        }
    }

    #[inline(always)]
    fn insert_null(&mut self) -> IdxSize {
        if self.null_idx == IdxSize::MAX {
            self.null_idx = self.keys.len().try_into().unwrap();
            self.keys.push(K::default());
        }
        self.null_idx
    }

    #[inline(always)]
    unsafe fn group_idx(&self, hash: u64, key: &K) -> Option<IdxSize> {
        let pos = self
            .table
            .find_slot(hash, |group| self.key(*group).tot_eq(key))
            .ok()?;
        Some(*self.table.slot(pos))
    }

    #[inline(always)]
    fn null_group_idx(&self) -> Option<IdxSize> {
        (self.null_idx < IdxSize::MAX).then_some(self.null_idx)
    }

    /// Prefetches the key of the first slot whose tag matches `hash`, as a
    /// key is compared after reading its group from the slot.
    #[inline(always)]
    unsafe fn prefetch_key(&self, hash: u64) {
        if self.table.prefetches()
            && let Some(group) = self.table.first_match(hash)
        {
            prefetch_l1(self.keys.as_ptr().add(*group as usize).cast());
        }
    }

    /// Inserts the keys in blocks. Each block first hashes all its keys, so
    /// the CPU can work on the lookups of several keys at once.
    #[inline(always)]
    unsafe fn insert_impl(
        &mut self,
        arr: &T::Array,
        null_is_valid: bool,
        subset: &[IdxSize],
        mut on_group: impl FnMut(IdxSize),
    ) {
        let validity = arr.validity().filter(|_| arr.has_nulls());
        let mut hashes = [const { MaybeUninit::<u64>::uninit() }; BLOCK_SIZE];
        for block in subset.chunks(BLOCK_SIZE) {
            // Grows for the whole block at once. Nulls and keys already in
            // the table don't take a slot, so this can grow a bit early.
            let num_keys = self.keys.len() + block.len();
            self.table.grow_for(num_keys, |group| {
                self.random_state
                    .tot_hash_one(*self.keys.get_unchecked(*group as usize))
            });
            for (hash, idx) in hashes.iter_mut().zip(block) {
                let key = arr.value_unchecked(*idx as usize);
                hash.write(self.random_state.tot_hash_one(key));
            }
            let hashes = &hashes[..block.len()];
            for hash in hashes {
                self.table.prefetch(hash.assume_init());
            }
            for hash in hashes {
                self.prefetch_key(hash.assume_init());
            }

            for (hash, idx) in hashes.iter().zip(block) {
                let idx = *idx as usize;
                if validity.is_some_and(|v| !v.get_bit_unchecked(idx)) {
                    if null_is_valid {
                        on_group(self.insert_null());
                    }
                } else {
                    let key = arr.value_unchecked(idx);
                    on_group(self.insert_key(hash.assume_init(), key));
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

        let mut hashes = [const { MaybeUninit::<u64>::uninit() }; BLOCK_SIZE];
        let mut partitions = [const { MaybeUninit::<usize>::uninit() }; BLOCK_SIZE];
        for block_start in (0..arr.len()).step_by(BLOCK_SIZE) {
            let block_len = BLOCK_SIZE.min(arr.len() - block_start);
            for i in 0..block_len {
                let key = arr.value_unchecked(block_start + i);
                let p = partitioner.hash_to_partition(hash_keys.random_state.tot_hash_one(key));
                let p_grouper = grouper(p);
                let hash = p_grouper.random_state.tot_hash_one(key);
                p_grouper.table.prefetch(hash);
                hashes.get_unchecked_mut(i).write(hash);
                partitions.get_unchecked_mut(i).write(p);
            }
            for i in 0..block_len {
                let p = partitions.get_unchecked(i).assume_init();
                let hash = hashes.get_unchecked(i).assume_init();
                grouper(p).prefetch_key(hash);
            }

            for i in 0..block_len {
                let idx = block_start + i;
                let found = if validity.is_some_and(|v| !v.get_bit_unchecked(idx)) {
                    null_group
                } else {
                    let p = partitions.get_unchecked(i).assume_init();
                    let hash = hashes.get_unchecked(i).assume_init();
                    let key = arr.value_unchecked(idx);
                    grouper(p).group_idx(hash, &key).map(|g| (p, g))
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
        let num_slots = slots_for(self.keys.len() + additional);
        if num_slots > self.table.num_slots() {
            self.table.resize(num_slots, |group| unsafe {
                self.random_state
                    .tot_hash_one(*self.keys.get_unchecked(*group as usize))
            });
        }
        self.keys.reserve(additional);
    }

    fn num_groups(&self) -> IdxSize {
        self.keys.len() as IdxSize
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
        self.finalize_keys(schema, self.keys.clone())
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
