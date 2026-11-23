#![allow(unsafe_op_in_unsafe_fn)]

use hashbrown::hash_table::{Entry, HashTable};
use polars_arrow::array::Array;
use polars_arrow::bitmap::MutableBitmap;
use polars_core::error::constants::LENGTH_LIMIT_MSG;
use polars_utils::float16::pf16;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::{BLOCK_SIZE, HashKeys, MIN_PREFETCH_BUCKETS};

/// Where a grouper keeps its keys and their groups.
pub trait KeyStore<K>: Send + Sync + 'static {
    fn new() -> Self;

    fn num_groups(&self) -> IdxSize;

    fn num_buckets(&self) -> usize;

    fn reserve(&mut self, additional: usize, random_state: &PlRandomState);

    fn prefetch_insert(&self, hash: u64);

    fn prefetch_get(&self, hash: u64);

    fn get(&self, hash: u64, key: &K) -> Option<IdxSize>;

    /// The group of `key`, a new group if `key` isn't stored yet.
    fn get_or_insert(&mut self, hash: u64, key: K, random_state: &PlRandomState) -> IdxSize;

    /// Adds a group without a key, for the null key.
    fn push_keyless_group(&mut self) -> IdxSize;

    /// The key of each group, a default key for a group without one.
    fn keys_in_group_order(&self) -> Vec<K>;

    /// As `keys_in_group_order`, leaving the store empty.
    fn take_keys_in_group_order(&mut self) -> Vec<K>;
}

/// A key type of the single-key grouper, with the store for it.
pub trait GroupKey: Default + TotalHash + TotalEq + Copy + Send + Sync + 'static {
    type Store: KeyStore<Self>;
}

macro_rules! impl_group_key {
    ($store:ident: $($t:ty),*) => {$(
        impl GroupKey for $t {
            type Store = $store<$t>;
        }
    )*};
}
impl_group_key!(InlineKeys: u8, u16, u32, u64, i8, i16, i32, i64, pf16, f32, f64);
// A 16-byte key next to its group would make every bucket half padding.
impl_group_key!(DenseKeys: u128, i128);

/// Each key with its group in the hash table.
pub struct InlineKeys<K> {
    table: HashTable<(K, IdxSize)>,
    num_groups: IdxSize,
}

impl<K: GroupKey> KeyStore<K> for InlineKeys<K> {
    fn new() -> Self {
        Self {
            table: HashTable::new(),
            num_groups: 0,
        }
    }

    #[inline(always)]
    fn num_groups(&self) -> IdxSize {
        self.num_groups
    }

    #[inline(always)]
    fn num_buckets(&self) -> usize {
        self.table.num_buckets()
    }

    #[inline(always)]
    fn reserve(&mut self, additional: usize, random_state: &PlRandomState) {
        self.table
            .reserve(additional, |(k, _)| random_state.tot_hash_one(*k));
    }

    #[inline(always)]
    fn prefetch_insert(&self, hash: u64) {
        self.table.prefetch_insert(hash);
    }

    #[inline(always)]
    fn prefetch_get(&self, hash: u64) {
        self.table.prefetch_get(hash);
    }

    #[inline(always)]
    fn get(&self, hash: u64, key: &K) -> Option<IdxSize> {
        self.table
            .find(hash, |(k, _)| k.tot_eq(key))
            .map(|(_, group)| *group)
    }

    #[inline(always)]
    fn get_or_insert(&mut self, hash: u64, key: K, random_state: &PlRandomState) -> IdxSize {
        let hasher = |(k, _): &(K, IdxSize)| random_state.tot_hash_one(*k);
        match self.table.entry(hash, |(k, _)| k.tot_eq(&key), hasher) {
            Entry::Occupied(o) => o.get().1,
            Entry::Vacant(v) => {
                let group = self.num_groups;
                v.insert((key, group));
                self.num_groups += 1;
                group
            },
        }
    }

    fn push_keyless_group(&mut self) -> IdxSize {
        self.num_groups += 1;
        self.num_groups - 1
    }

    fn keys_in_group_order(&self) -> Vec<K> {
        let mut keys = vec![K::default(); self.num_groups as usize];
        for (key, group) in self.table.iter() {
            unsafe { *keys.get_unchecked_mut(*group as usize) = *key };
        }
        keys
    }

    fn take_keys_in_group_order(&mut self) -> Vec<K> {
        let keys = self.keys_in_group_order();
        *self = Self::new();
        keys
    }
}

/// The group of each key in the hash table, and the keys in group order.
pub struct DenseKeys<K> {
    table: HashTable<IdxSize>,
    keys: Vec<K>,
}

impl<K: GroupKey> KeyStore<K> for DenseKeys<K> {
    fn new() -> Self {
        Self {
            table: HashTable::new(),
            keys: Vec::new(),
        }
    }

    #[inline(always)]
    fn num_groups(&self) -> IdxSize {
        self.keys.len() as IdxSize
    }

    #[inline(always)]
    fn num_buckets(&self) -> usize {
        self.table.num_buckets()
    }

    #[inline(always)]
    fn reserve(&mut self, additional: usize, random_state: &PlRandomState) {
        let keys = &self.keys;
        self.table.reserve(additional, |g| unsafe {
            random_state.tot_hash_one(*keys.get_unchecked(*g as usize))
        });
        self.keys.reserve(additional);
    }

    #[inline(always)]
    fn prefetch_insert(&self, hash: u64) {
        self.table.prefetch_insert(hash);
    }

    #[inline(always)]
    fn prefetch_get(&self, hash: u64) {
        self.table.prefetch_get(hash);
    }

    #[inline(always)]
    fn get(&self, hash: u64, key: &K) -> Option<IdxSize> {
        let keys = &self.keys;
        self.table
            .find(hash, |g| {
                unsafe { keys.get_unchecked(*g as usize) }.tot_eq(key)
            })
            .copied()
    }

    #[inline(always)]
    fn get_or_insert(&mut self, hash: u64, key: K, random_state: &PlRandomState) -> IdxSize {
        let keys = &self.keys;
        let entry = self.table.entry(
            hash,
            |g| unsafe { keys.get_unchecked(*g as usize) }.tot_eq(&key),
            |g| random_state.tot_hash_one(*unsafe { keys.get_unchecked(*g as usize) }),
        );
        match entry {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let group = self.keys.len() as IdxSize;
                v.insert(group);
                self.keys.push(key);
                group
            },
        }
    }

    fn push_keyless_group(&mut self) -> IdxSize {
        self.keys.push(K::default());
        self.keys.len() as IdxSize - 1
    }

    fn keys_in_group_order(&self) -> Vec<K> {
        self.keys.clone()
    }

    fn take_keys_in_group_order(&mut self) -> Vec<K> {
        self.table = HashTable::new();
        std::mem::take(&mut self.keys)
    }
}

pub struct SingleKeyHashGrouper<T: PolarsDataType>
where
    for<'a> T::Physical<'a>: GroupKey,
{
    keys: <T::Physical<'static> as GroupKey>::Store,
    /// The group of the null key, IdxSize::MAX if there is none.
    null_idx: IdxSize,
    random_state: PlRandomState,
}

impl<K, T: PolarsDataType> SingleKeyHashGrouper<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: GroupKey,
{
    pub fn new() -> Self {
        Self {
            keys: K::Store::new(),
            null_idx: IdxSize::MAX,
            random_state: PlRandomState::default(),
        }
    }

    #[inline(always)]
    fn should_prefetch(&self) -> bool {
        self.keys.num_buckets() >= MIN_PREFETCH_BUCKETS
    }

    #[inline(always)]
    fn group_idx(&self, hash: u64, key: &K) -> Option<IdxSize> {
        self.keys.get(hash, key)
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
            self.keys.num_groups() as usize + subset.len() < IdxSize::MAX as usize,
            "{}",
            LENGTH_LIMIT_MSG
        );

        let validity = arr.validity().filter(|_| arr.has_nulls());
        let random_state = &self.random_state;
        let mut hashes = [0; BLOCK_SIZE];
        for block in subset.chunks(BLOCK_SIZE) {
            // Reserve up front so the table doesn't move after prefetching.
            self.keys.reserve(block.len(), random_state);
            let prefetch = self.should_prefetch();
            for (hash, idx) in hashes.iter_mut().zip(block) {
                *hash = random_state.tot_hash_one(arr.value_unchecked(*idx as usize));
                if prefetch {
                    self.keys.prefetch_insert(*hash);
                }
            }

            for (hash, idx) in hashes.iter().zip(block) {
                let idx = *idx as usize;
                if validity.is_some_and(|v| !v.get_bit_unchecked(idx)) {
                    if null_is_valid {
                        if self.null_idx == IdxSize::MAX {
                            self.null_idx = self.keys.push_keyless_group();
                        }
                        on_group(self.null_idx);
                    }
                } else {
                    let key = arr.value_unchecked(idx);
                    on_group(self.keys.get_or_insert(*hash, key, random_state));
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
                    p_grouper.keys.prefetch_get(*hash);
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
    K: GroupKey,
{
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.keys.reserve(additional, &self.random_state);
    }

    fn num_groups(&self) -> IdxSize {
        self.keys.num_groups()
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
        self.finalize_keys(schema, self.keys.keys_in_group_order())
    }

    fn take_keys_in_group_order(&mut self, schema: &Schema) -> DataFrame {
        let keys = self.keys.take_keys_in_group_order();
        let out = self.finalize_keys(schema, keys);
        self.null_idx = IdxSize::MAX;
        out
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
