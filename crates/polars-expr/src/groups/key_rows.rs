use std::borrow::Cow;

use super::*;
use crate::hash_keys::{BLOCK_SIZE, HashKeys};
use crate::key_rows::{KeyRowIndexMap, KeyRowKeys, KeyRowLayout};

pub struct KeyRowHashGrouper {
    idx_map: KeyRowIndexMap<()>,
}

impl KeyRowHashGrouper {
    pub fn new(layout: Arc<KeyRowLayout>) -> Self {
        Self {
            idx_map: KeyRowIndexMap::new(layout),
        }
    }

    /// In debug builds, panics unless `keys` have the layout of every grouper.
    ///
    /// # Safety
    /// All groupers must be a KeyRowHashGrouper.
    unsafe fn debug_assert_layouts(groupers: &[Box<dyn Grouper>], keys: &KeyRowKeys) {
        if cfg!(debug_assertions) {
            for g in groupers {
                let grouper = unsafe { &*(&**g as *const dyn Grouper as *const KeyRowHashGrouper) };
                keys.assert_layout(grouper.idx_map.layout());
            }
        }
    }

    /// Calls `f` for each key with its partition and group, or `None` if the key
    /// is null or not found. Works in blocks, first prefetching the lookups of a
    /// whole block.
    ///
    /// # Safety
    /// All groupers must be a KeyRowHashGrouper for the key schema of `keys`.
    #[inline(always)]
    unsafe fn probe_partitions(
        groupers: &[Box<dyn Grouper>],
        keys: &KeyRowKeys,
        partitioner: &HashPartitioner,
        mut f: impl FnMut(IdxSize, Option<(usize, IdxSize)>),
    ) {
        unsafe {
            assert!(partitioner.num_partitions() == groupers.len());
            Self::debug_assert_layouts(groupers, keys);
            let grouper =
                |p: usize| &*(&**groupers.get_unchecked(p) as *const dyn Grouper as *const Self);
            let hashes = keys.hashes.values();
            let mut partitions = [0; BLOCK_SIZE];
            for block_start in (0..keys.len()).step_by(BLOCK_SIZE) {
                let block = block_start..keys.len().min(block_start + BLOCK_SIZE);
                for (p, idx) in partitions.iter_mut().zip(block.clone()) {
                    let hash = *hashes.get_unchecked(idx);
                    *p = partitioner.hash_to_partition(hash);
                    let idx_map = &grouper(*p).idx_map;
                    if idx_map.should_prefetch() {
                        idx_map.prefetch(hash);
                    }
                }

                for (p, idx) in partitions.iter().zip(block) {
                    let is_valid = keys
                        .validity
                        .as_ref()
                        .is_none_or(|v| v.get_bit_unchecked(idx));
                    let found = if is_valid {
                        grouper(*p).idx_map.get_index_of(keys, idx).map(|g| (*p, g))
                    } else {
                        None
                    };
                    f(idx as IdxSize, found);
                }
            }
        }
    }
}

impl Grouper for KeyRowHashGrouper {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new(self.idx_map.layout().clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    #[inline]
    fn num_groups(&self) -> IdxSize {
        self.idx_map.len()
    }

    unsafe fn insert_keys_subset(
        &mut self,
        keys: &HashKeys,
        subset: &[IdxSize],
        group_idxs: Option<&mut Vec<IdxSize>>,
    ) {
        let HashKeys::KeyRows(keys) = keys else {
            unreachable!()
        };

        let valid_subset: Cow<'_, [IdxSize]> = match &keys.validity {
            None => Cow::Borrowed(subset),
            Some(v) => Cow::Owned(
                subset
                    .iter()
                    .copied()
                    .filter(|i| unsafe { v.get_bit_unchecked(*i as usize) })
                    .collect(),
            ),
        };
        let mut scratch = Vec::new();
        let group_idxs = group_idxs.unwrap_or(&mut scratch);
        group_idxs.reserve(valid_subset.len());
        unsafe {
            self.idx_map
                .get_or_insert_batch(keys, &valid_subset, |_| (), group_idxs)
        };
    }

    fn get_keys_in_group_order(&self, schema: &Schema) -> DataFrame {
        self.idx_map.keys_frame(schema)
    }

    /// # Safety
    /// All groupers must be a KeyRowHashGrouper for the key schema of `keys`.
    unsafe fn probe_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        probe_matches: &mut Vec<IdxSize>,
    ) {
        let HashKeys::KeyRows(keys) = keys else {
            unreachable!()
        };
        unsafe {
            Self::probe_partitions(groupers, keys, partitioner, |idx, found| {
                if found.is_some() != invert {
                    probe_matches.push(idx);
                }
            });
        }
    }

    /// # Safety
    /// All groupers must be a KeyRowHashGrouper for the key schema of `keys`.
    unsafe fn contains_key_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        contains_key: &mut BitmapBuilder,
    ) {
        let HashKeys::KeyRows(keys) = keys else {
            unreachable!()
        };
        unsafe {
            Self::probe_partitions(groupers, keys, partitioner, |_, found| {
                contains_key.push(found.is_some() != invert);
            });
        }
    }

    /// # Safety
    /// All groupers must be a KeyRowHashGrouper for the key schema of `keys`.
    unsafe fn mark_groups_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        marks: &mut [MutableBitmap],
    ) {
        let HashKeys::KeyRows(keys) = keys else {
            unreachable!()
        };
        assert!(marks.len() == groupers.len());
        unsafe {
            Self::probe_partitions(groupers, keys, partitioner, |_, found| {
                if let Some((p, group_idx)) = found {
                    marks
                        .get_unchecked_mut(p)
                        .set_unchecked(group_idx as usize, true);
                }
            });
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
