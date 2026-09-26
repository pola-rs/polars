use std::borrow::Cow;

use super::*;
use crate::hash_keys::HashKeys;
use crate::key_rows::{KeyRowIndexMap, KeyRowKeys};

#[derive(Default)]
pub struct KeyRowHashGrouper {
    idx_map: KeyRowIndexMap<()>,
}

impl KeyRowHashGrouper {
    pub fn new() -> Self {
        Self::default()
    }

    /// # Safety
    /// All groupers must be a KeyRowHashGrouper, and `i` in-bounds.
    #[inline(always)]
    unsafe fn partition_group_idx(
        groupers: &[Box<dyn Grouper>],
        keys: &KeyRowKeys,
        partitioner: &HashPartitioner,
        i: usize,
    ) -> (usize, Option<IdxSize>) {
        unsafe {
            let p = partitioner.hash_to_partition(keys.hashes.value_unchecked(i));
            let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
            let grouper = &*(dyn_grouper as *const dyn Grouper as *const KeyRowHashGrouper);
            (p, grouper.idx_map.get_index_of(keys, i))
        }
    }
}

impl Grouper for KeyRowHashGrouper {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new())
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
    /// All groupers must be a KeyRowHashGrouper.
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
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            keys.for_each_hash(|idx, opt_hash| {
                let has_group = opt_hash.is_some()
                    && Self::partition_group_idx(groupers, keys, partitioner, idx as usize)
                        .1
                        .is_some();
                if has_group != invert {
                    probe_matches.push(idx);
                }
            });
        }
    }

    /// # Safety
    /// All groupers must be a KeyRowHashGrouper.
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
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            keys.for_each_hash(|idx, opt_hash| {
                let has_group = opt_hash.is_some()
                    && Self::partition_group_idx(groupers, keys, partitioner, idx as usize)
                        .1
                        .is_some();
                contains_key.push(has_group != invert);
            });
        }
    }

    /// # Safety
    /// All groupers must be a KeyRowHashGrouper.
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
        assert!(partitioner.num_partitions() == groupers.len());
        assert!(marks.len() == groupers.len());

        unsafe {
            for idx in (0..keys.len()).filter(|i| {
                keys.validity
                    .as_ref()
                    .is_none_or(|v| v.get_bit_unchecked(*i))
            }) {
                if let (p, Some(group_idx)) =
                    Self::partition_group_idx(groupers, keys, partitioner, idx)
                {
                    marks
                        .get_unchecked_mut(p)
                        .set_unchecked(group_idx as usize, true);
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
