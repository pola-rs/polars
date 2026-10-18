use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hot_groups::fixed_index_table::FixedIndexTable;
use crate::key_rows::{HotKeyRows, KeyRowCollector, KeyRowLayout, VERIFY_BATCH_SIZE};

pub struct KeyRowHashHotGrouper {
    layout: Arc<KeyRowLayout>,
    table: FixedIndexTable<IdxSize>,
    keys: HotKeyRows,
    evicted: KeyRowCollector,
    /// Per hot key whether it was replaced in the current batch.
    replaced: Vec<bool>,
}

impl KeyRowHashHotGrouper {
    pub fn new(layout: Arc<KeyRowLayout>, max_groups: usize) -> Self {
        let table = FixedIndexTable::new(max_groups.try_into().unwrap());
        Self {
            replaced: vec![false; table.num_slots()],
            table,
            keys: HotKeyRows::new(layout.clone()),
            evicted: KeyRowCollector::new(layout.clone()),
            layout,
        }
    }
}

impl HotGrouper for KeyRowHashHotGrouper {
    fn new_empty(&self, max_groups: usize) -> Box<dyn HotGrouper> {
        Box::new(Self::new(self.layout.clone(), max_groups))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn num_slots(&self) -> usize {
        self.table.num_slots()
    }

    fn double(&mut self) {
        self.table.double();
        self.replaced.resize(self.table.num_slots(), false);
    }

    fn insert_keys(
        &mut self,
        keys: &HashKeys,
        hot_idxs: &mut Vec<IdxSize>,
        hot_group_idxs: &mut Vec<EvictIdx>,
        cold_idxs: &mut Vec<IdxSize>,
        force_hot: bool,
    ) {
        let HashKeys::KeyRows(keys) = keys else {
            unreachable!()
        };
        keys.assert_layout(&self.layout);

        hot_idxs.reserve(keys.len());
        hot_group_idxs.reserve(keys.len());
        cold_idxs.reserve(keys.len());

        let hashes = keys.hashes.values().as_slice();
        let is_valid = |i: usize| unsafe {
            keys.validity
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(i))
        };
        let mut slots = [0 as IdxSize; VERIFY_BATCH_SIZE];
        let mut hot_keys = [0 as IdxSize; VERIFY_BATCH_SIZE];
        let mut ok = [false; VERIFY_BATCH_SIZE];
        let mut rows = Vec::with_capacity(VERIFY_BATCH_SIZE);
        let mut replaced_ks = Vec::new();
        for start in (0..keys.len()).step_by(VERIFY_BATCH_SIZE) {
            let n = VERIFY_BATCH_SIZE.min(keys.len() - start);
            let batch_hashes = &hashes[start..start + n];
            let (slots, hot_keys, ok) = (&mut slots[..n], &mut hot_keys[..n], &mut ok[..n]);

            // Find the candidate hot key of each key by its hash, hot key 0 standing in
            // for keys without one.
            let num_keys = self.table.len() as IdxSize;
            let mut num_found = 0;
            for (((h, slot), k), ok) in batch_hashes
                .iter()
                .zip(slots.iter_mut())
                .zip(hot_keys.iter_mut())
                .zip(ok.iter_mut())
            {
                let (found_slot, found_k) = self.table.find_hash(*h);
                let found = found_k < num_keys;
                *slot = found_slot as IdxSize;
                *k = if found { found_k } else { 0 };
                *ok = found;
                num_found += found as usize;
            }
            if let Some(validity) = &keys.validity {
                for (r, ok) in ok.iter_mut().enumerate() {
                    *ok &= unsafe { validity.get_bit_unchecked(start + r) };
                }
            }
            if num_found > 0 {
                unsafe { self.keys.verify(keys, start, hot_keys, &mut rows, ok) };
            }

            if num_found == n && ok.iter().all(|ok| *ok) {
                for (slot, h) in slots.iter().zip(batch_hashes) {
                    unsafe { self.table.touch(*slot as usize, *h) };
                }
                hot_idxs.extend(start as IdxSize..(start + n) as IdxSize);
                hot_group_idxs.extend(hot_keys.iter().map(|k| EvictIdx::new(*k, false)));
                continue;
            }

            // SAFETY: `insert_key` calls at most one of its closures at a time and none of
            // them outlives that call, so the accesses through `hot` never overlap.
            // `self.table` is a separate field.
            let hot: *mut HotKeyRows = &mut self.keys;
            let evicted = &mut self.evicted;
            let replaced = &mut self.replaced;
            for (r, h) in batch_hashes.iter().enumerate() {
                let i = start + r;
                let h = *h;
                unsafe {
                    if ok[r] {
                        let k = hot_keys[r];
                        if replaced_ks.is_empty() || !replaced[k as usize] {
                            self.table.touch(slots[r] as usize, h);
                            hot_idxs.push_unchecked(i as IdxSize);
                            hot_group_idxs.push_unchecked(EvictIdx::new(k, false));
                            continue;
                        }
                    } else if !is_valid(i) {
                        continue;
                    }

                    let opt_g = self.table.insert_key(
                        h,
                        i,
                        force_hot,
                        *hashes.get(i + 1).unwrap_or(&u64::MAX),
                        |i, k| (*hot).eq_key(*k, keys, *i),
                        |i| (*hot).push(keys, i),
                        |i, k| {
                            (*hot).collect(*k, evicted);
                            (*hot).replace(*k, keys, i);
                            replaced[*k as usize] = true;
                            replaced_ks.push(*k);
                        },
                    );
                    if let Some(g) = opt_g {
                        hot_idxs.push_unchecked(i as IdxSize);
                        hot_group_idxs.push_unchecked(g);
                    } else {
                        cold_idxs.push_unchecked(i as IdxSize);
                    }
                }
            }
            for k in replaced_ks.drain(..) {
                self.replaced[k as usize] = false;
            }
        }
    }

    fn keys(&self) -> HashKeys {
        HashKeys::KeyRows(self.keys.keys())
    }

    fn num_evictions(&self) -> usize {
        self.evicted.len()
    }

    fn take_evicted_keys(&mut self) -> HashKeys {
        HashKeys::KeyRows(self.evicted.take())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::PrimitiveArray;

    use super::*;
    use crate::key_rows::KeyRowKeys;

    // Maps to two different slots of a table of 1024 slots.
    const HASH: u64 = 0x123456789abcdef0;

    fn keys(a: &[i64], layout: &Arc<KeyRowLayout>) -> HashKeys {
        let df = df!("a" => a, "b" => a).unwrap();
        let mut keys = KeyRowKeys::from_columns(
            df.columns(),
            layout.clone(),
            &PlRandomState::default(),
            true,
        );
        keys.hashes = PrimitiveArray::from_vec(vec![HASH; a.len()]);
        HashKeys::KeyRows(keys)
    }

    #[test]
    fn colliding_hashes_get_their_own_groups() {
        let layout = Arc::new(KeyRowLayout::new(&[DataType::Int64, DataType::Int64]).unwrap());
        let mut grouper = KeyRowHashHotGrouper::new(layout.clone(), 1024);
        let (mut hot, mut groups, mut cold) = (Vec::new(), Vec::new(), Vec::new());
        grouper.insert_keys(
            &keys(&[1], &layout),
            &mut hot,
            &mut groups,
            &mut cold,
            false,
        );
        hot.clear();
        groups.clear();
        grouper.insert_keys(
            &keys(&[2, 1, 2], &layout),
            &mut hot,
            &mut groups,
            &mut cold,
            false,
        );
        assert_eq!(hot, [0, 1, 2]);
        assert!(cold.is_empty());
        assert!(groups.iter().all(|g| !g.should_evict()));
        assert_eq!(
            groups.iter().map(|g| g.idx()).collect::<Vec<_>>(),
            [1, 0, 1]
        );
    }
}
