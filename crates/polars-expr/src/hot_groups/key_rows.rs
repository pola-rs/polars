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
            evicted: KeyRowCollector::default(),
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

        hot_idxs.reserve(keys.len());
        hot_group_idxs.reserve(keys.len());
        cold_idxs.reserve(keys.len());

        let hashes = keys.hashes.values().as_slice();
        let is_valid = |i: usize| unsafe {
            keys.validity
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(i))
        };
        let mut found = Vec::with_capacity(VERIFY_BATCH_SIZE);
        let mut cand_idxs = Vec::with_capacity(VERIFY_BATCH_SIZE);
        let mut cand_ks = Vec::with_capacity(VERIFY_BATCH_SIZE);
        let mut rows = Vec::with_capacity(VERIFY_BATCH_SIZE);
        let mut ok = Vec::with_capacity(VERIFY_BATCH_SIZE);
        let mut replaced_ks = Vec::new();
        for start in (0..keys.len()).step_by(VERIFY_BATCH_SIZE) {
            let end = keys.len().min(start + VERIFY_BATCH_SIZE);

            found.clear();
            cand_idxs.clear();
            cand_ks.clear();
            for (i, h) in (start..end).zip(&hashes[start..end]) {
                let h = *h;
                let f = is_valid(i)
                    .then(|| {
                        self.table
                            .find_key(h, |k| unsafe { self.keys.hash(*k) } == h)
                    })
                    .flatten();
                if let Some((_, k)) = f {
                    cand_idxs.push(i as IdxSize);
                    cand_ks.push(k);
                }
                found.push(f);
            }
            ok.clear();
            ok.resize(cand_idxs.len(), true);
            unsafe {
                self.keys
                    .verify(keys, &cand_idxs, &cand_ks, &mut rows, &mut ok)
            };

            let mut c = 0;
            let hot: *mut HotKeyRows = &mut self.keys;
            let evicted = &mut self.evicted;
            let replaced = &mut self.replaced;
            for (i, f) in (start..end).zip(&found) {
                if !is_valid(i) {
                    continue;
                }
                let h = hashes[i];
                unsafe {
                    if let Some((slot, k)) = *f {
                        c += 1;
                        if ok[c - 1] && !replaced[k as usize] {
                            self.table.touch(slot, h);
                            hot_idxs.push_unchecked(i as IdxSize);
                            hot_group_idxs.push_unchecked(EvictIdx::new(k, false));
                            continue;
                        }
                    }

                    let opt_g = self.table.insert_key(
                        h,
                        i,
                        force_hot,
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
        HashKeys::KeyRows(self.evicted.take(self.layout.clone()))
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
