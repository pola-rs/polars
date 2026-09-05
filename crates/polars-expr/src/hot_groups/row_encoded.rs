use polars_array::{PlBinaryArray, PlPrimitiveArray};
use polars_buffer::Buffer;
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::RowEncodedKeys;
use crate::hot_groups::fixed_index_table::FixedIndexTable;

pub struct RowEncodedHashHotGrouper {
    key_schema: Arc<Schema>,
    table: FixedIndexTable<(u64, Vec<u8>)>,
    evicted_key_hashes: Vec<u64>,
    evicted_key_data: Vec<u8>,
    // The end of each evicted key in `evicted_key_data`, preceded by a leading zero —
    // the offsets a `PlBinaryArray` is built from.
    evicted_key_offsets: Vec<u64>,
}

impl RowEncodedHashHotGrouper {
    pub fn new(key_schema: Arc<Schema>, max_groups: usize) -> Self {
        Self {
            key_schema,
            table: FixedIndexTable::new(max_groups.try_into().unwrap()),
            evicted_key_hashes: Vec::new(),
            evicted_key_data: Vec::new(),
            evicted_key_offsets: vec![0],
        }
    }
}

impl HotGrouper for RowEncodedHashHotGrouper {
    fn new_empty(&self, max_groups: usize) -> Box<dyn HotGrouper> {
        Box::new(Self::new(self.key_schema.clone(), max_groups))
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
        let HashKeys::RowEncoded(keys) = keys else {
            unreachable!()
        };

        hot_idxs.reserve(keys.hashes.len());
        hot_group_idxs.reserve(keys.hashes.len());
        cold_idxs.reserve(keys.hashes.len());

        unsafe {
            keys.for_each_hash(|idx, opt_h| {
                if let Some(h) = opt_h {
                    let key = keys.keys.value_unchecked(idx as usize);
                    let opt_g = self.table.insert_key(
                        h,
                        key,
                        force_hot,
                        |a, b| *a == b.1,
                        |k| (h, k.to_owned()),
                        |k, ev_k| {
                            self.evicted_key_hashes.push(ev_k.0);
                            let end =
                                self.evicted_key_offsets.last().unwrap() + ev_k.1.len() as u64;
                            self.evicted_key_offsets.push(end);
                            self.evicted_key_data.extend_from_slice(&ev_k.1);
                            ev_k.0 = h;
                            ev_k.1.clear();
                            ev_k.1.extend_from_slice(k);
                        },
                    );
                    if let Some(g) = opt_g {
                        hot_idxs.push_unchecked(idx as IdxSize);
                        hot_group_idxs.push_unchecked(g);
                    } else {
                        cold_idxs.push_unchecked(idx as IdxSize);
                    }
                }
            });
        }
    }

    fn keys(&self) -> HashKeys {
        unsafe {
            let mut hashes = Vec::with_capacity(self.table.len());
            let keys = PlBinaryArray::from_values_iter(self.table.keys().iter().map(|(h, k)| {
                hashes.push_unchecked(*h);
                k
            }));
            let hashes = PlPrimitiveArray::from_vec(hashes);
            HashKeys::RowEncoded(RowEncodedKeys { hashes, keys })
        }
    }

    fn num_evictions(&self) -> usize {
        self.evicted_key_offsets.len() - 1
    }

    fn take_evicted_keys(&mut self) -> HashKeys {
        let hashes = PlPrimitiveArray::from_vec(core::mem::take(&mut self.evicted_key_hashes));
        let values = Buffer::from(core::mem::take(&mut self.evicted_key_data));
        // The offsets are drained too, so what is left behind is the empty run they started as.
        let offsets = Buffer::from(core::mem::replace(&mut self.evicted_key_offsets, vec![0]));
        let keys = PlBinaryArray::from_offsets(values, offsets);
        HashKeys::RowEncoded(RowEncodedKeys { hashes, keys })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_encoded(entries: &[(u64, &[u8])]) -> HashKeys {
        HashKeys::RowEncoded(RowEncodedKeys {
            hashes: PlPrimitiveArray::from_vec(entries.iter().map(|(h, _)| *h).collect()),
            keys: PlBinaryArray::from_values_iter(entries.iter().map(|(_, k)| *k)),
        })
    }

    fn evict(grouper: &mut RowEncodedHashHotGrouper, entries: &[(u64, &[u8])]) -> Vec<Vec<u8>> {
        let (mut hot, mut hot_groups, mut cold) = (Vec::new(), Vec::new(), Vec::new());
        grouper.insert_keys(
            &row_encoded(entries),
            &mut hot,
            &mut hot_groups,
            &mut cold,
            true,
        );

        let HashKeys::RowEncoded(evicted) = grouper.take_evicted_keys() else {
            unreachable!("a row-encoded grouper evicts row-encoded keys")
        };
        assert_eq!(evicted.hashes.len(), evicted.keys.len());
        evicted
            .keys
            .iter()
            .map(|k| k.expect("an evicted key is never null").to_vec())
            .collect()
    }

    /// The evicted keys are accumulated into one buffer behind a run of offsets, so a key that
    /// comes back whole is the only evidence that those offsets line up.
    #[test]
    fn evicted_keys_come_back_whole_and_draining_resets_the_offsets() {
        // Far fewer slots than keys, so the table has to evict.
        let mut grouper = RowEncodedHashHotGrouper::new(Arc::new(Schema::default()), 2);

        // Keys of differing lengths: a wrong offset would slice them into each other.
        let sent: &[(u64, &[u8])] = &[
            (1, b"aa"),
            (2, b"bbb"),
            (3, b"c"),
            (4, b"dddddd"),
            (5, b"ee"),
            (6, b"fff"),
        ];
        let evicted = evict(&mut grouper, sent);
        assert!(!evicted.is_empty(), "two hot slots cannot hold six keys");
        for key in &evicted {
            assert!(
                sent.iter().any(|(_, k)| *k == key.as_slice()),
                "{key:?} is not one of the keys that went in",
            );
        }

        // Draining took the offsets with it, so nothing is left over to be counted twice.
        assert_eq!(grouper.num_evictions(), 0);

        // A second round starts from the empty run the drain left behind.
        let more: &[(u64, &[u8])] = &[(7, b"ggggggg"), (8, b"h"), (9, b"ii"), (10, b"jjjj")];
        let again = evict(&mut grouper, more);
        for key in &again {
            assert!(
                sent.iter().chain(more).any(|(_, k)| *k == key.as_slice()),
                "{key:?} is not one of the keys that went in",
            );
        }
        assert_eq!(grouper.num_evictions(), 0);
    }
}
