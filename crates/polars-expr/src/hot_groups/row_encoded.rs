use arrow::array::{BinaryArray, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::RowEncodedKeys;
use crate::hot_groups::fixed_index_table::FixedIndexTable;

pub struct RowEncodedHashHotGrouper {
    key_schema: Arc<Schema>,
    table: FixedIndexTable<(u64, Vec<u8>)>,
    evicted_key_hashes: Vec<u64>,
    evicted_key_data: Vec<u8>,
    evicted_key_offsets: Offsets<i64>,
}

impl RowEncodedHashHotGrouper {
    pub fn new(key_schema: Arc<Schema>, max_groups: usize) -> Self {
        Self {
            key_schema,
            table: FixedIndexTable::new(max_groups.try_into().unwrap()),
            evicted_key_hashes: Vec::new(),
            evicted_key_data: Vec::new(),
            evicted_key_offsets: Offsets::new(),
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
                        |a, b| *a == b.1,
                        |k| (h, k.to_owned()),
                        |k, ev_k| {
                            self.evicted_key_hashes.push(ev_k.0);
                            self.evicted_key_offsets.try_push(ev_k.1.len()).unwrap();
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
            let keys = LargeBinaryArray::from_trusted_len_values_iter(
                self.table.keys().iter().map(|(h, k)| {
                    hashes.push_unchecked(*h);
                    k
                }),
            );
            let hashes = PrimitiveArray::from_vec(hashes);
            HashKeys::RowEncoded(RowEncodedKeys { hashes, keys })
        }
    }

    fn num_evictions(&self) -> usize {
        self.evicted_key_offsets.len_proxy()
    }

    fn take_evicted_keys(&mut self) -> HashKeys {
        let hashes = PrimitiveArray::from_vec(core::mem::take(&mut self.evicted_key_hashes));
        let values = Buffer::from(core::mem::take(&mut self.evicted_key_data));
        let offsets = OffsetsBuffer::from(core::mem::take(&mut self.evicted_key_offsets));
        let keys = BinaryArray::new(ArrowDataType::LargeBinary, offsets, values, None);
        HashKeys::RowEncoded(RowEncodedKeys { hashes, keys })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
