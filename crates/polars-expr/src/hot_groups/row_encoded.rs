use arrow::array::builder::StaticArrayBuilder;
use arrow::array::PrimitiveArray;
use arrow::offset::Offsets;
use polars_utils::vec::PushUnchecked;

use crate::hash_keys::RowEncodedKeys;
use crate::hot_groups::fixed_index_table::FixedIndexTable;

use super::*;

pub struct RowEncodedHashHotGrouper {
    key_schema: Arc<Schema>,
    table: FixedIndexTable<Vec<u8>>,
    evicted_key_data: Vec<u8>,
    evicted_key_offsets: Offsets<i64>,
}

impl RowEncodedHashHotGrouper {
    pub fn new(key_schema: Arc<Schema>, max_groups: usize) -> Self {
        Self {
            key_schema,
            table: FixedIndexTable::new(max_groups.try_into().unwrap()),
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
            let mut idx = 0;
            keys.for_each_hash(|opt_h| {
                if let Some(h) = opt_h {
                    let key = keys.keys.value_unchecked(idx);
                    let opt_g = self.table.insert_key(h, key, |evicted| {
                        self.evicted_key_offsets.try_push(evicted.len()).unwrap();
                        self.evicted_key_data.extend_from_slice(evicted);
                    });
                    if let Some(g) = opt_g {
                        hot_idxs.push_unchecked(idx as IdxSize);
                        hot_group_idxs.push_unchecked(g);
                    } else {
                        cold_idxs.push_unchecked(idx as IdxSize);
                    }
                }
                
                idx += 1;
            });
        }
    }

    fn keys(&self) -> HashKeys {
        let hashes = PrimitiveArray::from_slice(self.table.hashes());
        let keys = LargeBinaryArray::from_slice(self.table.keys());
        HashKeys::RowEncoded(RowEncodedKeys {
            hashes,
            keys,
        })
    }
    
    fn num_evictions(&self) -> usize {
        self.evicted_key_offsets.len()
    }
    
    fn take_evicted_keys(&mut self) -> HashKeys {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
