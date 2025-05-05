use arrow::array::builder::StaticArrayBuilder;
use arrow::array::{BinaryViewArrayGenericBuilder, PrimitiveArray, View};
use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::BinviewKeys;
use crate::hot_groups::fixed_index_table::FixedIndexTable;

pub struct BinviewHashHotGrouper {
    // The views in this table when not inline are stored in the vec.
    table: FixedIndexTable<(u64, View, Vec<u8>)>,
    evicted_key_hashes: Vec<u64>,
    evicted_keys: BinaryViewArrayGenericBuilder<[u8]>,
    null_idx: IdxSize,
}

impl BinviewHashHotGrouper {
    pub fn new(max_groups: usize) -> Self {
        Self {
            table: FixedIndexTable::new(max_groups.try_into().unwrap()),
            evicted_key_hashes: Vec::new(),
            evicted_keys: BinaryViewArrayGenericBuilder::new(ArrowDataType::BinaryView),
            null_idx: IdxSize::MAX,
        }
    }

    /// # Safety
    /// The view must be valid for the given buffer set.
    #[inline(always)]
    unsafe fn insert_key(
        &mut self,
        hash: u64,
        view: View,
        buffers: &Arc<[Buffer<u8>]>,
    ) -> Option<EvictIdx> {
        unsafe {
            let mut evict = |ev_h: &u64, ev_view: &View, ev_buffer: &Vec<u8>| {
                self.evicted_key_hashes.push(*ev_h);
                if ev_view.is_inline() {
                    self.evicted_keys.push_inline_view_ignore_validity(*ev_view);
                } else {
                    self.evicted_keys
                        .push_value_ignore_validity(ev_buffer.as_slice());
                }
            };
            if view.is_inline() {
                self.table.insert_key(
                    hash,
                    (),
                    |_, b| view == b.1,
                    |_| (hash, view, Vec::new()),
                    |_, ev_k| {
                        let (ev_h, ev_view, ev_buffer) = ev_k;
                        evict(ev_h, ev_view, ev_buffer);
                        *ev_h = hash;
                        *ev_view = view;
                        ev_buffer.clear();
                    },
                )
            } else {
                let bytes = view.get_external_slice_unchecked(buffers);
                self.table.insert_key(
                    hash,
                    (),
                    |_, b| {
                        // We only reach here if the hash matched, so jump straight to full comparison.
                        bytes == b.2
                    },
                    |_| (hash, view, bytes.to_vec()),
                    |_, ev_k| {
                        let (ev_h, ev_view, ev_buffer) = ev_k;
                        evict(ev_h, ev_view, ev_buffer);
                        *ev_h = hash;
                        *ev_view = view;
                        ev_buffer.clear();
                        ev_buffer.extend_from_slice(bytes);
                    },
                )
            }
        }
    }

    #[inline(always)]
    fn insert_null(&mut self) -> Option<EvictIdx> {
        if self.null_idx == IdxSize::MAX {
            self.null_idx = self
                .table
                .push_unmapped_key((0, View::default(), Vec::new()));
        }
        Some(EvictIdx::new(self.null_idx, false))
    }
}

impl HotGrouper for BinviewHashHotGrouper {
    fn new_empty(&self, max_groups: usize) -> Box<dyn HotGrouper> {
        Box::new(Self::new(max_groups))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn insert_keys(
        &mut self,
        hash_keys: &HashKeys,
        hot_idxs: &mut Vec<IdxSize>,
        hot_group_idxs: &mut Vec<EvictIdx>,
        cold_idxs: &mut Vec<IdxSize>,
    ) {
        let HashKeys::Binview(hash_keys) = hash_keys else {
            unreachable!()
        };

        hot_idxs.reserve(hash_keys.keys.len());
        hot_group_idxs.reserve(hash_keys.keys.len());
        cold_idxs.reserve(hash_keys.keys.len());

        let mut push_g = |idx: usize, opt_g: Option<EvictIdx>| unsafe {
            if let Some(g) = opt_g {
                hot_idxs.push_unchecked(idx as IdxSize);
                hot_group_idxs.push_unchecked(g);
            } else {
                cold_idxs.push_unchecked(idx as IdxSize);
            }
        };

        unsafe {
            let views = hash_keys.keys.views().as_slice();
            let buffers = hash_keys.keys.data_buffers();
            if hash_keys.null_is_valid {
                hash_keys.for_each_hash(|idx, opt_h| {
                    if let Some(h) = opt_h {
                        let view = views.get_unchecked(idx as usize);
                        push_g(idx as usize, self.insert_key(h, *view, buffers));
                    } else {
                        push_g(idx as usize, self.insert_null());
                    }
                });
            } else {
                hash_keys.for_each_hash(|idx, opt_h| {
                    if let Some(h) = opt_h {
                        let view = views.get_unchecked(idx as usize);
                        push_g(idx as usize, self.insert_key(h, *view, buffers));
                    }
                });
            }
        }
    }

    fn keys(&self) -> HashKeys {
        unsafe {
            let mut hashes = Vec::with_capacity(self.table.len());
            let mut keys_builder = BinaryViewArrayGenericBuilder::new(ArrowDataType::BinaryView);
            keys_builder.reserve(self.table.len());
            for (h, view, buf) in self.table.keys() {
                hashes.push_unchecked(*h);
                if view.is_inline() {
                    keys_builder.push_inline_view_ignore_validity(*view);
                } else {
                    keys_builder.push_value_ignore_validity(buf.as_slice());
                }
            }

            let hashes = PrimitiveArray::from_vec(hashes);
            let mut keys = keys_builder.freeze();
            let null_is_valid = self.null_idx < IdxSize::MAX;
            if null_is_valid {
                let mut validity = MutableBitmap::new();
                validity.extend_constant(keys.len(), true);
                validity.set(self.null_idx as usize, false);
                keys = keys.with_validity_typed(Some(validity.freeze()));
            }
            HashKeys::Binview(BinviewKeys {
                hashes,
                keys,
                null_is_valid,
            })
        }
    }

    fn num_evictions(&self) -> usize {
        self.evicted_keys.len()
    }

    fn take_evicted_keys(&mut self) -> HashKeys {
        let hashes = core::mem::take(&mut self.evicted_key_hashes);
        let keys = self.evicted_keys.freeze_reset();
        HashKeys::Binview(BinviewKeys {
            hashes: PrimitiveArray::from_vec(hashes),
            keys,
            null_is_valid: false,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
