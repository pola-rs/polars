use arrow::array::View;
use arrow::bitmap::MutableBitmap;
use polars_array::bitmap::PlBitmap;
use polars_array::builder::StaticArrayBuilder;
use polars_array::{ArrayRepr, PlBinaryViewArrayBuilder, PlPrimitiveArray};
use polars_buffer::Buffer;
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::BinviewKeys;
use crate::hot_groups::fixed_index_table::FixedIndexTable;

pub struct BinviewHashHotGrouper {
    // The views in this table when not inline are stored in the vec.
    table: FixedIndexTable<(u64, View, Vec<u8>)>,
    evicted_key_hashes: Vec<u64>,
    evicted_keys: PlBinaryViewArrayBuilder,
    null_idx: IdxSize,
}

impl BinviewHashHotGrouper {
    pub fn new(max_groups: usize) -> Self {
        Self {
            table: FixedIndexTable::new(max_groups.try_into().unwrap()),
            evicted_key_hashes: Vec::new(),
            evicted_keys: PlBinaryViewArrayBuilder::new(),
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
        force_hot: bool,
        buffers: &Buffer<Buffer<u8>>,
    ) -> Option<EvictIdx> {
        unsafe {
            let mut evict = |ev_h: &u64, ev_view: &View, ev_buffer: &Vec<u8>| {
                self.evicted_key_hashes.push(*ev_h);
                let bytes = ev_view
                    .get_inlined_slice()
                    .unwrap_or_else(|| ev_buffer.as_slice());
                self.evicted_keys.push_value(bytes);
            };
            if view.is_inline() {
                self.table.insert_key(
                    hash,
                    (),
                    force_hot,
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
                    force_hot,
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
        force_hot: bool,
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
            // A scalar chunk repeats one view over every element, so the view is read out of
            // the representation rather than out of a buffer that may hold only one slot.
            let views = hash_keys.keys.views_repr();
            let view_at = |idx: usize| match views {
                ArrayRepr::Scalar(view) => view,
                ArrayRepr::Flat(views) => unsafe { *views.get_unchecked(idx) },
            };
            let buffers = hash_keys.keys.data_buffers();
            if hash_keys.null_is_valid {
                hash_keys.for_each_hash(|idx, opt_h| {
                    if let Some(h) = opt_h {
                        push_g(
                            idx as usize,
                            self.insert_key(h, view_at(idx as usize), force_hot, buffers),
                        );
                    } else {
                        push_g(idx as usize, self.insert_null());
                    }
                });
            } else {
                hash_keys.for_each_hash(|idx, opt_h| {
                    if let Some(h) = opt_h {
                        push_g(
                            idx as usize,
                            self.insert_key(h, view_at(idx as usize), force_hot, buffers),
                        );
                    }
                });
            }
        }
    }

    fn keys(&self) -> HashKeys {
        unsafe {
            let mut hashes = Vec::with_capacity(self.table.len());
            let mut keys_builder = PlBinaryViewArrayBuilder::with_capacity(self.table.len());
            for (h, view, buf) in self.table.keys() {
                hashes.push_unchecked(*h);
                let bytes = view.get_inlined_slice().unwrap_or_else(|| buf.as_slice());
                keys_builder.push_value(bytes);
            }

            let hashes = PlPrimitiveArray::from_vec(hashes);
            let mut keys = keys_builder.freeze();
            let null_is_valid = self.null_idx < IdxSize::MAX;
            if null_is_valid {
                let mut validity = MutableBitmap::new();
                validity.extend_constant(keys.len(), true);
                validity.set(self.null_idx as usize, false);
                keys = keys.with_validity(Some(PlBitmap::from_bitmap(validity.freeze())));
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
            hashes: PlPrimitiveArray::from_vec(hashes),
            keys,
            null_is_valid: false,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::{Column, DataFrame, PlRandomState};

    use super::*;
    use crate::hash_keys::HashKeys;

    /// Evicted keys are rebuilt through the builder, and an inline view carries its own bytes
    /// while a long one points into a buffer — so both lengths have to come back intact.
    #[test]
    fn evicted_keys_come_back_whole_whether_inlined_or_not() {
        // Well over the 12-byte inline limit, and well under it.
        let sent = [
            "a",
            "a considerably longer key than fits inline",
            "bb",
            "another key that is far too long to inline",
            "ccc",
            "yet another long one that must live in a buffer",
        ];
        let values = Column::new("k".into(), sent);
        let df = DataFrame::new(values.len(), vec![values]).unwrap();
        let keys = HashKeys::from_df(&df, PlRandomState::default(), false, false);

        // Fewer slots than keys, so the table has to evict.
        let mut grouper = BinviewHashHotGrouper::new(2);
        let (mut hot, mut hot_groups, mut cold) = (Vec::new(), Vec::new(), Vec::new());
        grouper.insert_keys(&keys, &mut hot, &mut hot_groups, &mut cold, true);

        let HashKeys::Binview(evicted) = grouper.take_evicted_keys() else {
            unreachable!("a binview grouper evicts binview keys")
        };
        assert!(!evicted.keys.is_empty(), "two slots cannot hold six keys");
        assert_eq!(evicted.hashes.len(), evicted.keys.len());

        for key in evicted.keys.iter() {
            let key = std::str::from_utf8(key.expect("an evicted key is never null")).unwrap();
            assert!(
                sent.contains(&key),
                "{key:?} is not one of the keys sent in"
            );
        }

        // Draining took the keys with it.
        assert_eq!(grouper.num_evictions(), 0);
    }
}
