use arrow::array::{Array, BinaryViewArrayGeneric, View, ViewType};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use polars_compute::binview_index_map::{BinaryViewIndexMap, Entry};

use super::*;
use crate::hash_keys::HashKeys;

#[derive(Default)]
pub struct BinviewHashGrouper {
    idx_map: BinaryViewIndexMap<()>,
    null_idx: IdxSize,
}

impl BinviewHashGrouper {
    pub fn new() -> Self {
        Self {
            idx_map: BinaryViewIndexMap::default(),
            null_idx: IdxSize::MAX,
        }
    }

    /// # Safety
    /// The view must be valid for the given buffer set.
    #[inline(always)]
    unsafe fn insert_key(&mut self, hash: u64, view: View, buffers: &Arc<[Buffer<u8>]>) -> IdxSize {
        unsafe {
            match self.idx_map.entry_view(hash, view, buffers) {
                Entry::Occupied(o) => o.index(),
                Entry::Vacant(v) => {
                    let index = v.index();
                    v.insert(());
                    index
                },
            }
        }
    }

    #[inline(always)]
    fn insert_null(&mut self) -> IdxSize {
        if self.null_idx == IdxSize::MAX {
            self.null_idx = self.idx_map.push_unmapped_empty_entry(());
        }
        self.null_idx
    }

    /// # Safety
    /// The view must be valid for the given buffer set.
    #[inline(always)]
    unsafe fn contains_key(&self, hash: u64, view: &View, buffers: &Arc<[Buffer<u8>]>) -> bool {
        unsafe { self.idx_map.get_view(hash, view, buffers).is_some() }
    }

    #[inline(always)]
    fn contains_null(&self) -> bool {
        self.null_idx < IdxSize::MAX
    }

    /// # Safety
    /// The views must be valid for the given buffers.
    unsafe fn finalize_keys<V: ViewType + ?Sized>(
        &self,
        schema: &Schema,
        views: Buffer<View>,
        buffers: Arc<[Buffer<u8>]>,
        validity: Option<Bitmap>,
    ) -> DataFrame {
        let (name, dtype) = schema.get_at_index(0).unwrap();
        unsafe {
            let arrow_dtype = dtype.to_arrow(CompatLevel::newest());
            let keys = BinaryViewArrayGeneric::<V>::new_unchecked_unknown_md(
                arrow_dtype,
                views,
                buffers,
                validity,
                None,
            );
            let s =
                Series::from_chunks_and_dtype_unchecked(name.clone(), vec![Box::new(keys)], dtype);
            DataFrame::new(vec![Column::from(s)]).unwrap()
        }
    }
}

impl Grouper for BinviewHashGrouper {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_groups(&self) -> IdxSize {
        self.idx_map.len()
    }

    unsafe fn insert_keys_subset(
        &mut self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        group_idxs: Option<&mut Vec<IdxSize>>,
    ) {
        let HashKeys::Binview(hash_keys) = hash_keys else {
            unreachable!()
        };

        unsafe {
            let views = hash_keys.keys.views().as_slice();
            let buffers = hash_keys.keys.data_buffers();
            if let Some(validity) = hash_keys.keys.validity() {
                if hash_keys.null_is_valid {
                    let groups = subset.iter().map(|idx| {
                        if validity.get_bit_unchecked(*idx as usize) {
                            let hash = hash_keys.hashes.value_unchecked(*idx as usize);
                            let view = views.get_unchecked(*idx as usize);
                            self.insert_key(hash, *view, buffers)
                        } else {
                            self.insert_null()
                        }
                    });
                    if let Some(group_idxs) = group_idxs {
                        group_idxs.reserve(subset.len());
                        group_idxs.extend(groups);
                    } else {
                        groups.for_each(drop);
                    }
                } else {
                    let groups = subset.iter().filter_map(|idx| {
                        if validity.get_bit_unchecked(*idx as usize) {
                            let hash = hash_keys.hashes.value_unchecked(*idx as usize);
                            let view = views.get_unchecked(*idx as usize);
                            Some(self.insert_key(hash, *view, buffers))
                        } else {
                            None
                        }
                    });
                    if let Some(group_idxs) = group_idxs {
                        group_idxs.reserve(subset.len());
                        group_idxs.extend(groups);
                    } else {
                        groups.for_each(drop);
                    }
                }
            } else {
                let groups = subset.iter().map(|idx| {
                    let hash = hash_keys.hashes.value_unchecked(*idx as usize);
                    let view = views.get_unchecked(*idx as usize);
                    self.insert_key(hash, *view, buffers)
                });
                if let Some(group_idxs) = group_idxs {
                    group_idxs.reserve(subset.len());
                    group_idxs.extend(groups);
                } else {
                    groups.for_each(drop);
                }
            }
        }
    }

    fn get_keys_in_group_order(&self, schema: &Schema) -> DataFrame {
        let buffers: Arc<[_]> = self
            .idx_map
            .buffers()
            .iter()
            .map(|b| Buffer::from(b.to_vec()))
            .collect();
        let views = self.idx_map.iter_hash_views().map(|(_h, v)| v).collect();
        let validity = if self.null_idx < IdxSize::MAX {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(self.idx_map.len() as usize, true);
            validity.set(self.null_idx as usize, false);
            Some(validity.freeze())
        } else {
            None
        };

        unsafe {
            let (_name, dt) = schema.get_at_index(0).unwrap();
            match dt {
                DataType::Binary => self.finalize_keys::<[u8]>(schema, views, buffers, validity),
                DataType::String => self.finalize_keys::<str>(schema, views, buffers, validity),
                _ => unreachable!(),
            }
        }
    }

    /// # Safety
    /// All groupers must be a BinviewHashGrouper.
    unsafe fn probe_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        probe_matches: &mut Vec<IdxSize>,
    ) {
        let HashKeys::Binview(hash_keys) = hash_keys else {
            unreachable!()
        };
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            let null_p = partitioner.null_partition();
            let buffers = hash_keys.keys.data_buffers();
            let views = hash_keys.keys.views().as_slice();
            hash_keys.for_each_hash(|idx, opt_h| {
                let has_group = if let Some(h) = opt_h {
                    let p = partitioner.hash_to_partition(h);
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const BinviewHashGrouper);
                    let view = views.get_unchecked(idx as usize);
                    grouper.contains_key(h, view, buffers)
                } else {
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(null_p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const BinviewHashGrouper);
                    grouper.contains_null()
                };

                if has_group != invert {
                    probe_matches.push(idx);
                }
            });
        }
    }

    /// # Safety
    /// All groupers must be a BinviewHashGrouper.
    unsafe fn contains_key_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        contains_key: &mut BitmapBuilder,
    ) {
        let HashKeys::Binview(hash_keys) = hash_keys else {
            unreachable!()
        };
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            let null_p = partitioner.null_partition();
            let buffers = hash_keys.keys.data_buffers();
            let views = hash_keys.keys.views().as_slice();
            hash_keys.for_each_hash(|idx, opt_h| {
                let has_group = if let Some(h) = opt_h {
                    let p = partitioner.hash_to_partition(h);
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const BinviewHashGrouper);
                    let view = views.get_unchecked(idx as usize);
                    grouper.contains_key(h, view, buffers)
                } else {
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(null_p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const BinviewHashGrouper);
                    grouper.contains_null()
                };

                contains_key.push(has_group != invert);
            });
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
