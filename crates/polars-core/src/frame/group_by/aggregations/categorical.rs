use super::*;
use crate::chunked_array::arg_min_max::{arg_max_opt_iter, arg_min_opt_iter};

#[cfg(feature = "dtype-categorical")]
impl<T: PolarsCategoricalType> CategoricalChunked<T> {
    /// # Safety
    /// Groups must be in bounds of the array.
    pub(crate) unsafe fn agg_min(&self, groups: &GroupsType) -> Series {
        let mapping = self.get_mapping();
        // Rechunk to a single array so that random index lookups are O(1) rather than O(chunks).
        let phys = self.physical().rechunk();
        let arr = phys.downcast_as_array();
        let cats: ChunkedArray<T::PolarsPhysical> = match groups {
            GroupsType::Idx(groups) => {
                POOL.install(|| {
                    groups
                        .into_par_iter()
                        .map(|(first, idx)| {
                            if idx.is_empty() {
                                None
                            } else if idx.len() == 1 {
                                // SAFETY: group indices are valid by GroupsType invariant
                                unsafe { arr.get_unchecked(first as usize) }
                            } else {
                                let min_pos = arg_min_opt_iter(idx.iter().map(|&i| {
                                    // SAFETY: group indices are valid by GroupsType invariant
                                    unsafe { arr.get_unchecked(i as usize) }.map(|cat| unsafe {
                                        mapping.cat_to_str_unchecked(cat.as_cat())
                                    })
                                }));
                                // SAFETY: min_pos points to a non-null element by arg_min_opt_iter contract
                                min_pos.map(|pos| {
                                    unsafe { arr.get_unchecked(idx[pos] as usize) }.unwrap()
                                })
                            }
                        })
                        .collect()
                })
            },
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => {
                POOL.install(|| {
                    groups_slice
                        .par_iter()
                        .copied()
                        .map(|[first, len]| {
                            let min_pos = arg_min_opt_iter(
                                (first as usize..first as usize + len as usize).map(|i| {
                                    // SAFETY: slice bounds are valid by GroupsType invariant
                                    unsafe { arr.get_unchecked(i) }.map(|cat| unsafe {
                                        mapping.cat_to_str_unchecked(cat.as_cat())
                                    })
                                }),
                            );
                            // SAFETY: min_pos is within [0, len), so first+min_pos is a valid non-null index
                            min_pos.map(|pos| {
                                unsafe { arr.get_unchecked(first as usize + pos) }.unwrap()
                            })
                        })
                        .collect()
                })
            },
        };
        let result: CategoricalChunked<T> = unsafe {
            CategoricalChunked::from_cats_and_dtype_unchecked(cats, self.dtype().clone())
        };
        result.into_series()
    }

    /// # Safety
    /// Groups must be in bounds of the array.
    pub(crate) unsafe fn agg_max(&self, groups: &GroupsType) -> Series {
        let mapping = self.get_mapping();
        // Rechunk to a single array so that random index lookups are O(1) rather than O(chunks).
        let phys = self.physical().rechunk();
        let arr = phys.downcast_as_array();
        let cats: ChunkedArray<T::PolarsPhysical> = match groups {
            GroupsType::Idx(groups) => {
                POOL.install(|| {
                    groups
                        .into_par_iter()
                        .map(|(first, idx)| {
                            if idx.is_empty() {
                                None
                            } else if idx.len() == 1 {
                                // SAFETY: group indices are valid by GroupsType invariant
                                unsafe { arr.get_unchecked(first as usize) }
                            } else {
                                let max_pos = arg_max_opt_iter(idx.iter().map(|&i| {
                                    // SAFETY: group indices are valid by GroupsType invariant
                                    unsafe { arr.get_unchecked(i as usize) }.map(|cat| unsafe {
                                        mapping.cat_to_str_unchecked(cat.as_cat())
                                    })
                                }));
                                // SAFETY: max_pos points to a non-null element by arg_max_opt_iter contract
                                max_pos.map(|pos| {
                                    unsafe { arr.get_unchecked(idx[pos] as usize) }.unwrap()
                                })
                            }
                        })
                        .collect()
                })
            },
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => {
                POOL.install(|| {
                    groups_slice
                        .par_iter()
                        .copied()
                        .map(|[first, len]| {
                            let max_pos = arg_max_opt_iter(
                                (first as usize..first as usize + len as usize).map(|i| {
                                    // SAFETY: slice bounds are valid by GroupsType invariant
                                    unsafe { arr.get_unchecked(i) }.map(|cat| unsafe {
                                        mapping.cat_to_str_unchecked(cat.as_cat())
                                    })
                                }),
                            );
                            // SAFETY: max_pos is within [0, len), so first+max_pos is a valid non-null index
                            max_pos.map(|pos| {
                                unsafe { arr.get_unchecked(first as usize + pos) }.unwrap()
                            })
                        })
                        .collect()
                })
            },
        };
        let result: CategoricalChunked<T> = unsafe {
            CategoricalChunked::from_cats_and_dtype_unchecked(cats, self.dtype().clone())
        };
        result.into_series()
    }

    /// # Safety
    /// Groups must be in bounds of the array.
    pub(crate) unsafe fn agg_arg_min(&self, groups: &GroupsType) -> Series {
        let mapping = self.get_mapping();
        // Rechunk to a single array so that random index lookups are O(1) rather than O(chunks).
        let phys = self.physical().rechunk();
        let arr = phys.downcast_as_array();
        match groups {
            GroupsType::Idx(groups) => {
                _agg_helper_idx_idx(groups, |(_, idx)| {
                    if idx.is_empty() {
                        None
                    } else {
                        arg_min_opt_iter(idx.iter().map(|&i| {
                            // SAFETY: group indices are valid by GroupsType invariant
                            unsafe { arr.get_unchecked(i as usize) }
                                .map(|cat| unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) })
                        }))
                        .map(|pos| pos as IdxSize)
                    }
                })
            },
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => {
                _agg_helper_slice_idx(groups_slice, |[first, len]| {
                    if len == 0 {
                        None
                    } else {
                        arg_min_opt_iter((first as usize..first as usize + len as usize).map(|i| {
                            // SAFETY: slice bounds are valid by GroupsType invariant
                            unsafe { arr.get_unchecked(i) }
                                .map(|cat| unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) })
                        }))
                        .map(|pos| pos as IdxSize)
                    }
                })
            },
        }
    }

    /// # Safety
    /// Groups must be in bounds of the array.
    pub(crate) unsafe fn agg_arg_max(&self, groups: &GroupsType) -> Series {
        let mapping = self.get_mapping();
        // Rechunk to a single array so that random index lookups are O(1) rather than O(chunks).
        let phys = self.physical().rechunk();
        let arr = phys.downcast_as_array();
        match groups {
            GroupsType::Idx(groups) => {
                _agg_helper_idx_idx(groups, |(_, idx)| {
                    if idx.is_empty() {
                        None
                    } else {
                        arg_max_opt_iter(idx.iter().map(|&i| {
                            // SAFETY: group indices are valid by GroupsType invariant
                            unsafe { arr.get_unchecked(i as usize) }
                                .map(|cat| unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) })
                        }))
                        .map(|pos| pos as IdxSize)
                    }
                })
            },
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => {
                _agg_helper_slice_idx(groups_slice, |[first, len]| {
                    if len == 0 {
                        None
                    } else {
                        arg_max_opt_iter((first as usize..first as usize + len as usize).map(|i| {
                            // SAFETY: slice bounds are valid by GroupsType invariant
                            unsafe { arr.get_unchecked(i) }
                                .map(|cat| unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) })
                        }))
                        .map(|pos| pos as IdxSize)
                    }
                })
            },
        }
    }
}
