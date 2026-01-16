use arrow::bitmap::bitmask::BitMask;

use super::*;
use crate::chunked_array::cast::CastOptions;

pub fn _agg_helper_idx_bool<F>(groups: &GroupsIdx, f: F) -> Series
where
    F: Fn((IdxSize, &IdxVec)) -> Option<bool> + Send + Sync,
{
    let ca: BooleanChunked = POOL.install(|| groups.into_par_iter().map(f).collect());
    ca.into_series()
}

pub fn _agg_helper_slice_bool<F>(groups: &[[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<bool> + Send + Sync,
{
    let ca: BooleanChunked = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

#[cfg(feature = "bitwise")]
impl BooleanChunked {
    pub(crate) unsafe fn agg_and(&self, groups: &GroupsType) -> BooleanChunked {
        self.agg_all(groups, true)
    }

    pub(crate) unsafe fn agg_or(&self, groups: &GroupsType) -> BooleanChunked {
        self.agg_any(groups, true)
    }

    pub(crate) unsafe fn agg_xor(&self, groups: &GroupsType) -> BooleanChunked {
        self.bool_agg(
            groups,
            true,
            |values, idxs| {
                idxs.iter()
                    .map(|i| {
                        <IdxSize as From<bool>>::from(unsafe {
                            values.get_bit_unchecked(*i as usize)
                        })
                    })
                    .sum::<IdxSize>()
                    % 2
                    == 1
            },
            |values, validity, idxs| {
                idxs.iter()
                    .map(|i| {
                        <IdxSize as From<bool>>::from(unsafe {
                            validity.get_bit_unchecked(*i as usize)
                                & values.get_bit_unchecked(*i as usize)
                        })
                    })
                    .sum::<IdxSize>()
                    % 2
                    == 0
            },
            |_, _, _| unreachable!(),
            |values, start, length| {
                unsafe { values.sliced_unchecked(start as usize, length as usize) }.set_bits() % 2
                    == 1
            },
            |values, validity, start, length| {
                let values = unsafe { values.sliced_unchecked(start as usize, length as usize) };
                let validity =
                    unsafe { validity.sliced_unchecked(start as usize, length as usize) };
                values.num_intersections_with(validity) % 2 == 1
            },
            |_, _, _, _| unreachable!(),
        )
    }
}

impl BooleanChunked {
    pub(crate) unsafe fn agg_min(&self, groups: &GroupsType) -> Series {
        // faster paths
        if groups.is_sorted_flag() {
            match self.is_sorted_flag() {
                IsSorted::Ascending => {
                    return self.clone().into_series().agg_first_non_null(groups);
                },
                IsSorted::Descending => {
                    return self.clone().into_series().agg_last_non_null(groups);
                },
                _ => {},
            }
        }
        let ca_self = self.rechunk();
        let arr = ca_self.downcast_iter().next().unwrap();
        let no_nulls = arr.null_count() == 0;
        match groups {
            GroupsType::Idx(groups) => _agg_helper_idx_bool(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    arr.get(first as usize)
                } else if no_nulls {
                    take_min_bool_iter_unchecked_no_nulls(arr, idx2usize(idx))
                } else {
                    take_min_bool_iter_unchecked_nulls(arr, idx2usize(idx), idx.len() as IdxSize)
                }
            }),
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_bool(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        arr_group.min()
                    },
                }
            }),
        }
    }
    pub(crate) unsafe fn agg_max(&self, groups: &GroupsType) -> Series {
        // faster paths
        if groups.is_sorted_flag() {
            match self.is_sorted_flag() {
                IsSorted::Ascending => return self.clone().into_series().agg_last_non_null(groups),
                IsSorted::Descending => {
                    return self.clone().into_series().agg_first_non_null(groups);
                },
                _ => {},
            }
        }

        let ca_self = self.rechunk();
        let arr = ca_self.downcast_iter().next().unwrap();
        let no_nulls = arr.null_count() == 0;
        match groups {
            GroupsType::Idx(groups) => _agg_helper_idx_bool(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else if no_nulls {
                    take_max_bool_iter_unchecked_no_nulls(arr, idx2usize(idx))
                } else {
                    take_max_bool_iter_unchecked_nulls(arr, idx2usize(idx), idx.len() as IdxSize)
                }
            }),
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_bool(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        arr_group.max()
                    },
                }
            }),
        }
    }

    pub(crate) unsafe fn agg_sum(&self, groups: &GroupsType) -> Series {
        self.cast_with_options(&IDX_DTYPE, CastOptions::Overflowing)
            .unwrap()
            .agg_sum(groups)
    }

    /// # Safety
    ///
    /// Groups should be in correct.
    #[expect(clippy::too_many_arguments)]
    unsafe fn bool_agg(
        &self,
        groups: &GroupsType,
        ignore_nulls: bool,

        idx_no_valid: impl Fn(BitMask, &[IdxSize]) -> bool + Send + Sync,
        idx_validity: impl Fn(BitMask, BitMask, &[IdxSize]) -> bool + Send + Sync,
        idx_kleene: impl Fn(BitMask, BitMask, &[IdxSize]) -> Option<bool> + Send + Sync,

        slice_no_valid: impl Fn(BitMask, IdxSize, IdxSize) -> bool + Send + Sync,
        slice_validity: impl Fn(BitMask, BitMask, IdxSize, IdxSize) -> bool + Send + Sync,
        slice_kleene: impl Fn(BitMask, BitMask, IdxSize, IdxSize) -> Option<bool> + Send + Sync,
    ) -> BooleanChunked {
        let name = self.name().clone();
        let values = self.rechunk();
        let values = values.downcast_as_array();

        let ca: BooleanChunked = POOL.install(|| {
            let validity = values
                .validity()
                .filter(|v| v.unset_bits() > 0)
                .map(BitMask::from_bitmap);
            let values = BitMask::from_bitmap(values.values());

            if !ignore_nulls && let Some(validity) = validity {
                match groups {
                    GroupsType::Idx(idx) => idx
                        .into_par_iter()
                        .map(|(_, idx)| idx_kleene(values, validity, idx))
                        .collect(),
                    GroupsType::Slice {
                        groups,
                        overlapping: _,
                        monotonic: _,
                    } => groups
                        .into_par_iter()
                        .map(|[start, length]| slice_kleene(values, validity, *start, *length))
                        .collect(),
                }
            } else {
                match groups {
                    GroupsType::Idx(idx) => match validity {
                        None => idx
                            .into_par_iter()
                            .map(|(_, idx)| idx_no_valid(values, idx))
                            .collect(),
                        Some(validity) => idx
                            .into_par_iter()
                            .map(|(_, idx)| idx_validity(values, validity, idx))
                            .collect(),
                    },
                    GroupsType::Slice {
                        groups,
                        overlapping: _,
                        monotonic: _,
                    } => match validity {
                        None => groups
                            .into_par_iter()
                            .map(|[start, length]| slice_no_valid(values, *start, *length))
                            .collect(),
                        Some(validity) => groups
                            .into_par_iter()
                            .map(|[start, length]| {
                                slice_validity(values, validity, *start, *length)
                            })
                            .collect(),
                    },
                }
            }
        });
        ca.with_name(name)
    }

    /// # Safety
    ///
    /// Groups should be in correct.
    pub unsafe fn agg_any(&self, groups: &GroupsType, ignore_nulls: bool) -> BooleanChunked {
        self.bool_agg(
            groups,
            ignore_nulls,
            |values, idxs| {
                idxs.iter()
                    .any(|i| unsafe { values.get_bit_unchecked(*i as usize) })
            },
            |values, validity, idxs| {
                idxs.iter().any(|i| unsafe {
                    validity.get_bit_unchecked(*i as usize) & values.get_bit_unchecked(*i as usize)
                })
            },
            |values, validity, idxs| {
                let mut saw_null = false;
                for i in idxs.iter() {
                    let is_valid = unsafe { validity.get_bit_unchecked(*i as usize) };
                    let is_true = unsafe { values.get_bit_unchecked(*i as usize) };

                    if is_valid & is_true {
                        return Some(true);
                    }
                    saw_null |= !is_valid;
                }
                (!saw_null).then_some(false)
            },
            |values, start, length| {
                unsafe { values.sliced_unchecked(start as usize, length as usize) }.leading_zeros()
                    < length as usize
            },
            |values, validity, start, length| {
                let values = unsafe { values.sliced_unchecked(start as usize, length as usize) };
                let validity =
                    unsafe { validity.sliced_unchecked(start as usize, length as usize) };
                values.intersects_with(validity)
            },
            |values, validity, start, length| {
                let values = unsafe { values.sliced_unchecked(start as usize, length as usize) };
                let validity =
                    unsafe { validity.sliced_unchecked(start as usize, length as usize) };

                if values.intersects_with(validity) {
                    Some(true)
                } else if validity.unset_bits() == 0 {
                    Some(false)
                } else {
                    None
                }
            },
        )
    }

    /// # Safety
    ///
    /// Groups should be in correct.
    pub unsafe fn agg_all(&self, groups: &GroupsType, ignore_nulls: bool) -> BooleanChunked {
        self.bool_agg(
            groups,
            ignore_nulls,
            |values, idxs| {
                idxs.iter()
                    .all(|i| unsafe { values.get_bit_unchecked(*i as usize) })
            },
            |values, validity, idxs| {
                idxs.iter().all(|i| unsafe {
                    !validity.get_bit_unchecked(*i as usize) | values.get_bit_unchecked(*i as usize)
                })
            },
            |values, validity, idxs| {
                let mut saw_null = false;
                for i in idxs.iter() {
                    let is_valid = unsafe { validity.get_bit_unchecked(*i as usize) };
                    let is_true = unsafe { values.get_bit_unchecked(*i as usize) };

                    if is_valid & !is_true {
                        return Some(false);
                    }
                    saw_null |= !is_valid;
                }
                (!saw_null).then_some(true)
            },
            |values, start, length| {
                let values = unsafe { values.sliced_unchecked(start as usize, length as usize) };
                values.unset_bits() == 0
            },
            |values, validity, start, length| {
                let values = unsafe { values.sliced_unchecked(start as usize, length as usize) };
                let validity =
                    unsafe { validity.sliced_unchecked(start as usize, length as usize) };
                values.num_intersections_with(validity) == validity.set_bits()
            },
            |values, validity, start, length| {
                let values = unsafe { values.sliced_unchecked(start as usize, length as usize) };
                let validity =
                    unsafe { validity.sliced_unchecked(start as usize, length as usize) };

                let num_non_nulls = validity.set_bits();

                if values.num_intersections_with(validity) < num_non_nulls {
                    Some(false)
                } else if num_non_nulls < values.len() {
                    None
                } else {
                    Some(true)
                }
            },
        )
    }
}
