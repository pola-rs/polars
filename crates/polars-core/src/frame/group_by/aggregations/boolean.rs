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
unsafe fn bitwise_agg(
    ca: &BooleanChunked,
    groups: &GroupsProxy,
    f: fn(&BooleanChunked) -> Option<bool>,
) -> Series {
    // Prevent a rechunk for every individual group.
    let s = if groups.len() > 1 {
        ca.rechunk()
    } else {
        ca.clone()
    };

    match groups {
        GroupsProxy::Idx(groups) => _agg_helper_idx_bool::<_>(groups, |(_, idx)| {
            debug_assert!(idx.len() <= s.len());
            if idx.is_empty() {
                None
            } else {
                let take = s.take_unchecked(idx);
                f(&take)
            }
        }),
        GroupsProxy::Slice { groups, .. } => _agg_helper_slice_bool::<_>(groups, |[first, len]| {
            debug_assert!(len <= s.len() as IdxSize);
            if len == 0 {
                None
            } else {
                let take = _slice_from_offsets(&s, first, len);
                f(&take)
            }
        }),
    }
}

#[cfg(feature = "bitwise")]
impl BooleanChunked {
    pub(crate) unsafe fn agg_and(&self, groups: &GroupsProxy) -> Series {
        bitwise_agg(self, groups, ChunkBitwiseReduce::and_reduce)
    }

    pub(crate) unsafe fn agg_or(&self, groups: &GroupsProxy) -> Series {
        bitwise_agg(self, groups, ChunkBitwiseReduce::or_reduce)
    }

    pub(crate) unsafe fn agg_xor(&self, groups: &GroupsProxy) -> Series {
        bitwise_agg(self, groups, ChunkBitwiseReduce::xor_reduce)
    }
}

impl BooleanChunked {
    pub(crate) unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_first(groups);
            },
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_last(groups);
            },
            _ => {},
        }
        let ca_self = self.rechunk();
        let arr = ca_self.downcast_iter().next().unwrap();
        let no_nulls = arr.null_count() == 0;
        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx_bool(groups, |(first, idx)| {
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
            GroupsProxy::Slice {
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
    pub(crate) unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_last(groups);
            },
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_first(groups);
            },
            _ => {},
        }

        let ca_self = self.rechunk();
        let arr = ca_self.downcast_iter().next().unwrap();
        let no_nulls = arr.null_count() == 0;
        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx_bool(groups, |(first, idx)| {
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
            GroupsProxy::Slice {
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
    pub(crate) unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        self.cast_with_options(&IDX_DTYPE, CastOptions::Overflowing)
            .unwrap()
            .agg_sum(groups)
    }
}
