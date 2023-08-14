use super::*;

pub fn _agg_helper_idx_utf8<'a, F>(groups: &'a GroupsIdx, f: F) -> Series
where
    F: Fn((IdxSize, &'a Vec<IdxSize>)) -> Option<&'a str> + Send + Sync,
{
    let ca: Utf8Chunked = POOL.install(|| groups.into_par_iter().map(f).collect());
    ca.into_series()
}

pub fn _agg_helper_slice_utf8<'a, F>(groups: &'a [[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<&'a str> + Send + Sync,
{
    let ca: Utf8Chunked = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

impl Utf8Chunked {
    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_min<'a>(&'a self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (&self.is_sorted_flag(), &self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_first(groups);
            },
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_last(groups);
            },
            _ => {},
        }

        match groups {
            GroupsProxy::Idx(groups) => {
                let ca_self = self.rechunk();
                let arr = ca_self.downcast_iter().next().unwrap();
                let no_nulls = arr.null_count() == 0;
                _agg_helper_idx_utf8(groups, |(first, idx)| {
                    debug_assert!(idx.len() <= ca_self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        arr.get_unchecked(first as usize)
                    } else if no_nulls {
                        take_agg_utf8_iter_unchecked_no_null(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc < v { acc } else { v },
                        )
                    } else {
                        take_agg_utf8_iter_unchecked(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc < v { acc } else { v },
                            idx.len() as IdxSize,
                        )
                    }
                })
            },
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_utf8(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        let borrowed = arr_group.min_str();

                        // Safety:
                        // The borrowed has `arr_group`s lifetime, but it actually points to data
                        // hold by self. Here we tell the compiler that.
                        unsafe { std::mem::transmute::<Option<&str>, Option<&'a str>>(borrowed) }
                    },
                }
            }),
        }
    }

    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_max<'a>(&'a self, groups: &GroupsProxy) -> Series {
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

        match groups {
            GroupsProxy::Idx(groups) => {
                let ca_self = self.rechunk();
                let arr = ca_self.downcast_iter().next().unwrap();
                let no_nulls = arr.null_count() == 0;
                _agg_helper_idx_utf8(groups, |(first, idx)| {
                    debug_assert!(idx.len() <= self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        ca_self.get(first as usize)
                    } else if no_nulls {
                        take_agg_utf8_iter_unchecked_no_null(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc > v { acc } else { v },
                        )
                    } else {
                        take_agg_utf8_iter_unchecked(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc > v { acc } else { v },
                            idx.len() as IdxSize,
                        )
                    }
                })
            },
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_utf8(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        let borrowed = arr_group.max_str();

                        // Safety:
                        // The borrowed has `arr_group`s lifetime, but it actually points to data
                        // hold by self. Here we tell the compiler that.
                        unsafe { std::mem::transmute::<Option<&str>, Option<&'a str>>(borrowed) }
                    },
                }
            }),
        }
    }
}
