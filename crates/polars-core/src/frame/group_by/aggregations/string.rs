use super::*;

pub fn _agg_helper_idx_bin<'a, F>(groups: &'a GroupsIdx, f: F) -> Series
where
    F: Fn((IdxSize, &'a IdxVec)) -> Option<&'a [u8]> + Send + Sync,
{
    let ca: BinaryChunked = POOL.install(|| groups.into_par_iter().map(f).collect());
    ca.into_series()
}

pub fn _agg_helper_slice_bin<'a, F>(groups: &'a [[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<&'a [u8]> + Send + Sync,
{
    let ca: BinaryChunked = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

impl BinaryChunked {
    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_min<'a>(&'a self, groups: &GroupsType) -> Series {
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

        match groups {
            GroupsType::Idx(groups) => {
                let ca_self = self.rechunk();
                let arr = ca_self.downcast_as_array();
                let no_nulls = arr.null_count() == 0;
                _agg_helper_idx_bin(groups, |(first, idx)| {
                    debug_assert!(idx.len() <= ca_self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        arr.get_unchecked(first as usize)
                    } else if no_nulls {
                        take_agg_bin_iter_unchecked_no_null(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc < v { acc } else { v },
                        )
                    } else {
                        take_agg_bin_iter_unchecked(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc < v { acc } else { v },
                            idx.len() as IdxSize,
                        )
                    }
                })
            },
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_bin(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        let borrowed = arr_group.min_binary();

                        // SAFETY:
                        // The borrowed has `arr_group`s lifetime, but it actually points to data
                        // hold by self. Here we tell the compiler that.
                        unsafe { std::mem::transmute::<Option<&[u8]>, Option<&'a [u8]>>(borrowed) }
                    },
                }
            }),
        }
    }

    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_max<'a>(&'a self, groups: &GroupsType) -> Series {
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

        match groups {
            GroupsType::Idx(groups) => {
                let ca_self = self.rechunk();
                let arr = ca_self.downcast_as_array();
                let no_nulls = arr.null_count() == 0;
                _agg_helper_idx_bin(groups, |(first, idx)| {
                    debug_assert!(idx.len() <= self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        ca_self.get(first as usize)
                    } else if no_nulls {
                        take_agg_bin_iter_unchecked_no_null(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc > v { acc } else { v },
                        )
                    } else {
                        take_agg_bin_iter_unchecked(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc > v { acc } else { v },
                            idx.len() as IdxSize,
                        )
                    }
                })
            },
            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_bin(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        let borrowed = arr_group.max_binary();

                        // SAFETY:
                        // The borrowed has `arr_group`s lifetime, but it actually points to data
                        // hold by self. Here we tell the compiler that.
                        unsafe { std::mem::transmute::<Option<&[u8]>, Option<&'a [u8]>>(borrowed) }
                    },
                }
            }),
        }
    }

    pub(crate) unsafe fn agg_arg_min(&self, groups: &GroupsType) -> Series {
        // fast paths, consistent with other impls
        if groups.is_sorted_flag() {
            match self.is_sorted_flag() {
                IsSorted::Ascending => {
                    return self.clone().into_series().agg_arg_first_non_null(groups);
                },
                IsSorted::Descending => {
                    return self.clone().into_series().agg_arg_last_non_null(groups);
                },
                _ => {},
            }
        }

        let ca_self = self.rechunk();
        let arr = ca_self.downcast_as_array();
        let no_nulls = arr.null_count() == 0;

        let out: IdxCa = match groups {
            GroupsType::Idx(groups) => groups
                .all()
                .iter()
                .map(|idx| -> Option<IdxSize> {
                    if idx.is_empty() {
                        return None;
                    }

                    let mut best_pos: Option<IdxSize> = None;
                    let mut best_val: Option<&[u8]> = None;

                    if no_nulls {
                        for (pos, &i) in idx.iter().enumerate() {
                            // BinaryViewArray typically provides value_unchecked for no-nulls.
                            let v = arr.value_unchecked(i as usize);
                            match best_val {
                                None => {
                                    best_val = Some(v);
                                    best_pos = Some(pos as IdxSize);
                                },
                                Some(cur) => {
                                    if v < cur {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    }
                                },
                            }
                        }
                    } else {
                        for (pos, &i) in idx.iter().enumerate() {
                            if let Some(v) = arr.get(i as usize) {
                                match best_val {
                                    None => {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    },
                                    Some(cur) => {
                                        if v < cur {
                                            best_val = Some(v);
                                            best_pos = Some(pos as IdxSize);
                                        }
                                    },
                                }
                            }
                        }
                    }

                    best_pos
                })
                .collect_ca(PlSmallStr::EMPTY),

            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => groups_slice
                .iter()
                .map(|&[first, len]| -> Option<IdxSize> {
                    if len == 0 {
                        return None;
                    }

                    let start = first as usize;
                    let end = (first + len) as usize;

                    let mut best_pos: Option<IdxSize> = None;
                    let mut best_val: Option<&[u8]> = None;

                    if no_nulls {
                        for (pos, i) in (start..end).enumerate() {
                            let v = arr.value_unchecked(i);
                            match best_val {
                                None => {
                                    best_val = Some(v);
                                    best_pos = Some(pos as IdxSize);
                                },
                                Some(cur) => {
                                    if v < cur {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    }
                                },
                            }
                        }
                    } else {
                        for (pos, i) in (start..end).enumerate() {
                            if let Some(v) = arr.get(i) {
                                match best_val {
                                    None => {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    },
                                    Some(cur) => {
                                        if v < cur {
                                            best_val = Some(v);
                                            best_pos = Some(pos as IdxSize);
                                        }
                                    },
                                }
                            }
                        }
                    }

                    best_pos
                })
                .collect_ca(PlSmallStr::EMPTY),
        };

        out.into_series()
    }

    pub(crate) unsafe fn agg_arg_max(&self, groups: &GroupsType) -> Series {
        // fast paths
        if groups.is_sorted_flag() {
            match self.is_sorted_flag() {
                IsSorted::Ascending => {
                    return self.clone().into_series().agg_arg_last_non_null(groups);
                },
                IsSorted::Descending => {
                    return self.clone().into_series().agg_arg_first_non_null(groups);
                },
                _ => {},
            }
        }

        let ca_self = self.rechunk();
        let arr = ca_self.downcast_as_array();
        let no_nulls = arr.null_count() == 0;

        let out: IdxCa = match groups {
            GroupsType::Idx(groups) => groups
                .all()
                .iter()
                .map(|idx| -> Option<IdxSize> {
                    if idx.is_empty() {
                        return None;
                    }

                    let mut best_pos: Option<IdxSize> = None;
                    let mut best_val: Option<&[u8]> = None;

                    if no_nulls {
                        for (pos, &i) in idx.iter().enumerate() {
                            let v = arr.value_unchecked(i as usize);
                            match best_val {
                                None => {
                                    best_val = Some(v);
                                    best_pos = Some(pos as IdxSize);
                                },
                                Some(cur) => {
                                    if v > cur {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    }
                                },
                            }
                        }
                    } else {
                        for (pos, &i) in idx.iter().enumerate() {
                            if let Some(v) = arr.get(i as usize) {
                                match best_val {
                                    None => {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    },
                                    Some(cur) => {
                                        if v > cur {
                                            best_val = Some(v);
                                            best_pos = Some(pos as IdxSize);
                                        }
                                    },
                                }
                            }
                        }
                    }

                    best_pos
                })
                .collect_ca(PlSmallStr::EMPTY),

            GroupsType::Slice {
                groups: groups_slice,
                ..
            } => groups_slice
                .iter()
                .map(|&[first, len]| -> Option<IdxSize> {
                    if len == 0 {
                        return None;
                    }

                    let start = first as usize;
                    let end = (first + len) as usize;

                    let mut best_pos: Option<IdxSize> = None;
                    let mut best_val: Option<&[u8]> = None;

                    if no_nulls {
                        for (pos, i) in (start..end).enumerate() {
                            let v = arr.value_unchecked(i);
                            match best_val {
                                None => {
                                    best_val = Some(v);
                                    best_pos = Some(pos as IdxSize);
                                },
                                Some(cur) => {
                                    if v > cur {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    }
                                },
                            }
                        }
                    } else {
                        for (pos, i) in (start..end).enumerate() {
                            if let Some(v) = arr.get(i) {
                                match best_val {
                                    None => {
                                        best_val = Some(v);
                                        best_pos = Some(pos as IdxSize);
                                    },
                                    Some(cur) => {
                                        if v > cur {
                                            best_val = Some(v);
                                            best_pos = Some(pos as IdxSize);
                                        }
                                    },
                                }
                            }
                        }
                    }

                    best_pos
                })
                .collect_ca(PlSmallStr::EMPTY),
        };

        out.into_series()
    }
}

impl StringChunked {
    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_min<'a>(&'a self, groups: &GroupsType) -> Series {
        let out = self.as_binary().agg_min(groups);
        out.binary().unwrap().to_string_unchecked().into_series()
    }

    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_max<'a>(&'a self, groups: &GroupsType) -> Series {
        let out = self.as_binary().agg_max(groups);
        out.binary().unwrap().to_string_unchecked().into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    pub(crate) unsafe fn agg_arg_min(&self, groups: &GroupsType) -> Series {
        self.as_binary().agg_arg_min(groups)
    }

    #[cfg(feature = "algorithm_group_by")]
    pub(crate) unsafe fn agg_arg_max(&self, groups: &GroupsType) -> Series {
        self.as_binary().agg_arg_max(groups)
    }
}
