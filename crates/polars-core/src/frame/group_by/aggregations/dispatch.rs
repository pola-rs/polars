use super::*;

// implemented on the series because we don't need types
impl Series {
    fn slice_from_offsets(&self, first: IdxSize, len: IdxSize) -> Self {
        self.slice(first as i64, len as usize)
    }

    fn restore_logical(&self, out: Series) -> Series {
        if self.dtype().is_logical() {
            out.cast(self.dtype()).unwrap()
        } else {
            out
        }
    }

    #[doc(hidden)]
    pub fn agg_valid_count(&self, groups: &GroupsProxy) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 && self.null_count() > 0 {
            self.rechunk()
        } else {
            self.clone()
        };

        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idx| {
                debug_assert!(idx.len() <= s.len());
                if idx.is_empty() {
                    None
                } else if s.null_count() == 0 {
                    Some(idx.len() as IdxSize)
                } else {
                    let take = unsafe { s.take_slice_unchecked(idx) };
                    Some((take.len() - take.null_count()) as IdxSize)
                }
            }),
            GroupsProxy::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= s.len() as IdxSize);
                    if len == 0 {
                        None
                    } else if s.null_count() == 0 {
                        Some(len)
                    } else {
                        let take = s.slice_from_offsets(first, len);
                        Some((take.len() - take.null_count()) as IdxSize)
                    }
                })
            },
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        let mut out = match groups {
            GroupsProxy::Idx(groups) => {
                let indices = groups
                    .iter()
                    .map(
                        |(first, idx)| {
                            if idx.is_empty() {
                                None
                            } else {
                                Some(first)
                            }
                        },
                    )
                    .collect_ca("");
                // SAFETY: groups are always in bounds.
                s.take_unchecked(&indices)
            },
            GroupsProxy::Slice { groups, .. } => {
                let indices = groups
                    .iter()
                    .map(|&[first, len]| if len == 0 { None } else { Some(first) })
                    .collect_ca("");
                // SAFETY: groups are always in bounds.
                s.take_unchecked(&indices)
            },
        };
        if groups.is_sorted_flag() {
            out.set_sorted_flag(s.is_sorted_flag())
        }
        s.restore_logical(out)
    }

    #[doc(hidden)]
    pub unsafe fn agg_n_unique(&self, groups: &GroupsProxy) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        match groups {
            GroupsProxy::Idx(groups) => {
                agg_helper_idx_on_all_no_null::<IdxType, _>(groups, |idx| {
                    debug_assert!(idx.len() <= s.len());
                    if idx.is_empty() {
                        0
                    } else {
                        let take = s.take_slice_unchecked(idx);
                        take.n_unique().unwrap() as IdxSize
                    }
                })
            },
            GroupsProxy::Slice { groups, .. } => {
                _agg_helper_slice_no_null::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= s.len() as IdxSize);
                    if len == 0 {
                        0
                    } else {
                        let take = s.slice_from_offsets(first, len);
                        take.n_unique().unwrap() as IdxSize
                    }
                })
            },
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        use DataType::*;
        match s.dtype() {
            Boolean => s.cast(&Float64).unwrap().agg_mean(groups),
            Float32 => SeriesWrap(s.f32().unwrap().clone()).agg_mean(groups),
            Float64 => SeriesWrap(s.f64().unwrap().clone()).agg_mean(groups),
            dt if dt.is_numeric() => apply_method_physical_integer!(s, agg_mean, groups),
            #[cfg(feature = "dtype-datetime")]
            dt @ Datetime(_, _) => self
                .to_physical_repr()
                .agg_mean(groups)
                .cast(&Int64)
                .unwrap()
                .cast(dt)
                .unwrap(),
            #[cfg(feature = "dtype-duration")]
            dt @ Duration(_) => self
                .to_physical_repr()
                .agg_mean(groups)
                .cast(&Int64)
                .unwrap()
                .cast(dt)
                .unwrap(),
            #[cfg(feature = "dtype-time")]
            Time => self
                .to_physical_repr()
                .agg_mean(groups)
                .cast(&Int64)
                .unwrap()
                .cast(&Time)
                .unwrap(),
            #[cfg(feature = "dtype-date")]
            Date => (self
                .to_physical_repr()
                .agg_mean(groups)
                .cast(&Float64)
                .unwrap()
                * (MS_IN_DAY as f64))
                .cast(&Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
            _ => Series::full_null("", groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        use DataType::*;
        match s.dtype() {
            Boolean => s.cast(&Float64).unwrap().agg_median(groups),
            Float32 => SeriesWrap(s.f32().unwrap().clone()).agg_median(groups),
            Float64 => SeriesWrap(s.f64().unwrap().clone()).agg_median(groups),
            dt if dt.is_numeric() => apply_method_physical_integer!(s, agg_median, groups),
            #[cfg(feature = "dtype-datetime")]
            dt @ Datetime(_, _) => self
                .to_physical_repr()
                .agg_median(groups)
                .cast(&Int64)
                .unwrap()
                .cast(dt)
                .unwrap(),
            #[cfg(feature = "dtype-duration")]
            dt @ Duration(_) => self
                .to_physical_repr()
                .agg_median(groups)
                .cast(&Int64)
                .unwrap()
                .cast(dt)
                .unwrap(),
            #[cfg(feature = "dtype-time")]
            Time => self
                .to_physical_repr()
                .agg_median(groups)
                .cast(&Int64)
                .unwrap()
                .cast(&Time)
                .unwrap(),
            #[cfg(feature = "dtype-date")]
            Date => (self
                .to_physical_repr()
                .agg_median(groups)
                .cast(&Float64)
                .unwrap()
                * (MS_IN_DAY as f64))
                .cast(&Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
            _ => Series::full_null("", groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        use DataType::*;
        match s.dtype() {
            Float32 => s.f32().unwrap().agg_quantile(groups, quantile, interpol),
            Float64 => s.f64().unwrap().agg_quantile(groups, quantile, interpol),
            dt if dt.is_numeric() || dt.is_temporal() => {
                let ca = s.to_physical_repr();
                let physical_type = ca.dtype();
                let s =
                    apply_method_physical_integer!(ca, agg_quantile, groups, quantile, interpol);
                if dt.is_logical() {
                    // back to physical and then
                    // back to logical type
                    s.cast(physical_type).unwrap().cast(dt).unwrap()
                } else {
                    s
                }
            },
            _ => Series::full_null("", groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        let out = match groups {
            GroupsProxy::Idx(groups) => {
                let indices = groups
                    .all()
                    .iter()
                    .map(|idx| {
                        if idx.is_empty() {
                            None
                        } else {
                            Some(idx[idx.len() - 1])
                        }
                    })
                    .collect_ca("");
                s.take_unchecked(&indices)
            },
            GroupsProxy::Slice { groups, .. } => {
                let indices = groups
                    .iter()
                    .map(|&[first, len]| {
                        if len == 0 {
                            None
                        } else {
                            Some(first + len - 1)
                        }
                    })
                    .collect_ca("");
                s.take_unchecked(&indices)
            },
        };
        s.restore_logical(out)
    }
}
