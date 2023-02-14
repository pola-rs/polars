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
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if !self.has_validity() {
                    Some(idx.len() as IdxSize)
                } else {
                    let take =
                        unsafe { self.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize)) };
                    Some((take.len() - take.null_count()) as IdxSize)
                }
            }),
            GroupsProxy::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    if len == 0 {
                        None
                    } else if !self.has_validity() {
                        Some(len)
                    } else {
                        let take = self.slice_from_offsets(first, len);
                        Some((take.len() - take.null_count()) as IdxSize)
                    }
                })
            }
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Series {
        let out = match groups {
            GroupsProxy::Idx(groups) => {
                let mut iter = groups.iter().map(|(first, idx)| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some(first as usize)
                    }
                });
                // Safety:
                // groups are always in bounds
                self.take_opt_iter_unchecked(&mut iter)
            }
            GroupsProxy::Slice { groups, .. } => {
                let mut iter =
                    groups.iter().map(
                        |&[first, len]| {
                            if len == 0 {
                                None
                            } else {
                                Some(first as usize)
                            }
                        },
                    );
                // Safety:
                // groups are always in bounds
                self.take_opt_iter_unchecked(&mut iter)
            }
        };
        self.restore_logical(out)
    }

    #[doc(hidden)]
    pub unsafe fn agg_n_unique(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else {
                    let take = self.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize));
                    take.n_unique().ok().map(|v| v as IdxSize)
                }
            }),
            GroupsProxy::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    if len == 0 {
                        None
                    } else {
                        let take = self.slice_from_offsets(first, len);
                        take.n_unique().ok().map(|v| v as IdxSize)
                    }
                })
            }
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        use DataType::*;

        match self.dtype() {
            Float32 => SeriesWrap(self.f32().unwrap().clone()).agg_median(groups),
            Float64 => SeriesWrap(self.f64().unwrap().clone()).agg_median(groups),
            dt if dt.is_numeric() || dt.is_temporal() => {
                let ca = self.to_physical_repr();
                let physical_type = ca.dtype();
                let s = apply_method_physical_integer!(ca, agg_median, groups);
                if dt.is_logical() {
                    // back to physical and then
                    // back to logical type
                    s.cast(physical_type).unwrap().cast(dt).unwrap()
                } else {
                    s
                }
            }
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        use DataType::*;

        match self.dtype() {
            Float32 => self.f32().unwrap().agg_quantile(groups, quantile, interpol),
            Float64 => self.f64().unwrap().agg_quantile(groups, quantile, interpol),
            dt if dt.is_numeric() || dt.is_temporal() => {
                let ca = self.to_physical_repr();
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
            }
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        use DataType::*;

        match self.dtype() {
            Boolean => self.cast(&Float64).unwrap().agg_mean(groups),
            Float32 => SeriesWrap(self.f32().unwrap().clone()).agg_mean(groups),
            Float64 => SeriesWrap(self.f64().unwrap().clone()).agg_mean(groups),
            dt if dt.is_numeric() => {
                apply_method_physical_integer!(self, agg_mean, groups)
            }
            dt @ Duration(_) => {
                let s = self.to_physical_repr();
                // agg_mean returns Float64
                let out = s.agg_mean(groups);
                // cast back to Int64 and then to logical duration type
                out.cast(&Int64).unwrap().cast(dt).unwrap()
            }
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Series {
        let out = match groups {
            GroupsProxy::Idx(groups) => {
                let mut iter = groups.all().iter().map(|idx| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some(idx[idx.len() - 1] as usize)
                    }
                });
                self.take_opt_iter_unchecked(&mut iter)
            }
            GroupsProxy::Slice { groups, .. } => {
                let mut iter = groups.iter().map(|&[first, len]| {
                    if len == 0 {
                        None
                    } else {
                        Some((first + len - 1) as usize)
                    }
                });
                self.take_opt_iter_unchecked(&mut iter)
            }
        };
        self.restore_logical(out)
    }
}
