use super::*;
#[cfg(feature = "dtype-date")]
use crate::chunked_array::temporal::conversion::US_IN_DAY;

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
                    let take = unsafe { self.take_slice_unchecked(idx) };
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
            },
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Series {
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
                self.take_unchecked(&indices)
            },
            GroupsProxy::Slice { groups, .. } => {
                let indices = groups
                    .iter()
                    .map(|&[first, len]| if len == 0 { None } else { Some(first) })
                    .collect_ca("");
                // SAFETY: groups are always in bounds.
                self.take_unchecked(&indices)
            },
        };
        if groups.is_sorted_flag() {
            out.set_sorted_flag(self.is_sorted_flag())
        }
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
                    let take = self.take_slice_unchecked(idx);
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
            },
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        use DataType::*;

        match self.dtype() {
            Boolean => self.cast(&Float64).unwrap().agg_mean(groups),
            Float32 => SeriesWrap(self.f32().unwrap().clone()).agg_median(groups),
            Float64 => SeriesWrap(self.f64().unwrap().clone()).agg_median(groups),
            dt if dt.is_numeric() => {
                apply_method_physical_integer!(self, agg_median, groups)
            },
            #[cfg(feature = "dtype-date")]
            Date => {
                let s = self.cast(&Int64).unwrap() * (US_IN_DAY as f64);
                // agg_median returns Float64
                let out = s.agg_median(groups);
                // cast back to Int64 and then to logical duration type
                out.cast(&Int64)
                    .unwrap()
                    .cast(&Datetime(TimeUnit::Microseconds, None))
                    .unwrap()
            },
            dt @ (Datetime(_, _) | Duration(_) | Time) => {
                let s = self.to_physical_repr();
                // agg_median returns Float64
                let out = s.agg_median(groups);
                // cast back to Int64 and then to logical duration type
                out.cast(&Int64).unwrap().cast(dt).unwrap()
            },
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
            },
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
            },
            #[cfg(feature = "dtype-date")]
            Date => {
                let s = self.cast(&Int64).unwrap() * (US_IN_DAY as f64);
                // agg_mean returns Float64
                let out = s.agg_mean(groups);
                // cast back to Int64 and then to logical duration type
                out.cast(&Int64)
                    .unwrap()
                    .cast(&Datetime(TimeUnit::Microseconds, None))
                    .unwrap()
            },
            dt @ (Datetime(_, _) | Duration(_) | Time) => {
                let s = self.to_physical_repr();
                // agg_mean returns Float64
                let out = s.agg_mean(groups);
                // cast back to Int64 and then to logical duration type
                out.cast(&Int64).unwrap().cast(dt).unwrap()
            },
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Series {
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
                self.take_unchecked(&indices)
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
                self.take_unchecked(&indices)
            },
        };
        self.restore_logical(out)
    }
}
