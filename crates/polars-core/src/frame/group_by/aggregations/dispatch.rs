use arrow::bitmap::bitmask::BitMask;
use polars_compute::unique::{AmortizedUnique, amortized_unique_from_dtype};

use super::*;
use crate::prelude::row_encode::encode_rows_unordered;

// implemented on the series because we don't need types
impl Series {
    fn slice_from_offsets(&self, first: IdxSize, len: IdxSize) -> Self {
        self.slice(first as i64, len as usize)
    }

    unsafe fn restore_logical(&self, out: Series) -> Series {
        if self.dtype().is_logical() && !out.dtype().is_logical() {
            out.from_physical_unchecked(self.dtype()).unwrap()
        } else {
            out
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_valid_count(&self, groups: &GroupsType) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 && self.null_count() > 0 {
            self.rechunk()
        } else {
            self.clone()
        };

        match groups {
            GroupsType::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |_, idx| {
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
            GroupsType::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |_, [first, len]| {
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
    pub unsafe fn agg_first(&self, groups: &GroupsType) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        let mut out = match groups {
            GroupsType::Idx(groups) => {
                let indices = groups
                    .iter()
                    .map(
                        |(first, idx)| {
                            if idx.is_empty() { None } else { Some(first) }
                        },
                    )
                    .collect_ca(PlSmallStr::EMPTY);
                // SAFETY: groups are always in bounds.
                s.take_unchecked(&indices)
            },
            GroupsType::Slice { groups, .. } => {
                let indices = groups
                    .iter()
                    .map(|&[first, len]| if len == 0 { None } else { Some(first) })
                    .collect_ca(PlSmallStr::EMPTY);
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
    pub unsafe fn agg_first_non_null(&self, groups: &GroupsType) -> Series {
        if !self.has_nulls() {
            return self.agg_first(groups);
        }

        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        let validity = s.rechunk_validity().unwrap();
        let indices = match groups {
            GroupsType::Idx(groups) => {
                groups
                    .iter()
                    .map(|(_, idx)| {
                        let mut this_idx = None;
                        for &ii in idx.iter() {
                            // SAFETY: null_values has no null values
                            if validity.get_bit_unchecked(ii as usize) {
                                this_idx = Some(ii);
                                break;
                            }
                        }
                        this_idx
                    })
                    .collect_ca(PlSmallStr::EMPTY)
            },
            GroupsType::Slice { groups, .. } => {
                let mask = BitMask::from_bitmap(&validity);
                groups
                    .iter()
                    .map(|&[first, len]| {
                        // SAFETY: group slice is valid.
                        let validity = mask.sliced_unchecked(first as usize, len as usize);
                        let leading_zeros = validity.leading_zeros() as IdxSize;
                        if leading_zeros == len {
                            // All values are null, we have no first non-null.
                            None
                        } else {
                            Some(first + leading_zeros)
                        }
                    })
                    .collect_ca(PlSmallStr::EMPTY)
            },
        };
        // SAFETY: groups are always in bounds.
        let mut out = s.take_unchecked(&indices);
        if groups.is_sorted_flag() {
            out.set_sorted_flag(s.is_sorted_flag())
        }
        s.restore_logical(out)
    }

    #[doc(hidden)]
    pub unsafe fn agg_arg_first(&self, groups: &GroupsType) -> Series {
        let out: IdxCa = match groups {
            GroupsType::Idx(groups) => groups
                .iter()
                .map(|(_, idx)| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some(0 as IdxSize)
                    }
                })
                .collect_ca(PlSmallStr::EMPTY),

            GroupsType::Slice { groups, .. } => groups
                .iter()
                .map(|&[_first, len]| if len == 0 { None } else { Some(0 as IdxSize) })
                .collect_ca(PlSmallStr::EMPTY),
        };
        out.into_series()
    }

    #[doc(hidden)]
    pub unsafe fn agg_arg_first_non_null(&self, groups: &GroupsType) -> Series {
        if !self.has_nulls() {
            return self.agg_arg_first(groups);
        }

        let validity = self.rechunk_validity().unwrap();

        let out: IdxCa = match groups {
            GroupsType::Idx(groups) => groups
                .iter()
                .map(|(_, idx)| {
                    let mut pos: Option<IdxSize> = None;
                    for (p, &ii) in idx.iter().enumerate() {
                        if validity.get_bit_unchecked(ii as usize) {
                            pos = Some(p as IdxSize);
                            break;
                        }
                    }
                    pos
                })
                .collect_ca(PlSmallStr::EMPTY),

            GroupsType::Slice { groups, .. } => {
                let mask = BitMask::from_bitmap(&validity);
                groups
                    .iter()
                    .map(|&[first, len]| {
                        if len == 0 {
                            return None;
                        }
                        let v = mask.sliced_unchecked(first as usize, len as usize);
                        let lz = v.leading_zeros() as IdxSize;
                        if lz == len { None } else { Some(lz) }
                    })
                    .collect_ca(PlSmallStr::EMPTY)
            },
        };

        out.into_series()
    }

    #[doc(hidden)]
    pub unsafe fn agg_arg_last(&self, groups: &GroupsType) -> Series {
        let out: IdxCa = match groups {
            GroupsType::Idx(groups) => groups
                .all()
                .iter()
                .map(|idx| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some((idx.len() - 1) as IdxSize)
                    }
                })
                .collect_ca(PlSmallStr::EMPTY),

            GroupsType::Slice { groups, .. } => groups
                .iter()
                .map(|&[_first, len]| {
                    if len == 0 {
                        None
                    } else {
                        Some((len - 1) as IdxSize)
                    }
                })
                .collect_ca(PlSmallStr::EMPTY),
        };

        out.into_series()
    }

    #[doc(hidden)]
    pub unsafe fn agg_arg_last_non_null(&self, groups: &GroupsType) -> Series {
        if !self.has_nulls() {
            return self.agg_arg_last(groups);
        }

        let validity = self.rechunk_validity().unwrap();

        let out: IdxCa = match groups {
            GroupsType::Idx(groups) => groups
                .iter()
                .map(|(_, idx)| {
                    for (p, &ii) in idx.iter().enumerate().rev() {
                        if validity.get_bit_unchecked(ii as usize) {
                            return Some(p as IdxSize);
                        }
                    }
                    None
                })
                .collect_ca(PlSmallStr::EMPTY),

            GroupsType::Slice { groups, .. } => {
                let mask = BitMask::from_bitmap(&validity);
                groups
                    .iter()
                    .map(|&[first, len]| {
                        if len == 0 {
                            return None;
                        }
                        let v = mask.sliced_unchecked(first as usize, len as usize);
                        let tz = v.trailing_zeros() as IdxSize;
                        if tz == len { None } else { Some(len - tz - 1) }
                    })
                    .collect_ca(PlSmallStr::EMPTY)
            },
        };

        out.into_series()
    }

    #[doc(hidden)]
    pub unsafe fn agg_n_unique(&self, groups: &GroupsType) -> Series {
        let values = self.to_physical_repr();
        let dtype = values.dtype();
        let values = if dtype.contains_objects() {
            panic!("{}", polars_err!(opq = unique, dtype));
        } else if let Some(ca) = values.try_str() {
            ca.as_binary().into_column()
        } else if dtype.is_nested() {
            encode_rows_unordered(&[values.into_owned().into_column()])
                .unwrap()
                .into_column()
        } else {
            values.into_owned().into_column()
        };

        let values = values.rechunk_to_arrow(CompatLevel::newest());
        let values = values.as_ref();
        let state = amortized_unique_from_dtype(values.dtype());

        struct CloneWrapper(Box<dyn AmortizedUnique>);
        impl Clone for CloneWrapper {
            fn clone(&self) -> Self {
                Self(self.0.new_empty())
            }
        }

        POOL.install(|| match groups {
            GroupsType::Idx(idx) => idx
                .all()
                .into_par_iter()
                .map_with(CloneWrapper(state), |state, idxs| unsafe {
                    state.0.n_unique_idx(values, idxs.as_slice())
                })
                .collect::<NoNull<IdxCa>>(),
            GroupsType::Slice {
                groups,
                overlapping: _,
                monotonic: _,
            } => groups
                .into_par_iter()
                .map_with(CloneWrapper(state), |state, [start, len]| {
                    state.0.n_unique_slice(values, *start, *len)
                })
                .collect::<NoNull<IdxCa>>(),
        })
        .into_inner()
        .into_series()
    }

    #[doc(hidden)]
    pub unsafe fn agg_mean(&self, groups: &GroupsType) -> Series {
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
            dt if dt.is_primitive_numeric() => apply_method_physical_integer!(s, agg_mean, groups),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => self.cast(&Float64).unwrap().agg_mean(groups),
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
                * (US_IN_DAY as f64))
                .cast(&Datetime(TimeUnit::Microseconds, None))
                .unwrap(),
            _ => Series::full_null(PlSmallStr::EMPTY, groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_median(&self, groups: &GroupsType) -> Series {
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
            dt if dt.is_primitive_numeric() => {
                apply_method_physical_integer!(s, agg_median, groups)
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => self.cast(&Float64).unwrap().agg_median(groups),
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
                * (US_IN_DAY as f64))
                .cast(&Datetime(TimeUnit::Microseconds, None))
                .unwrap(),
            _ => Series::full_null(PlSmallStr::EMPTY, groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_quantile(
        &self,
        groups: &GroupsType,
        quantile: f64,
        method: QuantileMethod,
    ) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        use DataType::*;
        match s.dtype() {
            Float32 => s.f32().unwrap().agg_quantile(groups, quantile, method),
            Float64 => s.f64().unwrap().agg_quantile(groups, quantile, method),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => s
                .cast(&DataType::Float64)
                .unwrap()
                .agg_quantile(groups, quantile, method),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => self
                .to_physical_repr()
                .agg_quantile(groups, quantile, method)
                .cast(&Int64)
                .unwrap()
                .into_datetime(*tu, tz.clone()),
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => self
                .to_physical_repr()
                .agg_quantile(groups, quantile, method)
                .cast(&Int64)
                .unwrap()
                .into_duration(*tu),
            #[cfg(feature = "dtype-time")]
            Time => self
                .to_physical_repr()
                .agg_quantile(groups, quantile, method)
                .cast(&Int64)
                .unwrap()
                .into_time(),
            #[cfg(feature = "dtype-date")]
            Date => (self
                .to_physical_repr()
                .agg_quantile(groups, quantile, method)
                .cast(&Float64)
                .unwrap()
                * (US_IN_DAY as f64))
                .cast(&DataType::Int64)
                .unwrap()
                .into_datetime(TimeUnit::Microseconds, None),
            dt if dt.is_primitive_numeric() => {
                apply_method_physical_integer!(s, agg_quantile, groups, quantile, method)
            },
            _ => Series::full_null(PlSmallStr::EMPTY, groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_varying_quantile(
        &self,
        groups: &GroupsType,
        quantiles: &[f64],
        method: QuantileMethod,
    ) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        use DataType::*;
        match s.dtype() {
            Float32 => s
                .f32()
                .unwrap()
                .agg_varying_quantile(groups, quantiles, method),
            Float64 => s
                .f64()
                .unwrap()
                .agg_varying_quantile(groups, quantiles, method),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => s
                .cast(&DataType::Float64)
                .unwrap()
                .agg_varying_quantile(groups, quantiles, method),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => self
                .to_physical_repr()
                .agg_varying_quantile(groups, quantiles, method)
                .cast(&Int64)
                .unwrap()
                .into_datetime(*tu, tz.clone()),
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => self
                .to_physical_repr()
                .agg_varying_quantile(groups, quantiles, method)
                .cast(&Int64)
                .unwrap()
                .into_duration(*tu),
            #[cfg(feature = "dtype-time")]
            Time => self
                .to_physical_repr()
                .agg_varying_quantile(groups, quantiles, method)
                .cast(&Int64)
                .unwrap()
                .into_time(),
            #[cfg(feature = "dtype-date")]
            Date => (self
                .to_physical_repr()
                .agg_varying_quantile(groups, quantiles, method)
                .cast(&Float64)
                .unwrap()
                * (US_IN_DAY as f64))
                .cast(&DataType::Int64)
                .unwrap()
                .into_datetime(TimeUnit::Microseconds, None),
            dt if dt.is_primitive_numeric() => {
                apply_method_physical_integer!(s, agg_varying_quantile, groups, quantiles, method)
            },
            _ => Series::full_null(PlSmallStr::EMPTY, groups.len(), s.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_last(&self, groups: &GroupsType) -> Series {
        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        let out = match groups {
            GroupsType::Idx(groups) => {
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
                    .collect_ca(PlSmallStr::EMPTY);
                s.take_unchecked(&indices)
            },
            GroupsType::Slice { groups, .. } => {
                let indices = groups
                    .iter()
                    .map(|&[first, len]| {
                        if len == 0 {
                            None
                        } else {
                            Some(first + len - 1)
                        }
                    })
                    .collect_ca(PlSmallStr::EMPTY);
                s.take_unchecked(&indices)
            },
        };
        s.restore_logical(out)
    }

    #[doc(hidden)]
    pub unsafe fn agg_last_non_null(&self, groups: &GroupsType) -> Series {
        if !self.has_nulls() {
            return self.agg_last(groups);
        }

        // Prevent a rechunk for every individual group.
        let s = if groups.len() > 1 {
            self.rechunk()
        } else {
            self.clone()
        };

        let validity = s.rechunk_validity().unwrap();
        let indices = match groups {
            GroupsType::Idx(groups) => {
                groups
                    .iter()
                    .map(|(_, idx)| {
                        // We may or may not find a valid value.
                        let mut opt_idx = None;
                        for &ii in idx.iter().rev() {
                            // SAFETY: index is always in range.
                            if validity.get_bit_unchecked(ii as usize) {
                                opt_idx = Some(ii);
                                break;
                            }
                        }
                        opt_idx
                    })
                    .collect_ca(PlSmallStr::EMPTY)
            },
            GroupsType::Slice { groups, .. } => {
                let mask = BitMask::from_bitmap(&validity);
                groups
                    .iter()
                    .map(|&[first, len]| {
                        // SAFETY: group slice is valid.
                        let validity = mask.sliced_unchecked(first as usize, len as usize);
                        let trailing_zeros = validity.trailing_zeros() as IdxSize;
                        if trailing_zeros == len {
                            // All values are null, we have no last non-null.
                            None
                        } else {
                            Some(first + len - trailing_zeros - 1)
                        }
                    })
                    .collect_ca(PlSmallStr::EMPTY)
            },
        };
        // SAFETY: groups are always in bounds.
        let mut out = s.take_unchecked(&indices);
        if groups.is_sorted_flag() {
            out.set_sorted_flag(s.is_sorted_flag())
        }
        s.restore_logical(out)
    }
}
