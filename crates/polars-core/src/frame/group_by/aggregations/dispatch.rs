use arrow::bitmap::bitmask::BitMask;
use polars_compute::unique::{AmortizedUnique, amortized_unique_from_dtype};

use super::*;
use crate::prelude::row_encode::encode_rows_unordered;

// Groups larger than this route to `Series::n_unique` (radix sort + scan
// for primitives, hashset for binary) instead of the amortized
// `AmortizedUnique` hashset: sort wins on cost, and keeping big groups
// out of the amortized hashset bounds its capacity, which avoids the
// O(capacity) `.clear()` storm of polars#27655.
const N_UNIQUE_SORT_FALLBACK_THRESHOLD: usize = 16384;

// implemented on the series because we don't need types
impl Series {
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
        let valid = self.rechunk_validity();

        match groups {
            GroupsType::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idxs| {
                debug_assert!(idxs.len() <= self.len());
                if let Some(v) = &valid {
                    let mut count = 0;
                    for idx in idxs.iter() {
                        count += unsafe { v.get_bit_unchecked(*idx as usize) as IdxSize };
                    }
                    Some(count)
                } else {
                    Some(self.len() as IdxSize)
                }
            }),
            GroupsType::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    if let Some(v) = &valid {
                        let m = BitMask::from_bitmap(v).sliced(first as usize, len as usize);
                        Some(m.set_bits() as IdxSize)
                    } else {
                        Some(self.len() as IdxSize)
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
        if groups.is_sorted_by_first_idx() {
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
        if matches!(groups, GroupsType::Slice { .. }) && !groups.is_overlapping() {
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

        // Keep the Column for the sort-fallback path. Big groups go through
        // `Series::n_unique`, bypassing the amortized hashset.
        let col = values.clone();
        let values = values.rechunk_to_arrow(CompatLevel::newest());
        let values = values.as_ref();
        let state = amortized_unique_from_dtype(values.dtype());

        struct CloneWrapper(Box<dyn AmortizedUnique>);
        impl Clone for CloneWrapper {
            fn clone(&self) -> Self {
                Self(self.0.new_empty())
            }
        }

        // SAFETY for the `.unwrap()` on `Series::n_unique()` below: we've
        // already filtered out dtypes that can fail (objects panic above;
        // nested types are row-encoded to binary). All remaining dtypes
        // have a `ChunkUnique` impl that infallibly returns `Ok(_)`.
        RAYON
            .install(|| match groups {
                GroupsType::Idx(idx) => idx
                    .all()
                    .into_par_iter()
                    .map_with(CloneWrapper(state), |state, idxs| unsafe {
                        let idxs = idxs.as_slice();
                        if idxs.len() > N_UNIQUE_SORT_FALLBACK_THRESHOLD {
                            col.take_slice_unchecked(idxs).n_unique().unwrap() as IdxSize
                        } else {
                            state.0.n_unique_idx(values, idxs)
                        }
                    })
                    .collect::<NoNull<IdxCa>>(),
                GroupsType::Slice {
                    groups,
                    overlapping: _,
                    monotonic: _,
                } => groups
                    .into_par_iter()
                    .map_with(CloneWrapper(state), |state, &[start, len]| {
                        let len_us = len as usize;
                        if len_us > N_UNIQUE_SORT_FALLBACK_THRESHOLD {
                            col.slice(start as i64, len_us).n_unique().unwrap() as IdxSize
                        } else {
                            state.0.n_unique_slice(values, start, len)
                        }
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
        if groups.is_monotonic() {
            out.set_sorted_flag(s.is_sorted_flag())
        }
        s.restore_logical(out)
    }
}
