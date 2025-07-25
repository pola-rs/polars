use arrow::array::builder::{ShareStrategy, make_builder};
use arrow::array::{Array, FixedSizeListArray};
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::arity::unary_kernel;
use polars_core::utils::slice_offsets;

use super::min_max::AggType;
use super::*;
#[cfg(feature = "array_count")]
use crate::chunked_array::array::count::array_count_matches;
use crate::chunked_array::array::count::count_boolean_bits;
use crate::chunked_array::array::sum_mean::sum_with_nulls;
#[cfg(feature = "array_any_all")]
use crate::prelude::array::any_all::{array_all, array_any};
use crate::prelude::array::get::array_get;
use crate::prelude::array::join::array_join;
use crate::prelude::array::sum_mean::sum_array_numerical;
use crate::series::ArgAgg;

pub fn has_inner_nulls(ca: &ArrayChunked) -> bool {
    for arr in ca.downcast_iter() {
        if arr.values().null_count() > 0 {
            return true;
        }
    }
    false
}

fn get_agg(ca: &ArrayChunked, agg_type: AggType) -> Series {
    let values = ca.get_inner();
    let width = ca.width();
    min_max::array_dispatch(ca.name().clone(), &values, width, agg_type)
}

pub trait ArrayNameSpace: AsArray {
    fn array_max(&self) -> Series {
        let ca = self.as_array();
        get_agg(ca, AggType::Max)
    }

    fn array_min(&self) -> Series {
        let ca = self.as_array();
        get_agg(ca, AggType::Min)
    }

    fn array_sum(&self) -> PolarsResult<Series> {
        let ca = self.as_array();

        if has_inner_nulls(ca) {
            return sum_with_nulls(ca, ca.inner_dtype());
        };

        match ca.inner_dtype() {
            DataType::Boolean => Ok(count_boolean_bits(ca).into_series()),
            dt if dt.is_primitive_numeric() => Ok(sum_array_numerical(ca, dt)),
            dt => sum_with_nulls(ca, dt),
        }
    }

    fn array_mean(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::mean_with_nulls(ca)
    }

    fn array_median(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::median_with_nulls(ca)
    }

    fn array_std(&self, ddof: u8) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::std_with_nulls(ca, ddof)
    }

    fn array_var(&self, ddof: u8) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::var_with_nulls(ca, ddof)
    }

    fn array_unique(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_array();
        ca.try_apply_amortized_to_list(|s| s.as_ref().unique())
    }

    fn array_unique_stable(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_array();
        ca.try_apply_amortized_to_list(|s| s.as_ref().unique_stable())
    }

    fn array_n_unique(&self) -> PolarsResult<IdxCa> {
        let ca = self.as_array();
        ca.try_apply_amortized_generic(|opt_s| {
            let opt_v = opt_s.map(|s| s.as_ref().n_unique()).transpose()?;
            Ok(opt_v.map(|idx| idx as IdxSize))
        })
    }

    #[cfg(feature = "array_any_all")]
    fn array_any(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_any(ca)
    }

    #[cfg(feature = "array_any_all")]
    fn array_all(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_all(ca)
    }

    fn array_sort(&self, options: SortOptions) -> PolarsResult<ArrayChunked> {
        let ca = self.as_array();
        // SAFETY: Sort only changes the order of the elements in each subarray.
        unsafe { ca.try_apply_amortized_same_type(|s| s.as_ref().sort_with(options)) }
    }

    fn array_reverse(&self) -> ArrayChunked {
        let ca = self.as_array();
        // SAFETY: Reverse only changes the order of the elements in each subarray
        unsafe { ca.apply_amortized_same_type(|s| s.as_ref().reverse()) }
    }

    fn array_arg_min(&self) -> IdxCa {
        let ca = self.as_array();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_min().map(|idx| idx as IdxSize))
        })
    }

    fn array_arg_max(&self) -> IdxCa {
        let ca = self.as_array();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_max().map(|idx| idx as IdxSize))
        })
    }

    fn array_get(&self, index: &Int64Chunked, null_on_oob: bool) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_get(ca, index, null_on_oob)
    }

    fn array_join(&self, separator: &StringChunked, ignore_nulls: bool) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_join(ca, separator, ignore_nulls).map(|ok| ok.into_series())
    }

    #[cfg(feature = "array_count")]
    fn array_count_matches(&self, element: AnyValue) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_count_matches(ca, element)
    }

    fn array_shift(&self, n: &Series) -> PolarsResult<Series> {
        let ca = self.as_array();
        let n_s = n.cast(&DataType::Int64)?;
        let n = n_s.i64()?;
        let out = match (ca.len(), n.len()) {
            (a, b) if a == b => {
                // SAFETY: Shift does not change the dtype and number of elements of sub-array.
                unsafe {
                    ca.zip_and_apply_amortized_same_type(n, |opt_s, opt_periods| {
                        match (opt_s, opt_periods) {
                            (Some(s), Some(n)) => Some(s.as_ref().shift(n)),
                            _ => None,
                        }
                    })
                }
            },
            (_, 1) => {
                if let Some(n) = n.get(0) {
                    // SAFETY: Shift does not change the dtype and number of elements of sub-array.
                    unsafe { ca.apply_amortized_same_type(|s| s.as_ref().shift(n)) }
                } else {
                    ArrayChunked::full_null_with_dtype(
                        ca.name().clone(),
                        ca.len(),
                        ca.inner_dtype(),
                        ca.width(),
                    )
                }
            },
            (1, _) => {
                if ca.get(0).is_some() {
                    // Optimize: This does not need to broadcast first.
                    let ca = ca.new_from_index(0, n.len());
                    // SAFETY: Shift does not change the dtype and number of elements of sub-array.
                    unsafe {
                        ca.zip_and_apply_amortized_same_type(n, |opt_s, opt_periods| {
                            match (opt_s, opt_periods) {
                                (Some(s), Some(n)) => Some(s.as_ref().shift(n)),
                                _ => None,
                            }
                        })
                    }
                } else {
                    ArrayChunked::full_null_with_dtype(
                        ca.name().clone(),
                        ca.len(),
                        ca.inner_dtype(),
                        ca.width(),
                    )
                }
            },
            _ => polars_bail!(length_mismatch = "arr.shift", ca.len(), n.len()),
        };
        Ok(out.into_series())
    }

    fn array_slice(&self, offset: i64, length: i64) -> PolarsResult<Series> {
        let slice_arr: ArrayChunked = unary_kernel(
            self.as_array(),
            move |arr: &FixedSizeListArray| -> FixedSizeListArray {
                let length: usize = if length < 0 {
                    (arr.size() as i64 + length).max(0)
                } else {
                    length
                }
                .try_into()
                .expect("Length can not be larger than i64::MAX");
                let (raw_offset, slice_len) = slice_offsets(offset, length, arr.size());

                let mut builder = make_builder(arr.values().dtype());
                builder.reserve(slice_len * arr.len());

                let mut validity = BitmapBuilder::with_capacity(arr.len());

                let values = arr.values().as_ref();
                for row in 0..arr.len() {
                    if !arr.is_valid(row) {
                        validity.push(false);
                        continue;
                    }
                    let inner_offset = row * arr.size() + raw_offset;
                    builder.subslice_extend(values, inner_offset, slice_len, ShareStrategy::Always);
                    validity.push(true);
                }
                let values = builder.freeze_reset();
                let sliced_dtype = match arr.dtype() {
                    ArrowDataType::FixedSizeList(inner, _) => {
                        ArrowDataType::FixedSizeList(inner.clone(), slice_len)
                    },
                    _ => unreachable!(),
                };
                FixedSizeListArray::new(
                    sliced_dtype,
                    arr.len(),
                    values,
                    validity.into_opt_validity(),
                )
            },
        );
        Ok(slice_arr.into_series())
    }
}

impl ArrayNameSpace for ArrayChunked {}
