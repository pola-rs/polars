use arrow::bitmap::BitmapBuilder;
use polars_array::arrow::export;
use polars_array::builder::{PlArrayBuilder, ShareStrategy, builder_like};
use polars_core::chunked_array::builder::fixed_size_list::get_fixed_size_list_builder;
use polars_core::utils::slice_offsets;

use super::min_max::AggType;
use super::*;
#[cfg(feature = "array_count")]
use crate::chunked_array::array::count::array_count_matches;
use crate::chunked_array::array::count::count_boolean_bits;
use crate::chunked_array::array::sum_mean::sum_with_nulls;
use crate::prelude::array::get::array_get;
use crate::prelude::array::join::array_join;
use crate::prelude::array::sum_mean::sum_array_numerical;
use crate::series::ArgAgg;

pub fn has_inner_nulls(ca: &ArrayChunked) -> bool {
    for arr in ca.downcast_iter() {
        // The values are either flat or scalar; either way they are what the elements read, so a
        // null among them is a null inside an element. An empty array has neither.
        let values = arr.flat_values().or_else(|| arr.scalar_values());
        if values.is_some_and(|values| values.null_count() > 0) {
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

    fn array_dot(&self, other: &ArrayChunked) -> PolarsResult<Series> {
        dot::array_dot(self.as_array(), other)
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

    fn array_sort(&self, options: SortOptions) -> PolarsResult<ArrayChunked> {
        let ca = self.as_array();
        // SAFETY: Sort only changes the order of the elements in each subarray.
        unsafe { ca.try_apply_amortized_same_type(|s| s.as_ref().sort_with(options)) }
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
                let target_len = n.len();
                let single_array = ca.get_as_series(0);
                shift_broadcast_array(
                    single_array,
                    n,
                    ca.width(),
                    target_len,
                    ca.name().clone(),
                    ca.inner_dtype(),
                )?
            },
            _ => polars_bail!(length_mismatch = "arr.shift", ca.len(), n.len()),
        };
        Ok(out.into_series())
    }

    fn array_slice(&self, offset: i64, length: i64) -> PolarsResult<Series> {
        let ca = self.as_array();
        let width = ca.width();
        let length: usize = if length < 0 {
            (width as i64 + length).max(0)
        } else {
            length
        }
        .try_into()
        .expect("Length can not be larger than i64::MAX");
        let (raw_offset, slice_len) = slice_offsets(offset, length, width);

        let chunks = ca
            .downcast_iter()
            .map(|arr| {
                // TODO(polars-array-scalar): the slice is taken row by row, so a scalar chunk is
                // written out here rather than the one element it stands for being sliced once.
                let arr = arr.to_flat();
                let values = arr.values();

                let mut builder = builder_like(values);
                builder.reserve(slice_len * arr.len());
                let mut validity = BitmapBuilder::with_capacity(arr.len());

                for row in 0..arr.len() {
                    validity.push(arr.is_valid(row));
                    if !arr.is_valid(row) {
                        // A null row still holds a slot per value, undetermined though they are.
                        builder.extend_nulls(slice_len);
                        continue;
                    }
                    let inner_offset = row * width + raw_offset;
                    builder.subslice_extend(values, inner_offset, slice_len, ShareStrategy::Always);
                }

                PlFixedSizeListArray::new(
                    PlArrayBuilder::freeze(builder),
                    slice_len,
                    arr.len(),
                    validity.into_opt_validity(),
                )
                .into_boxed()
            })
            .collect::<Vec<_>>();

        // The chunks carry no logical type, so the width the slice left is named here.
        let dtype = DataType::Array(Box::new(ca.inner_dtype().clone()), slice_len);
        let slice_arr =
            unsafe { ArrayChunked::from_chunks_and_dtype(ca.name().clone(), chunks, dtype) };
        Ok(slice_arr.into_series())
    }
}

impl ArrayNameSpace for ArrayChunked {}

fn shift_broadcast_array(
    single_array: Option<Series>,
    n: &Int64Chunked,
    width: usize,
    target_len: usize,
    name: PlSmallStr,
    inner_dtype: &DataType,
) -> PolarsResult<ArrayChunked> {
    debug_assert!(target_len == n.len());

    let Some(single_array) = single_array else {
        return Ok(ArrayChunked::full_null_with_dtype(
            name,
            target_len,
            inner_dtype,
            width,
        ));
    };

    let mut builder = get_fixed_size_list_builder(inner_dtype, target_len, width, name)?;

    // SAFETY: `arr` is a chunk of a shifted `single_array`, so it has exactly
    // `width` elements of `inner_dtype`, matching the builder's expectations
    unsafe {
        for index in 0..target_len {
            match n.get(index) {
                Some(period) => {
                    let shifted = single_array.shift(period);
                    let shifted = if shifted.n_chunks() > 1 {
                        shifted.rechunk()
                    } else {
                        shifted
                    };
                    // TODO(polars-array-scalar): the builder is an Arrow one, so a scalar chunk
                    // is written out here rather than the one element it stands for being pushed.
                    let arr = export::to_arrow(shifted.chunks()[0].as_ref());
                    builder.push_unchecked(&*arr, 0);
                },
                None => builder.push_null(),
            }
        }
    }

    Ok(builder.finish())
}
