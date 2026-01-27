use arrow::array::Array;
use polars_utils::arg_min_max::ArgMinMax;
use polars_utils::min_max::{MaxIgnoreNan, MinIgnoreNan, MinMaxPolicy};

use crate::chunked_array::ChunkedArray;
use crate::chunked_array::ops::float_sorted_arg_max::{
    float_arg_max_sorted_ascending, float_arg_max_sorted_descending,
};
use crate::datatypes::{
    BinaryChunked, BooleanChunked, PolarsDataType, PolarsNumericType, StringChunked,
};
#[cfg(feature = "dtype-categorical")]
use crate::datatypes::{CategoricalChunked, PolarsCategoricalType};
use crate::series::IsSorted;

pub fn arg_min_opt_iter<T, I>(iter: I) -> Option<usize>
where
    I: IntoIterator<Item = Option<T>>,
    T: Ord,
{
    iter.into_iter()
        .enumerate()
        .flat_map(|(idx, val)| Some((idx, val?)))
        .min_by(|x, y| Ord::cmp(&x.1, &y.1))
        .map(|x| x.0)
}

pub fn arg_max_opt_iter<T, I>(iter: I) -> Option<usize>
where
    I: IntoIterator<Item = Option<T>>,
    T: Ord,
{
    iter.into_iter()
        .enumerate()
        .flat_map(|(idx, val)| Some((idx, val?)))
        .max_by(|x, y| Ord::cmp(&x.1, &y.1))
        .map(|x| x.0)
}

pub fn arg_min_numeric<T>(ca: &ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    if ca.null_count() == ca.len() {
        None
    } else if let Ok(vals) = ca.cont_slice() {
        arg_min_numeric_slice(vals, ca.is_sorted_flag())
    } else {
        arg_min_numeric_chunked(ca)
    }
}

pub fn arg_max_numeric<T>(ca: &ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    if ca.null_count() == ca.len() {
        None
    } else if T::get_static_dtype().is_float() && !matches!(ca.is_sorted_flag(), IsSorted::Not) {
        arg_max_float_sorted(ca)
    } else if let Ok(vals) = ca.cont_slice() {
        arg_max_numeric_slice(vals, ca.is_sorted_flag())
    } else {
        arg_max_numeric_chunked(ca)
    }
}

/// # Safety
/// `ca` has a float dtype, has at least one non-null value and is sorted.
fn arg_max_float_sorted<T>(ca: &ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
{
    let out = match ca.is_sorted_flag() {
        IsSorted::Ascending => float_arg_max_sorted_ascending(ca),
        IsSorted::Descending => float_arg_max_sorted_descending(ca),
        _ => unreachable!(),
    };
    Some(out)
}

#[cfg(feature = "dtype-categorical")]
pub fn arg_min_cat<T: PolarsCategoricalType>(ca: &CategoricalChunked<T>) -> Option<usize> {
    if ca.null_count() == ca.len() {
        return None;
    }
    arg_min_opt_iter(ca.iter_str())
}

#[cfg(feature = "dtype-categorical")]
pub fn arg_max_cat<T: PolarsCategoricalType>(ca: &CategoricalChunked<T>) -> Option<usize> {
    if ca.null_count() == ca.len() {
        return None;
    }
    arg_max_opt_iter(ca.iter_str())
}

pub fn arg_min_bool(ca: &BooleanChunked) -> Option<usize> {
    ca.first_false_idx().or_else(|| ca.first_true_idx())
}

pub fn arg_max_bool(ca: &BooleanChunked) -> Option<usize> {
    ca.first_true_idx().or_else(|| ca.first_false_idx())
}

pub fn arg_min_str(ca: &StringChunked) -> Option<usize> {
    arg_min_physical_generic(ca)
}

pub fn arg_max_str(ca: &StringChunked) -> Option<usize> {
    arg_max_physical_generic(ca)
}

pub fn arg_min_binary(ca: &BinaryChunked) -> Option<usize> {
    arg_min_physical_generic(ca)
}

pub fn arg_max_binary(ca: &BinaryChunked) -> Option<usize> {
    arg_max_physical_generic(ca)
}

fn arg_min_physical_generic<T>(ca: &ChunkedArray<T>) -> Option<usize>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: Ord,
{
    if ca.null_count() == ca.len() {
        return None;
    }
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.first_non_null(),
        IsSorted::Descending => ca.last_non_null(),
        IsSorted::Not => arg_min_opt_iter(ca.iter()),
    }
}

fn arg_max_physical_generic<T>(ca: &ChunkedArray<T>) -> Option<usize>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: Ord,
{
    if ca.null_count() == ca.len() {
        return None;
    }
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.last_non_null(),
        IsSorted::Descending => ca.first_non_null(),
        IsSorted::Not => arg_max_opt_iter(ca.iter()),
    }
}

fn arg_min_numeric_chunked<'a, T>(ca: &'a ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.first_non_null(),
        IsSorted::Descending => ca.last_non_null(),
        IsSorted::Not => {
            let mut chunk_start_offset = 0;
            let mut min_idx: Option<usize> = None;
            let mut min_val: Option<T::Native> = None;
            for arr in ca.downcast_iter() {
                if arr.len() == arr.null_count() {
                    chunk_start_offset += arr.len();
                    continue;
                }

                let chunk_min: Option<(usize, T::Native)> = if arr.null_count() > 0 {
                    arr.into_iter()
                        .enumerate()
                        .flat_map(|(idx, val)| Some((idx, *(val?))))
                        .reduce(|acc, (idx, val)| {
                            if MinIgnoreNan::is_better(&val, &acc.1) {
                                (idx, val)
                            } else {
                                acc
                            }
                        })
                } else {
                    // When no nulls & array not empty => we can use fast argmin.
                    let min_idx: usize = arr.values().as_slice().argmin();
                    Some((min_idx, arr.value(min_idx)))
                };

                if let Some((chunk_min_idx, chunk_min_val)) = chunk_min {
                    if min_val.is_none()
                        || MinIgnoreNan::is_better(&chunk_min_val, &min_val.unwrap())
                    {
                        min_val = Some(chunk_min_val);
                        min_idx = Some(chunk_start_offset + chunk_min_idx);
                    }
                }
                chunk_start_offset += arr.len();
            }
            min_idx
        },
    }
}

fn arg_max_numeric_chunked<'a, T>(ca: &'a ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.last_non_null(),
        IsSorted::Descending => ca.first_non_null(),
        IsSorted::Not => {
            let mut chunk_start_offset = 0;
            let mut max_idx: Option<usize> = None;
            let mut max_val: Option<T::Native> = None;
            for arr in ca.downcast_iter() {
                if arr.len() == arr.null_count() {
                    chunk_start_offset += arr.len();
                    continue;
                }

                let chunk_max: Option<(usize, T::Native)> = if arr.null_count() > 0 {
                    arr.into_iter()
                        .enumerate()
                        .flat_map(|(idx, val)| Some((idx, *(val?))))
                        .reduce(|acc, (idx, val)| {
                            if MaxIgnoreNan::is_better(&val, &acc.1) {
                                (idx, val)
                            } else {
                                acc
                            }
                        })
                } else {
                    // When no nulls & array not empty => we can use fast argmax.
                    let max_idx: usize = arr.values().as_slice().argmax();
                    Some((max_idx, arr.value(max_idx)))
                };

                if let Some((chunk_max_idx, chunk_max_val)) = chunk_max {
                    if max_val.is_none()
                        || MaxIgnoreNan::is_better(&chunk_max_val, &max_val.unwrap())
                    {
                        max_val = Some(chunk_max_val);
                        max_idx = Some(chunk_start_offset + chunk_max_idx);
                    }
                }
                chunk_start_offset += arr.len();
            }
            max_idx
        },
    }
}

fn arg_min_numeric_slice<T>(vals: &[T], is_sorted: IsSorted) -> Option<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    match is_sorted {
        // all vals are not null guarded by cont_slice
        IsSorted::Ascending => Some(0),
        // all vals are not null guarded by cont_slice
        IsSorted::Descending => Some(vals.len() - 1),
        IsSorted::Not => Some(vals.argmin()), // assumes not empty
    }
}

fn arg_max_numeric_slice<T>(vals: &[T], is_sorted: IsSorted) -> Option<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    match is_sorted {
        // all vals are not null guarded by cont_slice
        IsSorted::Ascending => Some(vals.len() - 1),
        // all vals are not null guarded by cont_slice
        IsSorted::Descending => Some(0),
        IsSorted::Not => Some(vals.argmax()), // assumes not empty
    }
}
