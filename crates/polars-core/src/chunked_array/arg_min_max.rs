use polars_array::{PlArray, StaticArray};
use polars_utils::arg_min_max::ArgMinMax;
use polars_utils::min_max::{MaxIgnoreNan, MinIgnoreNan, MinMaxPolicy};

use crate::chunked_array::ChunkedArray;
use crate::chunked_array::flat::FlatNumericChunkedArray;
use crate::chunked_array::ops::float_sorted_arg_max::{
    float_arg_max_sorted_ascending, float_arg_max_sorted_descending,
};
use crate::datatypes::{
    BinaryChunked, BinaryOffsetChunked, BooleanChunked, PolarsDataType, PolarsNumericType,
    StringChunked,
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
    } else if let Some(vals) = ca.as_flat().and_then(|ca| ca.cont_slice().ok()) {
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
    } else if let Some(vals) = ca.as_flat().and_then(|ca| ca.cont_slice().ok()) {
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
    if ca.physical().scalar_value_ignore_validity().is_some() {
        return ca.physical().first_non_null();
    }
    arg_min_opt_iter(ca.iter_str())
}

#[cfg(feature = "dtype-categorical")]
pub fn arg_max_cat<T: PolarsCategoricalType>(ca: &CategoricalChunked<T>) -> Option<usize> {
    if ca.null_count() == ca.len() {
        return None;
    }
    if ca.physical().scalar_value_ignore_validity().is_some() {
        return ca.physical().last_non_null();
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

pub fn arg_min_binary_offset(ca: &BinaryOffsetChunked) -> Option<usize> {
    arg_min_physical_generic(ca)
}

pub fn arg_max_binary_offset(ca: &BinaryOffsetChunked) -> Option<usize> {
    arg_max_physical_generic(ca)
}

/// The index of the element no other one is `better` than, walking a chunk at a time.
///
/// `better` decides ties, and so which of several equal extremes the index names: `min` keeps
/// the first of them and `max` the last, as [`Iterator::min_by`] and [`Iterator::max_by`] do.
fn arg_extreme_physical_generic<'a, T, F>(ca: &'a ChunkedArray<T>, better: F) -> Option<usize>
where
    T: PolarsDataType,
    F: Fn(&T::Physical<'a>, &T::Physical<'a>) -> bool,
{
    // Threading the index through the fold as part of its accumulator is what
    // `.enumerate().flat_map(..).max_by(..)` does, and it makes the accumulator wide enough to
    // spill: the comparison then reads the best value back out of memory on every element it
    // keeps.  Holding the best outside the walk leaves the fold nothing to carry, so the whole
    // chain stays in registers -- and `for_each` still folds, so the chunk's representation is
    // hoisted out of the loop the way [`Iterator::fold`] hoists it.
    let mut best: Option<(usize, T::Physical<'a>)> = None;
    let mut offset = 0;

    for arr in ca.downcast_iter() {
        let mut i = 0;
        arr.iter().for_each(|value| {
            let idx = offset + i;
            i += 1;

            if let Some(value) = value {
                match &best {
                    Some((_, best_value)) if !better(&value, best_value) => {},
                    _ => best = Some((idx, value)),
                }
            }
        });
        offset += arr.len();
    }

    best.map(|(idx, _)| idx)
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
        // A later element that merely ties is not smaller, so the first minimum wins.
        IsSorted::Not => arg_extreme_physical_generic(ca, |value, best| value < best),
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
        // A later element that ties is taken, so the last maximum wins.
        IsSorted::Not => arg_extreme_physical_generic(ca, |value, best| value >= best),
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
            if ca.scalar_value_ignore_validity().is_some() {
                return ca.first_non_null();
            }

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
                        .flat_map(|(idx, val)| Some((idx, val?)))
                        .reduce(|acc, (idx, val)| {
                            if MinIgnoreNan::is_better(&val, &acc.1) {
                                (idx, val)
                            } else {
                                acc
                            }
                        })
                } else {
                    // When no nulls & array not empty => we can use fast argmin.
                    let values = arr.to_flat_values();
                    let min_idx: usize = values.as_slice().argmin();
                    Some((min_idx, values[min_idx]))
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
            if ca.scalar_value_ignore_validity().is_some() {
                return ca.first_non_null();
            }

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
                        .flat_map(|(idx, val)| Some((idx, val?)))
                        .reduce(|acc, (idx, val)| {
                            if MaxIgnoreNan::is_better(&val, &acc.1) {
                                (idx, val)
                            } else {
                                acc
                            }
                        })
                } else {
                    // When no nulls & array not empty => we can use fast argmax.
                    let values = arr.to_flat_values();
                    let max_idx: usize = values.as_slice().argmax();
                    Some((max_idx, values[max_idx]))
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
