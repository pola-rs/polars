use std::cmp::Ordering;

use arrow::array::{BooleanArray, MutableBooleanArray};
use arrow::bitmap::MutableBitmap;
use either::Either;
use polars_core::chunked_array::ops::sort::arg_bottom_k::_arg_bottom_k;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::total_ord::TotalOrd;

fn arg_partition<T, C: Fn(&T, &T) -> Ordering>(
    v: &mut [T],
    k: usize,
    descending: bool,
    cmp: C,
) -> &[T] {
    let (lower, _el, upper) = v.select_nth_unstable_by(k, &cmp);
    if descending {
        lower.sort_unstable_by(cmp);
        lower
    } else {
        upper.sort_unstable_by(|a, b| cmp(b, a));
        upper
    }
}

fn extract_target_and_k(s: &[Series]) -> PolarsResult<(usize, &Series)> {
    let k_s = &s[1];

    polars_ensure!(
        k_s.len() == 1,
        ComputeError: "`k` must be a single value for `top_k`."
    );

    let Some(k) = k_s.cast(&IDX_DTYPE)?.idx()?.get(0) else {
        polars_bail!(ComputeError: "`k` must be set for `top_k`")
    };

    let src = &s[0];

    Ok((k as usize, src))
}

fn top_k_num_impl<T>(ca: &ChunkedArray<T>, k: usize, descending: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkSort<T>,
{
    if k >= ca.len() {
        return ca.sort(!descending);
    }

    // descending is opposite from sort as top-k returns largest
    let k = if descending {
        std::cmp::min(k, ca.len())
    } else {
        ca.len().saturating_sub(k + 1)
    };

    match ca.to_vec_null_aware() {
        Either::Left(mut v) => {
            let values = arg_partition(&mut v, k, descending, TotalOrd::tot_cmp);
            ChunkedArray::from_slice(ca.name(), values)
        },
        Either::Right(mut v) => {
            let values = arg_partition(&mut v, k, descending, TotalOrd::tot_cmp);
            let mut out = ChunkedArray::from_iter(values.iter().copied());
            out.rename(ca.name());
            out
        },
    }
}

fn top_k_bool_impl(
    ca: &ChunkedArray<BooleanType>,
    k: usize,
    descending: bool,
) -> ChunkedArray<BooleanType> {
    if ca.null_count() == 0 {
        let true_count = ca.sum().unwrap() as usize;
        let mut bitmap = MutableBitmap::with_capacity(k);
        if !descending {
            // true first
            bitmap.extend_constant(std::cmp::min(k, true_count), true);
            bitmap.extend_constant(k.saturating_sub(true_count), false);
        } else {
            let false_count = ca.len().saturating_sub(true_count);
            bitmap.extend_constant(std::cmp::min(k, false_count), false);
            bitmap.extend_constant(k.saturating_sub(false_count), true);
        }
        let arr = BooleanArray::from_data_default(bitmap.into(), None);
        unsafe {
            ChunkedArray::from_chunks_and_dtype(ca.name(), vec![Box::new(arr)], DataType::Boolean)
        }
    } else {
        let null_count = ca.null_count();
        let true_count = ca.sum().unwrap() as usize;
        let false_count = ca.len() - true_count - null_count;
        let mut remaining = k;

        fn extend_constant_check_remaining(
            array: &mut MutableBooleanArray,
            remaining: &mut usize,
            additional: usize,
            value: Option<bool>,
        ) {
            array.extend_constant(std::cmp::min(additional, *remaining), value);
            *remaining = remaining.saturating_sub(additional);
        }

        let mut array = MutableBooleanArray::with_capacity(k);
        if !descending {
            // Null -> True -> False
            extend_constant_check_remaining(&mut array, &mut remaining, null_count, None);
            extend_constant_check_remaining(&mut array, &mut remaining, true_count, Some(true));
            extend_constant_check_remaining(&mut array, &mut remaining, false_count, Some(false));
        } else {
            // False -> True -> Null
            extend_constant_check_remaining(&mut array, &mut remaining, false_count, Some(false));
            extend_constant_check_remaining(&mut array, &mut remaining, true_count, Some(true));
            extend_constant_check_remaining(&mut array, &mut remaining, null_count, None);
        }
        let mut new_ca: ChunkedArray<BooleanType> = BooleanArray::from(array).into();
        new_ca.rename(ca.name());
        new_ca
    }
}

fn top_k_binary_impl(
    ca: &ChunkedArray<BinaryType>,
    k: usize,
    descending: bool,
) -> ChunkedArray<BinaryType> {
    if k >= ca.len() {
        return ca.sort(!descending);
    }

    // descending is opposite from sort as top-k returns largest
    let k = if descending {
        std::cmp::min(k, ca.len())
    } else {
        ca.len().saturating_sub(k + 1)
    };

    if ca.null_count() == 0 {
        let mut v: Vec<&[u8]> = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            v.extend(arr.non_null_values_iter());
        }
        let values = arg_partition(&mut v, k, descending, TotalOrd::tot_cmp);
        ChunkedArray::from_slice(ca.name(), values)
    } else {
        let mut v = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            v.extend(arr.iter());
        }
        let values = arg_partition(&mut v, k, descending, TotalOrd::tot_cmp);
        let mut out = ChunkedArray::from_iter(values.iter().copied());
        out.rename(ca.name());
        out
    }
}

pub fn top_k(s: &[Series], descending: bool) -> PolarsResult<Series> {
    let (k, src) = extract_target_and_k(s)?;

    if src.is_empty() {
        return Ok(src.clone());
    }

    match src.is_sorted_flag() {
        polars_core::series::IsSorted::Ascending => {
            // TopK is the k element in the bottom of ascending sorted array
            return Ok(src.slice((src.len() - k) as i64, k).reverse());
        },
        polars_core::series::IsSorted::Descending => {
            return Ok(src.slice(0, k));
        },
        _ => {},
    }

    let origin_dtype = src.dtype();

    let s = src.to_physical_repr();

    match s.dtype() {
        DataType::Boolean => Ok(top_k_bool_impl(s.bool().unwrap(), k, descending).into_series()),
        DataType::String => {
            let ca = top_k_binary_impl(&s.str().unwrap().as_binary(), k, descending);
            let ca = unsafe { ca.to_string_unchecked() };
            Ok(ca.into_series())
        },
        DataType::Binary => Ok(top_k_binary_impl(s.binary().unwrap(), k, descending).into_series()),
        _dt => {
            macro_rules! dispatch {
                ($ca:expr) => {{
                    top_k_num_impl($ca, k, descending).into_series()
                }};
            }
            unsafe { downcast_as_macro_arg_physical!(&s, dispatch).cast_unchecked(origin_dtype) }
        },
    }
}

pub fn top_k_by(
    s: &[Series],
    by: &[Series],
    sort_options: SortMultipleOptions,
) -> PolarsResult<Series> {
    let (k, src) = extract_target_and_k(s)?;

    if src.is_empty() {
        return Ok(src.clone());
    }

    let multithreaded = sort_options.multithreaded;

    let idx = _arg_bottom_k(k, by, &mut sort_options.with_order_reversed())?;

    let result = unsafe {
        if multithreaded {
            src.take_unchecked_threaded(&idx.into_inner(), false)
        } else {
            src.take_unchecked(&idx.into_inner())
        }
    };
    Ok(result)
}
