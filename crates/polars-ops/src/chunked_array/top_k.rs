use arrow::array::{BinaryViewArray, BooleanArray, PrimitiveArray, StaticArray, View};
use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_core::chunked_array::ops::sort::arg_bottom_k::_arg_bottom_k;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::{downcast_as_macro_arg_physical, POOL};
use polars_utils::total_ord::TotalOrd;

fn first_n_valid_mask(num_valid: usize, out_len: usize) -> Option<Bitmap> {
    if num_valid < out_len {
        let mut bm = MutableBitmap::with_capacity(out_len);
        bm.extend_constant(num_valid, true);
        bm.extend_constant(out_len - num_valid, false);
        Some(bm.freeze())
    } else {
        None
    }
}

fn top_k_bool_impl(
    ca: &ChunkedArray<BooleanType>,
    k: usize,
    descending: bool,
) -> ChunkedArray<BooleanType> {
    if k >= ca.len() && ca.null_count() == 0 {
        return ca.clone();
    }

    let null_count = ca.null_count();
    let non_null_count = ca.len() - ca.null_count();
    let true_count = ca.sum().unwrap() as usize;
    let false_count = non_null_count - true_count;
    let mut out_len = k.min(ca.len());
    let validity = first_n_valid_mask(non_null_count, out_len);

    // Logical sequence of physical bits.
    let sequence = if descending {
        [
            (false_count, false),
            (true_count, true),
            (null_count, false),
        ]
    } else {
        [
            (true_count, true),
            (false_count, false),
            (null_count, false),
        ]
    };

    let mut bm = MutableBitmap::with_capacity(out_len);
    for (n, value) in sequence {
        if out_len == 0 {
            break;
        }
        let extra = out_len.min(n);
        bm.extend_constant(extra, value);
        out_len -= extra;
    }

    let arr = BooleanArray::from_data_default(bm.into(), validity);
    ChunkedArray::with_chunk_like(ca, arr)
}

fn top_k_num_impl<T>(ca: &ChunkedArray<T>, k: usize, descending: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    if k >= ca.len() && ca.null_count() == 0 {
        return ca.clone();
    }

    // Get rid of all the nulls and transform into Vec<T::Native>.
    let nnca = ca.drop_nulls().rechunk();
    let chunk = nnca.downcast_into_iter().next().unwrap();
    let (_, buffer, _) = chunk.into_inner();
    let mut vec = buffer.make_mut();

    // Partition.
    if k < vec.len() {
        if descending {
            vec.select_nth_unstable_by(k, TotalOrd::tot_cmp);
        } else {
            vec.select_nth_unstable_by(k, |a, b| TotalOrd::tot_cmp(b, a));
        }
    }

    // Reconstruct output (with nulls at the end).
    let out_len = k.min(ca.len());
    let non_null_count = ca.len() - ca.null_count();
    vec.resize(out_len, T::Native::default());
    let validity = first_n_valid_mask(non_null_count, out_len);

    let arr = PrimitiveArray::from_vec(vec).with_validity_typed(validity);
    ChunkedArray::with_chunk_like(ca, arr)
}

fn top_k_binary_impl(
    ca: &ChunkedArray<BinaryType>,
    k: usize,
    descending: bool,
) -> ChunkedArray<BinaryType> {
    if k >= ca.len() && ca.null_count() == 0 {
        return ca.clone();
    }

    // Get rid of all the nulls and transform into mutable views.
    let nnca = ca.drop_nulls().rechunk();
    let chunk = nnca.downcast_into_iter().next().unwrap();
    let buffers = chunk.data_buffers().clone();
    let mut views = chunk.into_views();

    // Partition.
    if k < views.len() {
        if descending {
            views.select_nth_unstable_by(k, |a, b| unsafe {
                let a_sl = a.get_slice_unchecked(&buffers);
                let b_sl = b.get_slice_unchecked(&buffers);
                a_sl.cmp(b_sl)
            });
        } else {
            views.select_nth_unstable_by(k, |a, b| unsafe {
                let a_sl = a.get_slice_unchecked(&buffers);
                let b_sl = b.get_slice_unchecked(&buffers);
                b_sl.cmp(a_sl)
            });
        }
    }

    // Reconstruct output (with nulls at the end).
    let out_len = k.min(ca.len());
    let non_null_count = ca.len() - ca.null_count();
    views.resize(out_len, View::default());
    let validity = first_n_valid_mask(non_null_count, out_len);

    let arr = unsafe {
        BinaryViewArray::new_unchecked_unknown_md(
            ArrowDataType::BinaryView,
            views.into(),
            buffers,
            validity,
            None,
        )
    };
    ChunkedArray::with_chunk_like(ca, arr)
}

pub fn top_k(s: &[Series], descending: bool) -> PolarsResult<Series> {
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

    let (k, src) = extract_target_and_k(s)?;

    if src.is_empty() {
        return Ok(src.clone());
    }

    let sorted_flag = src.is_sorted_flag();
    let is_sorted = match src.is_sorted_flag() {
        IsSorted::Ascending => true,
        IsSorted::Descending => true,
        IsSorted::Not => false,
    };
    if is_sorted {
        let out_len = k.min(src.len());
        let ignored_len = src.len() - out_len;

        let slice_at_start = (sorted_flag == IsSorted::Ascending) ^ descending;
        let nulls_at_start = src.get(0).unwrap() == AnyValue::Null;
        let offset = if nulls_at_start == slice_at_start {
            src.null_count().min(ignored_len)
        } else {
            0
        };

        return if slice_at_start {
            Ok(src.slice(offset as i64, out_len))
        } else {
            Ok(src.slice(-(offset as i64) - (out_len as i64), out_len))
        };
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
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => {
            let src = src.decimal().unwrap();
            let ca = top_k_num_impl(src, k, descending);
            let mut lca = DecimalChunked::new_logical(ca);
            lca.2 = Some(DataType::Decimal(src.precision(), Some(src.scale())));
            Ok(lca.into_series())
        },
        DataType::Null => Ok(src.slice(0, k)),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => {
            // Fallback to more generic impl.
            top_k_by_impl(k, src, &[src.clone()], vec![descending])
        },
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

pub fn top_k_by(s: &[Series], descending: Vec<bool>) -> PolarsResult<Series> {
    /// Return (k, src, by)
    fn extract_parameters(s: &[Series]) -> PolarsResult<(usize, &Series, &[Series])> {
        let k_s = &s[1];

        polars_ensure!(
            k_s.len() == 1,
            ComputeError: "`k` must be a single value for `top_k`."
        );

        let Some(k) = k_s.cast(&IDX_DTYPE)?.idx()?.get(0) else {
            polars_bail!(ComputeError: "`k` must be set for `top_k`")
        };

        let src = &s[0];

        let by = &s[2..];

        Ok((k as usize, src, by))
    }

    let (k, src, by) = extract_parameters(s)?;

    if src.is_empty() {
        return Ok(src.clone());
    }

    if by.first().map(|x| x.is_empty()).unwrap_or(false) {
        return Ok(src.clone());
    }

    for s in by {
        if s.len() != src.len() {
            polars_bail!(ComputeError: "`by` column's ({}) length ({}) should have the same length as the source column length ({}) in `top_k`", s.name(), s.len(), src.len())
        }
    }

    top_k_by_impl(k, src, by, descending)
}

fn top_k_by_impl(
    k: usize,
    src: &Series,
    by: &[Series],
    descending: Vec<bool>,
) -> PolarsResult<Series> {
    if src.is_empty() {
        return Ok(src.clone());
    }

    let multithreaded = k >= 10000 && POOL.current_num_threads() > 1;
    let mut sort_options = SortMultipleOptions {
        descending: descending.into_iter().map(|x| !x).collect(),
        nulls_last: vec![true; by.len()],
        multithreaded,
        maintain_order: false,
    };

    let idx = _arg_bottom_k(k, by, &mut sort_options)?;

    let result = unsafe { src.take_unchecked(&idx.into_inner()) };
    Ok(result)
}
