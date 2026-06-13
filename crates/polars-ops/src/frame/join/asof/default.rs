use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::Bitmap;
use num_traits::Zero;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_utils::abs_diff::AbsDiff;
use polars_utils::float::IsFloat;
use polars_utils::total_ord::TotalOrd;
use rayon::prelude::*;

use super::{
    AsofJoinBackwardState, AsofJoinForwardState, AsofJoinNearestState, AsofJoinState, AsofStrategy,
};

// Below this, rayon chunk setup can dominate the asof state-machine work.
const PARALLEL_ASOF_MIN_ROWS: usize = 131_072;

fn join_asof_impl<'a, T, S, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    mut filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    S: AsofJoinState<T::Physical<'a>>,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    if left.len() == left.null_count() || right.len() == right.null_count() {
        return IdxCa::full_null(PlSmallStr::EMPTY, left.len());
    }

    let mut out = vec![0; left.len()];
    let mut mask = vec![0; left.len().div_ceil(8)];
    let mut state = S::new(allow_eq);

    if left.null_count() == 0 && right.null_count() == 0 {
        for (i, val_l) in left.values_iter().enumerate() {
            if let Some(r_idx) = state.next(
                &val_l,
                // SAFETY: next() only calls with indices < right.len().
                |j| Some(unsafe { right.value_unchecked(j as usize) }),
                right.len() as IdxSize,
            ) {
                // SAFETY: r_idx is non-null and valid.
                unsafe {
                    let val_r = right.value_unchecked(r_idx as usize);
                    *out.get_unchecked_mut(i) = r_idx;
                    *mask.get_unchecked_mut(i / 8) |= (filter(val_l, val_r) as u8) << (i % 8);
                }
            }
        }
    } else {
        for (i, opt_val_l) in left.iter().enumerate() {
            if let Some(val_l) = opt_val_l {
                if let Some(r_idx) = state.next(
                    &val_l,
                    // SAFETY: next() only calls with indices < right.len().
                    |j| unsafe { right.get_unchecked(j as usize) },
                    right.len() as IdxSize,
                ) {
                    // SAFETY: r_idx is non-null and valid.
                    unsafe {
                        let val_r = right.value_unchecked(r_idx as usize);
                        *out.get_unchecked_mut(i) = r_idx;
                        *mask.get_unchecked_mut(i / 8) |= (filter(val_l, val_r) as u8) << (i % 8);
                    }
                }
            }
        }
    }

    let bitmap = Bitmap::try_new(mask, out.len()).unwrap();
    IdxCa::from_vec_validity(PlSmallStr::EMPTY, out, Some(bitmap))
}

fn join_asof_forward<'a, T, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: TotalOrd,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinForwardState, _>(left, right, filter, allow_eq)
}

fn join_asof_backward<'a, T, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: TotalOrd,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinBackwardState, _>(left, right, filter, allow_eq)
}

fn join_asof_nearest<'a, T, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: NumericNative,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinNearestState, _>(left, right, filter, allow_eq)
}

/// Parallel no-tolerance kernel. Splits the left into roughly
/// `RAYON.current_num_threads()` 8-aligned chunks, pre-computes the
/// per-chunk state via binary search over the right array, and processes
/// the chunks in parallel. Each chunk has its own state and writes to
/// its own slice of `out` / `mask`; no cross-thread synchronization is
/// needed.
fn join_asof_impl_parallel<'a, T, S>(
    left: &'a PrimitiveArray<T::Native>,
    right: &'a PrimitiveArray<T::Native>,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsNumericType,
    S: AsofJoinState<T::Native> + Send + Sync,
{
    if left.len() == left.null_count() || right.len() == right.null_count() {
        return IdxCa::full_null(PlSmallStr::EMPTY, left.len());
    }

    let n = left.len();
    let n_right = right.len() as IdxSize;
    let n_threads = RAYON.current_num_threads().max(1);
    // At least 65K rows per chunk so the per-chunk setup cost (binary
    // search + state construction) is amortized.
    let raw_chunk = (n / n_threads).max(65_536);
    // Round chunk size down to a multiple of 8 so chunks never straddle
    // a mask byte boundary (each thread flushes its own cur_mask register
    // into its slice of `mask` without atomic writes).
    let chunk_size = (raw_chunk / 8) * 8;
    let n_chunks = if chunk_size == 0 {
        1
    } else {
        n.div_ceil(chunk_size)
    };

    let left_slice: &[T::Native] = left.values().as_slice();
    let right_slice: &[T::Native] = right.values().as_slice();

    // Seed states for each chunk. Binary search per chunk is O(log n)
    // per chunk, O(n_chunks * log n) total - negligible.
    let template = S::new(allow_eq);
    let initial_states: Vec<S> = (0..n_chunks)
        .map(|i| {
            let chunk_start = i * chunk_size;
            let mut state = template;
            let (scan_offset, state_bound) =
                state.initial_for_left(right_slice, &left_slice[chunk_start]);
            state.reset_to(scan_offset, state_bound);
            state
        })
        .collect();

    let mut out = vec![IdxSize::default(); n];
    let mut mask = vec![0u8; n.div_ceil(8)];

    // Process chunks in parallel writing directly to disjoint slices of
    // the shared `out` and `mask` buffers. Chunks are 8-aligned and
    // chunk_size is a multiple of 8, so the mask byte ranges are
    // disjoint. We split the mutable references manually because rayon
    // does not provide a way to express "disjoint mutable sub-slices
    // across threads" through safe APIs.
    //
    // SAFETY: chunk `i` writes to out[chunk_start..chunk_end] and to
    // mask[chunk_start/8 .. (chunk_end-1)/8 + 1]. These ranges are
    // disjoint across chunks because chunks are 8-aligned and
    // chunk_size is a multiple of 8.
    let out_ptr = out.as_mut_ptr() as usize;
    let mask_ptr = mask.as_mut_ptr() as usize;
    let left_ptr = left_slice.as_ptr() as usize;
    let right_ptr = right_slice.as_ptr() as usize;

    RAYON.install(|| {
        (0..n_chunks).into_par_iter().for_each(|i| {
            let chunk_start = i * chunk_size;
            let chunk_end = if i + 1 == n_chunks {
                n
            } else {
                (i + 1) * chunk_size
            };
            let chunk_len = chunk_end - chunk_start;
            let mut state = initial_states[i];
            let mut cur_mask: u8 = 0;
            for j in 0..chunk_len {
                // SAFETY: `chunk_start + j < n`, `state.next` only asks for
                // `k < n_right`, and each chunk writes to disjoint output/mask bytes.
                unsafe {
                    let val_l = &*(left_ptr as *const T::Native).add(chunk_start + j);
                    let r_idx_opt = state.next(
                        val_l,
                        |k: IdxSize| Some(*(right_ptr as *const T::Native).add(k as usize)),
                        n_right,
                    );
                    if let Some(r_idx) = r_idx_opt {
                        *(out_ptr as *mut IdxSize).add(chunk_start + j) = r_idx;
                        cur_mask |= 1u8 << (j & 7);
                    }
                }
                if (j & 7) == 7 {
                    // SAFETY: chunk_len rounded up; only the last byte may
                    // be a partial write and we flush it after the loop.
                    unsafe { *(mask_ptr as *mut u8).add((chunk_start / 8) + (j >> 3)) = cur_mask };
                    cur_mask = 0;
                }
            }
            if !chunk_len.is_multiple_of(8) {
                unsafe {
                    *(mask_ptr as *mut u8).add((chunk_start / 8) + ((chunk_len - 1) >> 3)) =
                        cur_mask
                };
            }
        });
    });

    let bitmap = Bitmap::try_new(mask, n).unwrap();
    IdxCa::from_vec_validity(PlSmallStr::EMPTY, out, Some(bitmap))
}

/// Parallel no-tolerance kernel with a filter closure (tolerance path).
///
/// Identical structure to [`join_asof_impl_parallel`] but applies the
/// tolerance filter inside the per-row hot loop. The filter is a
/// `Fn(T::Native, T::Native) -> bool` so it can be monomorphized for
/// each tolerance value and shared across rayon workers.
fn join_asof_impl_parallel_with_filter<'a, T, S, F>(
    left: &'a PrimitiveArray<T::Native>,
    right: &'a PrimitiveArray<T::Native>,
    allow_eq: bool,
    filter: F,
) -> IdxCa
where
    T: PolarsNumericType,
    S: AsofJoinState<T::Native> + Send + Sync,
    F: Fn(T::Native, T::Native) -> bool + Send + Sync,
{
    if left.len() == left.null_count() || right.len() == right.null_count() {
        return IdxCa::full_null(PlSmallStr::EMPTY, left.len());
    }

    let n = left.len();
    let n_right = right.len() as IdxSize;
    let n_threads = RAYON.current_num_threads().max(1);
    let raw_chunk = (n / n_threads).max(65_536);
    let chunk_size = (raw_chunk / 8) * 8;
    let n_chunks = if chunk_size == 0 {
        1
    } else {
        n.div_ceil(chunk_size)
    };

    let left_slice: &[T::Native] = left.values().as_slice();
    let right_slice: &[T::Native] = right.values().as_slice();

    let template = S::new(allow_eq);
    let initial_states: Vec<S> = (0..n_chunks)
        .map(|i| {
            let chunk_start = i * chunk_size;
            let mut state = template;
            let (scan_offset, state_bound) =
                state.initial_for_left(right_slice, &left_slice[chunk_start]);
            state.reset_to(scan_offset, state_bound);
            state
        })
        .collect();

    let mut out = vec![IdxSize::default(); n];
    let mut mask = vec![0u8; n.div_ceil(8)];

    // Same disjoint-write invariant as the no-filter kernel above: chunk `i`
    // writes to out[chunk_start..chunk_end] and to
    // mask[chunk_start/8 .. (chunk_end-1)/8 + 1]. These ranges are disjoint
    // across chunks because chunk starts are 8-aligned and chunk_size is a
    // multiple of 8.
    let out_ptr = out.as_mut_ptr() as usize;
    let mask_ptr = mask.as_mut_ptr() as usize;
    let left_ptr = left_slice.as_ptr() as usize;
    let right_ptr = right_slice.as_ptr() as usize;

    RAYON.install(|| {
        (0..n_chunks).into_par_iter().for_each(|i| {
            let chunk_start = i * chunk_size;
            let chunk_end = if i + 1 == n_chunks {
                n
            } else {
                (i + 1) * chunk_size
            };
            let chunk_len = chunk_end - chunk_start;
            let mut state = initial_states[i];
            let mut cur_mask: u8 = 0;
            for j in 0..chunk_len {
                // SAFETY: `chunk_start + j < n`, `state.next` only asks for
                // `k < n_right`, and each chunk writes to disjoint output/mask bytes.
                unsafe {
                    let val_l = &*(left_ptr as *const T::Native).add(chunk_start + j);
                    let r_idx_opt = state.next(
                        val_l,
                        |k: IdxSize| Some(*(right_ptr as *const T::Native).add(k as usize)),
                        n_right,
                    );
                    if let Some(r_idx) = r_idx_opt {
                        let val_r = *(right_ptr as *const T::Native).add(r_idx as usize);
                        *(out_ptr as *mut IdxSize).add(chunk_start + j) = r_idx;
                        if filter(*val_l, val_r) {
                            cur_mask |= 1u8 << (j & 7);
                        }
                    }
                }
                if (j & 7) == 7 {
                    unsafe { *(mask_ptr as *mut u8).add((chunk_start / 8) + (j >> 3)) = cur_mask };
                    cur_mask = 0;
                }
            }
            if !chunk_len.is_multiple_of(8) {
                unsafe {
                    *(mask_ptr as *mut u8).add((chunk_start / 8) + ((chunk_len - 1) >> 3)) =
                        cur_mask
                };
            }
        });
    });

    let bitmap = Bitmap::try_new(mask, n).unwrap();
    IdxCa::from_vec_validity(PlSmallStr::EMPTY, out, Some(bitmap))
}

pub(crate) fn join_asof_numeric<T: PolarsNumericType>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
    allow_eq: bool,
) -> PolarsResult<IdxCa> {
    let other = input_ca.unpack_series_matching_type(other)?;

    let ca = input_ca.rechunk();
    let other = other.rechunk();
    let left = ca.downcast_as_array();
    let right = other.downcast_as_array();

    // The parallel kernel reads primitive value buffers directly and only pays
    // off once chunk setup overhead is amortized. Float joins stay on the
    // existing serial path so NaN ordering semantics remain unchanged.
    if T::Native::is_float()
        || left.len() < PARALLEL_ASOF_MIN_ROWS
        || left.null_count() > 0
        || right.null_count() > 0
    {
        let out = if let Some(t) = tolerance {
            let native_tolerance = t.try_extract::<T::Native>()?;
            let abs_tolerance = native_tolerance.abs_diff(T::Native::zero());
            let filter = |l: T::Native, r: T::Native| l.abs_diff(r) <= abs_tolerance;
            match strategy {
                AsofStrategy::Forward => join_asof_forward::<T, _>(left, right, filter, allow_eq),
                AsofStrategy::Backward => join_asof_backward::<T, _>(left, right, filter, allow_eq),
                AsofStrategy::Nearest => join_asof_nearest::<T, _>(left, right, filter, allow_eq),
            }
        } else {
            let filter = |_l: T::Native, _r: T::Native| true;
            match strategy {
                AsofStrategy::Forward => join_asof_forward::<T, _>(left, right, filter, allow_eq),
                AsofStrategy::Backward => join_asof_backward::<T, _>(left, right, filter, allow_eq),
                AsofStrategy::Nearest => join_asof_nearest::<T, _>(left, right, filter, allow_eq),
            }
        };
        return Ok(out);
    }

    let out = if let Some(t) = tolerance {
        let native_tolerance = t.try_extract::<T::Native>()?;
        let abs_tolerance = native_tolerance.abs_diff(T::Native::zero());
        let filter = |l: T::Native, r: T::Native| l.abs_diff(r) <= abs_tolerance;
        match strategy {
            AsofStrategy::Forward => {
                join_asof_impl_parallel_with_filter::<T, AsofJoinForwardState, _>(
                    left, right, allow_eq, filter,
                )
            },
            AsofStrategy::Backward => {
                join_asof_impl_parallel_with_filter::<T, AsofJoinBackwardState, _>(
                    left, right, allow_eq, filter,
                )
            },
            AsofStrategy::Nearest => {
                join_asof_impl_parallel_with_filter::<T, AsofJoinNearestState, _>(
                    left, right, allow_eq, filter,
                )
            },
        }
    } else {
        match strategy {
            AsofStrategy::Forward => {
                join_asof_impl_parallel::<T, AsofJoinForwardState>(left, right, allow_eq)
            },
            AsofStrategy::Backward => {
                join_asof_impl_parallel::<T, AsofJoinBackwardState>(left, right, allow_eq)
            },
            AsofStrategy::Nearest => {
                join_asof_impl_parallel::<T, AsofJoinNearestState>(left, right, allow_eq)
            },
        }
    };
    Ok(out)
}

pub(crate) fn join_asof<T>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    allow_eq: bool,
) -> PolarsResult<IdxCa>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: TotalOrd,
{
    let other = input_ca.unpack_series_matching_type(other)?;

    let ca = input_ca.rechunk();
    let other = other.rechunk();
    let left = ca.downcast_iter().next().unwrap();
    let right = other.downcast_iter().next().unwrap();

    let filter = |_l: T::Physical<'_>, _r: T::Physical<'_>| true;
    Ok(match strategy {
        AsofStrategy::Forward => {
            join_asof_impl::<T, AsofJoinForwardState, _>(left, right, filter, allow_eq)
        },
        AsofStrategy::Backward => {
            join_asof_impl::<T, AsofJoinBackwardState, _>(left, right, filter, allow_eq)
        },
        AsofStrategy::Nearest => polars_bail!(InvalidOperation:
            "AsOf strategy \"nearest\" is not supported for {} data type",
            T::get_static_dtype()
        ),
    })
}

#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;

    use super::super::_join_asof_dispatch;
    use super::*;

    fn raw_values(idx: &IdxCa) -> Vec<IdxSize> {
        idx.downcast_iter()
            .flat_map(|chunk| chunk.values_iter().copied())
            .collect()
    }

    fn assert_idx_equal(label: &str, actual: &IdxCa, expected: &IdxCa) {
        assert_eq!(
            actual.to_vec(),
            expected.to_vec(),
            "{label}: logical values"
        );
        assert_eq!(
            raw_values(actual),
            raw_values(expected),
            "{label}: raw values"
        );
    }

    fn dispatch_numeric_asof_i64(
        left_vals: &[i64],
        right_vals: &[i64],
        strategy: AsofStrategy,
        tolerance: Option<i64>,
        allow_eq: bool,
    ) -> IdxCa {
        let left = Series::new("l".into(), left_vals.to_vec());
        let right = Series::new("r".into(), right_vals.to_vec());
        _join_asof_dispatch(
            &left,
            &right,
            strategy,
            tolerance.map(AnyValue::Int64),
            allow_eq,
        )
        .unwrap()
    }

    fn serial_numeric_asof_i64(
        left_vals: &[i64],
        right_vals: &[i64],
        strategy: AsofStrategy,
        tolerance: Option<i64>,
        allow_eq: bool,
    ) -> IdxCa {
        let left = PrimitiveArray::from_slice(left_vals);
        let right = PrimitiveArray::from_slice(right_vals);
        let abs_tolerance = tolerance.map(|t| t.abs_diff(0));
        let filter = |l: i64, r: i64| match abs_tolerance {
            Some(t) => l.abs_diff(r) <= t,
            None => true,
        };

        match strategy {
            AsofStrategy::Forward => {
                join_asof_forward::<Int64Type, _>(&left, &right, filter, allow_eq)
            },
            AsofStrategy::Backward => {
                join_asof_backward::<Int64Type, _>(&left, &right, filter, allow_eq)
            },
            AsofStrategy::Nearest => {
                join_asof_nearest::<Int64Type, _>(&left, &right, filter, allow_eq)
            },
        }
    }

    fn parallel_chunk_starts(n: usize) -> Vec<usize> {
        let n_threads = RAYON.current_num_threads().max(1);
        let raw_chunk = (n / n_threads).max(65_536);
        let chunk_size = (raw_chunk / 8) * 8;
        if chunk_size == 0 {
            return Vec::new();
        }
        (chunk_size..n).step_by(chunk_size).collect()
    }

    #[test]
    fn test_asof_backward() {
        let a = PrimitiveArray::from_slice([-1, 2, 3, 3, 3, 4]);
        let b = PrimitiveArray::from_slice([1, 2, 3, 3]);

        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]
        );

        let b = PrimitiveArray::from_slice([1, 2, 4, 5]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]
        );

        let a = PrimitiveArray::from_slice([2, 4, 4, 4]);
        let b = PrimitiveArray::from_slice([1, 2, 3, 3]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(tuples.to_vec(), &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_backward_tolerance() {
        let a = PrimitiveArray::from_slice([-1, 20, 25, 30, 30, 40]);
        let b = PrimitiveArray::from_slice([10, 20, 30, 30]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |l, r| l.abs_diff(r) <= 4u32, true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(3), Some(3), None]
        );
    }

    #[test]
    fn test_asof_forward_tolerance() {
        let a = PrimitiveArray::from_slice([-1, 20, 25, 30, 30, 40, 52]);
        let b = PrimitiveArray::from_slice([10, 20, 33, 55]);
        let tuples = join_asof_forward::<Int32Type, _>(&a, &b, |l, r| l.abs_diff(r) <= 4u32, true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(2), Some(2), None, Some(3)]
        );
    }

    #[test]
    fn test_asof_forward() {
        let a = PrimitiveArray::from_slice([-1, 1, 2, 4, 6]);
        let b = PrimitiveArray::from_slice([1, 2, 4, 5]);

        let tuples = join_asof_forward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples.to_vec(), &[Some(0), Some(0), Some(1), Some(2), None]);
    }

    #[test]
    fn test_asof_parallel_numeric_matches_serial_across_chunk_boundaries() {
        let n = PARALLEL_ASOF_MIN_ROWS + 9;
        let right_n = n + 257;
        let left_vals = (0..n).map(|i| (i as i64 * 3) / 2).collect::<Vec<_>>();
        let right_vals = (0..right_n).map(|i| (i as i64 / 2) * 2).collect::<Vec<_>>();

        for strategy in [
            AsofStrategy::Backward,
            AsofStrategy::Forward,
            AsofStrategy::Nearest,
        ] {
            for allow_eq in [false, true] {
                for tolerance in [None, Some(0), Some(2)] {
                    let actual = dispatch_numeric_asof_i64(
                        &left_vals,
                        &right_vals,
                        strategy,
                        tolerance,
                        allow_eq,
                    );
                    let expected = serial_numeric_asof_i64(
                        &left_vals,
                        &right_vals,
                        strategy,
                        tolerance,
                        allow_eq,
                    );
                    assert_idx_equal(
                        &format!("{strategy:?} allow_eq={allow_eq} tolerance={tolerance:?}"),
                        &actual,
                        &expected,
                    );
                }
            }
        }
    }

    #[test]
    fn test_asof_parallel_nearest_duplicate_and_tie_boundaries() {
        let n = PARALLEL_ASOF_MIN_ROWS + 17;
        let boundaries = parallel_chunk_starts(n);
        assert!(!boundaries.is_empty());

        let left_vals = (0..n).map(|i| i as i64).collect::<Vec<_>>();
        let mut right_vals = Vec::with_capacity(n + boundaries.len() * 2 + 32);
        for i in 0..(n + 32) {
            let value = i as i64;
            right_vals.push(value);
            if boundaries.contains(&i) {
                right_vals.push(value);
                right_vals.push(value);
            }
        }

        for allow_eq in [false, true] {
            let actual = dispatch_numeric_asof_i64(
                &left_vals,
                &right_vals,
                AsofStrategy::Nearest,
                None,
                allow_eq,
            );
            let expected = serial_numeric_asof_i64(
                &left_vals,
                &right_vals,
                AsofStrategy::Nearest,
                None,
                allow_eq,
            );
            assert_idx_equal(
                &format!("nearest duplicate boundary allow_eq={allow_eq}"),
                &actual,
                &expected,
            );
        }

        let left_vals = (0..n).map(|i| i as i64 * 4 + 2).collect::<Vec<_>>();
        let right_vals = (0..(n + 32)).map(|i| i as i64 * 4).collect::<Vec<_>>();

        for allow_eq in [false, true] {
            for tolerance in [None, Some(2)] {
                let actual = dispatch_numeric_asof_i64(
                    &left_vals,
                    &right_vals,
                    AsofStrategy::Nearest,
                    tolerance,
                    allow_eq,
                );
                let expected = serial_numeric_asof_i64(
                    &left_vals,
                    &right_vals,
                    AsofStrategy::Nearest,
                    tolerance,
                    allow_eq,
                );
                assert_idx_equal(
                    &format!(
                        "nearest equal-distance tie allow_eq={allow_eq} tolerance={tolerance:?}"
                    ),
                    &actual,
                    &expected,
                );
            }
        }
    }

    #[test]
    fn test_asof_parallel_large_left_empty_and_single_right() {
        let n = PARALLEL_ASOF_MIN_ROWS + 9;
        let left_vals = (0..n).map(|i| i as i64).collect::<Vec<_>>();

        for right_vals in [Vec::new(), vec![n as i64 / 2]] {
            for strategy in [
                AsofStrategy::Backward,
                AsofStrategy::Forward,
                AsofStrategy::Nearest,
            ] {
                for allow_eq in [false, true] {
                    for tolerance in [None, Some(0), Some(2)] {
                        let actual = dispatch_numeric_asof_i64(
                            &left_vals,
                            &right_vals,
                            strategy,
                            tolerance,
                            allow_eq,
                        );
                        let expected = serial_numeric_asof_i64(
                            &left_vals,
                            &right_vals,
                            strategy,
                            tolerance,
                            allow_eq,
                        );
                        assert_idx_equal(
                            &format!(
                                "{strategy:?} right_len={} allow_eq={allow_eq} tolerance={tolerance:?}",
                                right_vals.len()
                            ),
                            &actual,
                            &expected,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_asof_parallel_tolerance_preserves_serial_raw_values() {
        let n = PARALLEL_ASOF_MIN_ROWS + 9;
        let left_vals = (0..n).map(|i| i as i64 * 10 + 5).collect::<Vec<_>>();
        let right_vals = (0..n).map(|i| i as i64 * 10).collect::<Vec<_>>();

        for strategy in [
            AsofStrategy::Backward,
            AsofStrategy::Forward,
            AsofStrategy::Nearest,
        ] {
            let actual =
                dispatch_numeric_asof_i64(&left_vals, &right_vals, strategy, Some(0), true);
            let expected =
                serial_numeric_asof_i64(&left_vals, &right_vals, strategy, Some(0), true);
            assert_idx_equal(&format!("{strategy:?}"), &actual, &expected);
        }
    }

    #[test]
    fn test_asof_numeric_with_partial_nulls_keeps_serial_semantics() {
        let left = Int64Chunked::from_iter_options(
            "l".into(),
            [None, Some(0), Some(2), Some(4), Some(6)].into_iter(),
        )
        .into_series();
        let right = Int64Chunked::from_iter_options(
            "r".into(),
            [Some(0), None, Some(4), Some(6)].into_iter(),
        )
        .into_series();

        let out = _join_asof_dispatch(&left, &right, AsofStrategy::Backward, None, true).unwrap();
        assert_eq!(out.to_vec(), &[None, Some(0), Some(0), Some(2), Some(3)]);

        let left = Int64Chunked::from_iter_options(
            "l".into(),
            [Some(1), Some(2), None, Some(6), Some(8)].into_iter(),
        )
        .into_series();
        let right = Int64Chunked::from_iter_options(
            "r".into(),
            [Some(0), None, Some(5), Some(8)].into_iter(),
        )
        .into_series();

        let out = _join_asof_dispatch(
            &left,
            &right,
            AsofStrategy::Nearest,
            Some(AnyValue::Int64(1)),
            true,
        )
        .unwrap();
        assert_eq!(out.to_vec(), &[Some(0), None, None, Some(2), Some(3)]);
    }
}
