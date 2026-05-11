use std::ops::{Add, Div, Mul, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::{Bitmap, MutableBitmap};
use bytemuck::allocation::zeroed_vec;
use num_traits::{NumCast, Zero};
use polars_core::prelude::*;
use polars_utils::slice::SliceAble;

use super::linear_itp;

/// # Safety
/// - `x` must be non-empty.
#[inline]
unsafe fn signed_interp_by_sorted<T, F>(y_start: T, y_end: T, x: &[F], out: &mut Vec<T>)
where
    T: Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + Copy
        + Zero,
    F: Sub<Output = F> + NumCast + Copy,
{
    let range_y = y_end - y_start;
    let x_start;
    let range_x;
    let iter;
    unsafe {
        x_start = x.get_unchecked(0);
        range_x = NumCast::from(*x.get_unchecked(x.len() - 1) - *x_start).unwrap();
        iter = x.slice_unchecked(1..x.len() - 1).iter();
    }
    let slope = range_y / range_x;
    for x_i in iter {
        let x_delta = NumCast::from(*x_i - *x_start).unwrap();
        let v = linear_itp(y_start, x_delta, slope);
        out.push(v)
    }
}

/// # Safety
/// - `x` must be non-empty.
/// - `sorting_indices` must be the same size as `x`
#[inline]
unsafe fn signed_interp_by<T, F>(
    y_start: T,
    y_end: T,
    x: &[F],
    out: &mut [T],
    sorting_indices: &[IdxSize],
) where
    T: Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + Copy
        + Zero,
    F: Sub<Output = F> + NumCast + Copy,
{
    let range_y = y_end - y_start;
    let x_start;
    let range_x;
    let iter;
    unsafe {
        x_start = x.get_unchecked(0);
        range_x = NumCast::from(*x.get_unchecked(x.len() - 1) - *x_start).unwrap();
        iter = x.slice_unchecked(1..x.len() - 1).iter();
    }
    let slope = range_y / range_x;
    for (idx, x_i) in iter.enumerate() {
        let x_delta = NumCast::from(*x_i - *x_start).unwrap();
        let v = linear_itp(y_start, x_delta, slope);
        unsafe {
            let out_idx = sorting_indices.get_unchecked(idx + 1);
            *out.get_unchecked_mut(*out_idx as usize) = v;
        }
    }
}

fn interpolate_impl_by_sorted<T, F, I>(
    chunked_arr: &ChunkedArray<T>,
    by: &ChunkedArray<F>,
    interpolation_branch: I,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    F: PolarsNumericType,
    I: Fn(T::Native, T::Native, &[F::Native], &mut Vec<T::Native>),
{
    // Early exits identical to original semantics.
    if !chunked_arr.has_nulls() || chunked_arr.null_count() == chunked_arr.len() {
        return Ok(chunked_arr.clone());
    }

    let by = by.rechunk();

    // ----- Access raw values + validity without copies -----
    // SAFETY: rechunk() ensures a single chunk.
    let by_arr = by.downcast_iter().next().unwrap();
    let by_values: &[F::Native] = by_arr.values();
    // None => all-valid
    let by_valid = by_arr.validity();

    let is_by_valid = |idx: usize| -> bool {
        match by_valid {
            None => true,
            Some(valid) => unsafe { valid.get_bit_unchecked(idx) },
        }
    };

    // ----- Find first/last non-null y as before -----
    let first = chunked_arr.first_non_null().unwrap();
    let last = chunked_arr.last_non_null().unwrap() + 1;

    // We'll build both the values and the validity explicitly.
    let n = chunked_arr.len();
    let mut out = Vec::with_capacity(n);
    // Start pessimistic (all invalid), flip to true as we push "real" values.
    let mut validity = MutableBitmap::with_capacity(n);
    validity.extend_constant(n, false);

    // Pre-fill the leading nulls
    for i in 0..first {
        out.push(Zero::zero());
        unsafe { validity.set_unchecked(i, false) };
    }

    // Iterator positioned at first non-null y
    let mut iter = chunked_arr.iter().enumerate().skip(first);
    let (mut low_idx, opt_low) = iter.next().unwrap();
    let mut low = opt_low.unwrap();
    out.push(low);
    unsafe { validity.set_unchecked(low_idx, true) };

    // Reusable scratch buffer for the x's we pass to interpolation_branch.
    // Note: we DO NOT clone/filter the whole `by`, we just push the few values we need per gap.
    let mut scratch_x: Vec<F::Native> = Vec::new();

    // Helper that, given low/high anchors and index bounds, pushes interpolations
    // while skipping indices where `by` is null.
    let mut interpolate_gap = |low_idx: usize,
                               low: T::Native,
                               high_idx: usize,
                               high: T::Native,
                               out: &mut Vec<T::Native>,
                               validity: &mut MutableBitmap| {
        // (low_idx, high_idx) interior range
        let start = low_idx + 1;
        let end = high_idx.saturating_sub(1);

        if start > end {
            return; // no interior points
        }

        // We’ll walk interior indices and handle contiguous runs where by is valid.
        let mut i = start;
        while i <= end {
            // 1) Emit placeholders for any leading invalid-by indices (kept null).
            while i <= end && !is_by_valid(i) {
                out.push(Zero::zero());
                // leave validity[i] = false
                i += 1;
            }
            if i > end {
                break;
            }

            // 2) Find the contiguous run [s..e] where by is valid
            let s = i;
            while i <= end && is_by_valid(i) {
                i += 1;
            }
            let e = i - 1;

            // 3) Feed exactly the x's for this run into interpolation_branch.
            //    The branch expects a slice that includes the interior x's; it gets
            //    the y-anchors (low/high) separately as args.
            //    To preserve the original math (which used [low..=high]),
            //    we include both endpoints too: [x_low, x_s..=x_e, x_high].
            scratch_x.clear();
            // x_low
            debug_assert!(
                is_by_valid(low_idx),
                "We only interpolate if anchors have non-null `by`"
            );
            scratch_x.push(unsafe { *by_values.get_unchecked(low_idx) });
            // interior x's (valid-by only)
            scratch_x.extend_from_slice(&by_values[s..=e]);
            // x_high
            debug_assert!(
                is_by_valid(high_idx),
                "We only interpolate if anchors have non-null `by`"
            );
            scratch_x.push(unsafe { *by_values.get_unchecked(high_idx) });

            // Before calling, the output currently has values up to position (current gap start - 1).
            // interpolation_branch appends exactly (scratch_x.len() - 2) values in-order,
            // one per interior x.
            interpolation_branch(low, high, &scratch_x, out);

            // 4) Mark validity=true for the run we just filled.
            //    The branch appended exactly (e - s + 1) values.
            for idx in s..=e {
                unsafe { validity.set_unchecked(idx, true) };
            }
            // Loop continues; next iteration will either push placeholders for invalid-by
            // or run another valid-by block.
        }
    };

    // Walk forward exactly like the original loop, but only interpolate a gap
    // if BOTH anchors have non-null `by`. If not, we leave the whole gap null
    // (placeholders) to match “as-if filtered out” semantics.
    for (idx, next) in iter {
        if let Some(v) = next {
            // If both anchors have valid-by, interpolate the preceding gap.
            // Else, leave the preceding gap as nulls (already handled below).
            if is_by_valid(low_idx) && is_by_valid(idx) {
                // We *haven’t* emitted placeholders for the gap yet; do that here
                // interleaved with interpolation in interpolate_gap.
                interpolate_gap(low_idx, low, idx, v, &mut out, &mut validity);
            } else {
                // Fill the entire interior with placeholders (remain null)
                for _ in (low_idx + 1)..idx {
                    out.push(Zero::zero());
                    // validity[j] stays false
                }
            }
            out.push(v);
            unsafe { validity.set_unchecked(idx, true) };
            // Advance anchor
            low = v;
            low_idx = idx;
        } else {
            // Do nothing now; we only append once we see the high anchor,
            // keeping behavior identical to original in terms of push order.
            continue;
        }
    }

    // Trailing nulls behavior identical to original, plus mask.
    if first != 0 || last != n {
        for i in last..n {
            out.push(Zero::zero());
            unsafe { validity.set_unchecked(i, false) };
        }
        let array = PrimitiveArray::new(
            T::get_static_dtype().to_arrow(CompatLevel::newest()),
            out.into(),
            Some(validity.into()),
        );
        Ok(ChunkedArray::with_chunk(chunked_arr.name().clone(), array))
    } else {
        // All interior handled; just build with explicit validity.
        let array = PrimitiveArray::new(
            T::get_static_dtype().to_arrow(CompatLevel::newest()),
            out.into(),
            Some(validity.into()),
        );
        Ok(ChunkedArray::with_chunk(chunked_arr.name().clone(), array))
    }
}

fn interpolate_impl_by<T, F, I>(
    ca: &ChunkedArray<T>,
    by: &ChunkedArray<F>,
    interpolation_branch: I,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    F: PolarsNumericType,
    I: Fn(T::Native, T::Native, &[F::Native], &mut [T::Native], &[IdxSize]),
{
    // Early exit (same semantics).
    if !ca.has_nulls() || ca.null_count() == ca.len() {
        return Ok(ca.clone());
    }

    // Sort "by" and gather both columns.
    let sorting_indices = by.arg_sort(Default::default());
    let sorting_indices = sorting_indices
        .cont_slice()
        .expect("arg_sort produces single chunk");

    // SAFETY: gather without bounds checks; preserves nulls.
    let by_sorted = unsafe { by.take_unchecked(sorting_indices) };
    let ca_sorted = unsafe { ca.take_unchecked(sorting_indices) };

    // ----- Access raw values + validity (no cont_slice on possibly-null array) -----
    let by_arr = by_sorted.downcast_iter().next().unwrap(); // single chunk after take
    let by_vals: &[F::Native] = by_arr.values();
    let by_valid_opt = by_arr.validity();

    #[inline(always)]
    fn is_valid(valid: Option<&Bitmap>, idx: usize) -> bool {
        match valid {
            None => true,
            Some(bm) => unsafe { bm.get_bit_unchecked(idx) },
        }
    }

    // Find first/last non-null y in the *sorted* y.
    let first = ca_sorted.first_non_null().unwrap();
    //let last = ca_sorted.last_non_null().unwrap() + 1;

    let n = ca_sorted.len();
    // Pre-allocate output (all zeros) and validity (all false).
    let mut out = zeroed_vec::<T::Native>(n);
    let mut validity = MutableBitmap::with_capacity(n);
    validity.extend_constant(n, false);

    // Iterator positioned at first non-null y (sorted order).
    let mut iter = ca_sorted.iter().enumerate().skip(first);
    let (mut low_idx, opt_low) = iter.next().unwrap();
    let mut low = opt_low.unwrap();

    // Map the sorted index to original position via sorting_indices
    unsafe {
        let out_idx = *sorting_indices.get_unchecked(low_idx) as usize;
        *out.get_unchecked_mut(out_idx) = low;
        validity.set_unchecked(out_idx, true);
    }

    // Scratch buffers reused per valid-by run inside a gap.
    let mut scratch_x: Vec<F::Native> = Vec::new();
    let mut scratch_idx: Vec<IdxSize> = Vec::new();

    // Helper: interpolate a *subrun* [s..=e] of interior indices whose `by` is valid,
    // using anchors at low_idx/high_idx. Writes directly into `out` at mapped positions.
    let mut interp_subrun = |low_idx: usize,
                             low: T::Native,
                             s: usize,
                             e: usize,
                             high_idx: usize,
                             high: T::Native,
                             out: &mut [T::Native],
                             validity: &mut MutableBitmap| {
        // Build x and index slices: [x_low, x_s..=x_e, x_high] / [idx_low, idx_s..=idx_e, idx_high]
        scratch_x.clear();
        scratch_idx.clear();

        // Safety: we only call this when anchors are valid-by.
        unsafe {
            scratch_x.push(*by_vals.get_unchecked(low_idx));
            scratch_idx.push(*sorting_indices.get_unchecked(low_idx));
            // interior chunk
            scratch_x.extend_from_slice(&by_vals[s..=e]);
            scratch_idx.extend_from_slice(&sorting_indices[s..=e]);
            // right anchor
            scratch_x.push(*by_vals.get_unchecked(high_idx));
            scratch_idx.push(*sorting_indices.get_unchecked(high_idx));
        }

        // Interpolate into `out` at the positions given by `scratch_idx`.
        interpolation_branch(low, high, &scratch_x, out, &scratch_idx);

        // Mark interior indices (original positions) as valid.
        for k in s..=e {
            unsafe {
                let out_i = *sorting_indices.get_unchecked(k) as usize;
                validity.set_unchecked(out_i, true);
            }
        }
    };

    // Main walk over sorted y, identical structure, with null-aware "by" handling.
    for (idx, next) in iter {
        if let Some(v) = next {
            // If both anchors have non-null `by`, interpolate the interior.
            let anchors_ok: bool = is_valid(by_valid_opt, low_idx) && is_valid(by_valid_opt, idx);
            if anchors_ok {
                let start = low_idx + 1;
                let end = idx.saturating_sub(1);
                if start <= end {
                    // Scan interior into valid-by subruns.
                    let mut i = start;
                    while i <= end {
                        // skip invalid-by positions
                        while i <= end && !is_valid(by_valid_opt, i) {
                            i += 1;
                        }
                        if i > end {
                            break;
                        }
                        // contiguous valid-by run [s..=e]
                        let s = i;
                        while i <= end && is_valid(by_valid_opt, i) {
                            i += 1;
                        }
                        let e = i - 1;

                        // Interpolate this subrun.
                        interp_subrun(low_idx, low, s, e, idx, v, &mut out, &mut validity);
                    }
                }
            }
            // Write the high anchor and mark valid (always keep observed y’s).
            unsafe {
                let out_idx = *sorting_indices.get_unchecked(idx) as usize;
                *out.get_unchecked_mut(out_idx) = v;
                validity.set_unchecked(out_idx, true);
            }
            // Advance anchor
            low = v;
            low_idx = idx;
        } else {
            // keep scanning until we find the next high anchor
            continue;
        }
    }

    // Leading/trailing y-nulls in the sorted order should remain null.
    // They were never marked valid; ensure mask reflects that (it already does).
    // Build array with explicit validity.
    let array = PrimitiveArray::new(
        T::get_static_dtype().to_arrow(CompatLevel::newest()),
        out.into(),
        Some(validity.into()),
    );
    Ok(ChunkedArray::with_chunk(ca_sorted.name().clone(), array))
}

pub fn interpolate_by(s: &Column, by: &Column, by_is_sorted: bool) -> PolarsResult<Column> {
    polars_ensure!(s.len() == by.len(), InvalidOperation: "`by` column must be the same length as Series ({}), got {}", s.len(), by.len());

    fn func<T, F>(
        ca: &ChunkedArray<T>,
        by: &ChunkedArray<F>,
        is_sorted: bool,
    ) -> PolarsResult<Column>
    where
        T: PolarsNumericType,
        F: PolarsNumericType,
        ChunkedArray<T>: IntoColumn,
    {
        if is_sorted {
            interpolate_impl_by_sorted(ca, by, |y_start, y_end, x, out| unsafe {
                signed_interp_by_sorted(y_start, y_end, x, out)
            })
            .map(|x| x.into_column())
        } else {
            interpolate_impl_by(ca, by, |y_start, y_end, x, out, sorting_indices| unsafe {
                signed_interp_by(y_start, y_end, x, out, sorting_indices)
            })
            .map(|x| x.into_column())
        }
    }

    match (s.dtype(), by.dtype()) {
        (DataType::Float64, DataType::Float64) => {
            func(s.f64().unwrap(), by.f64().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::Float32) => {
            func(s.f64().unwrap(), by.f32().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Float64) => {
            func(s.f32().unwrap(), by.f64().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Float32) => {
            func(s.f32().unwrap(), by.f32().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::Int64) => {
            func(s.f64().unwrap(), by.i64().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::Int32) => {
            func(s.f64().unwrap(), by.i32().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::UInt64) => {
            func(s.f64().unwrap(), by.u64().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::UInt32) => {
            func(s.f64().unwrap(), by.u32().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Int64) => {
            func(s.f32().unwrap(), by.i64().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Int32) => {
            func(s.f32().unwrap(), by.i32().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::UInt64) => {
            func(s.f32().unwrap(), by.u64().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::UInt32) => {
            func(s.f32().unwrap(), by.u32().unwrap(), by_is_sorted)
        },
        #[cfg(feature = "dtype-date")]
        (_, DataType::Date) => interpolate_by(s, &by.cast(&DataType::Int32).unwrap(), by_is_sorted),
        #[cfg(feature = "dtype-datetime")]
        (_, DataType::Datetime(_, _)) => {
            interpolate_by(s, &by.cast(&DataType::Int64).unwrap(), by_is_sorted)
        },
        (DataType::UInt64 | DataType::UInt32 | DataType::Int64 | DataType::Int32, _) => {
            interpolate_by(&s.cast(&DataType::Float64).unwrap(), by, by_is_sorted)
        },
        _ => {
            polars_bail!(InvalidOperation: "expected series to be Float64, Float32, \
                Int64, Int32, UInt64, UInt32, and `by` to be Date, Datetime, Int64, Int32, \
                UInt64, UInt32, Float32 or Float64")
        },
    }
}
