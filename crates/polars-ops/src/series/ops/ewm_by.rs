use arrow::compute::concatenate::concatenate_validities;
use arrow::compute::utils::combine_validities_and;
use bytemuck::allocation::zeroed_vec;
use num_traits::{Float, FromPrimitive, One, Zero};
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

pub fn ewm_mean_by(
    s: &Series,
    times: &Series,
    half_life: i64,
    assume_sorted: bool,
) -> PolarsResult<Series> {
    match (s.dtype(), times.dtype()) {
        (DataType::Float64, DataType::Int64) => Ok((if assume_sorted {
            ewm_mean_by_impl_sorted(s.f64().unwrap(), times.i64().unwrap(), half_life)
        } else {
            ewm_mean_by_impl(s.f64().unwrap(), times.i64().unwrap(), half_life)
        })
        .into_series()),
        (DataType::Float32, DataType::Int64) => Ok((if assume_sorted {
            ewm_mean_by_impl_sorted(s.f32().unwrap(), times.i64().unwrap(), half_life)
        } else {
            ewm_mean_by_impl(s.f32().unwrap(), times.i64().unwrap(), half_life)
        })
        .into_series()),
        #[cfg(feature = "dtype-datetime")]
        (_, DataType::Datetime(time_unit, _)) => {
            let half_life = adjust_half_life_to_time_unit(half_life, time_unit);
            ewm_mean_by(s, &times.cast(&DataType::Int64)?, half_life, assume_sorted)
        },
        #[cfg(feature = "dtype-date")]
        (_, DataType::Date) => ewm_mean_by(
            s,
            &times.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
            half_life,
            assume_sorted,
        ),
        (_, DataType::UInt64 | DataType::UInt32 | DataType::Int32) => {
            ewm_mean_by(s, &times.cast(&DataType::Int64)?, half_life, assume_sorted)
        },
        (DataType::UInt64 | DataType::UInt32 | DataType::Int64 | DataType::Int32, _) => {
            ewm_mean_by(
                &s.cast(&DataType::Float64)?,
                times,
                half_life,
                assume_sorted,
            )
        },
        _ => {
            polars_bail!(InvalidOperation: "expected series to be Float64, Float32, \
                Int64, Int32, UInt64, UInt32, and `by` to be Date, Datetime, Int64, Int32, \
                UInt64, or UInt32")
        },
    }
}

/// Sort on behalf of user
fn ewm_mean_by_impl<T>(
    values: &ChunkedArray<T>,
    times: &Int64Chunked,
    half_life: i64,
) -> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float + Zero + One,
    ChunkedArray<T>: ChunkTakeUnchecked<IdxCa>,
{
    let sorting_indices = times.arg_sort(Default::default());
    let sorted_values = unsafe { values.take_unchecked(&sorting_indices) };
    let sorted_times = unsafe { times.take_unchecked(&sorting_indices) };
    let sorting_indices = sorting_indices
        .cont_slice()
        .expect("`arg_sort` should have returned a single chunk");

    let mut out: Vec<_> = zeroed_vec(sorted_times.len());

    let mut skip_rows: usize = 0;
    let mut prev_time: i64 = 0;
    let mut prev_result = T::Native::zero();
    for (idx, (value, time)) in sorted_values.iter().zip(sorted_times.iter()).enumerate() {
        if let (Some(time), Some(value)) = (time, value) {
            prev_time = time;
            prev_result = value;
            unsafe {
                let out_idx = sorting_indices.get_unchecked(idx);
                *out.get_unchecked_mut(*out_idx as usize) = prev_result;
            }
            skip_rows = idx + 1;
            break;
        };
    }
    sorted_values
        .iter()
        .zip(sorted_times.iter())
        .enumerate()
        .skip(skip_rows)
        .for_each(|(idx, (value, time))| {
            if let (Some(time), Some(value)) = (time, value) {
                let result = update(value, prev_result, time, prev_time, half_life);
                prev_time = time;
                prev_result = result;
                unsafe {
                    let out_idx = sorting_indices.get_unchecked(idx);
                    *out.get_unchecked_mut(*out_idx as usize) = result;
                }
            };
        });
    let mut arr = T::Array::from_zeroable_vec(out, values.dtype().to_arrow(true));
    if (times.null_count() > 0) || (values.null_count() > 0) {
        let (times, values) = align_chunks_binary(times, values);
        let times_chunk_refs: Vec<_> = times.chunks().iter().map(|c| &**c).collect();
        let times_validity = concatenate_validities(&times_chunk_refs);
        let values_chunk_refs: Vec<_> = values.chunks().iter().map(|c| &**c).collect();
        let values_validity = concatenate_validities(&values_chunk_refs);
        let validity = combine_validities_and(times_validity.as_ref(), values_validity.as_ref());
        arr = arr.with_validity_typed(validity);
    }
    ChunkedArray::with_chunk(values.name(), arr)
}

/// Fastpath if `times` is known to already be sorted.
fn ewm_mean_by_impl_sorted<T>(
    values: &ChunkedArray<T>,
    times: &Int64Chunked,
    half_life: i64,
) -> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float + Zero + One,
{
    let mut out = Vec::with_capacity(times.len());

    let mut skip_rows: usize = 0;
    let mut prev_time: i64 = 0;
    let mut prev_result = T::Native::zero();
    for (idx, (value, time)) in values.iter().zip(times.iter()).enumerate() {
        if let (Some(time), Some(value)) = (time, value) {
            prev_time = time;
            prev_result = value;
            out.push(Some(prev_result));
            skip_rows = idx + 1;
            break;
        } else {
            out.push(None)
        }
    }
    values
        .iter()
        .zip(times.iter())
        .skip(skip_rows)
        .for_each(|(value, time)| {
            let result_opt = match (time, value) {
                (Some(time), Some(value)) => {
                    let result = update(value, prev_result, time, prev_time, half_life);
                    prev_time = time;
                    prev_result = result;
                    Some(result)
                },
                _ => None,
            };
            out.push(result_opt);
        });
    ChunkedArray::<T>::from_iter_options(values.name(), out.into_iter())
}

fn adjust_half_life_to_time_unit(half_life: i64, time_unit: &TimeUnit) -> i64 {
    match time_unit {
        TimeUnit::Milliseconds => half_life / 1_000_000,
        TimeUnit::Microseconds => half_life / 1_000,
        TimeUnit::Nanoseconds => half_life,
    }
}

fn update<T>(value: T, prev_result: T, time: i64, prev_time: i64, half_life: i64) -> T
where
    T: Float + Zero + One + FromPrimitive,
{
    if value != prev_result {
        let delta_time = time - prev_time;
        // equivalent to: alpha = 1 - exp(-delta_time*ln(2) / half_life)
        let one_minus_alpha = T::from_f64(0.5)
            .unwrap()
            .powf(T::from_i64(delta_time).unwrap() / T::from_i64(half_life).unwrap());
        let alpha = T::one() - one_minus_alpha;
        alpha * value + one_minus_alpha * prev_result
    } else {
        value
    }
}
