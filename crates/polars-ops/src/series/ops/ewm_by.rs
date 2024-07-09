use bytemuck::allocation::zeroed_vec;
use num_traits::{Float, FromPrimitive, One, Zero};
use polars_core::prelude::*;
use polars_core::utils::binary_concatenate_validities;

pub fn ewm_mean_by(
    s: &Series,
    times: &Series,
    half_life: i64,
    times_is_sorted: bool,
) -> PolarsResult<Series> {
    fn func<T>(
        values: &ChunkedArray<T>,
        times: &Int64Chunked,
        half_life: i64,
        times_is_sorted: bool,
    ) -> PolarsResult<Series>
    where
        T: PolarsFloatType,
        T::Native: Float + Zero + One,
        ChunkedArray<T>: IntoSeries,
    {
        if times_is_sorted {
            Ok(ewm_mean_by_impl_sorted(values, times, half_life).into_series())
        } else {
            Ok(ewm_mean_by_impl(values, times, half_life).into_series())
        }
    }

    match (s.dtype(), times.dtype()) {
        (DataType::Float64, DataType::Int64) => func(
            s.f64().unwrap(),
            times.i64().unwrap(),
            half_life,
            times_is_sorted,
        ),
        (DataType::Float32, DataType::Int64) => func(
            s.f32().unwrap(),
            times.i64().unwrap(),
            half_life,
            times_is_sorted,
        ),
        #[cfg(feature = "dtype-datetime")]
        (_, DataType::Datetime(time_unit, _)) => {
            let half_life = adjust_half_life_to_time_unit(half_life, time_unit);
            ewm_mean_by(
                s,
                &times.cast(&DataType::Int64)?,
                half_life,
                times_is_sorted,
            )
        },
        #[cfg(feature = "dtype-date")]
        (_, DataType::Date) => ewm_mean_by(
            s,
            &times.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
            half_life,
            times_is_sorted,
        ),
        (_, DataType::UInt64 | DataType::UInt32 | DataType::Int32) => ewm_mean_by(
            s,
            &times.cast(&DataType::Int64)?,
            half_life,
            times_is_sorted,
        ),
        (DataType::UInt64 | DataType::UInt32 | DataType::Int64 | DataType::Int32, _) => {
            ewm_mean_by(
                &s.cast(&DataType::Float64)?,
                times,
                half_life,
                times_is_sorted,
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
    let mut arr = T::Array::from_zeroable_vec(out, values.dtype().to_arrow(CompatLevel::newest()));
    if (times.null_count() > 0) || (values.null_count() > 0) {
        let validity = binary_concatenate_validities(times, values);
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
    let mut out: Vec<_> = zeroed_vec(times.len());

    let mut skip_rows: usize = 0;
    let mut prev_time: i64 = 0;
    let mut prev_result = T::Native::zero();
    for (idx, (value, time)) in values.iter().zip(times.iter()).enumerate() {
        if let (Some(time), Some(value)) = (time, value) {
            prev_time = time;
            prev_result = value;
            unsafe {
                *out.get_unchecked_mut(idx) = prev_result;
            }
            skip_rows = idx + 1;
            break;
        }
    }
    values
        .iter()
        .zip(times.iter())
        .enumerate()
        .skip(skip_rows)
        .for_each(|(idx, (value, time))| {
            if let (Some(time), Some(value)) = (time, value) {
                let result = update(value, prev_result, time, prev_time, half_life);
                prev_time = time;
                prev_result = result;
                unsafe {
                    *out.get_unchecked_mut(idx) = result;
                }
            };
        });
    let mut arr = T::Array::from_zeroable_vec(out, values.dtype().to_arrow(CompatLevel::newest()));
    if (times.null_count() > 0) || (values.null_count() > 0) {
        let validity = binary_concatenate_validities(times, values);
        arr = arr.with_validity_typed(validity);
    }
    ChunkedArray::with_chunk(values.name(), arr)
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
