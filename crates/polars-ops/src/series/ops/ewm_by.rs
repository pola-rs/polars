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
    dispatch_ewm_by::<true>(s, times, half_life, times_is_sorted, "ewm_mean_by")
}

pub fn ewm_sum_by(
    s: &Series,
    times: &Series,
    half_life: i64,
    times_is_sorted: bool,
) -> PolarsResult<Series> {
    dispatch_ewm_by::<false>(s, times, half_life, times_is_sorted, "ewm_sum_by")
}

fn dispatch_ewm_by<const IS_MEAN: bool>(
    s: &Series,
    times: &Series,
    half_life: i64,
    times_is_sorted: bool,
    op_name: &'static str,
) -> PolarsResult<Series> {
    fn func<T, const IS_MEAN: bool>(
        values: &ChunkedArray<T>,
        times: &Int64Chunked,
        half_life: i64,
        times_is_sorted: bool,
    ) -> PolarsResult<Series>
    where
        T: PolarsFloatType,
        T::Native: Float + Zero + One + FromPrimitive,
        ChunkedArray<T>: ChunkTakeUnchecked<IdxCa>,
    {
        let ca = if times_is_sorted {
            ewm_by_impl_sorted::<T, IS_MEAN>(values, times, half_life)
        } else {
            ewm_by_impl::<T, IS_MEAN>(values, times, half_life)
        };
        Ok(ca.into_series())
    }

    polars_ensure!(
        s.len() == times.len(),
        length_mismatch = op_name,
        s.len(),
        times.len()
    );

    match (s.dtype(), times.dtype()) {
        (DataType::Float64, DataType::Int64) => func::<_, IS_MEAN>(
            s.f64().unwrap(),
            times.i64().unwrap(),
            half_life,
            times_is_sorted,
        ),
        (DataType::Float32, DataType::Int64) => func::<_, IS_MEAN>(
            s.f32().unwrap(),
            times.i64().unwrap(),
            half_life,
            times_is_sorted,
        ),
        #[cfg(feature = "dtype-f16")]
        (DataType::Float16, DataType::Int64) => func::<_, IS_MEAN>(
            s.f16().unwrap(),
            times.i64().unwrap(),
            half_life,
            times_is_sorted,
        ),
        #[cfg(feature = "dtype-datetime")]
        (_, DataType::Datetime(time_unit, _)) => {
            let half_life = adjust_half_life_to_time_unit(half_life, time_unit);
            dispatch_ewm_by::<IS_MEAN>(
                s,
                &times.cast(&DataType::Int64)?,
                half_life,
                times_is_sorted,
                op_name,
            )
        },
        #[cfg(feature = "dtype-date")]
        (_, DataType::Date) => dispatch_ewm_by::<IS_MEAN>(
            s,
            &times.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?,
            half_life,
            times_is_sorted,
            op_name,
        ),
        (_, DataType::UInt64 | DataType::UInt32 | DataType::Int32) => dispatch_ewm_by::<IS_MEAN>(
            s,
            &times.cast(&DataType::Int64)?,
            half_life,
            times_is_sorted,
            op_name,
        ),
        (DataType::UInt64 | DataType::UInt32 | DataType::Int64 | DataType::Int32, _) => {
            dispatch_ewm_by::<IS_MEAN>(
                &s.cast(&DataType::Float64)?,
                times,
                half_life,
                times_is_sorted,
                op_name,
            )
        },
        _ => {
            polars_bail!(InvalidOperation: "expected series to be Float64, Float32, Float16, \
                Int64, Int32, UInt64, UInt32, and `by` to be Date, Datetime, Int64, Int32, \
                UInt64, or UInt32")
        },
    }
}

/// Sort on behalf of user
fn ewm_by_impl<T, const IS_MEAN: bool>(
    values: &ChunkedArray<T>,
    times: &Int64Chunked,
    half_life: i64,
) -> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float + Zero + One + FromPrimitive,
    ChunkedArray<T>: ChunkTakeUnchecked<IdxCa>,
{
    let sorting_indices = times.arg_sort(Default::default());
    let sorted_values = unsafe { values.take_unchecked(&sorting_indices) };
    let sorted_times = unsafe { times.take_unchecked(&sorting_indices) };
    let sorting_indices = sorting_indices
        .cont_slice()
        .expect("`arg_sort` should have returned a single chunk");

    let mut out: Vec<_> = zeroed_vec(sorted_times.len());
    ewm_by_core::<T, IS_MEAN, _>(
        sorted_values
            .iter()
            .zip(sorted_times.iter())
            .enumerate()
            .map(|(idx, (value, time))| (sorting_indices[idx] as usize, value, time)),
        half_life,
        |out_idx, result| unsafe {
            *out.get_unchecked_mut(out_idx) = result;
        },
    );
    ewm_by_finish(values, times, out)
}

/// Fastpath if `times` is known to already be sorted.
fn ewm_by_impl_sorted<T, const IS_MEAN: bool>(
    values: &ChunkedArray<T>,
    times: &Int64Chunked,
    half_life: i64,
) -> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float + Zero + One + FromPrimitive,
{
    let mut out: Vec<_> = zeroed_vec(times.len());
    ewm_by_core::<T, IS_MEAN, _>(
        values
            .iter()
            .zip(times.iter())
            .enumerate()
            .map(|(idx, (value, time))| (idx, value, time)),
        half_life,
        |idx, result| unsafe {
            *out.get_unchecked_mut(idx) = result;
        },
    );
    ewm_by_finish(values, times, out)
}

#[inline]
fn ewm_by_core<T, const IS_MEAN: bool, F>(
    pairs: impl Iterator<Item = (usize, Option<T::Native>, Option<i64>)>,
    half_life: i64,
    mut write: F,
) where
    T: PolarsFloatType,
    T::Native: Float + Zero + One + FromPrimitive,
    F: FnMut(usize, T::Native),
{
    let mut prev_time: i64 = 0;
    let mut prev_result = T::Native::zero();
    let mut started = false;

    for (out_idx, value, time) in pairs {
        if let (Some(time), Some(value)) = (time, value) {
            if !started {
                prev_time = time;
                prev_result = value;
                write(out_idx, prev_result);
                started = true;
            } else {
                let result =
                    update::<T::Native, IS_MEAN>(value, prev_result, time, prev_time, half_life);
                prev_time = time;
                prev_result = result;
                write(out_idx, result);
            }
        }
    }
}

fn ewm_by_finish<T>(
    values: &ChunkedArray<T>,
    times: &Int64Chunked,
    out: Vec<T::Native>,
) -> ChunkedArray<T>
where
    T: PolarsFloatType,
{
    let mut arr = T::Array::from_zeroable_vec(out, values.dtype().to_arrow(CompatLevel::newest()));
    if (times.null_count() > 0) || (values.null_count() > 0) {
        let validity = binary_concatenate_validities(times, values);
        arr = arr.with_validity_typed(validity);
    }
    ChunkedArray::with_chunk(values.name().clone(), arr)
}

fn adjust_half_life_to_time_unit(half_life: i64, time_unit: &TimeUnit) -> i64 {
    match time_unit {
        TimeUnit::Milliseconds => half_life / 1_000_000,
        TimeUnit::Microseconds => half_life / 1_000,
        TimeUnit::Nanoseconds => half_life,
    }
}

#[inline]
fn update<T, const IS_MEAN: bool>(
    value: T,
    prev_result: T,
    time: i64,
    prev_time: i64,
    half_life: i64,
) -> T
where
    T: Float + Zero + One + FromPrimitive,
{
    if IS_MEAN && value == prev_result {
        return value;
    }
    let delta_time = time - prev_time;
    // 0.5^(delta_time/half_life) == exp(-delta_time*ln(2) / half_life) == (1 - alpha)
    let one_minus_alpha = T::from_f64(0.5)
        .unwrap()
        .powf(T::from_i64(delta_time).unwrap() / T::from_i64(half_life).unwrap());
    let weight = if IS_MEAN {
        T::one() - one_minus_alpha
    } else {
        T::one()
    };
    one_minus_alpha * prev_result + weight * value
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ewm_sum_by_uniform_times() {
        let values = Series::new("x".into(), [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let times = Series::new("t".into(), [0i64, 1, 2, 5, 6, 10]);
        let result = ewm_sum_by(&values, &times, 1, true).unwrap();
        let out = result.f64().unwrap();
        let actual: Vec<f64> = out.iter().map(|v| v.unwrap()).collect();
        let expected = [1.0, 2.5, 4.25, 4.53125, 7.265625, 6.4541015625];
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10, "actual={actual:?}");
        }
    }

    #[test]
    fn test_ewm_sum_by_constant() {
        let values = Series::new("values".into(), [1.0f64, 1.0, 1.0]);
        let times = Series::new("times".into(), [0i64, 7, 12]);
        let result = ewm_sum_by(&values, &times, 2, true).unwrap();
        let out = result.f64().unwrap();
        let actual: Vec<f64> = out.iter().map(|v| v.unwrap()).collect();
        assert!((actual[0] - 1.0).abs() < 1e-10, "actual={actual:?}");
        assert!(
            (actual[1] - 1.0883883476483184).abs() < 1e-10,
            "actual={actual:?}"
        );
        assert!(
            (actual[2] - 1.192401695296637).abs() < 1e-10,
            "actual={actual:?}"
        );
    }
}
