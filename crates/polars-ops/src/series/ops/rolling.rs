use polars_compute::rolling;
use polars_core::prelude::*;
#[cfg(feature = "moment")]
use {
    num_traits::{Float, pow::Pow},
    std::ops::SubAssign,
};

#[cfg(feature = "moment")]
fn rolling_skew_ca<T>(
    ca: &ChunkedArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + SubAssign + Pow<T::Native, Output = T::Native>,
{
    let ca = ca.rechunk();
    let out = rolling::dispatch::rolling_skew(
        ca.downcast_as_array(),
        window_size,
        min_periods,
        center,
        params,
    )?;
    Ok(unsafe { ca.with_chunks(vec![out]) })
}

#[cfg(feature = "moment")]
pub fn rolling_skew(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    let window_size = options.window_size;
    let min_periods = options.min_periods;
    let center = options.center;
    let params = options.fn_params;

    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16().unwrap();
            rolling_skew_ca(ca, window_size, min_periods, center, params).map(|ca| ca.into_series())
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            rolling_skew_ca(ca, window_size, min_periods, center, params).map(|ca| ca.into_series())
        },
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            rolling_skew_ca(ca, window_size, min_periods, center, params).map(|ca| ca.into_series())
        },
        dt if dt.is_primitive_numeric() => {
            let s = s.cast(&DataType::Float64).unwrap();
            rolling_skew(&s, options)
        },
        dt => polars_bail!(opq = rolling_skew, dt),
    }
}

#[cfg(feature = "moment")]
fn rolling_kurtosis_ca<T>(
    ca: &ChunkedArray<T>,
    window_size: usize,
    params: Option<RollingFnParams>,
    min_periods: usize,
    center: bool,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + SubAssign + Pow<T::Native, Output = T::Native>,
{
    let ca = ca.rechunk();
    let out = rolling::dispatch::rolling_kurtosis(
        ca.downcast_as_array(),
        window_size,
        min_periods,
        center,
        params,
    )?;
    Ok(unsafe { ca.with_chunks(vec![out]) })
}

#[cfg(feature = "moment")]
pub fn rolling_kurtosis(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    let window_size = options.window_size;
    let min_periods = options.min_periods;
    let center = options.center;
    let params = options.fn_params;

    match s.dtype() {
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            rolling_kurtosis_ca(ca, window_size, params, min_periods, center)
                .map(|ca| ca.into_series())
        },
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            rolling_kurtosis_ca(ca, window_size, params, min_periods, center)
                .map(|ca| ca.into_series())
        },
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16().unwrap();
            rolling_kurtosis_ca(ca, window_size, params, min_periods, center)
                .map(|ca| ca.into_series())
        },
        dt if dt.is_primitive_numeric() => {
            let s = s.cast(&DataType::Float64).unwrap();
            rolling_kurtosis(&s, options)
        },
        dt => polars_bail!(opq = rolling_kurtosis, dt),
    }
}
