use polars_core::prelude::*;
#[cfg(feature = "moment")]
use {
    num_traits::{Float, pow::Pow},
    std::ops::SubAssign,
};

use crate::series::ops::SeriesSealed;

#[cfg(feature = "moment")]
fn rolling_skew<T>(
    ca: &ChunkedArray<T>,
    window_size: usize,
    bias: bool,
    min_periods: usize,
    center: bool,
) -> PolarsResult<ChunkedArray<T>>
where
    ChunkedArray<T>: IntoSeries,
    T: PolarsFloatType,
    T::Native: Float + SubAssign + Pow<T::Native, Output = T::Native>,
{
    use arrow::array::Array;

    let ca = ca.rechunk();
    let arr = ca.downcast_get(0).unwrap();
    let params = Some(RollingFnParams::Skew { bias });
    let arr = if arr.has_nulls() {
        polars_compute::rolling::nulls::rolling_skew(arr, window_size, min_periods, center, params)
    } else {
        let values = arr.values();
        polars_compute::rolling::no_nulls::rolling_skew(
            values,
            window_size,
            min_periods,
            center,
            params,
        )?
    };
    Ok(unsafe { ca.with_chunks(vec![arr]) })
}

pub trait RollingSeries: SeriesSealed {
    #[cfg(feature = "moment")]
    fn rolling_skew(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let window_size = options.window_size;
        let min_periods = options.min_periods;
        let center = options.center;

        let Some(RollingFnParams::Skew { bias }) = options.fn_params else {
            unreachable!("must be set")
        };

        let s = self.as_series();

        match s.dtype() {
            DataType::Float64 => {
                let ca = s.f64().unwrap();
                rolling_skew(ca, window_size, bias, min_periods, center).map(|ca| ca.into_series())
            },
            DataType::Float32 => {
                let ca = s.f32().unwrap();
                rolling_skew(ca, window_size, bias, min_periods, center).map(|ca| ca.into_series())
            },
            dt if dt.is_primitive_numeric() => {
                let s = s.cast(&DataType::Float64).unwrap();
                s.rolling_skew(options)
            },
            dt => polars_bail!(opq = rolling_skew, dt),
        }
    }
}

impl RollingSeries for Series {}
