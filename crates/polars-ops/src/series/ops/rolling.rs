use polars_core::prelude::*;
#[cfg(feature = "moment")]
use {
    crate::series::ops::moment::MomentSeries,
    polars_core::export::num::{self, Float, FromPrimitive},
    polars_core::utils::with_unstable_series,
    std::ops::SubAssign,
};

use crate::series::ops::SeriesSealed;

#[cfg(feature = "moment")]
fn rolling_skew<T>(
    ca: &ChunkedArray<T>,
    window_size: usize,
    bias: bool,
) -> PolarsResult<ChunkedArray<T>>
where
    ChunkedArray<T>: IntoSeries,
    T: PolarsFloatType,
    T::Native: Float + SubAssign + num::pow::Pow<T::Native, Output = T::Native>,
{
    with_unstable_series(ca.dtype(), |us| {
        ca.rolling_map_float(window_size, |arr| {
            let arr = unsafe { arr.chunks_mut().get_mut(0).unwrap() };

            // SAFETY: dtype is correct.
            unsafe {
                us.with_array(arr, |us| {
                    us.as_ref()
                        .skew(bias)
                        .unwrap()
                        .map(|flt| T::Native::from_f64(flt).unwrap())
                })
            }
        })
    })
}

pub trait RollingSeries: SeriesSealed {
    #[cfg(feature = "moment")]
    fn rolling_skew(&self, window_size: usize, bias: bool) -> PolarsResult<Series> {
        let s = self.as_series();

        match s.dtype() {
            DataType::Float64 => {
                let ca = s.f64().unwrap();
                rolling_skew(ca, window_size, bias).map(|ca| ca.into_series())
            },
            DataType::Float32 => {
                let ca = s.f32().unwrap();
                rolling_skew(ca, window_size, bias).map(|ca| ca.into_series())
            },
            dt if dt.is_numeric() => {
                let s = s.cast(&DataType::Float64).unwrap();
                s.rolling_skew(window_size, bias)
            },
            dt => polars_bail!(opq = rolling_skew, dt),
        }
    }
}

impl RollingSeries for Series {}
