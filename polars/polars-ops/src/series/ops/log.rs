use polars_core::prelude::*;

use crate::series::ops::SeriesSealed;

fn log<T: PolarsNumericType>(ca: &ChunkedArray<T>, base: f64) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.log(base))
}

fn exp<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.exp())
}

pub trait LogSeries: SeriesSealed {
    /// Compute the logarithm to a given base
    fn log(&self, base: f64) -> Series {
        let s = self.as_series().to_physical_repr();
        let s = s.as_ref();

        use DataType::*;
        match s.dtype() {
            Int32 => log(s.i32().unwrap(), base).into_series(),
            Int64 => log(s.i64().unwrap(), base).into_series(),
            UInt32 => log(s.u32().unwrap(), base).into_series(),
            UInt64 => log(s.u64().unwrap(), base).into_series(),
            Float32 => s.f32().unwrap().apply(|v| v.log(base as f32)).into_series(),
            Float64 => s.f64().unwrap().apply(|v| v.log(base)).into_series(),
            _ => s.cast(&DataType::Float64).unwrap().log(base),
        }
    }

    /// Calculate the exponential of all elements in the input array.
    fn exp(&self) -> Series {
        let s = self.as_series().to_physical_repr();
        let s = s.as_ref();

        use DataType::*;
        match s.dtype() {
            Int32 => exp(s.i32().unwrap()).into_series(),
            Int64 => exp(s.i64().unwrap()).into_series(),
            UInt32 => exp(s.u32().unwrap()).into_series(),
            UInt64 => exp(s.u64().unwrap()).into_series(),
            Float32 => s.f32().unwrap().apply(|v| v.exp()).into_series(),
            Float64 => s.f64().unwrap().apply(|v| v.exp()).into_series(),
            _ => s.cast(&DataType::Float64).unwrap().exp(),
        }
    }

    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    fn entropy(&self, base: f64, normalize: bool) -> Option<f64> {
        let s = self.as_series().to_physical_repr();
        match s.dtype() {
            DataType::Float32 | DataType::Float64 => {
                let pk = s.as_ref();

                let pk = if normalize {
                    let sum = pk.sum_as_series();

                    if sum.get(0).unwrap().extract::<f64>()? != 1.0 {
                        pk / &sum
                    } else {
                        pk.clone()
                    }
                } else {
                    pk.clone()
                };

                let log_pk = pk.log(base);
                (&pk * &log_pk).sum::<f64>().map(|v| -v)
            }
            _ => s
                .cast(&DataType::Float64)
                .ok()
                .and_then(|s| s.entropy(base, normalize)),
        }
    }
}

impl LogSeries for Series {}
