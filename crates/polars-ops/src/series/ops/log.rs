use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;

use crate::series::ops::SeriesSealed;

fn log<T: PolarsNumericType>(ca: &ChunkedArray<T>, base: f64) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.log(base))
}

fn log1p<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.ln_1p())
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
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    log(ca, base).into_series()
                })
            },
            Float32 => s
                .f32()
                .unwrap()
                .apply_values(|v| v.log(base as f32))
                .into_series(),
            Float64 => s.f64().unwrap().apply_values(|v| v.log(base)).into_series(),
            _ => s.cast(&DataType::Float64).unwrap().log(base),
        }
    }

    /// Compute the natural logarithm of all elements plus one in the input array
    fn log1p(&self) -> Series {
        let s = self.as_series().to_physical_repr();
        let s = s.as_ref();

        use DataType::*;
        match s.dtype() {
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    log1p(ca).into_series()
                })
            },
            Float32 => s.f32().unwrap().apply_values(|v| v.ln_1p()).into_series(),
            Float64 => s.f64().unwrap().apply_values(|v| v.ln_1p()).into_series(),
            _ => s.cast(&DataType::Float64).unwrap().log1p(),
        }
    }

    /// Calculate the exponential of all elements in the input array.
    fn exp(&self) -> Series {
        let s = self.as_series().to_physical_repr();
        let s = s.as_ref();

        use DataType::*;
        match s.dtype() {
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    exp(ca).into_series()
                })
            },
            Float32 => s.f32().unwrap().apply_values(|v| v.exp()).into_series(),
            Float64 => s.f64().unwrap().apply_values(|v| v.exp()).into_series(),
            _ => s.cast(&DataType::Float64).unwrap().exp(),
        }
    }

    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    fn entropy(&self, base: f64, normalize: bool) -> PolarsResult<f64> {
        let s = self.as_series().to_physical_repr();
        polars_ensure!(s.dtype().is_numeric(), InvalidOperation: "expected numerical input for 'entropy'");
        // if there is only one value in the series, return 0.0 to prevent the
        // function from returning -0.0
        if s.len() == 1 {
            return Ok(0.0);
        }
        match s.dtype() {
            DataType::Float32 | DataType::Float64 => {
                let pk = s.as_ref();

                let pk = if normalize {
                    let sum = pk.sum_reduce().unwrap().into_series("");

                    if sum.get(0).unwrap().extract::<f64>().unwrap() != 1.0 {
                        (pk / &sum)?
                    } else {
                        pk.clone()
                    }
                } else {
                    pk.clone()
                };

                let log_pk = pk.log(base);
                (&pk * &log_pk)?.sum::<f64>().map(|v| -v)
            },
            _ => s
                .cast(&DataType::Float64)
                .map(|s| s.entropy(base, normalize))?,
        }
    }
}

impl LogSeries for Series {}
