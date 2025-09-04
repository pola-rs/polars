use polars_core::prelude::arity::broadcast_binary_elementwise_values;
use polars_core::prelude::*;
use polars_core::{with_match_physical_float_polars_type, with_match_physical_integer_polars_type};

use crate::series::ops::SeriesSealed;

fn log1p<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.ln_1p())
}

fn exp<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.exp())
}

pub trait LogSeries: SeriesSealed {
    /// Compute the logarithm to a given base
    fn log(&self, base: &Series) -> Series {
        let s = self.as_series();

        use DataType::*;
        match (s.dtype(), base.dtype()) {
            (dt1, dt2) if dt1 == dt2 && dt1.is_float() => {
                let s = s.to_physical_repr();
                let base = base.to_physical_repr();
                with_match_physical_float_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    let base_ca: &ChunkedArray<$T> = base.as_ref().as_ref().as_ref();
                    let out: ChunkedArray<$T> = broadcast_binary_elementwise_values(ca, base_ca,
                        |x, base| x.log(base)
                    );
                    out.into_series()
                })
            },
            (_, Float64) => s.cast(&DataType::Float64).unwrap().log(base),
            (Float64, _) => s.log(&base.cast(&DataType::Float64).unwrap()),
            (_, _) => s
                .cast(&DataType::Float64)
                .unwrap()
                .log(&base.cast(&DataType::Float64).unwrap()),
        }
    }

    /// Compute the natural logarithm of all elements plus one in the input array
    fn log1p(&self) -> Series {
        let s = self.as_series();
        if s.dtype().is_decimal() {
            return s.cast(&DataType::Float64).unwrap().log1p();
        }

        let s = s.to_physical_repr();
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
        let s = self.as_series();
        if s.dtype().is_decimal() {
            return s.cast(&DataType::Float64).unwrap().exp();
        }

        let s = s.to_physical_repr();
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

    /// Compute the entropy as `-sum(pk * log(pk))`.
    /// where `pk` are discrete probabilities.
    fn entropy(&self, base: f64, normalize: bool) -> PolarsResult<f64> {
        let s = self.as_series().to_physical_repr();
        polars_ensure!(s.dtype().is_primitive_numeric(), InvalidOperation: "expected numerical input for 'entropy'");
        // if there is only one value in the series, return 0.0 to prevent the
        // function from returning -0.0
        if s.len() == 1 {
            return Ok(0.0);
        }
        match s.dtype() {
            DataType::Float32 | DataType::Float64 => {
                let pk = s.as_ref();

                let pk = if normalize {
                    let sum = pk.sum_reduce().unwrap().into_series(PlSmallStr::EMPTY);

                    if sum.get(0).unwrap().extract::<f64>().unwrap() != 1.0 {
                        (pk / &sum)?
                    } else {
                        pk.clone()
                    }
                } else {
                    pk.clone()
                };

                let base = &Series::new(PlSmallStr::EMPTY, [base]);
                (&pk * &pk.log(base))?.sum::<f64>().map(|v| -v)
            },
            _ => s
                .cast(&DataType::Float64)
                .map(|s| s.entropy(base, normalize))?,
        }
    }
}

impl LogSeries for Series {}
