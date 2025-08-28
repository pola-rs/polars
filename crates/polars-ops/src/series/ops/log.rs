use arrow::array::{Array, PrimitiveArray};
use arrow::compute::utils::combine_validities_and;
use arrow::types::NativeType;
use num_traits::Float;
use polars_core::prelude::arity::apply_binary_kernel_broadcast;
use polars_core::prelude::*;
use polars_core::{
    with_match_physical_float_polars_type, with_match_physical_float_type,
    with_match_physical_integer_polars_type,
};

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
                    let ca: ChunkedArray<$T> = apply_binary_kernel_broadcast(
                        ca,
                        base_ca,
                        |l, r| log_kernel_binary::<$T>(l, r)                        ,
                        |l, r| log_kernel_unary_left::<$T>(l, r),
                        |l, r| log_kernel_unary_right::<$T>(l, r),
                    );
                    ca.into_series()
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

    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    fn entropy(&self, base: f64, normalize: bool) -> PolarsResult<f64> {
        let s: std::borrow::Cow<'_, Series> = self.as_series().to_physical_repr();
        polars_ensure!(s.dtype().is_primitive_numeric(), InvalidOperation: "expected numerical input for 'entropy'");
        // if there is only one value in the series, return 0.0 to prevent the
        // function from returning -0.0
        if s.len() == 1 {
            return Ok(0.0);
        }
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

        use DataType::*;
        match s.dtype() {
            Float32 => {
                let a = s.f32().unwrap();
                Ok(-a
                    .iter()
                    .map(|p| match p {
                        Some(p) if p > 0.0 => (p * p.log(base as f32)) as f64,
                        _ => 0.0,
                    })
                    .sum::<f64>())
            },
            Float64 => {
                let a = s.f64().unwrap();
                Ok(-a
                    .iter()
                    .map(|p| match p {
                        Some(p) if p > 0.0 => p * p.log(base),
                        _ => 0.0,
                    })
                    .sum::<f64>())
            },
            _ => s
                .cast(&DataType::Float64)
                .map(|s| s.entropy(base, normalize))?,
        }
    }
}

impl LogSeries for Series {}

fn log_kernel_binary<'a, T>(x_arr: &'a T::Array, base_arr: &'a T::Array) -> T::Array
where
    T: PolarsDataType,
    T::Physical<'a>: num_traits::Float + NativeType,
    T::Array: ArrayFromIter<T::Physical<'a>>,
{
    let validity = combine_validities_and(x_arr.validity(), base_arr.validity());
    let element_iter = x_arr
        .values_iter()
        .zip(base_arr.values_iter())
        .map(|(x, base): (T::Physical<'a>, T::Physical<'a>)| x.log(base));
    let result: T::Array = element_iter.collect_arr();
    result.with_validity_typed(validity)
}

fn log_kernel_unary_right<'a, T>(x_arr: &'a T::Array, base: T::Physical<'a>) -> T::Array
where
    T: PolarsDataType,
    T::Physical<'a>: num_traits::Float + NativeType,
    T::Array: ArrayFromIter<T::Physical<'a>>,
{
    let validity = x_arr.validity().cloned();
    let element_iter = x_arr.values_iter().map(|x: T::Physical<'a>| x.log(base));
    let result: T::Array = element_iter.collect_arr();
    result.with_validity_typed(validity)
}

fn log_kernel_unary_left<'a, T>(x_arr: T::Physical<'a>, base: &'a T::Array) -> T::Array
where
    T: PolarsDataType,
    T::Physical<'a>: num_traits::Float + NativeType,
    T::Array: ArrayFromIter<T::Physical<'a>>,
{
    let validity = base.validity().cloned();
    let element_iter = base
        .values_iter()
        .map(|base: T::Physical<'a>| x_arr.log(base));
    let result: T::Array = element_iter.collect_arr();
    result.with_validity_typed(validity)
}
