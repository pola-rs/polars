use num_traits::{Float, NumCast, One, ToPrimitive, Zero};
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

fn erf_f64(v: f64) -> f64 {
    #[cfg(feature = "nightly")]
    {
        v.erf()
    }
    #[cfg(not(feature = "nightly"))]
    {
        let _ = v;
        unimplemented!("'erf' requires the 'nightly' feature")
    }
}

fn erfc_f64(v: f64) -> f64 {
    #[cfg(feature = "nightly")]
    {
        v.erfc()
    }
    #[cfg(not(feature = "nightly"))]
    {
        let _ = v;
        unimplemented!("'erfc' requires the 'nightly' feature")
    }
}

fn erf<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(erf_f64)
}

fn erfc<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(erfc_f64)
}

pub trait LogSeries: SeriesSealed {
    /// Compute the logarithm to a given base
    fn log(&self, base: &Series) -> PolarsResult<Series> {
        let s = self.as_series();
        polars_ensure!(
            s.dtype().is_numeric() || s.dtype().is_bool(),
            InvalidOperation: "expected numeric or boolean input for `log`, got `{}`", s.dtype()
        );
        polars_ensure!(
            base.dtype().is_numeric() || base.dtype().is_bool(),
            InvalidOperation: "expected numeric or boolean input for `base`, got `{}`", base.dtype()
        );

        match (s.dtype(), base.dtype()) {
            (dt1, dt2) if dt1 == dt2 && dt1.is_float() => {
                with_match_physical_float_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                    let base_ca: &ChunkedArray<$T> = base.as_ref().as_ref();
                    let out: ChunkedArray<$T> = broadcast_binary_elementwise_values(ca, base_ca,
                        |x, base| x.log(base)
                    );
                    Ok(out.into_series())
                })
            },
            (dt1, _) if dt1.is_float() => s.log(&base.cast(dt1)?),
            (_, dt2) if dt2.is_float() => s.cast(base.dtype())?.log(base),
            (_, _) => s.cast(&DataType::Float64)?.log(base),
        }
    }

    /// Compute the natural logarithm of all elements plus one in the input array
    fn log1p(&self) -> PolarsResult<Series> {
        let s = self.as_series();

        use DataType::*;
        match s.dtype() {
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                    Ok(log1p(ca).into_series())
                })
            },
            #[cfg(feature = "dtype-f16")]
            Float16 => Ok(s.f16().unwrap().apply_values(|v| v.ln_1p()).into_series()),
            Float32 => Ok(s.f32().unwrap().apply_values(|v| v.ln_1p()).into_series()),
            Float64 => Ok(s.f64().unwrap().apply_values(|v| v.ln_1p()).into_series()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => s.cast(&DataType::Float64)?.log1p(),
            Boolean => s.cast(&DataType::Float64)?.log1p(),
            dt => polars_bail!(opq = log1p, dt),
        }
    }

    /// Calculate the exponential of all elements in the input array.
    fn exp(&self) -> PolarsResult<Series> {
        let s = self.as_series();

        use DataType::*;
        match s.dtype() {
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                    Ok(exp(ca).into_series())
                })
            },
            #[cfg(feature = "dtype-f16")]
            Float16 => Ok(s.f16().unwrap().apply_values(|v| v.exp()).into_series()),
            Float32 => Ok(s.f32().unwrap().apply_values(|v| v.exp()).into_series()),
            Float64 => Ok(s.f64().unwrap().apply_values(|v| v.exp()).into_series()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => s.cast(&DataType::Float64)?.exp(),
            Boolean => s.cast(&DataType::Float64)?.exp(),
            dt => polars_bail!(opq = exp, dt),
        }
    }

    /// Compute the error function of all elements in the input array.
    fn erf(&self) -> PolarsResult<Series> {
        let s = self.as_series();

        use DataType::*;
        match s.dtype() {
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                    Ok(erf(ca).into_series())
                })
            },
            #[cfg(feature = "dtype-f16")]
            Float16 => Ok(s
                .f16()
                .unwrap()
                .apply_values(|v| erf_f64(v.into()).into())
                .into_series()),
            Float32 => Ok(s
                .f32()
                .unwrap()
                .apply_values(|v| erf_f64(v as f64) as f32)
                .into_series()),
            Float64 => Ok(s.f64().unwrap().apply_values(erf_f64).into_series()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => s.cast(&DataType::Float64)?.erf(),
            Boolean => s.cast(&DataType::Float64)?.erf(),
            dt => polars_bail!(opq = erf, dt),
        }
    }

    /// Compute the complementary error function of all elements in the input array.
    fn erfc(&self) -> PolarsResult<Series> {
        let s = self.as_series();

        use DataType::*;
        match s.dtype() {
            dt if dt.is_integer() => {
                with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                    Ok(erfc(ca).into_series())
                })
            },
            #[cfg(feature = "dtype-f16")]
            Float16 => Ok(s
                .f16()
                .unwrap()
                .apply_values(|v| erfc_f64(v.into()).into())
                .into_series()),
            Float32 => Ok(s
                .f32()
                .unwrap()
                .apply_values(|v| erfc_f64(v as f64) as f32)
                .into_series()),
            Float64 => Ok(s.f64().unwrap().apply_values(erfc_f64).into_series()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => s.cast(&DataType::Float64)?.erfc(),
            Boolean => s.cast(&DataType::Float64)?.erfc(),
            dt => polars_bail!(opq = erfc, dt),
        }
    }

    /// Compute the entropy as `-sum(pk * log(pk))`.
    /// where `pk` are discrete probabilities.
    fn entropy(&self, base: f64, normalize: bool) -> PolarsResult<f64> {
        let s = self.as_series();
        match s.dtype() {
            dt if dt.is_float() => {
                with_match_physical_float_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                    Ok(entropy_float(ca, base, normalize))
                })
            },
            dt if dt.is_integer() || dt.is_decimal() || dt.is_bool() => s
                .cast(&DataType::Float64)
                .map(|s| s.entropy(base, normalize))?,
            dt => polars_bail!(opq = entropy, dt),
        }
    }
}

/// Entropy with `0 * log(0) = 0`, NaN for negative weights and `+0.0` for a zero result.
fn entropy_float<T>(ca: &ChunkedArray<T>, base: f64, normalize: bool) -> f64
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    if ca.len() == ca.null_count() {
        return 0.0;
    }
    let total = if normalize {
        let total = ca.sum().unwrap();
        // Also catches NaN input and a sum that overflowed, which would normalize to zeros.
        if !(total.is_finite() && total > T::Native::zero()) {
            return f64::NAN;
        }
        total
    } else {
        T::Native::one()
    };
    let base: T::Native = NumCast::from(base).unwrap();
    let plogp = ca.apply_values(|x| {
        if x < T::Native::zero() {
            return T::Native::nan();
        }
        let p = x / total;
        if p == T::Native::zero() {
            T::Native::zero()
        } else {
            p * p.log(base)
        }
    });
    0.0 - plogp.sum().unwrap().to_f64().unwrap()
}

impl LogSeries for Series {}
