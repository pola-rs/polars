use arrow::legacy::kernels::pow::pow as pow_kernel;
use num::pow::Pow;
use polars_core::export::num;
use polars_core::export::num::{Float, ToPrimitive};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum PowFunction {
    Generic,
    Sqrt,
    Cbrt,
}

impl Display for PowFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            PowFunction::Generic => write!(f, "pow"),
            PowFunction::Sqrt => write!(f, "sqrt"),
            PowFunction::Cbrt => write!(f, "cbrt"),
        }
    }
}

fn pow_on_chunked_arrays<T, F>(
    base: &ChunkedArray<T>,
    exponent: &ChunkedArray<F>,
) -> PolarsResult<Option<Series>>
where
    T: PolarsNumericType,
    F: PolarsNumericType,
    T::Native: num::pow::Pow<F::Native, Output = T::Native> + ToPrimitive,
    ChunkedArray<T>: IntoSeries,
{
    if (base.len() == 1) && (exponent.len() != 1) {
        let base = base
            .get(0)
            .ok_or_else(|| polars_err!(ComputeError: "base is null"))?;

        Ok(Some(
            exponent
                .apply_values_generic(|exp| Pow::pow(base, exp))
                .into_series(),
        ))
    } else {
        Ok(Some(
            polars_core::chunked_array::ops::arity::binary(base, exponent, pow_kernel)
                .into_series(),
        ))
    }
}

fn pow_on_floats<T>(
    base: &ChunkedArray<T>,
    exponent: &ChunkedArray<T>,
) -> PolarsResult<Option<Series>>
where
    T: PolarsFloatType,
    T::Native: num::pow::Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoSeries,
{
    let dtype = T::get_dtype();

    if exponent.len() == 1 {
        let Some(exponent_value) = exponent.get(0) else {
            return Ok(Some(Series::full_null(base.name(), base.len(), &dtype)));
        };
        let s = match exponent_value.to_f64().unwrap() {
            a if a == 1.0 => base.clone().into_series(),
            // specialized sqrt will ensure (-inf)^0.5 = NaN
            // and will likely be faster as well.
            a if a == 0.5 => base.apply_values(|v| v.sqrt()).into_series(),
            a if a.fract() == 0.0 && a < 10.0 && a > 1.0 => {
                let mut out = base.clone();

                for _ in 1..exponent_value.to_u8().unwrap() {
                    out = out * base.clone()
                }
                out.into_series()
            },
            _ => base
                .apply_values(|v| Pow::pow(v, exponent_value))
                .into_series(),
        };
        Ok(Some(s))
    } else {
        pow_on_chunked_arrays(base, exponent)
    }
}

fn pow_to_uint_dtype<T, F>(
    base: &ChunkedArray<T>,
    exponent: &ChunkedArray<F>,
) -> PolarsResult<Option<Series>>
where
    T: PolarsIntegerType,
    F: PolarsIntegerType,
    T::Native: num::pow::Pow<F::Native, Output = T::Native> + ToPrimitive,
    ChunkedArray<T>: IntoSeries,
{
    let dtype = T::get_dtype();

    if exponent.len() == 1 {
        let Some(exponent_value) = exponent.get(0) else {
            return Ok(Some(Series::full_null(base.name(), base.len(), &dtype)));
        };
        let s = match exponent_value.to_u64().unwrap() {
            1 => base.clone().into_series(),
            2..=10 => {
                let mut out = base.clone();

                for _ in 1..exponent_value.to_u8().unwrap() {
                    out = out * base.clone()
                }
                out.into_series()
            },
            _ => base
                .apply_values(|v| Pow::pow(v, exponent_value))
                .into_series(),
        };
        Ok(Some(s))
    } else {
        pow_on_chunked_arrays(base, exponent)
    }
}

fn pow_on_series(base: &Series, exponent: &Series) -> PolarsResult<Option<Series>> {
    use DataType::*;
    match (base.dtype(), exponent.dtype()) {
        #[cfg(feature = "dtype-u8")]
        (UInt8, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.u8().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        #[cfg(feature = "dtype-i8")]
        (Int8, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.i8().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        #[cfg(feature = "dtype-u16")]
        (UInt16, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.u16().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        #[cfg(feature = "dtype-i16")]
        (Int16, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.i16().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        (UInt32, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.u32().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        (Int32, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.i32().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        (UInt64, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.u64().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        (Int64, UInt8 | UInt16 | UInt32 | UInt64) => {
            let ca = base.i64().unwrap();
            let exponent = exponent.strict_cast(&DataType::UInt32)?;
            pow_to_uint_dtype(ca, exponent.u32().unwrap())
        },
        (Float32, _) => {
            let ca = base.f32().unwrap();
            let exponent = exponent.strict_cast(&DataType::Float32)?;
            pow_on_floats(ca, exponent.f32().unwrap())
        },
        (Float64, _) => {
            let ca = base.f64().unwrap();
            let exponent = exponent.strict_cast(&DataType::Float64)?;
            pow_on_floats(ca, exponent.f64().unwrap())
        },
        _ => {
            let base = base.cast(&DataType::Float64)?;
            pow_on_series(&base, exponent)
        },
    }
}

pub(super) fn pow(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let base = &s[0];
    let exponent = &s[1];

    let base_len = base.len();
    let exp_len = exponent.len();
    match (base_len, exp_len) {
        (1, _) | (_, 1) => pow_on_series(base, exponent),
        (len_a, len_b) if len_a == len_b => pow_on_series(base, exponent),
        _ => polars_bail!(
            ComputeError:
            "exponent shape: {} in `pow` expression does not match that of the base: {}",
            exp_len, base_len,
        ),
    }
}

pub(super) fn sqrt(base: &Series) -> PolarsResult<Series> {
    use DataType::*;
    match base.dtype() {
        Float32 => {
            let ca = base.f32().unwrap();
            sqrt_on_floats(ca)
        },
        Float64 => {
            let ca = base.f64().unwrap();
            sqrt_on_floats(ca)
        },
        _ => {
            let base = base.cast(&DataType::Float64)?;
            sqrt(&base)
        },
    }
}

fn sqrt_on_floats<T>(base: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::pow::Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(base.apply_values(|v| v.sqrt()).into_series())
}

pub(super) fn cbrt(base: &Series) -> PolarsResult<Series> {
    use DataType::*;
    match base.dtype() {
        Float32 => {
            let ca = base.f32().unwrap();
            cbrt_on_floats(ca)
        },
        Float64 => {
            let ca = base.f64().unwrap();
            cbrt_on_floats(ca)
        },
        _ => {
            let base = base.cast(&DataType::Float64)?;
            cbrt(&base)
        },
    }
}

fn cbrt_on_floats<T>(base: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::pow::Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(base.apply_values(|v| v.cbrt()).into_series())
}
