use num_traits::pow::Pow;
use num_traits::{Float, One, ToPrimitive, Zero};
use polars_core::prelude::arity::{broadcast_binary_elementwise, unary_elementwise_values};
use polars_core::with_match_physical_integer_type;

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
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    F: PolarsNumericType,
    T::Native: Pow<F::Native, Output = T::Native> + ToPrimitive,
{
    if exponent.len() == 1 {
        if let Some(e) = exponent.get(0) {
            if e == F::Native::zero() {
                return unary_elementwise_values(base, |_| T::Native::one());
            }
            if e == F::Native::one() {
                return base.clone();
            }
            if e == F::Native::one() + F::Native::one() {
                return base * base;
            }
        }
    }

    broadcast_binary_elementwise(base, exponent, |b, e| Some(Pow::pow(b?, e?)))
}

fn pow_on_floats<T>(
    base: &ChunkedArray<T>,
    exponent: &ChunkedArray<T>,
) -> PolarsResult<Option<Column>>
where
    T: PolarsFloatType,
    T::Native: Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoColumn,
{
    let dtype = T::get_dtype();

    if exponent.len() == 1 {
        let Some(exponent_value) = exponent.get(0) else {
            return Ok(Some(Column::full_null(
                base.name().clone(),
                base.len(),
                &dtype,
            )));
        };
        let s = match exponent_value.to_f64().unwrap() {
            1.0 => base.clone().into_column(),
            // specialized sqrt will ensure (-inf)^0.5 = NaN
            // and will likely be faster as well.
            0.5 => base.apply_values(|v| v.sqrt()).into_column(),
            a if a.fract() == 0.0 && a < 10.0 && a > 1.0 => {
                let mut out = base.clone();

                for _ in 1..exponent_value.to_u8().unwrap() {
                    out = out * base.clone()
                }
                out.into_column()
            },
            _ => base
                .apply_values(|v| Pow::pow(v, exponent_value))
                .into_column(),
        };
        Ok(Some(s))
    } else {
        Ok(Some(pow_on_chunked_arrays(base, exponent).into_column()))
    }
}

fn pow_to_uint_dtype<T, F>(
    base: &ChunkedArray<T>,
    exponent: &ChunkedArray<F>,
) -> PolarsResult<Option<Column>>
where
    T: PolarsIntegerType,
    F: PolarsIntegerType,
    T::Native: Pow<F::Native, Output = T::Native> + ToPrimitive,
    ChunkedArray<T>: IntoColumn,
{
    let dtype = T::get_dtype();

    if exponent.len() == 1 {
        let Some(exponent_value) = exponent.get(0) else {
            return Ok(Some(Column::full_null(
                base.name().clone(),
                base.len(),
                &dtype,
            )));
        };
        let s = match exponent_value.to_u64().unwrap() {
            1 => base.clone().into_column(),
            2..=10 => {
                let mut out = base.clone();

                for _ in 1..exponent_value.to_u8().unwrap() {
                    out = out * base.clone()
                }
                out.into_column()
            },
            _ => base
                .apply_values(|v| Pow::pow(v, exponent_value))
                .into_column(),
        };
        Ok(Some(s))
    } else {
        Ok(Some(pow_on_chunked_arrays(base, exponent).into_column()))
    }
}

fn pow_on_series(base: &Column, exponent: &Column) -> PolarsResult<Option<Column>> {
    use DataType::*;

    let base_dtype = base.dtype();
    polars_ensure!(
        base_dtype.is_primitive_numeric(),
        InvalidOperation: "`pow` operation not supported for dtype `{}` as base", base_dtype
    );
    let exponent_dtype = exponent.dtype();
    polars_ensure!(
        exponent_dtype.is_primitive_numeric(),
        InvalidOperation: "`pow` operation not supported for dtype `{}` as exponent", exponent_dtype
    );

    // if false, dtype is float
    if base_dtype.is_integer() {
        with_match_physical_integer_type!(base_dtype, |$native_type| {
            if exponent_dtype.is_float() {
                match exponent_dtype {
                    Float32 => {
                        let ca = base.cast(&DataType::Float32)?;
                        pow_on_floats(ca.f32().unwrap(), exponent.f32().unwrap())
                    },
                    Float64 => {
                        let ca = base.cast(&DataType::Float64)?;
                        pow_on_floats(ca.f64().unwrap(), exponent.f64().unwrap())
                    },
                    _ => unreachable!(),
                }
            } else {
                let ca = base.$native_type().unwrap();
                let exponent = exponent.strict_cast(&DataType::UInt32).map_err(|err| polars_err!(
                    InvalidOperation:
                    "{}\n\nHint: if you were trying to raise an integer to a negative integer power, please cast your base or exponent to float first.",
                    err
                ))?;
                pow_to_uint_dtype(ca, exponent.u32().unwrap())
            }
        })
    } else {
        match base_dtype {
            Float32 => {
                let ca = base.f32().unwrap();
                let exponent = exponent.strict_cast(&DataType::Float32)?;
                pow_on_floats(ca, exponent.f32().unwrap())
            },
            Float64 => {
                let ca = base.f64().unwrap();
                let exponent = exponent.strict_cast(&DataType::Float64)?;
                pow_on_floats(ca, exponent.f64().unwrap())
            },
            _ => unreachable!(),
        }
    }
}

pub(super) fn pow(s: &mut [Column]) -> PolarsResult<Option<Column>> {
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

pub(super) fn sqrt(base: &Column) -> PolarsResult<Column> {
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

fn sqrt_on_floats<T>(base: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(base.apply_values(|v| v.sqrt()).into_column())
}

pub(super) fn cbrt(base: &Column) -> PolarsResult<Column> {
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

fn cbrt_on_floats<T>(base: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(base.apply_values(|v| v.cbrt()).into_column())
}
