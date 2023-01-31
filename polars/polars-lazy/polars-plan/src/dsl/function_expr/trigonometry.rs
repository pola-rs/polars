use num::Float;
use polars_core::export::num;

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum TrigonometricFunction {
    Sin,
    Cos,
    Tan,
    ArcSin,
    ArcCos,
    ArcTan,
    Sinh,
    Cosh,
    Tanh,
    ArcSinh,
    ArcCosh,
    ArcTanh,
}

impl Display for TrigonometricFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            TrigonometricFunction::Sin => write!(f, "sin"),
            TrigonometricFunction::Cos => write!(f, "cos"),
            TrigonometricFunction::Tan => write!(f, "tan"),
            TrigonometricFunction::ArcSin => write!(f, "arcsin"),
            TrigonometricFunction::ArcCos => write!(f, "arccos"),
            TrigonometricFunction::ArcTan => write!(f, "arctan"),
            TrigonometricFunction::Sinh => write!(f, "sinh"),
            TrigonometricFunction::Cosh => write!(f, "cosh"),
            TrigonometricFunction::Tanh => write!(f, "tanh"),
            TrigonometricFunction::ArcSinh => write!(f, "arcsinh"),
            TrigonometricFunction::ArcCosh => write!(f, "arccosh"),
            TrigonometricFunction::ArcTanh => write!(f, "arctanh"),
        }
    }
}

pub(super) fn apply_trigonometric_function(
    s: &Series,
    trig_function: TrigonometricFunction,
) -> PolarsResult<Series> {
    use DataType::*;
    match s.dtype() {
        Float32 => {
            let ca = s.f32().unwrap();
            apply_trigonometric_function_to_float(ca, trig_function)
        }
        Float64 => {
            let ca = s.f64().unwrap();
            apply_trigonometric_function_to_float(ca, trig_function)
        }
        dt if dt.is_numeric() => {
            let s = s.cast(&DataType::Float64)?;
            apply_trigonometric_function(&s, trig_function)
        }
        dt => Err(PolarsError::ComputeError(
            format!("cannot use trigonometric function on Series of dtype: {dt:?}",).into(),
        )),
    }
}

fn apply_trigonometric_function_to_float<T>(
    ca: &ChunkedArray<T>,
    trig_function: TrigonometricFunction,
) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    match trig_function {
        TrigonometricFunction::Sin => sin(ca),
        TrigonometricFunction::Cos => cos(ca),
        TrigonometricFunction::Tan => tan(ca),
        TrigonometricFunction::ArcSin => arcsin(ca),
        TrigonometricFunction::ArcCos => arccos(ca),
        TrigonometricFunction::ArcTan => arctan(ca),
        TrigonometricFunction::Sinh => sinh(ca),
        TrigonometricFunction::Cosh => cosh(ca),
        TrigonometricFunction::Tanh => tanh(ca),
        TrigonometricFunction::ArcSinh => arcsinh(ca),
        TrigonometricFunction::ArcCosh => arccosh(ca),
        TrigonometricFunction::ArcTanh => arctanh(ca),
    }
}

fn sin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sin()).into_series())
}

fn cos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cos()).into_series())
}

fn tan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.tan()).into_series())
}

fn arcsin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.asin()).into_series())
}

fn arccos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.acos()).into_series())
}

fn arctan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.atan()).into_series())
}

fn sinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sinh()).into_series())
}

fn cosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cosh()).into_series())
}

fn tanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.tanh()).into_series())
}

fn arcsinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.asinh()).into_series())
}

fn arccosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.acosh()).into_series())
}

fn arctanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.atanh()).into_series())
}
