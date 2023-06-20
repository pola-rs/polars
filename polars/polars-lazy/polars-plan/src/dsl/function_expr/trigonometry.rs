use num::Float;
use polars_core::export::num;

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum TrigonometricFunction {
    Cos,
    Cot,
    Sin,
    Tan,
    ArcCos,
    ArcSin,
    ArcTan,
    Cosh,
    Sinh,
    Tanh,
    ArcCosh,
    ArcSinh,
    ArcTanh,
    Degrees,
    Radians,
}

impl Display for TrigonometricFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            TrigonometricFunction::Cos => write!(f, "cos"),
            TrigonometricFunction::Cot => write!(f, "cot"),
            TrigonometricFunction::Sin => write!(f, "sin"),
            TrigonometricFunction::Tan => write!(f, "tan"),
            TrigonometricFunction::ArcCos => write!(f, "arccos"),
            TrigonometricFunction::ArcSin => write!(f, "arcsin"),
            TrigonometricFunction::ArcTan => write!(f, "arctan"),
            TrigonometricFunction::Cosh => write!(f, "cosh"),
            TrigonometricFunction::Sinh => write!(f, "sinh"),
            TrigonometricFunction::Tanh => write!(f, "tanh"),
            TrigonometricFunction::ArcCosh => write!(f, "arccosh"),
            TrigonometricFunction::ArcSinh => write!(f, "arcsinh"),
            TrigonometricFunction::ArcTanh => write!(f, "arctanh"),
            TrigonometricFunction::Degrees => write!(f, "degrees"),
            TrigonometricFunction::Radians => write!(f, "radians"),
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
            let s = s.cast(&Float64)?;
            apply_trigonometric_function(&s, trig_function)
        }
        dt => polars_bail!(op = "trigonometry", dt),
    }
}

fn apply_trigonometric_function_to_float<T>(
    ca: &ChunkedArray<T>,
    trig_function: TrigonometricFunction,
) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    match trig_function {
        TrigonometricFunction::Cos => cos(ca),
        TrigonometricFunction::Cot => cot(ca),
        TrigonometricFunction::Sin => sin(ca),
        TrigonometricFunction::Tan => tan(ca),
        TrigonometricFunction::ArcCos => arccos(ca),
        TrigonometricFunction::ArcSin => arcsin(ca),
        TrigonometricFunction::ArcTan => arctan(ca),
        TrigonometricFunction::Cosh => cosh(ca),
        TrigonometricFunction::Sinh => sinh(ca),
        TrigonometricFunction::Tanh => tanh(ca),
        TrigonometricFunction::ArcCosh => arccosh(ca),
        TrigonometricFunction::ArcSinh => arcsinh(ca),
        TrigonometricFunction::ArcTanh => arctanh(ca),
        TrigonometricFunction::Degrees => degrees(ca),
        TrigonometricFunction::Radians => radians(ca),
    }
}

fn cos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cos()).into_series())
}

fn cot<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cos() / v.sin()).into_series())
}

fn sin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sin()).into_series())
}

fn tan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.tan()).into_series())
}

fn arccos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.acos()).into_series())
}

fn arcsin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.asin()).into_series())
}

fn arctan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.atan()).into_series())
}

fn cosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cosh()).into_series())
}

fn sinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sinh()).into_series())
}

fn tanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.tanh()).into_series())
}

fn arccosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.acosh()).into_series())
}

fn arcsinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.asinh()).into_series())
}

fn arctanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.atanh()).into_series())
}

fn degrees<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.to_degrees()).into_series())
}

fn radians<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.to_radians()).into_series())
}
