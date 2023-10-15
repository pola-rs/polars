use arrow::legacy::kernels::atan2::atan2 as atan2_kernel;
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
        },
        Float64 => {
            let ca = s.f64().unwrap();
            apply_trigonometric_function_to_float(ca, trig_function)
        },
        dt if dt.is_numeric() => {
            let s = s.cast(&Float64)?;
            apply_trigonometric_function(&s, trig_function)
        },
        dt => polars_bail!(op = "trigonometry", dt),
    }
}

pub(super) fn apply_arctan2(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let y = &s[0];
    let x = &s[1];

    let y_len = y.len();
    let x_len = x.len();

    match (y_len, x_len) {
        (1, _) | (_, 1) => arctan2_on_series(y, x),
        (len_a, len_b) if len_a == len_b => arctan2_on_series(y, x),
        _ => polars_bail!(
            ComputeError:
            "y shape: {} in `arctan2` expression does not match that of x: {}",
            y_len, x_len,
        ),
    }
}

fn arctan2_on_series(y: &Series, x: &Series) -> PolarsResult<Option<Series>> {
    use DataType::*;
    match y.dtype() {
        Float32 => {
            let y_ca: &ChunkedArray<Float32Type> = y.f32().unwrap();
            arctan2_on_floats(y_ca, x)
        },
        Float64 => {
            let y_ca: &ChunkedArray<Float64Type> = y.f64().unwrap();
            arctan2_on_floats(y_ca, x)
        },
        _ => {
            let y = y.cast(&DataType::Float64)?;
            arctan2_on_series(&y, x)
        },
    }
}

fn arctan2_on_floats<T>(y: &ChunkedArray<T>, x: &Series) -> PolarsResult<Option<Series>>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    let dtype = T::get_dtype();
    let x = x.cast(&dtype)?;
    let x = y.unpack_series_matching_type(&x).unwrap();

    if x.len() == 1 {
        let x_value = x
            .get(0)
            .ok_or_else(|| polars_err!(ComputeError: "arctan2 x value is null"))?;

        Ok(Some(y.apply_values(|v| v.atan2(x_value)).into_series()))
    } else if y.len() == 1 {
        let y_value = y
            .get(0)
            .ok_or_else(|| polars_err!(ComputeError: "arctan2 y value is null"))?;

        Ok(Some(x.apply_values(|v| y_value.atan2(v)).into_series()))
    } else {
        Ok(Some(
            polars_core::prelude::arity::binary(y, x, atan2_kernel).into_series(),
        ))
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
    Ok(ca.apply_values(|v| v.cos()).into_series())
}

fn cot<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.tan().powi(-1)).into_series())
}

fn sin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.sin()).into_series())
}

fn tan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.tan()).into_series())
}

fn arccos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.acos()).into_series())
}

fn arcsin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.asin()).into_series())
}

fn arctan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.atan()).into_series())
}

fn cosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.cosh()).into_series())
}

fn sinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.sinh()).into_series())
}

fn tanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.tanh()).into_series())
}

fn arccosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.acosh()).into_series())
}

fn arcsinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.asinh()).into_series())
}

fn arctanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.atanh()).into_series())
}

fn degrees<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.to_degrees()).into_series())
}

fn radians<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply_values(|v| v.to_radians()).into_series())
}
