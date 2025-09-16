use num_traits::Float;
use polars_core::chunked_array::ops::arity::broadcast_binary_elementwise;

use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum IRTrigonometricFunction {
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

impl Display for IRTrigonometricFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            IRTrigonometricFunction::Cos => write!(f, "cos"),
            IRTrigonometricFunction::Cot => write!(f, "cot"),
            IRTrigonometricFunction::Sin => write!(f, "sin"),
            IRTrigonometricFunction::Tan => write!(f, "tan"),
            IRTrigonometricFunction::ArcCos => write!(f, "arccos"),
            IRTrigonometricFunction::ArcSin => write!(f, "arcsin"),
            IRTrigonometricFunction::ArcTan => write!(f, "arctan"),
            IRTrigonometricFunction::Cosh => write!(f, "cosh"),
            IRTrigonometricFunction::Sinh => write!(f, "sinh"),
            IRTrigonometricFunction::Tanh => write!(f, "tanh"),
            IRTrigonometricFunction::ArcCosh => write!(f, "arccosh"),
            IRTrigonometricFunction::ArcSinh => write!(f, "arcsinh"),
            IRTrigonometricFunction::ArcTanh => write!(f, "arctanh"),
            IRTrigonometricFunction::Degrees => write!(f, "degrees"),
            IRTrigonometricFunction::Radians => write!(f, "radians"),
        }
    }
}

impl From<IRTrigonometricFunction> for IRFunctionExpr {
    fn from(value: IRTrigonometricFunction) -> Self {
        Self::Trigonometry(value)
    }
}

pub(super) fn apply_trigonometric_function(
    s: &Column,
    trig_function: IRTrigonometricFunction,
) -> PolarsResult<Column> {
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
        dt if dt.is_primitive_numeric() => {
            let s = s.cast(&Float64)?;
            apply_trigonometric_function(&s, trig_function)
        },
        dt => polars_bail!(op = "trigonometry", dt),
    }
}

pub(super) fn apply_arctan2(s: &mut [Column]) -> PolarsResult<Column> {
    let y = &s[0];
    let x = &s[1];

    let y_len = y.len();
    let x_len = x.len();

    match (y_len, x_len) {
        (1, _) | (_, 1) => arctan2_on_columns(y, x),
        (len_a, len_b) if len_a == len_b => arctan2_on_columns(y, x),
        _ => polars_bail!(
            ComputeError:
            "y shape: {} in `arctan2` expression does not match that of x: {}",
            y_len, x_len,
        ),
    }
}

fn arctan2_on_columns(y: &Column, x: &Column) -> PolarsResult<Column> {
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
            arctan2_on_columns(&y, x)
        },
    }
}

fn arctan2_on_floats<T>(y: &ChunkedArray<T>, x: &Column) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    let dtype = T::get_static_dtype();
    let x = x.cast(&dtype)?;
    let x = y
        .unpack_series_matching_type(x.as_materialized_series())
        .unwrap();

    Ok(broadcast_binary_elementwise(y, x, |yv, xv| Some(yv?.atan2(xv?))).into_column())
}

fn apply_trigonometric_function_to_float<T>(
    ca: &ChunkedArray<T>,
    trig_function: IRTrigonometricFunction,
) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    match trig_function {
        IRTrigonometricFunction::Cos => cos(ca),
        IRTrigonometricFunction::Cot => cot(ca),
        IRTrigonometricFunction::Sin => sin(ca),
        IRTrigonometricFunction::Tan => tan(ca),
        IRTrigonometricFunction::ArcCos => arccos(ca),
        IRTrigonometricFunction::ArcSin => arcsin(ca),
        IRTrigonometricFunction::ArcTan => arctan(ca),
        IRTrigonometricFunction::Cosh => cosh(ca),
        IRTrigonometricFunction::Sinh => sinh(ca),
        IRTrigonometricFunction::Tanh => tanh(ca),
        IRTrigonometricFunction::ArcCosh => arccosh(ca),
        IRTrigonometricFunction::ArcSinh => arcsinh(ca),
        IRTrigonometricFunction::ArcTanh => arctanh(ca),
        IRTrigonometricFunction::Degrees => degrees(ca),
        IRTrigonometricFunction::Radians => radians(ca),
    }
}

fn cos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.cos()).into_column())
}

fn cot<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.tan().powi(-1)).into_column())
}

fn sin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.sin()).into_column())
}

fn tan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.tan()).into_column())
}

fn arccos<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.acos()).into_column())
}

fn arcsin<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.asin()).into_column())
}

fn arctan<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.atan()).into_column())
}

fn cosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.cosh()).into_column())
}

fn sinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.sinh()).into_column())
}

fn tanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.tanh()).into_column())
}

fn arccosh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.acosh()).into_column())
}

fn arcsinh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.asinh()).into_column())
}

fn arctanh<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.atanh()).into_column())
}

fn degrees<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.to_degrees()).into_column())
}

fn radians<T>(ca: &ChunkedArray<T>) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: IntoColumn,
{
    Ok(ca.apply_values(|v| v.to_radians()).into_column())
}
