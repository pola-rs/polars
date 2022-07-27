use super::*;
use num::Float;
use polars_core::export::num;

pub(super) fn apply_trigonometric_function(
    s: &Series,
    trig_function: TrigonometricFunction,
) -> Result<Series> {
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
            format!(
                "cannot use trigonometric function on Series of dtype: {:?}",
                dt
            )
            .into(),
        )),
    }
}

fn apply_trigonometric_function_to_float<T>(
    ca: &ChunkedArray<T>,
    trig_function: TrigonometricFunction,
) -> Result<Series>
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

fn sin<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sin()).into_series())
}

fn cos<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cos()).into_series())
}

fn tan<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.tan()).into_series())
}

fn arcsin<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.asin()).into_series())
}

fn arccos<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.acos()).into_series())
}

fn arctan<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.atan()).into_series())
}

fn sinh<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sinh()).into_series())
}

fn cosh<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cosh()).into_series())
}

fn tanh<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.tanh()).into_series())
}

fn arcsinh<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.asinh()).into_series())
}

fn arccosh<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.acosh()).into_series())
}

fn arctanh<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.atanh()).into_series())
}
