use super::*;
use num::Float;
use polars_core::export::num;

pub(super) fn trigo(s: &Series, trigotype: TrigoType) -> Result<Series> {
    use DataType::*;
    match s.dtype() {
        Float32 => {
            let ca = s.f32().unwrap();
            trigo_float(ca, trigotype)
        }
        Float64 => {
            let ca = s.f64().unwrap();
            trigo_float(ca, trigotype)
        }
        _ => {
            let s = s.cast(&DataType::Float64)?;
            trigo(&s, trigotype)
        }
    }
}

fn trigo_float<T>(ca: &ChunkedArray<T>, trigotype: TrigoType) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    match trigotype {
        TrigoType::Sin => sin(ca),
        TrigoType::Cos => cos(ca),
        TrigoType::Tan => tan(ca),
        TrigoType::ArcSin => arcsin(ca),
        TrigoType::ArcCos => arccos(ca),
        TrigoType::ArcTan => arctan(ca),
        TrigoType::Sinh => sinh(ca),
        TrigoType::Cosh => cosh(ca),
        TrigoType::Tanh => tanh(ca),
        TrigoType::ArcSinh => arcsinh(ca),
        TrigoType::ArcCosh => arccosh(ca),
        TrigoType::ArcTanh => arctanh(ca),
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
