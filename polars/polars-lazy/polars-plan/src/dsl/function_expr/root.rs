use num::Float;
use polars_core::export::num;

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum RootFunction {
    Sqrt,
    Cbrt,
}

impl Display for RootFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            RootFunction::Sqrt => write!(f, "sqrt"),
            RootFunction::Cbrt => write!(f, "cbrt"),
        }
    }
}

pub(super) fn apply_root_function(s: &Series, root_function: RootFunction) -> PolarsResult<Series> {
    use DataType::*;
    match s.dtype() {
        Float32 => {
            let ca = s.f32().unwrap();
            apply_root_function_to_float(ca, root_function)
        }
        Float64 => {
            let ca = s.f64().unwrap();
            apply_root_function_to_float(ca, root_function)
        }
        dt if dt.is_numeric() => {
            let s = s.cast(&DataType::Float64)?;
            apply_root_function(&s, root_function)
        }
        dt => Err(PolarsError::ComputeError(
            format!("cannot use root function on Series of dtype: {dt:?}",).into(),
        )),
    }
}

fn apply_root_function_to_float<T>(
    ca: &ChunkedArray<T>,
    root_function: RootFunction,
) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    match root_function {
        RootFunction::Sqrt => sqrt(ca),
        RootFunction::Cbrt => cbrt(ca),
    }
}

fn sqrt<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.sqrt()).into_series())
}

fn cbrt<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    Ok(ca.apply(|v| v.cbrt()).into_series())
}
