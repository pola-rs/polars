use num_traits::Signed;
use polars_core::prelude::*;

fn negate_numeric<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    ca.apply_values(|v| -v)
}

pub fn negate(s: &Series) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Int8 => negate_numeric(s.i8().unwrap()).into_series(),
        Int16 => negate_numeric(s.i16().unwrap()).into_series(),
        Int32 => negate_numeric(s.i32().unwrap()).into_series(),
        Int64 => negate_numeric(s.i64().unwrap()).into_series(),
        Float32 => negate_numeric(s.f32().unwrap()).into_series(),
        Float64 => negate_numeric(s.f64().unwrap()).into_series(),
        dt => polars_bail!(opq = neg, dt),
    };
    Ok(out)
}
