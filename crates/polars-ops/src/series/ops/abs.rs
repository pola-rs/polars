use num_traits::Signed;
use polars_core::prelude::*;

fn abs_numeric<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    ca.apply_values(|v| v.abs())
}

/// Convert numerical values to their absolute value.
pub fn abs(s: &Series) -> PolarsResult<Series> {
    let physical_s = s.to_physical_repr();
    use DataType::*;
    let out = match physical_s.dtype() {
        #[cfg(feature = "dtype-i8")]
        Int8 => abs_numeric(physical_s.i8()?).into_series(),
        #[cfg(feature = "dtype-i16")]
        Int16 => abs_numeric(physical_s.i16()?).into_series(),
        Int32 => abs_numeric(physical_s.i32()?).into_series(),
        Int64 => abs_numeric(physical_s.i64()?).into_series(),
        UInt8 | UInt16 | UInt32 | UInt64 => s.clone(),
        Float32 => abs_numeric(physical_s.f32()?).into_series(),
        Float64 => abs_numeric(physical_s.f64()?).into_series(),
        dt => polars_bail!(opq = abs, dt),
    };
    out.cast(s.dtype())
}
