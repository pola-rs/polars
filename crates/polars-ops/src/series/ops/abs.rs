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
    use DataType::*;
    let out = match s.dtype() {
        #[cfg(feature = "dtype-i8")]
        Int8 => abs_numeric(s.i8().unwrap()).into_series(),
        #[cfg(feature = "dtype-i16")]
        Int16 => abs_numeric(s.i16().unwrap()).into_series(),
        Int32 => abs_numeric(s.i32().unwrap()).into_series(),
        Int64 => abs_numeric(s.i64().unwrap()).into_series(),
        #[cfg(feature = "dtype-u8")]
        UInt8 => s.clone(),
        #[cfg(feature = "dtype-u16")]
        UInt16 => s.clone(),
        UInt32 | UInt64 => s.clone(),
        Float32 => abs_numeric(s.f32().unwrap()).into_series(),
        Float64 => abs_numeric(s.f64().unwrap()).into_series(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => {
            let ca = s.decimal().unwrap();
            let precision = ca.precision();
            let scale = ca.scale();

            let out = abs_numeric(ca.as_ref());
            out.into_decimal_unchecked(precision, scale).into_series()
        },
        #[cfg(feature = "dtype-duration")]
        Duration(_) => {
            let physical = s.to_physical_repr();
            let ca = physical.i64().unwrap();
            let out = abs_numeric(ca).into_series();
            out.cast(s.dtype())?
        },
        dt => polars_bail!(opq = abs, dt),
    };
    Ok(out)
}
