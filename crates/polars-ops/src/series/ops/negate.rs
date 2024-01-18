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
        #[cfg(feature = "dtype-i8")]
        Int8 => negate_numeric(s.i8().unwrap()).into_series(),
        #[cfg(feature = "dtype-i16")]
        Int16 => negate_numeric(s.i16().unwrap()).into_series(),
        Int32 => negate_numeric(s.i32().unwrap()).into_series(),
        Int64 => negate_numeric(s.i64().unwrap()).into_series(),
        Float32 => negate_numeric(s.f32().unwrap()).into_series(),
        Float64 => negate_numeric(s.f64().unwrap()).into_series(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => {
            let ca = s.decimal().unwrap();
            let precision = ca.precision();
            let scale = ca.scale();

            let out = negate_numeric(ca.as_ref());
            out.into_decimal_unchecked(precision, scale).into_series()
        },
        #[cfg(feature = "dtype-duration")]
        Duration(_) => {
            let physical = s.to_physical_repr();
            let ca = physical.i64().unwrap();
            let s = negate_numeric(ca).into_series();
            s.cast(s.dtype())?
        },
        dt => polars_bail!(opq = neg, dt),
    };
    Ok(out)
}
