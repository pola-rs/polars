use polars_core::prelude::*;

pub fn negate(s: &Series) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        #[cfg(feature = "dtype-i8")]
        Int8 => s.i8().unwrap().wrapping_neg().into_series(),
        #[cfg(feature = "dtype-i16")]
        Int16 => s.i16().unwrap().wrapping_neg().into_series(),
        Int32 => s.i32().unwrap().wrapping_neg().into_series(),
        Int64 => s.i64().unwrap().wrapping_neg().into_series(),
        Float32 => s.f32().unwrap().wrapping_neg().into_series(),
        Float64 => s.f64().unwrap().wrapping_neg().into_series(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => {
            let ca = s.decimal().unwrap();
            let precision = ca.precision();
            let scale = ca.scale();

            let out = ca.as_ref().wrapping_neg();
            out.into_decimal_unchecked(precision, scale).into_series()
        },
        #[cfg(feature = "dtype-duration")]
        Duration(_) => {
            let physical = s.to_physical_repr();
            let ca = physical.i64().unwrap();
            let out = ca.wrapping_neg().into_series();
            out.cast(s.dtype())?
        },
        dt => polars_bail!(opq = neg, dt),
    };
    Ok(out)
}
