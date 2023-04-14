use super::*;

pub(super) fn upper_bound(s: &Series) -> PolarsResult<Series> {
    let name = s.name();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Series::new(name, &[i8::MAX]),
        #[cfg(feature = "dtype-i16")]
        Int16 => Series::new(name, &[i16::MAX]),
        Int32 => Series::new(name, &[i32::MAX]),
        Int64 => Series::new(name, &[i64::MAX]),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Series::new(name, &[u8::MAX]),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Series::new(name, &[u16::MAX]),
        UInt32 => Series::new(name, &[u32::MAX]),
        UInt64 => Series::new(name, &[u64::MAX]),
        Float32 => Series::new(name, &[f32::INFINITY]),
        Float64 => Series::new(name, &[f64::INFINITY]),
        dt => polars_bail!(
            ComputeError: "cannot determine upper bound for dtype `{}`", dt,
        ),
    };
    Ok(s)
}

pub(super) fn lower_bound(s: &Series) -> PolarsResult<Series> {
    let name = s.name();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Series::new(name, &[i8::MIN]),
        #[cfg(feature = "dtype-i16")]
        Int16 => Series::new(name, &[i16::MIN]),
        Int32 => Series::new(name, &[i32::MIN]),
        Int64 => Series::new(name, &[i64::MIN]),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Series::new(name, &[u8::MIN]),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Series::new(name, &[u16::MIN]),
        UInt32 => Series::new(name, &[u32::MIN]),
        UInt64 => Series::new(name, &[u64::MIN]),
        Float32 => Series::new(name, &[f32::NEG_INFINITY]),
        Float64 => Series::new(name, &[f64::NEG_INFINITY]),
        dt => polars_bail!(
            ComputeError: "cannot determine lower bound for dtype `{}`", dt,
        ),
    };
    Ok(s)
}
