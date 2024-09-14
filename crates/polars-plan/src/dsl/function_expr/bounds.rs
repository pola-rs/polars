use super::*;

pub(super) fn upper_bound(s: &Column) -> PolarsResult<Column> {
    let name = s.name().clone();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Column::new_scalar(name, Scalar::from(i8::MAX), 1),
        #[cfg(feature = "dtype-i16")]
        Int16 => Column::new_scalar(name, Scalar::from(i16::MAX), 1),
        Int32 => Column::new_scalar(name, Scalar::from(i32::MAX), 1),
        Int64 => Column::new_scalar(name, Scalar::from(i64::MAX), 1),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Column::new_scalar(name, Scalar::from(u8::MAX), 1),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Column::new_scalar(name, Scalar::from(u16::MAX), 1),
        UInt32 => Column::new_scalar(name, Scalar::from(u32::MAX), 1),
        UInt64 => Column::new_scalar(name, Scalar::from(u64::MAX), 1),
        Float32 => Column::new_scalar(name, Scalar::from(f32::INFINITY), 1),
        Float64 => Column::new_scalar(name, Scalar::from(f64::INFINITY), 1),
        dt => polars_bail!(
            ComputeError: "cannot determine upper bound for dtype `{}`", dt,
        ),
    };
    Ok(s)
}

pub(super) fn lower_bound(s: &Column) -> PolarsResult<Column> {
    let name = s.name().clone();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Column::new_scalar(name, Scalar::from(i8::MIN), 1),
        #[cfg(feature = "dtype-i16")]
        Int16 => Column::new_scalar(name, Scalar::from(i16::MIN), 1),
        Int32 => Column::new_scalar(name, Scalar::from(i32::MIN), 1),
        Int64 => Column::new_scalar(name, Scalar::from(i64::MIN), 1),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Column::new_scalar(name, Scalar::from(u8::MIN), 1),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Column::new_scalar(name, Scalar::from(u16::MIN), 1),
        UInt32 => Column::new_scalar(name, Scalar::from(u32::MIN), 1),
        UInt64 => Column::new_scalar(name, Scalar::from(u64::MIN), 1),
        Float32 => Column::new_scalar(name, Scalar::from(f32::NEG_INFINITY), 1),
        Float64 => Column::new_scalar(name, Scalar::from(f64::NEG_INFINITY), 1),
        dt => polars_bail!(
            ComputeError: "cannot determine lower bound for dtype `{}`", dt,
        ),
    };
    Ok(s)
}
