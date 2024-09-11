use super::*;

pub(super) fn upper_bound(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    let name = s.name().clone();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Column::new(name, &[i8::MAX]),
        #[cfg(feature = "dtype-i16")]
        Int16 => Column::new(name, &[i16::MAX]),
        Int32 => Column::new(name, &[i32::MAX]),
        Int64 => Column::new(name, &[i64::MAX]),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Column::new(name, &[u8::MAX]),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Column::new(name, &[u16::MAX]),
        UInt32 => Column::new(name, &[u32::MAX]),
        UInt64 => Column::new(name, &[u64::MAX]),
        Float32 => Column::new(name, &[f32::INFINITY]),
        Float64 => Column::new(name, &[f64::INFINITY]),
        dt => polars_bail!(
            ComputeError: "cannot determine upper bound for dtype `{}`", dt,
        ),
    };
    Ok(s)
}

pub(super) fn lower_bound(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    let name = s.name().clone();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Column::new(name, &[i8::MIN]),
        #[cfg(feature = "dtype-i16")]
        Int16 => Column::new(name, &[i16::MIN]),
        Int32 => Column::new(name, &[i32::MIN]),
        Int64 => Column::new(name, &[i64::MIN]),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Column::new(name, &[u8::MIN]),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Column::new(name, &[u16::MIN]),
        UInt32 => Column::new(name, &[u32::MIN]),
        UInt64 => Column::new(name, &[u64::MIN]),
        Float32 => Column::new(name, &[f32::NEG_INFINITY]),
        Float64 => Column::new(name, &[f64::NEG_INFINITY]),
        dt => polars_bail!(
            ComputeError: "cannot determine lower bound for dtype `{}`", dt,
        ),
    };
    Ok(s)
}
