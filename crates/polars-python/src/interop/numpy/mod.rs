macro_rules! with_match_physical_numpy_polars_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use polars_core::datatypes::DataType as D;
    match $key_type {
        #[cfg(feature = "dtype-i8")]
        D::Int8 => __with_ty__! { Int8Type },
        #[cfg(feature = "dtype-i16")]
        D::Int16 => __with_ty__! { Int16Type },
        D::Int32 => __with_ty__! { Int32Type },
        D::Int64 => __with_ty__! { Int64Type },
        #[cfg(feature = "dtype-u8")]
        D::UInt8 => __with_ty__! { UInt8Type },
        #[cfg(feature = "dtype-u16")]
        D::UInt16 => __with_ty__! { UInt16Type },
        D::UInt32 => __with_ty__! { UInt32Type },
        D::UInt64 => __with_ty__! { UInt64Type },
        D::Float32 => __with_ty__! { Float32Type },
        D::Float64 => __with_ty__! { Float64Type },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

pub mod to_numpy_df;
pub mod to_numpy_series;
mod utils;
