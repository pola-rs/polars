//! Macros that run a body once per [`DataType`](crate::DataType), with the type it holds bound.
//!
//! The `#[cfg]`s the arms carry are expanded into the caller's crate, so an arm is kept or dropped
//! by the `dtype-*` features of the *calling* crate rather than by those of `polars-dtype`. A crate
//! that calls one of these therefore needs the `dtype-*` features it wants arms for.

/// Runs a body with `$T` bound to the native type a primitive numeric [`DataType`](crate::DataType)
/// holds.
///
/// Panics on any other data type.
#[macro_export]
macro_rules! with_match_physical_numeric_type {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    #[cfg(feature = "dtype-f16")]
    use polars_utils::float16::pf16;
    use $crate::DataType::*;
    match $dtype {
        #[cfg(feature = "dtype-i8")]
        Int8 => __with_ty__! { i8 },
        #[cfg(feature = "dtype-i16")]
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        #[cfg(feature = "dtype-i128")]
        Int128 => __with_ty__! { i128 },
        #[cfg(feature = "dtype-u8")]
        UInt8 => __with_ty__! { u8 },
        #[cfg(feature = "dtype-u16")]
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        #[cfg(feature = "dtype-u128")]
        UInt128 => __with_ty__! { u128 },
        #[cfg(feature = "dtype-f16")]
        Float16 => __with_ty__! { pf16 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

/// Runs a body with `$T` bound to the native type an integer [`DataType`](crate::DataType) holds.
///
/// Panics on any other data type.
#[macro_export]
macro_rules! with_match_physical_integer_type {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::DataType::*;
    match $dtype {
        #[cfg(feature = "dtype-i8")]
        Int8 => __with_ty__! { i8 },
        #[cfg(feature = "dtype-i16")]
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        #[cfg(feature = "dtype-i128")]
        Int128 => __with_ty__! { i128 },
        #[cfg(feature = "dtype-u8")]
        UInt8 => __with_ty__! { u8 },
        #[cfg(feature = "dtype-u16")]
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        #[cfg(feature = "dtype-u128")]
        UInt128 => __with_ty__! { u128 },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

/// Runs a body with `$T` bound to the native type a float [`DataType`](crate::DataType) holds.
///
/// Panics on any other data type.
#[macro_export]
macro_rules! with_match_physical_float_type {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    #[cfg(feature = "dtype-f16")]
    use polars_utils::float16::pf16;
    use $crate::DataType::*;
    match $dtype {
        #[cfg(feature = "dtype-f16")]
        Float16 => __with_ty__! { pf16 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}
