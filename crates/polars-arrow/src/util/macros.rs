#[macro_export]
macro_rules! with_match_primitive_type {(
    $key_type:expr, impl<$T:ident> $($body:tt)*
) => ({
    use polars_utils::float16::pf16;
    use $crate::datatypes::PrimitiveType::*;

    match $key_type {
        Int8 => { #[allow(dead_code)] type $T = i8; $($body)* },
        Int16 => { #[allow(dead_code)] type $T = i16; $($body)* },
        Int32 => { #[allow(dead_code)] type $T = i32; $($body)* },
        Int64 => { #[allow(dead_code)] type $T = i64; $($body)* },
        Int128 => { #[allow(dead_code)] type $T = i128; $($body)* },
        UInt8 => { #[allow(dead_code)] type $T = u8; $($body)* },
        UInt16 => { #[allow(dead_code)] type $T = u16; $($body)* },
        UInt32 => { #[allow(dead_code)] type $T = u32; $($body)* },
        UInt64 => { #[allow(dead_code)] type $T = u64; $($body)* },
        UInt128 => { #[allow(dead_code)] type $T = u128; $($body)* },
        Float16 => { #[allow(dead_code)] type $T = pf16; $($body)* },
        Float32 => { #[allow(dead_code)] type $T = f32; $($body)* },
        Float64 => { #[allow(dead_code)] type $T = f64; $($body)* },
        _ => panic!("operator does not support primitive `{:?}`",
            $key_type)
    }
})}

#[macro_export]
macro_rules! with_match_primitive_type_full {(
    $key_type:expr, impl<$T:ident> $($body:tt)*
) => ({
    use polars_utils::float16::pf16;
    use $crate::datatypes::PrimitiveType::*;

    match $key_type {
        Int8 => { #[allow(dead_code)] type $T = i8; $($body)* },
        Int16 => { #[allow(dead_code)] type $T = i16; $($body)* },
        Int32 => { #[allow(dead_code)] type $T = i32; $($body)* },
        Int64 => { #[allow(dead_code)] type $T = i64; $($body)* },
        Int128 => { #[allow(dead_code)] type $T = i128; $($body)* },
        UInt8 => { #[allow(dead_code)] type $T = u8; $($body)* },
        UInt16 => { #[allow(dead_code)] type $T = u16; $($body)* },
        UInt32 => { #[allow(dead_code)] type $T = u32; $($body)* },
        UInt64 => { #[allow(dead_code)] type $T = u64; $($body)* },
        UInt128 => { #[allow(dead_code)] type $T = u128; $($body)* },
        Float16 => { #[allow(dead_code)] type $T = pf16; $($body)* },
        Float32 => { #[allow(dead_code)] type $T = f32; $($body)* },
        Float64 => { #[allow(dead_code)] type $T = f64; $($body)* },
        _ => panic!("operator does not support primitive `{:?}`",
            $key_type)
    }
})}

#[macro_export]
macro_rules! match_integer_type {(
    $key_type:expr, impl<$T:ident> $($body:tt)*
) => ({
    use $crate::datatypes::IntegerType::*;
    match $key_type {
        Int8 => { #[allow(dead_code)] type $T = i8; $($body)* },
        Int16 => { #[allow(dead_code)] type $T = i16; $($body)* },
        Int32 => { #[allow(dead_code)] type $T = i32; $($body)* },
        Int64 => { #[allow(dead_code)] type $T = i64; $($body)* },
        Int128 => { #[allow(dead_code)] type $T = i128; $($body)* },
        UInt8 => { #[allow(dead_code)] type $T = u8; $($body)* },
        UInt16 => { #[allow(dead_code)] type $T = u16; $($body)* },
        UInt32 => { #[allow(dead_code)] type $T = u32; $($body)* },
        UInt64 => { #[allow(dead_code)] type $T = u64; $($body)* },
        UInt128 => { #[allow(dead_code)] type $T = u128; $($body)* },
    }
})}
