#[macro_export]
macro_rules! matches_any_order {
    ($expression1:expr, $expression2:expr,  $( $pattern1:pat_param )|+,  $( $pattern2:pat_param )|+) => {
        (matches!($expression1, $( $pattern1 )|+) && matches!($expression2, $( $pattern2)|+)) ||
        matches!($expression2, $( $pattern1 ) |+) && matches!($expression1, $( $pattern2)|+)
    }
}

#[macro_export]
macro_rules! no_call_const {
    () => {{
        const { assert!(false, "should not be called") }
        unreachable!()
    }};
}

// Same as OSS except for the feature gates.
#[macro_export]
macro_rules! with_match_physical_numeric_polars_type {(
    $key_type:expr, |$T:ident| $($body:tt)*
) => ({
    use $crate::datatypes::DataType::*;
    match $key_type {
        Int8 => { #[allow(dead_code)] type $T = Int8Type; $($body)* },
        Int16 => { #[allow(dead_code)] type $T = Int16Type; $($body)* },
        Int32 => { #[allow(dead_code)] type $T = Int32Type; $($body)* },
        Int64 => { #[allow(dead_code)] type $T = Int64Type; $($body)* },
        Int128 => { #[allow(dead_code)] type $T = Int128Type; $($body)* },
        UInt8 => { #[allow(dead_code)] type $T = UInt8Type; $($body)* },
        UInt16 => { #[allow(dead_code)] type $T = UInt16Type; $($body)* },
        UInt32 => { #[allow(dead_code)] type $T = UInt32Type; $($body)* },
        UInt64 => { #[allow(dead_code)] type $T = UInt64Type; $($body)* },
        UInt128 => { #[allow(dead_code)] type $T = UInt128Type; $($body)* },
        Float16 => { #[allow(dead_code)] type $T = Float16Type; $($body)* },
        Float32 => { #[allow(dead_code)] type $T = Float32Type; $($body)* },
        Float64 => { #[allow(dead_code)] type $T = Float64Type; $($body)* },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}
