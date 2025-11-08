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
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    match $key_type {
        Int8 => __with_ty__! { Int8Type },
        Int16 => __with_ty__! { Int16Type },
        Int32 => __with_ty__! { Int32Type },
        Int64 => __with_ty__! { Int64Type },
        Int128 => __with_ty__! { Int128Type },
        UInt8 => __with_ty__! { UInt8Type },
        UInt16 => __with_ty__! { UInt16Type },
        UInt32 => __with_ty__! { UInt32Type },
        UInt64 => __with_ty__! { UInt64Type },
        UInt128 => __with_ty__! { UInt128Type },
        Float32 => __with_ty__! { Float32Type },
        Float64 => __with_ty__! { Float64Type },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}
