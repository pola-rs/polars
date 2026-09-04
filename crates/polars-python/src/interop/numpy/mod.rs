macro_rules! with_match_physical_numpy_polars_type {(
    $key_type:expr, |$T:ident| $($body:tt)*
) => ({
    use polars_core::datatypes::DataType as D;
    match $key_type {
        #[cfg(feature = "dtype-i8")]
        D::Int8 => { #[allow(dead_code)] type $T = Int8Type; $($body)* },
        #[cfg(feature = "dtype-i16")]
        D::Int16 => { #[allow(dead_code)] type $T = Int16Type; $($body)* },
        D::Int32 => { #[allow(dead_code)] type $T = Int32Type; $($body)* },
        D::Int64 => { #[allow(dead_code)] type $T = Int64Type; $($body)* },
        #[cfg(feature = "dtype-u8")]
        D::UInt8 => { #[allow(dead_code)] type $T = UInt8Type; $($body)* },
        #[cfg(feature = "dtype-u16")]
        D::UInt16 => { #[allow(dead_code)] type $T = UInt16Type; $($body)* },
        D::UInt32 => { #[allow(dead_code)] type $T = UInt32Type; $($body)* },
        D::UInt64 => { #[allow(dead_code)] type $T = UInt64Type; $($body)* },
        #[cfg(feature = "dtype-f16")]
        D::Float16 => { #[allow(dead_code)] type $T = Float16Type; $($body)* },
        D::Float32 => { #[allow(dead_code)] type $T = Float32Type; $($body)* },
        D::Float64 => { #[allow(dead_code)] type $T = Float64Type; $($body)* },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

pub mod to_numpy_df;
pub mod to_numpy_series;
mod utils;
