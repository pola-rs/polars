macro_rules! with_match_physical_numpy_polars_type {(
    $key_type:expr, impl<$T:ident> $($body:tt)*
) => ({
    use polars_core::datatypes::DataType as D;
    match $key_type {
        #[cfg(feature = "dtype-i8")]
        D::Int8 => { type $T = Int8Type; $($body)* },
        #[cfg(feature = "dtype-i16")]
        D::Int16 => { type $T = Int16Type; $($body)* },
        D::Int32 => { type $T = Int32Type; $($body)* },
        D::Int64 => { type $T = Int64Type; $($body)* },
        #[cfg(feature = "dtype-u8")]
        D::UInt8 => { type $T = UInt8Type; $($body)* },
        #[cfg(feature = "dtype-u16")]
        D::UInt16 => { type $T = UInt16Type; $($body)* },
        D::UInt32 => { type $T = UInt32Type; $($body)* },
        D::UInt64 => { type $T = UInt64Type; $($body)* },
        #[cfg(feature = "dtype-f16")]
        D::Float16 => { type $T = Float16Type; $($body)* },
        D::Float32 => { type $T = Float32Type; $($body)* },
        D::Float64 => { type $T = Float64Type; $($body)* },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

pub mod to_numpy_df;
pub mod to_numpy_series;
mod utils;
