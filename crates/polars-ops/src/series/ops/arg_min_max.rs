use polars_core::chunked_array::arg_min_max::{
    arg_max_binary, arg_max_binary_offset, arg_max_bool, arg_max_numeric, arg_max_str,
    arg_min_binary, arg_min_binary_offset, arg_min_bool, arg_min_numeric, arg_min_str,
};
#[cfg(feature = "dtype-categorical")]
use polars_core::chunked_array::arg_min_max::{arg_max_cat, arg_min_cat};
#[cfg(feature = "dtype-categorical")]
use polars_core::with_match_categorical_physical_type;

use super::*;

/// Argmin/ Argmax
pub trait ArgAgg {
    /// Get the index of the minimal value
    fn arg_min(&self) -> Option<usize>;
    /// Get the index of the maximal value
    fn arg_max(&self) -> Option<usize>;
}

macro_rules! with_match_physical_numeric_polars_type {(
    $key_type:expr, impl<$T:ident> $($body:tt)*
) => ({
    use DataType::*;
    match $key_type {
        #[cfg(feature = "dtype-i8")]
        Int8 => { #[allow(dead_code)] type $T = Int8Type; $($body)* },
        #[cfg(feature = "dtype-i16")]
        Int16 => { #[allow(dead_code)] type $T = Int16Type; $($body)* },
        Int32 => { #[allow(dead_code)] type $T = Int32Type; $($body)* },
        Int64 => { #[allow(dead_code)] type $T = Int64Type; $($body)* },
        #[cfg(feature = "dtype-i128")]
        Int128 => { #[allow(dead_code)] type $T = Int128Type; $($body)* },
        #[cfg(feature = "dtype-u8")]
        UInt8 => { #[allow(dead_code)] type $T = UInt8Type; $($body)* },
        #[cfg(feature = "dtype-u16")]
        UInt16 => { #[allow(dead_code)] type $T = UInt16Type; $($body)* },
        UInt32 => { #[allow(dead_code)] type $T = UInt32Type; $($body)* },
        UInt64 => { #[allow(dead_code)] type $T = UInt64Type; $($body)* },
        #[cfg(feature = "dtype-u128")]
        UInt128 => { #[allow(dead_code)] type $T = UInt128Type; $($body)* },
        #[cfg(feature = "dtype-f16")]
        Float16 => { #[allow(dead_code)] type $T = Float16Type; $($body)* },
        Float32 => { #[allow(dead_code)] type $T = Float32Type; $($body)* },
        Float64 => { #[allow(dead_code)] type $T = Float64Type; $($body)* },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

impl ArgAgg for Series {
    fn arg_min(&self) -> Option<usize> {
        use DataType::*;
        let phys_s = self.to_physical_repr();
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            Categorical(cats, _) => {
                with_match_categorical_physical_type!(cats.physical(), impl<C> {
                    arg_min_cat(self.cat::<C>().unwrap())
                })
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _) => phys_s.arg_min(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => phys_s.arg_min(),
            Date | Datetime(_, _) | Duration(_) | Time => phys_s.arg_min(),
            String => arg_min_str(self.str().unwrap()),
            Binary => arg_min_binary(self.binary().unwrap()),
            BinaryOffset => arg_min_binary_offset(self.binary_offset().unwrap()),
            Boolean => arg_min_bool(self.bool().unwrap()),
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(phys_s.dtype(), impl<T> {
                    let ca: &ChunkedArray<T> = phys_s.as_ref().as_ref().as_ref();
                    arg_min_numeric(ca)
                })
            },
            dt if dt.is_nested() => self
                .row_encode_ordered(false, false)
                .ok()?
                .into_series()
                .arg_min(),
            _ => None,
        }
    }

    fn arg_max(&self) -> Option<usize> {
        use DataType::*;
        let phys_s = self.to_physical_repr();
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            Categorical(cats, _) => {
                with_match_categorical_physical_type!(cats.physical(), impl<C> {
                    arg_max_cat(self.cat::<C>().unwrap())
                })
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _) => phys_s.arg_max(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => phys_s.arg_max(),
            Date | Datetime(_, _) | Duration(_) | Time => phys_s.arg_max(),
            String => arg_max_str(self.str().unwrap()),
            Binary => arg_max_binary(self.binary().unwrap()),
            BinaryOffset => arg_max_binary_offset(self.binary_offset().unwrap()),
            Boolean => arg_max_bool(self.bool().unwrap()),
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(phys_s.dtype(), impl<T> {
                    let ca: &ChunkedArray<T> = phys_s.as_ref().as_ref().as_ref();
                    arg_max_numeric(ca)
                })
            },
            dt if dt.is_nested() => self
                .row_encode_ordered(false, false)
                .ok()?
                .into_series()
                .arg_max(),
            _ => None,
        }
    }
}
