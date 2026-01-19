use polars_core::chunked_array::arg_min_max::{
    arg_max_binary, arg_max_bool, arg_max_numeric, arg_max_str, arg_min_binary, arg_min_bool,
    arg_min_numeric, arg_min_str,
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
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use DataType::*;
    match $key_type {
        #[cfg(feature = "dtype-i8")]
        Int8 => __with_ty__! { Int8Type },
        #[cfg(feature = "dtype-i16")]
        Int16 => __with_ty__! { Int16Type },
        Int32 => __with_ty__! { Int32Type },
        Int64 => __with_ty__! { Int64Type },
        #[cfg(feature = "dtype-i128")]
        Int128 => __with_ty__! { Int128Type },
        #[cfg(feature = "dtype-u8")]
        UInt8 => __with_ty__! { UInt8Type },
        #[cfg(feature = "dtype-u16")]
        UInt16 => __with_ty__! { UInt16Type },
        UInt32 => __with_ty__! { UInt32Type },
        UInt64 => __with_ty__! { UInt64Type },
        #[cfg(feature = "dtype-u128")]
        UInt128 => __with_ty__! { UInt128Type },
        #[cfg(feature = "dtype-f16")]
        Float16 => __with_ty__! { Float16Type },
        Float32 => __with_ty__! { Float32Type },
        Float64 => __with_ty__! { Float64Type },
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
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    arg_min_cat(self.cat::<$C>().unwrap())
                })
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _) => phys_s.arg_min(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => phys_s.arg_min(),
            Date | Datetime(_, _) | Duration(_) | Time => phys_s.arg_min(),
            String => arg_min_str(self.str().unwrap()),
            Binary => arg_min_binary(self.binary().unwrap()),
            Boolean => arg_min_bool(self.bool().unwrap()),
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(phys_s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = phys_s.as_ref().as_ref().as_ref();
                    arg_min_numeric(ca)
                })
            },
            _ => None,
        }
    }

    fn arg_max(&self) -> Option<usize> {
        use DataType::*;
        let phys_s = self.to_physical_repr();
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            Categorical(cats, _) => {
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    arg_max_cat(self.cat::<$C>().unwrap())
                })
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _) => phys_s.arg_max(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => phys_s.arg_max(),
            Date | Datetime(_, _) | Duration(_) | Time => phys_s.arg_max(),
            String => arg_max_str(self.str().unwrap()),
            Binary => arg_max_binary(self.binary().unwrap()),
            Boolean => arg_max_bool(self.bool().unwrap()),
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(phys_s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = phys_s.as_ref().as_ref().as_ref();
                    arg_max_numeric(ca)
                })
            },
            _ => None,
        }
    }
}
