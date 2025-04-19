use polars_utils::pl_str::PlSmallStr;

use super::{AnyValue, DataType, Scalar};

macro_rules! impl_from {
    ($(($t:ty, $av:ident, $dt:ident))+) => {
        $(
            impl From<$t> for Scalar {
                #[inline]
                fn from(v: $t) -> Self {
                    Self::new(DataType::$dt, AnyValue::$av(v))
                }
            }
        )+
    }
}

impl_from! {
    (bool, Boolean, Boolean)
    (i8, Int8, Int8)
    (i16, Int16, Int16)
    (i32, Int32, Int32)
    (i64, Int64, Int64)
    (i128, Int128, Int128)
    (u8, UInt8, UInt8)
    (u16, UInt16, UInt16)
    (u32, UInt32, UInt32)
    (u64, UInt64, UInt64)
    (f32, Float32, Float32)
    (f64, Float64, Float64)
    (PlSmallStr, StringOwned, String)
    (Vec<u8>, BinaryOwned, Binary)
}
