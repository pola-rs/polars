use polars_error::{polars_bail, PolarsResult};

use super::{AnyValue, DataType, Scalar};

pub trait IntoScalar {
    fn into_scalar(self, dtype: DataType) -> PolarsResult<Scalar>;
}

macro_rules! impl_into_scalar {
    ($(
        $ty:ty: ($($dt:pat),+ $(,)?),
    )+) => {
        $(
        impl IntoScalar for $ty {
            fn into_scalar(self, dtype: DataType) -> PolarsResult<Scalar> {
                Ok(match &dtype {
                    T::Null => Scalar::new(dtype, AnyValue::Null),
                    $($dt => Scalar::new(dtype, self.into()),)+
                    _ => polars_bail!(InvalidOperation: "Cannot cast `{}` to `Scalar` with dtype={dtype}", stringify!($ty)),
                })
            }
        }
        )+
    };
}

use DataType as T;
impl_into_scalar! {
    bool: (T::Boolean),
    u8: (T::UInt8),
    u16: (T::UInt16),
    u32: (T::UInt32), // T::Categorical, T::Enum
    u64: (T::UInt64),
    i8: (T::Int8),
    i16: (T::Int16),
    i32: (T::Int32), // T::Date
    i64: (T::Int64), // T::Datetime, T::Duration, T::Time
    // i128: (T::Decimal),
    f32: (T::Float32),
    f64: (T::Float64),
    // Vec<u8>: (T::Binary),
    // String: (T::String),
    // Series: (T::List, T::Array),
    // /// Can be used to fmt and implements Any, so can be downcasted to the proper value type.
    // #[cfg(feature = "object")]
    // Object(&'a dyn PolarsObjectSafe),
    // #[cfg(feature = "object")]
    // ObjectOwned(OwnedObject),
    // #[cfg(feature = "dtype-struct")]
    // // 3 pointers and thus not larger than string/vec
    // // - idx in the `&StructArray`
    // // - The array itself
    // // - The fields
    // Struct(usize, &'a StructArray, &'a [Field]),
    // #[cfg(feature = "dtype-struct")]
    // StructOwned(Box<(Vec<AnyValue<'a>>, Vec<Field>)>),
}
