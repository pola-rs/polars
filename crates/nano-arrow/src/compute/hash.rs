//! Contains the [`hash`] and typed (e.g. [`hash_primitive`]) operators.
// multiversion does not copy documentation, causing a false positive
#![allow(missing_docs)]
use ahash::RandomState;
use multiversion::multiversion;
use std::hash::Hash;

macro_rules! new_state {
    () => {
        RandomState::with_seeds(0, 0, 0, 0)
    };
}

use crate::{
    array::{Array, BinaryArray, BooleanArray, PrimitiveArray, Utf8Array},
    datatypes::{DataType, PhysicalType, PrimitiveType},
    error::{Error, Result},
    offset::Offset,
    types::NativeType,
};

use super::arity::unary;

#[multiversion(targets("x86_64+aes+sse3+ssse3+avx+avx2"))]
/// Element-wise hash of a [`PrimitiveArray`]. Validity is preserved.
pub fn hash_primitive<T: NativeType + Hash>(array: &PrimitiveArray<T>) -> PrimitiveArray<u64> {
    let state = new_state!();

    unary(array, |x| state.hash_one(x), DataType::UInt64)
}

#[multiversion(targets("x86_64+aes+sse3+ssse3+avx+avx2"))]
/// Element-wise hash of a [`BooleanArray`]. Validity is preserved.
pub fn hash_boolean(array: &BooleanArray) -> PrimitiveArray<u64> {
    let state = new_state!();

    let values = array
        .values_iter()
        .map(|x| state.hash_one(x))
        .collect::<Vec<_>>()
        .into();

    PrimitiveArray::<u64>::new(DataType::UInt64, values, array.validity().cloned())
}

#[multiversion(targets("x86_64+aes+sse3+ssse3+avx+avx2"))]
/// Element-wise hash of a [`Utf8Array`]. Validity is preserved.
pub fn hash_utf8<O: Offset>(array: &Utf8Array<O>) -> PrimitiveArray<u64> {
    let state = new_state!();

    let values = array
        .values_iter()
        .map(|x| state.hash_one(x.as_bytes()))
        .collect::<Vec<_>>()
        .into();

    PrimitiveArray::<u64>::new(DataType::UInt64, values, array.validity().cloned())
}

/// Element-wise hash of a [`BinaryArray`]. Validity is preserved.
pub fn hash_binary<O: Offset>(array: &BinaryArray<O>) -> PrimitiveArray<u64> {
    let state = new_state!();
    let values = array
        .values_iter()
        .map(|x| state.hash_one(x))
        .collect::<Vec<_>>()
        .into();

    PrimitiveArray::<u64>::new(DataType::UInt64, values, array.validity().cloned())
}

macro_rules! with_match_primitive_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use crate::datatypes::PrimitiveType::*;
    use crate::types::{days_ms, months_days_ns};
    match $key_type {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        Int128 => __with_ty__! { i128 },
        DaysMs => __with_ty__! { days_ms },
        MonthDayNano => __with_ty__! { months_days_ns },
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        _ => return Err(Error::NotYetImplemented(format!(
            "Hash not implemented for type {:?}",
            $key_type
        )))
    }
})}

/// Returns the element-wise hash of an [`Array`]. Validity is preserved.
/// Supported DataTypes:
/// * Boolean types
/// * All primitive types except `Float32` and `Float64`
/// * `[Large]Utf8`;
/// * `[Large]Binary`.
/// # Errors
/// This function errors whenever it does not support the specific `DataType`.
pub fn hash(array: &dyn Array) -> Result<PrimitiveArray<u64>> {
    use PhysicalType::*;
    Ok(match array.data_type().to_physical_type() {
        Boolean => hash_boolean(array.as_any().downcast_ref().unwrap()),
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            hash_primitive::<$T>(array.as_any().downcast_ref().unwrap())
        }),
        Binary => hash_binary::<i32>(array.as_any().downcast_ref().unwrap()),
        LargeBinary => hash_binary::<i64>(array.as_any().downcast_ref().unwrap()),
        Utf8 => hash_utf8::<i32>(array.as_any().downcast_ref().unwrap()),
        LargeUtf8 => hash_utf8::<i64>(array.as_any().downcast_ref().unwrap()),
        t => {
            return Err(Error::NotYetImplemented(format!(
                "Hash not implemented for type {t:?}"
            )))
        }
    })
}

/// Checks if an array of type `datatype` can be used in [`hash`].
///
/// # Examples
/// ```
/// use arrow2::compute::hash::can_hash;
/// use arrow2::datatypes::{DataType};
///
/// let data_type = DataType::Int8;
/// assert_eq!(can_hash(&data_type), true);

/// let data_type = DataType::Null;
/// assert_eq!(can_hash(&data_type), false);
/// ```
pub fn can_hash(data_type: &DataType) -> bool {
    matches!(
        data_type.to_physical_type(),
        PhysicalType::Boolean
            | PhysicalType::Primitive(PrimitiveType::Int8)
            | PhysicalType::Primitive(PrimitiveType::Int16)
            | PhysicalType::Primitive(PrimitiveType::Int32)
            | PhysicalType::Primitive(PrimitiveType::Int64)
            | PhysicalType::Primitive(PrimitiveType::Int128)
            | PhysicalType::Primitive(PrimitiveType::DaysMs)
            | PhysicalType::Primitive(PrimitiveType::MonthDayNano)
            | PhysicalType::Primitive(PrimitiveType::UInt8)
            | PhysicalType::Primitive(PrimitiveType::UInt16)
            | PhysicalType::Primitive(PrimitiveType::UInt32)
            | PhysicalType::Primitive(PrimitiveType::UInt64)
            | PhysicalType::Binary
            | PhysicalType::LargeBinary
            | PhysicalType::Utf8
            | PhysicalType::LargeUtf8
    )
}
