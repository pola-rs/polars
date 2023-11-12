use std::ops::Rem;

use num_traits::NumCast;
use strength_reduce::{
    StrengthReducedU16, StrengthReducedU32, StrengthReducedU64, StrengthReducedU8,
};

use super::NativeArithmetics;
use crate::array::{Array, PrimitiveArray};
use crate::compute::arity::{binary, unary};
use crate::datatypes::PrimitiveType;

/// Remainder of two primitive arrays with the same type.
/// Panics if the divisor is zero of one pair of values overflows.
pub fn rem<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Rem<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a % b)
}

/// Remainder a primitive array of type T by a scalar T.
/// Panics if the divisor is zero.
pub fn rem_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Rem<Output = T> + NumCast,
{
    let rhs = *rhs;

    match T::PRIMITIVE {
        PrimitiveType::UInt64 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u64>>().unwrap();
            let rhs = rhs.to_u64().unwrap();

            let reduced_rem = StrengthReducedU64::new(rhs);

            // small hack to avoid a transmute of `PrimitiveArray<u64>` to `PrimitiveArray<T>`
            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt32 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u32>>().unwrap();
            let rhs = rhs.to_u32().unwrap();

            let reduced_rem = StrengthReducedU32::new(rhs);

            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            // small hack to avoid an unsafe transmute of `PrimitiveArray<u64>` to `PrimitiveArray<T>`
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt16 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u16>>().unwrap();
            let rhs = rhs.to_u16().unwrap();

            let reduced_rem = StrengthReducedU16::new(rhs);

            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            // small hack to avoid an unsafe transmute of `PrimitiveArray<u16>` to `PrimitiveArray<T>`
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt8 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u8>>().unwrap();
            let rhs = rhs.to_u8().unwrap();

            let reduced_rem = StrengthReducedU8::new(rhs);

            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            // small hack to avoid an unsafe transmute of `PrimitiveArray<u8>` to `PrimitiveArray<T>`
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        _ => unary(lhs, |a| a % rhs, lhs.data_type().clone()),
    }
}
