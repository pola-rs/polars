use std::convert::identity;

use arrow::bitmap::binary_fold;
use arrow::types::NativeType;
use polars_array::{PlBitmap, PlBooleanArray, PlPrimitiveArray};
use polars_utils::float16::pf16;

use crate::boolean::{all, any, flat_validity};

pub trait BitwiseKernel {
    type Scalar;

    fn count_ones(&self) -> PlPrimitiveArray<u32>;
    fn count_zeros(&self) -> PlPrimitiveArray<u32>;

    fn leading_ones(&self) -> PlPrimitiveArray<u32>;
    fn leading_zeros(&self) -> PlPrimitiveArray<u32>;

    fn trailing_ones(&self) -> PlPrimitiveArray<u32>;
    fn trailing_zeros(&self) -> PlPrimitiveArray<u32>;

    fn reduce_and(&self) -> Option<Self::Scalar>;
    fn reduce_or(&self) -> Option<Self::Scalar>;
    fn reduce_xor(&self) -> Option<Self::Scalar>;

    fn bit_and(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
    fn bit_or(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
    fn bit_xor(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
}

/// The counts of an array whose values buffer holds a single slot, counted once and repeated in
/// `O(1)` memory, or of one that holds a slot per element, counted one by one.
fn count_values<T, I, F>(
    scalar_value: Option<T>,
    values: I,
    length: usize,
    validity: Option<PlBitmap>,
    op: F,
) -> PlPrimitiveArray<u32>
where
    I: Iterator<Item = T>,
    F: Fn(T) -> u32,
{
    match scalar_value {
        Some(value) => PlPrimitiveArray::new_scalar(op(value), length),
        None => PlPrimitiveArray::from_vec(values.map(op).collect()),
    }
    .with_validity(validity)
}

/// The value every element of `arr` reads and the number of its elements that are not null, if
/// its values buffer holds a single slot and at least one element reads it as non-null.
#[inline]
fn repeated_value<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Option<(T, usize)> {
    let count = arr.len() - arr.null_count();
    arr.scalar_values()
        .filter(|_| count > 0)
        .map(|v| (v, count))
}

/// As [`repeated_value`], for a boolean array.
#[inline]
fn repeated_bit(arr: &PlBooleanArray) -> Option<(bool, usize)> {
    let count = arr.len() - arr.null_count();
    arr.scalar_values()
        .filter(|_| count > 0)
        .map(|v| (v, count))
}

/// Counts the bits of every value of a primitive array with `$count`, keeping the scalar
/// representation where the array has one.
macro_rules! count_bits {
    ($arr:expr, $count:ident, $to_bits:expr) => {{
        let arr = $arr;
        count_values(
            arr.scalar_values(),
            arr.values_iter(),
            arr.len(),
            arr.validity().map(PlBitmap::from),
            |v| $to_bits(v).$count(),
        )
    }};
}

/// Folds the non-null values of a primitive array over their bits.
macro_rules! reduce_bits {
    ($arr:expr, $to_bits:expr, $from_bits:expr, $op:expr) => {{
        let arr = $arr;
        if arr.has_nulls() {
            arr.iter()
                .flatten()
                .map($to_bits)
                .reduce($op)
                .map($from_bits)
        } else {
            arr.values_iter().map($to_bits).reduce($op).map($from_bits)
        }
    }};
}

macro_rules! impl_bitwise_kernel {
    ($(($T:ty, $to_bits:expr, $from_bits:expr)),+ $(,)?) => {
        $(
        impl BitwiseKernel for PlPrimitiveArray<$T> {
            type Scalar = $T;

            #[inline(never)]
            fn count_ones(&self) -> PlPrimitiveArray<u32> {
                count_bits!(self, count_ones, $to_bits)
            }

            #[inline(never)]
            fn count_zeros(&self) -> PlPrimitiveArray<u32> {
                count_bits!(self, count_zeros, $to_bits)
            }

            #[inline(never)]
            fn leading_ones(&self) -> PlPrimitiveArray<u32> {
                count_bits!(self, leading_ones, $to_bits)
            }

            #[inline(never)]
            fn leading_zeros(&self) -> PlPrimitiveArray<u32> {
                count_bits!(self, leading_zeros, $to_bits)
            }

            #[inline(never)]
            fn trailing_ones(&self) -> PlPrimitiveArray<u32> {
                count_bits!(self, trailing_ones, $to_bits)
            }

            #[inline(never)]
            fn trailing_zeros(&self) -> PlPrimitiveArray<u32> {
                count_bits!(self, trailing_zeros, $to_bits)
            }

            // `and` and `or` are idempotent, so an array that repeats one value reduces to that
            // value without a single element being walked.
            #[inline(never)]
            fn reduce_and(&self) -> Option<Self::Scalar> {
                match repeated_value(self) {
                    Some((value, _)) => Some(value),
                    None => reduce_bits!(self, $to_bits, $from_bits, |a, b| a & b),
                }
            }

            #[inline(never)]
            fn reduce_or(&self) -> Option<Self::Scalar> {
                match repeated_value(self) {
                    Some((value, _)) => Some(value),
                    None => reduce_bits!(self, $to_bits, $from_bits, |a, b| a | b),
                }
            }

            // `xor` cancels in pairs, so an even number of copies of one value leaves nothing of
            // it and an odd number leaves a single copy.
            #[inline(never)]
            fn reduce_xor(&self) -> Option<Self::Scalar> {
                match repeated_value(self) {
                    Some((value, count)) if count % 2 == 1 => Some(value),
                    Some((value, _)) => Some($from_bits($to_bits(value) ^ $to_bits(value))),
                    None => reduce_bits!(self, $to_bits, $from_bits, |a, b| a ^ b),
                }
            }

            fn bit_and(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar {
                $from_bits($to_bits(lhs) & $to_bits(rhs))
            }
            fn bit_or(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar {
                $from_bits($to_bits(lhs) | $to_bits(rhs))
            }
            fn bit_xor(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar {
                $from_bits($to_bits(lhs) ^ $to_bits(rhs))
            }
        }
        )+
    };
}

impl_bitwise_kernel! {
    (i8, identity, identity),
    (i16, identity, identity),
    (i32, identity, identity),
    (i64, identity, identity),
    (u8, identity, identity),
    (u16, identity, identity),
    (u32, identity, identity),
    (u64, identity, identity),
    (pf16, pf16::to_bits, pf16::from_bits),
    (f32, f32::to_bits, f32::from_bits),
    (f64, f64::to_bits, f64::from_bits),
}

#[cfg(feature = "dtype-u128")]
impl_bitwise_kernel! {
    (u128, identity, identity),
}

#[cfg(feature = "dtype-i128")]
impl_bitwise_kernel! {
    (i128, identity, identity),
}

impl BitwiseKernel for PlBooleanArray {
    type Scalar = bool;

    #[inline(never)]
    fn count_ones(&self) -> PlPrimitiveArray<u32> {
        count_values(
            self.scalar_values(),
            self.values_iter(),
            self.len(),
            self.validity().map(PlBitmap::from),
            u32::from,
        )
    }

    #[inline(never)]
    fn count_zeros(&self) -> PlPrimitiveArray<u32> {
        count_values(
            self.scalar_values(),
            self.values_iter(),
            self.len(),
            self.validity().map(PlBitmap::from),
            |v| u32::from(!v),
        )
    }

    #[inline(always)]
    fn leading_ones(&self) -> PlPrimitiveArray<u32> {
        self.count_ones()
    }

    #[inline(always)]
    fn leading_zeros(&self) -> PlPrimitiveArray<u32> {
        self.count_zeros()
    }

    #[inline(always)]
    fn trailing_ones(&self) -> PlPrimitiveArray<u32> {
        self.count_ones()
    }

    #[inline(always)]
    fn trailing_zeros(&self) -> PlPrimitiveArray<u32> {
        self.count_zeros()
    }

    #[inline(always)]
    fn reduce_and(&self) -> Option<Self::Scalar> {
        all(self)
    }

    #[inline(always)]
    fn reduce_or(&self) -> Option<Self::Scalar> {
        any(self)
    }

    fn reduce_xor(&self) -> Option<Self::Scalar> {
        // As for the primitive arrays: an even number of copies of one bit cancels to `false`,
        // and an odd number leaves that bit.
        if let Some((value, count)) = repeated_bit(self) {
            return Some(value && count % 2 == 1);
        }
        if self.len() == self.null_count() {
            return None;
        }

        // A scalar bitmap is what the two checks above have already answered for: either it
        // cancels to a parity, or every element under it is null.
        let values = self.flat_values()?;

        match flat_validity(self) {
            Some(validity) => {
                let nonnull_parity =
                    binary_fold(values, validity, |lhs, rhs| lhs & rhs, 0, |a, b| a ^ b);
                Some(nonnull_parity.count_ones() % 2 == 1)
            },
            // Either there is no mask, or it marks every element valid: a scalar mask that marks
            // them all null is what the check above has caught.
            None => Some(values.set_bits() % 2 == 1),
        }
    }

    fn bit_and(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar {
        lhs & rhs
    }

    fn bit_or(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar {
        lhs | rhs
    }

    fn bit_xor(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar {
        lhs ^ rhs
    }
}
