use std::convert::identity;

use arrow::array::{Array, BooleanArray, PrimitiveArray};
use arrow::datatypes::ArrowDataType;
use arrow::legacy::utils::CustomIterTools;

pub trait BitwiseKernel {
    type Scalar;

    fn count_ones(&self) -> PrimitiveArray<u32>;
    fn count_zeros(&self) -> PrimitiveArray<u32>;

    fn leading_ones(&self) -> PrimitiveArray<u32>;
    fn leading_zeros(&self) -> PrimitiveArray<u32>;

    fn trailing_ones(&self) -> PrimitiveArray<u32>;
    fn trailing_zeros(&self) -> PrimitiveArray<u32>;

    fn reduce_and(&self) -> Option<Self::Scalar>;
    fn reduce_or(&self) -> Option<Self::Scalar>;
    fn reduce_xor(&self) -> Option<Self::Scalar>;

    fn bit_and(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
    fn bit_or(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
    fn bit_xor(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
}

macro_rules! impl_bitwise_kernel {
    ($(($T:ty, $to_bits:expr, $from_bits:expr)),+ $(,)?) => {
        $(
        impl BitwiseKernel for PrimitiveArray<$T> {
            type Scalar = $T;

            #[inline(never)]
            fn count_ones(&self) -> PrimitiveArray<u32> {
                PrimitiveArray::new(
                    ArrowDataType::UInt32,
                    self.values_iter()
                        .map(|&v| $to_bits(v).count_ones())
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn count_zeros(&self) -> PrimitiveArray<u32> {
                PrimitiveArray::new(
                    ArrowDataType::UInt32,
                    self.values_iter()
                        .map(|&v| $to_bits(v).count_zeros())
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn leading_ones(&self) -> PrimitiveArray<u32> {
                PrimitiveArray::new(
                    ArrowDataType::UInt32,
                    self.values_iter()
                        .map(|&v| $to_bits(v).leading_ones())
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn leading_zeros(&self) -> PrimitiveArray<u32> {
                PrimitiveArray::new(
                    ArrowDataType::UInt32,
                    self.values_iter()
                        .map(|&v| $to_bits(v).leading_zeros())
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn trailing_ones(&self) -> PrimitiveArray<u32> {
                PrimitiveArray::new(
                    ArrowDataType::UInt32,
                    self.values_iter()
                        .map(|&v| $to_bits(v).trailing_ones())
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn trailing_zeros(&self) -> PrimitiveArray<u32> {
                PrimitiveArray::new(
                    ArrowDataType::UInt32,
                    self.values().iter()
                        .map(|&v| $to_bits(v).trailing_zeros())
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn reduce_and(&self) -> Option<Self::Scalar> {
                if !self.has_nulls() {
                    self.values_iter().copied().map($to_bits).reduce(|a, b| a & b).map($from_bits)
                } else {
                    self.non_null_values_iter().map($to_bits).reduce(|a, b| a & b).map($from_bits)
                }
            }

            #[inline(never)]
            fn reduce_or(&self) -> Option<Self::Scalar> {
                if !self.has_nulls() {
                    self.values_iter().copied().map($to_bits).reduce(|a, b| a | b).map($from_bits)
                } else {
                    self.non_null_values_iter().map($to_bits).reduce(|a, b| a | b).map($from_bits)
                }
            }

            #[inline(never)]
            fn reduce_xor(&self) -> Option<Self::Scalar> {
                if !self.has_nulls() {
                    self.values_iter().copied().map($to_bits).reduce(|a, b| a ^ b).map($from_bits)
                } else {
                    self.non_null_values_iter().map($to_bits).reduce(|a, b| a ^ b).map($from_bits)
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
    (f32, f32::to_bits, f32::from_bits),
    (f64, f64::to_bits, f64::from_bits),
}

impl BitwiseKernel for BooleanArray {
    type Scalar = bool;

    #[inline(never)]
    fn count_ones(&self) -> PrimitiveArray<u32> {
        PrimitiveArray::new(
            ArrowDataType::UInt32,
            self.values_iter()
                .map(u32::from)
                .collect_trusted::<Vec<_>>()
                .into(),
            self.validity().cloned(),
        )
    }

    #[inline(never)]
    fn count_zeros(&self) -> PrimitiveArray<u32> {
        PrimitiveArray::new(
            ArrowDataType::UInt32,
            self.values_iter()
                .map(|v| u32::from(!v))
                .collect_trusted::<Vec<_>>()
                .into(),
            self.validity().cloned(),
        )
    }

    #[inline(always)]
    fn leading_ones(&self) -> PrimitiveArray<u32> {
        self.count_ones()
    }

    #[inline(always)]
    fn leading_zeros(&self) -> PrimitiveArray<u32> {
        self.count_zeros()
    }

    #[inline(always)]
    fn trailing_ones(&self) -> PrimitiveArray<u32> {
        self.count_ones()
    }

    #[inline(always)]
    fn trailing_zeros(&self) -> PrimitiveArray<u32> {
        self.count_zeros()
    }

    fn reduce_and(&self) -> Option<Self::Scalar> {
        if self.len() == self.null_count() {
            None
        } else if !self.has_nulls() {
            Some(self.values().unset_bits() == 0)
        } else {
            Some((self.values() & self.validity().unwrap()).unset_bits() == 0)
        }
    }

    fn reduce_or(&self) -> Option<Self::Scalar> {
        if self.len() == self.null_count() {
            None
        } else if !self.has_nulls() {
            Some(self.values().set_bits() > 0)
        } else {
            Some((self.values() & self.validity().unwrap()).set_bits() > 0)
        }
    }

    fn reduce_xor(&self) -> Option<Self::Scalar> {
        if self.len() == self.null_count() {
            None
        } else if !self.has_nulls() {
            Some(self.values().set_bits() % 2 == 1)
        } else {
            Some((self.values() & self.validity().unwrap()).set_bits() % 2 == 1)
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
