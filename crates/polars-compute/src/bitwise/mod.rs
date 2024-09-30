use std::convert::identity;

use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::datatypes::ArrowDataType;
use arrow::legacy::utils::CustomIterTools;
use bytemuck::Zeroable;

pub trait BitwiseKernel {
    type Scalar;

    fn count_ones(&self) -> PrimitiveArray<u8>;
    fn count_zeros(&self) -> PrimitiveArray<u8>;

    fn leading_ones(&self) -> PrimitiveArray<u8>;
    fn leading_zeros(&self) -> PrimitiveArray<u8>;

    fn trailing_ones(&self) -> PrimitiveArray<u8>;
    fn trailing_zeros(&self) -> PrimitiveArray<u8>;

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
            fn count_ones(&self) -> PrimitiveArray<u8> {
                PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    self.values()
                        .iter()
                        .map(|&v| ($to_bits(v).count_ones() & 0xFF) as u8)
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn count_zeros(&self) -> PrimitiveArray<u8> {
                PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    self
                        .values()
                        .iter()
                        .map(|&v| ($to_bits(v).count_zeros() & 0xFF) as u8)
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn leading_ones(&self) -> PrimitiveArray<u8> {
                PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    self.values()
                        .iter()
                        .map(|&v| ($to_bits(v).leading_ones() & 0xFF) as u8)
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn leading_zeros(&self) -> PrimitiveArray<u8> {
                PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    self.values()
                        .iter()
                        .map(|&v| ($to_bits(v).leading_zeros() & 0xFF) as u8)
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn trailing_ones(&self) -> PrimitiveArray<u8> {
                PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    self.values()
                        .iter()
                        .map(|&v| ($to_bits(v).trailing_ones() & 0xFF) as u8)
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn trailing_zeros(&self) -> PrimitiveArray<u8> {
                PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    self.values().iter()
                        .map(|&v| ($to_bits(v).trailing_zeros() & 0xFF) as u8)
                        .collect_trusted::<Vec<_>>()
                        .into(),
                    self.validity().cloned(),
                )
            }

            #[inline(never)]
            fn reduce_and(&self) -> Option<Self::Scalar> {
                if self.validity().map_or(false, |v| v.unset_bits() > 0) {
                    return None;
                }

                let values = self.values();

                if values.is_empty() {
                    return None;
                }

                Some($from_bits(values.iter().fold(!$to_bits(<$T>::zeroed()), |a, &b| a & $to_bits(b))))
            }

            #[inline(never)]
            fn reduce_or(&self) -> Option<Self::Scalar> {
                if self.validity().map_or(false, |v| v.unset_bits() > 0) {
                    return None;
                }

                let values = self.values();

                if values.is_empty() {
                    return None;
                }

                Some($from_bits(values.iter().fold($to_bits(<$T>::zeroed()), |a, &b| a | $to_bits(b))))
            }

            #[inline(never)]
            fn reduce_xor(&self) -> Option<Self::Scalar> {
                if self.validity().map_or(false, |v| v.unset_bits() > 0) {
                    return None;
                }

                let values = self.values();

                if values.is_empty() {
                    return None;
                }

                Some($from_bits(values.iter().fold($to_bits(<$T>::zeroed()), |a, &b| a ^ $to_bits(b))))
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
    fn count_ones(&self) -> PrimitiveArray<u8> {
        PrimitiveArray::new(
            ArrowDataType::UInt8,
            self.values()
                .iter()
                .map(u8::from)
                .collect_trusted::<Vec<_>>()
                .into(),
            self.validity().cloned(),
        )
    }

    #[inline(never)]
    fn count_zeros(&self) -> PrimitiveArray<u8> {
        PrimitiveArray::new(
            ArrowDataType::UInt8,
            self.values()
                .iter()
                .map(|v| u8::from(!v))
                .collect_trusted::<Vec<_>>()
                .into(),
            self.validity().cloned(),
        )
    }

    #[inline(always)]
    fn leading_ones(&self) -> PrimitiveArray<u8> {
        self.count_ones()
    }

    #[inline(always)]
    fn leading_zeros(&self) -> PrimitiveArray<u8> {
        self.count_zeros()
    }

    #[inline(always)]
    fn trailing_ones(&self) -> PrimitiveArray<u8> {
        self.count_ones()
    }

    #[inline(always)]
    fn trailing_zeros(&self) -> PrimitiveArray<u8> {
        self.count_zeros()
    }

    fn reduce_and(&self) -> Option<Self::Scalar> {
        if self.validity().map_or(false, |v| v.unset_bits() > 0) {
            return None;
        }

        let values = self.values();

        if values.is_empty() {
            return None;
        }

        Some(values.unset_bits() == 0)
    }

    fn reduce_or(&self) -> Option<Self::Scalar> {
        if self.validity().map_or(false, |v| v.unset_bits() > 0) {
            return None;
        }

        let values = self.values();

        if values.is_empty() {
            return None;
        }

        Some(values.set_bits() > 0)
    }

    fn reduce_xor(&self) -> Option<Self::Scalar> {
        if self.validity().map_or(false, |v| v.unset_bits() > 0) {
            return None;
        }

        let values = self.values();

        if values.is_empty() {
            return None;
        }

        Some(values.set_bits() % 2 == 1)
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
