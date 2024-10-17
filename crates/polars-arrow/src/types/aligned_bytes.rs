use bytemuck::{Pod, Zeroable};

use super::{days_ms, f16, i256, months_days_ns};
use crate::array::View;

/// Define that a type has the same byte alignment and size as `B`.
///
/// # Safety
///
/// This is safe to implement if both types have the same alignment and size.
pub unsafe trait AlignedBytesCast<B: AlignedBytes>: Pod {}

/// A representation of a type as raw bytes with the same alignment as the original type.
pub trait AlignedBytes: Pod + Zeroable + Copy + Default + Eq {
    const ALIGNMENT: usize;
    const SIZE: usize;

    type Unaligned: AsRef<[u8]>
        + AsMut<[u8]>
        + std::ops::Index<usize, Output = u8>
        + std::ops::IndexMut<usize, Output = u8>
        + for<'a> TryFrom<&'a [u8]>
        + std::fmt::Debug
        + Default
        + IntoIterator<Item = u8>
        + Pod;

    fn to_unaligned(&self) -> Self::Unaligned;
    fn from_unaligned(unaligned: Self::Unaligned) -> Self;

    /// Safely cast a mutable reference to a [`Vec`] of `T` to a mutable reference of `Self`.
    fn cast_vec_ref_mut<T: AlignedBytesCast<Self>>(vec: &mut Vec<T>) -> &mut Vec<Self> {
        if cfg!(debug_assertions) {
            assert_eq!(size_of::<T>(), size_of::<Self>());
            assert_eq!(align_of::<T>(), align_of::<Self>());
        }

        // SAFETY: SameBytes guarantees that T:
        // 1. has the same size
        // 2. has the same alignment
        // 3. is Pod (therefore has no life-time issues)
        unsafe { std::mem::transmute(vec) }
    }
}

macro_rules! impl_aligned_bytes {
    (
        $(($name:ident, $size:literal, $alignment:literal, [$($eq_type:ty),*]),)+
    ) => {
        $(
        /// Bytes with a size and alignment.
        /// 
        /// This is used to reduce the monomorphizations for routines that solely rely on the size
        /// and alignment of types.
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Pod, Zeroable)]
        #[repr(C, align($alignment))]
        pub struct $name([u8; $size]);

        impl AlignedBytes for $name {
            const ALIGNMENT: usize = $alignment;
            const SIZE: usize = $size;

            type Unaligned = [u8; $size];

            #[inline(always)]
            fn to_unaligned(&self) -> Self::Unaligned {
                self.0
            }
            #[inline(always)]
            fn from_unaligned(unaligned: Self::Unaligned) -> Self {
                Self(unaligned)
            }
        }

        impl AsRef<[u8; $size]> for $name {
            #[inline(always)]
            fn as_ref(&self) -> &[u8; $size] {
                &self.0
            }
        }

        $(
        impl From<$eq_type> for $name {
            #[inline(always)]
            fn from(value: $eq_type) -> Self {
                bytemuck::must_cast(value)
            }
        }
        impl From<$name> for $eq_type {
            #[inline(always)]
            fn from(value: $name) -> Self {
                bytemuck::must_cast(value)
            }
        }
        unsafe impl AlignedBytesCast<$name> for $eq_type {}
        )*
        )+
    }
}

impl_aligned_bytes! {
    (Bytes1Alignment1, 1, 1, [u8, i8]),
    (Bytes2Alignment2, 2, 2, [u16, i16, f16]),
    (Bytes4Alignment4, 4, 4, [u32, i32, f32]),
    (Bytes8Alignment8, 8, 8, [u64, i64, f64]),
    (Bytes8Alignment4, 8, 4, [days_ms]),
    (Bytes12Alignment4, 12, 4, [[u32; 3]]),
    (Bytes16Alignment4, 16, 4, [View]),
    (Bytes16Alignment8, 16, 8, [months_days_ns]),
    (Bytes16Alignment16, 16, 16, [u128, i128]),
    (Bytes32Alignment16, 32, 16, [i256]),
}
