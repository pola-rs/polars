use bytemuck::{Pod, Zeroable};

use super::{days_ms, f16, i256, months_days_ns};
use crate::array::View;

/// A representation of a type as raw bytes with the same alignment as the original type.
pub trait AlignedBytes: Pod + Default + Eq {
    const ALIGNMENT: usize;
    const SIZE: usize;
    const SIZE_ALIGNMENT_PAIR: PrimitiveSizeAlignmentPair;

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

    fn zeros() -> Self;
    fn ones() -> Self;

    fn unsigned_leq(self, other: Self) -> bool;
}

macro_rules! impl_aligned_bytes {
    (
        $(($name:ident, $size:literal, $alignment:literal, $sap:ident, [$($eq_type:ty),*]$(, $unsigned:ty)?),)+
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
            const SIZE_ALIGNMENT_PAIR: PrimitiveSizeAlignmentPair = PrimitiveSizeAlignmentPair::$sap;

            type Unaligned = [u8; $size];

            #[inline(always)]
            fn to_unaligned(&self) -> Self::Unaligned {
                self.0
            }
            #[inline(always)]
            fn from_unaligned(unaligned: Self::Unaligned) -> Self {
                Self(unaligned)
            }
            fn zeros() -> Self {
                Self([0u8; _])
            }
            fn ones() -> Self {
                Self([0xFFu8; _])
            }
            #[inline(always)]
            fn unsigned_leq(self, _other: Self) -> bool {
                $(
                    return <$unsigned>::from(self) <= <$unsigned>::from(_other);
                )?

                #[allow(unreachable_code)]
                {
                    unreachable!()
                }
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
        )*
        )+
    }
}

#[derive(Clone, Copy)]
pub enum PrimitiveSizeAlignmentPair {
    S1A1,
    S2A2,
    S4A4,
    S8A4,
    S8A8,
    S12A4,
    S16A4,
    S16A8,
    S16A16,
    S32A16,
}

impl PrimitiveSizeAlignmentPair {
    pub const fn size(self) -> usize {
        match self {
            Self::S1A1 => 1,
            Self::S2A2 => 2,
            Self::S4A4 => 4,
            Self::S8A4 | Self::S8A8 => 8,
            Self::S12A4 => 12,
            Self::S16A4 | Self::S16A8 | Self::S16A16 => 16,
            Self::S32A16 => 32,
        }
    }

    pub const fn alignment(self) -> usize {
        match self {
            Self::S1A1 => 1,
            Self::S2A2 => 2,
            Self::S4A4 | Self::S8A4 | Self::S12A4 | Self::S16A4 => 4,
            Self::S8A8 | Self::S16A8 => 8,
            Self::S16A16 | Self::S32A16 => 16,
        }
    }
}

impl_aligned_bytes! {
    (Bytes1Alignment1, 1, 1, S1A1, [u8, i8], u8),
    (Bytes2Alignment2, 2, 2, S2A2, [u16, i16, f16], u16),
    (Bytes4Alignment4, 4, 4, S4A4, [u32, i32, f32], u32),
    (Bytes8Alignment8, 8, 8, S8A8, [u64, i64, f64], u64),
    (Bytes8Alignment4, 8, 4, S8A4, [days_ms]),
    (Bytes12Alignment4, 12, 4, S12A4, [[u32; 3]]),
    (Bytes16Alignment4, 16, 4, S16A4, [View]),
    (Bytes16Alignment8, 16, 8, S16A8, [months_days_ns]),
    (Bytes16Alignment16, 16, 16, S16A16, [u128, i128]),
    (Bytes32Alignment16, 32, 16, S32A16, [i256]),
}
