//! Contains traits and implementations of multi-data used in SIMD.
//! The actual representation is driven by the feature flag `"simd"`, which, if set,
//! uses [`std::simd`].
use super::{days_ms, f16, i256, months_days_ns, BitChunk, BitChunkIter, NativeType};

/// Describes the ability to convert itself from a [`BitChunk`].
pub trait FromMaskChunk<T> {
    /// Convert itself from a slice.
    fn from_chunk(v: T) -> Self;
}

/// A struct that lends itself well to be compiled leveraging SIMD
/// # Safety
/// The `NativeType` and the `NativeSimd` must have possible a matching alignment.
/// e.g. slicing `&[NativeType]` by `align_of<NativeSimd>()` must be properly aligned/safe.
pub unsafe trait NativeSimd: Sized + Default + Copy {
    /// Number of lanes
    const LANES: usize;
    /// The [`NativeType`] of this struct. E.g. `f32` for a `NativeSimd = f32x16`.
    type Native: NativeType;
    /// The type holding bits for masks.
    type Chunk: BitChunk;
    /// Type used for masking.
    type Mask: FromMaskChunk<Self::Chunk>;

    /// Sets values to `default` based on `mask`.
    fn select(self, mask: Self::Mask, default: Self) -> Self;

    /// Convert itself from a slice.
    /// # Panics
    /// * iff `v.len()` != `T::LANES`
    fn from_chunk(v: &[Self::Native]) -> Self;

    /// creates a new Self from `v` by populating items from `v` up to its length.
    /// Items from `v` at positions larger than the number of lanes are ignored;
    /// remaining items are populated with `remaining`.
    fn from_incomplete_chunk(v: &[Self::Native], remaining: Self::Native) -> Self;

    /// Returns a tuple of 3 items whose middle item is itself, and the remaining
    /// are the head and tail of the un-aligned parts.
    fn align(values: &[Self::Native]) -> (&[Self::Native], &[Self], &[Self::Native]);
}

/// Trait implemented by some [`NativeType`] that have a SIMD representation.
pub trait Simd: NativeType {
    /// The SIMD type associated with this trait.
    /// This type supports SIMD operations
    type Simd: NativeSimd<Native = Self>;
}

#[cfg(not(feature = "simd"))]
mod native;
#[cfg(not(feature = "simd"))]
pub use native::*;
#[cfg(feature = "simd")]
mod packed;
#[cfg(feature = "simd")]
pub use packed::*;

macro_rules! native_simd {
    ($name:tt, $type:ty, $lanes:expr, $mask:ty) => {
        /// Multi-Data correspondence of the native type
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        pub struct $name(pub [$type; $lanes]);

        unsafe impl NativeSimd for $name {
            const LANES: usize = $lanes;
            type Native = $type;
            type Chunk = $mask;
            type Mask = $mask;

            #[inline]
            fn select(self, mask: $mask, default: Self) -> Self {
                let mut reduced = default;
                let iter = BitChunkIter::new(mask, Self::LANES);
                for (i, b) in (0..Self::LANES).zip(iter) {
                    reduced[i] = if b { self[i] } else { reduced[i] };
                }
                reduced
            }

            #[inline]
            fn from_chunk(v: &[$type]) -> Self {
                ($name)(v.try_into().unwrap())
            }

            #[inline]
            fn from_incomplete_chunk(v: &[$type], remaining: $type) -> Self {
                let mut a = [remaining; $lanes];
                a.iter_mut().zip(v.iter()).for_each(|(a, b)| *a = *b);
                Self(a)
            }

            #[inline]
            fn align(values: &[Self::Native]) -> (&[Self::Native], &[Self], &[Self::Native]) {
                unsafe { values.align_to::<Self>() }
            }
        }

        impl std::ops::Index<usize> for $name {
            type Output = $type;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl std::ops::IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl Default for $name {
            #[inline]
            fn default() -> Self {
                ($name)([<$type>::default(); $lanes])
            }
        }
    };
}

#[cfg(not(feature = "simd"))]
pub(super) use native_simd;

// Types do not have specific intrinsics and thus SIMD can't be specialized.
// Therefore, we can declare their MD representation as `[$t; 8]` irrespectively
// of how they are represented in the different channels.
native_simd!(f16x32, f16, 32, u32);
native_simd!(days_msx8, days_ms, 8, u8);
native_simd!(months_days_nsx8, months_days_ns, 8, u8);
native_simd!(i128x8, i128, 8, u8);
native_simd!(i256x8, i256, 8, u8);

// In the native implementation, a mask is 1 bit wide, as per AVX512.
impl<T: BitChunk> FromMaskChunk<T> for T {
    #[inline]
    fn from_chunk(v: T) -> Self {
        v
    }
}

macro_rules! native {
    ($type:ty, $simd:ty) => {
        impl Simd for $type {
            type Simd = $simd;
        }
    };
}

native!(u8, u8x64);
native!(u16, u16x32);
native!(u32, u32x16);
native!(u64, u64x8);
native!(i8, i8x64);
native!(i16, i16x32);
native!(i32, i32x16);
native!(i64, i64x8);
native!(f16, f16x32);
native!(f32, f32x16);
native!(f64, f64x8);
native!(i128, i128x8);
native!(i256, i256x8);
native!(days_ms, days_msx8);
native!(months_days_ns, months_days_nsx8);
