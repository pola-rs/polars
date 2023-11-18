use crate::types::NativeType;

/// [`NativeType`] that supports a representation of 8 lanes
pub trait Simd8: NativeType {
    /// The 8 lane representation of `Self`
    type Simd: Simd8Lanes<Self>;
}

/// Trait declaring an 8-lane multi-data.
pub trait Simd8Lanes<T>: Copy {
    /// loads a complete chunk
    fn from_chunk(v: &[T]) -> Self;
    /// loads an incomplete chunk, filling the remaining items with `remaining`.
    fn from_incomplete_chunk(v: &[T], remaining: T) -> Self;
}

/// Trait implemented by implementors of [`Simd8Lanes`] whose [`Simd8`] implements [PartialEq].
pub trait Simd8PartialEq: Copy {
    /// Equal
    fn eq(self, other: Self) -> u8;
    /// Not equal
    fn neq(self, other: Self) -> u8;
}

/// Trait implemented by implementors of [`Simd8Lanes`] whose [`Simd8`] implements [PartialOrd].
pub trait Simd8PartialOrd: Copy {
    /// Less than or equal to
    fn lt_eq(self, other: Self) -> u8;
    /// Less than
    fn lt(self, other: Self) -> u8;
    /// Greater than
    fn gt(self, other: Self) -> u8;
    /// Greater than or equal to
    fn gt_eq(self, other: Self) -> u8;
}

#[inline]
pub(super) fn set<T: Copy, F: Fn(T, T) -> bool>(lhs: [T; 8], rhs: [T; 8], op: F) -> u8 {
    let mut byte = 0u8;
    lhs.iter()
        .zip(rhs.iter())
        .enumerate()
        .for_each(|(i, (lhs, rhs))| {
            byte |= if op(*lhs, *rhs) { 1 << i } else { 0 };
        });
    byte
}

/// Types that implement Simd8
macro_rules! simd8_native {
    ($type:ty) => {
        impl Simd8 for $type {
            type Simd = [$type; 8];
        }

        impl Simd8Lanes<$type> for [$type; 8] {
            #[inline]
            fn from_chunk(v: &[$type]) -> Self {
                v.try_into().unwrap()
            }

            #[inline]
            fn from_incomplete_chunk(v: &[$type], remaining: $type) -> Self {
                let mut a = [remaining; 8];
                a.iter_mut().zip(v.iter()).for_each(|(a, b)| *a = *b);
                a
            }
        }
    };
}

/// Types that implement PartialEq
macro_rules! simd8_native_partial_eq {
    ($type:ty) => {
        impl Simd8PartialEq for [$type; 8] {
            #[inline]
            fn eq(self, other: Self) -> u8 {
                set(self, other, |x, y| x == y)
            }

            #[inline]
            fn neq(self, other: Self) -> u8 {
                #[allow(clippy::float_cmp)]
                set(self, other, |x, y| x != y)
            }
        }
    };
}

/// Types that implement PartialOrd
macro_rules! simd8_native_partial_ord {
    ($type:ty) => {
        impl Simd8PartialOrd for [$type; 8] {
            #[inline]
            fn lt_eq(self, other: Self) -> u8 {
                set(self, other, |x, y| x <= y)
            }

            #[inline]
            fn lt(self, other: Self) -> u8 {
                set(self, other, |x, y| x < y)
            }

            #[inline]
            fn gt_eq(self, other: Self) -> u8 {
                set(self, other, |x, y| x >= y)
            }

            #[inline]
            fn gt(self, other: Self) -> u8 {
                set(self, other, |x, y| x > y)
            }
        }
    };
}

/// Types that implement simd8, PartialEq and PartialOrd
macro_rules! simd8_native_all {
    ($type:ty) => {
        simd8_native! {$type}
        simd8_native_partial_eq! {$type}
        simd8_native_partial_ord! {$type}
    };
}

#[cfg(not(feature = "simd"))]
mod native;

#[cfg(feature = "simd")]
mod packed;
