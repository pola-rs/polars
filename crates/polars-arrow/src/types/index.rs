use super::NativeType;
use crate::trusted_len::TrustedLen;

/// Sealed trait describing the subset of [`NativeType`] (`i32`, `i64`, `u32` and `u64`)
/// that can be used to index a slot of an array.
pub trait Index:
    NativeType
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + num_traits::One
    + num_traits::Num
    + num_traits::CheckedAdd
    + PartialOrd
    + Ord
{
    /// Convert itself to [`usize`].
    fn to_usize(&self) -> usize;
    /// Convert itself from [`usize`].
    fn from_usize(index: usize) -> Option<Self>;

    /// Convert itself from [`usize`].
    fn from_as_usize(index: usize) -> Self;

    /// An iterator from (inclusive) `start` to (exclusive) `end`.
    fn range(start: usize, end: usize) -> Option<IndexRange<Self>> {
        let start = Self::from_usize(start);
        let end = Self::from_usize(end);
        match (start, end) {
            (Some(start), Some(end)) => Some(IndexRange::new(start, end)),
            _ => None,
        }
    }
}

macro_rules! index {
    ($t:ty) => {
        impl Index for $t {
            #[inline]
            fn to_usize(&self) -> usize {
                *self as usize
            }

            #[inline]
            fn from_usize(value: usize) -> Option<Self> {
                Self::try_from(value).ok()
            }

            #[inline]
            fn from_as_usize(value: usize) -> Self {
                value as $t
            }
        }
    };
}

index!(i8);
index!(i16);
index!(i32);
index!(i64);
index!(u8);
index!(u16);
index!(u32);
index!(u64);

/// Range of [`Index`], equivalent to `(a..b)`.
/// `Step` is unstable in Rust, which does not allow us to implement (a..b) for [`Index`].
pub struct IndexRange<I: Index> {
    start: I,
    end: I,
}

impl<I: Index> IndexRange<I> {
    /// Returns a new [`IndexRange`].
    pub fn new(start: I, end: I) -> Self {
        assert!(end >= start);
        Self { start, end }
    }
}

impl<I: Index> Iterator for IndexRange<I> {
    type Item = I;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }
        let old = self.start;
        self.start += I::one();
        Some(old)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end - self.start).to_usize();
        (len, Some(len))
    }
}

/// # Safety
///
/// A range is always of known length.
unsafe impl<I: Index> TrustedLen for IndexRange<I> {}
