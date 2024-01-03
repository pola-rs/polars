use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::nulls::IsNull;
use crate::slice::GetSaferUnchecked;
use crate::IdxSize;

pub trait Bounded {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait NullCount {
    fn null_count(&self) -> usize {
        0
    }
}

impl<T: NullCount> NullCount for &T {
    fn null_count(&self) -> usize {
        (*self).null_count()
    }
}

impl<T> Bounded for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T> NullCount for &[T] {
    fn null_count(&self) -> usize {
        0
    }
}

pub trait Indexable {
    type Item: IsNull;

    fn get(&self, i: usize) -> Self::Item;

    /// # Safety
    /// Doesn't do any bound checks.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}

impl<T: Copy + IsNull> Indexable for &[T] {
    type Item = T;

    fn get(&self, i: usize) -> Self::Item {
        self[i]
    }

    /// # Safety
    /// Doesn't do any bound checks.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        *self.get_unchecked_release(i)
    }
}

pub fn check_bounds(idx: &[IdxSize], len: IdxSize) -> PolarsResult<()> {
    // We iterate in large uninterrupted chunks to help auto-vectorization.
    let mut in_bounds = true;
    for chunk in idx.chunks(1024) {
        for i in chunk {
            if *i >= len {
                in_bounds = false;
            }
        }
        if !in_bounds {
            break;
        }
    }
    polars_ensure!(in_bounds, OutOfBounds: "indices are out of bounds");
    Ok(())
}

pub trait ToIdx {
    fn to_idx(self, len: u64) -> IdxSize;
}

macro_rules! impl_to_idx {
    ($ty:ty) => {
        impl ToIdx for $ty {
            #[inline]
            fn to_idx(self, _len: u64) -> IdxSize {
                self as IdxSize
            }
        }
    };
    ($ty:ty, $ity:ty) => {
        impl ToIdx for $ty {
            #[inline]
            fn to_idx(self, len: u64) -> IdxSize {
                let idx = self as $ity;
                if idx < 0 {
                    (idx + len as $ity) as IdxSize
                } else {
                    idx as IdxSize
                }
            }
        }
    };
}

impl_to_idx!(u8);
impl_to_idx!(u16);
impl_to_idx!(u32);
impl_to_idx!(u64);
impl_to_idx!(i8, i16);
impl_to_idx!(i16, i32);
impl_to_idx!(i32, i64);
impl_to_idx!(i64, i64);
