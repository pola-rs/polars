use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::nulls::IsNull;
use crate::slice::GetSaferUnchecked;
use crate::IdxSize;

pub trait Bounded {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn null_count(&self) -> usize;
}

impl<T> Bounded for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

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
    polars_ensure!(in_bounds, ComputeError: "indices are out of bounds");
    Ok(())
}
