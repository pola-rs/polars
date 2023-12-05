use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::slice::GetSaferUnchecked;
use crate::IdxSize;

pub trait Indexable {
    type Item<'a>
    where
        Self: 'a;

    fn get(&self, i: usize) -> Self::Item<'_>;

    /// # Safety
    /// Doesn't do any bound checks.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item<'_>;
}

impl<T> Indexable for &[T] {
    type Item<'a> = &'a T where Self: 'a;

    fn get(&self, i: usize) -> Self::Item<'_> {
        &self[i]
    }

    /// # Safety
    /// Doesn't do any bound checks.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item<'_> {
        self.get_unchecked_release(i)
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
