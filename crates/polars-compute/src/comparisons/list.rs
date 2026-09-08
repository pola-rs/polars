//! The equality kernels over a [`PlListArray`], whose lengths often settle them.

use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlListArray};

use super::dyn_array::{pl_array_tot_eq_missing_kernel, pl_array_tot_ne_missing_kernel};
use super::{Condense, PlTotalEqKernel, condense, repeated};

impl PlTotalEqKernel for PlListArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        list_compare_values(
            self,
            other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        list_compare_values(
            self,
            other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        list_compare_scalar(
            self,
            &**other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        list_compare_scalar(
            self,
            &**other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }
}

/// Compares the lists of `lhs` against `rhs`'s, element for element.
fn list_compare_values(
    lhs: &PlListArray,
    rhs: &PlListArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let length = lhs.len();

    if lhs.values().array_type() != rhs.values().array_type() {
        return repeated(mismatch, length);
    }

    // Both sides repeat one list, so comparing those two lists once answers for every element.
    if let (Some(lhs), Some(rhs)) = (lhs.scalar_value(), rhs.scalar_value()) {
        // A null element is one the missing-aware kernel answers for, not this.
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            return repeated(compare_lists(&*lhs, &*rhs, how, inner, mismatch), length);
        }
    }

    PlBitmap::from_iter((0..length).map(|i| {
        // A null element has no list to read; the missing-aware kernel answers for it.
        if lhs.is_null(i) || rhs.is_null(i) {
            return !mismatch;
        }
        compare_lists(&*lhs.value(i), &*rhs.value(i), how, inner, mismatch)
    }))
}

/// Compares the lists of `lhs` against the single list `rhs`.
fn list_compare_scalar(
    lhs: &PlListArray,
    rhs: &dyn PlArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let length = lhs.len();

    if lhs.values().array_type() != rhs.array_type() {
        return repeated(mismatch, length);
    }

    if let Some(Some(lhs)) = lhs.scalar_value() {
        return repeated(compare_lists(&*lhs, rhs, how, inner, mismatch), length);
    }

    PlBitmap::from_iter((0..length).map(|i| {
        if lhs.is_null(i) {
            return !mismatch;
        }
        compare_lists(&*lhs.value(i), rhs, how, inner, mismatch)
    }))
}

/// Whether the two lists answer `how` over the values they hold, one against one.
fn compare_lists(
    lhs: &dyn PlArray,
    rhs: &dyn PlArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> bool {
    // Lists of different lengths hold no pair of values to compare.
    if lhs.len() != rhs.len() {
        return mismatch;
    }
    // Two empty lists are the same list, whatever the value type under them.
    if lhs.is_empty() {
        return !mismatch;
    }
    condense(inner(lhs, rhs), 1, lhs.len(), how).get(0)
}
