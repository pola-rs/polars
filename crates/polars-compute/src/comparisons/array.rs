//! The equality kernels over a [`PlFixedSizeListArray`], whose width often settles them.

use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlFixedSizeListArray};

use super::dyn_array::{pl_array_tot_eq_missing_kernel, pl_array_tot_ne_missing_kernel};
use super::{Condense, PlTotalEqKernel, condense, repeated};

impl PlTotalEqKernel for PlFixedSizeListArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        fsl_compare_values(
            self,
            other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        fsl_compare_values(
            self,
            other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        fsl_compare_scalar(
            self,
            &**other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        fsl_compare_scalar(
            self,
            &**other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }
}

/// Compares the lists of `lhs` against `rhs`'s, element for element.
fn fsl_compare_values(
    lhs: &PlFixedSizeListArray,
    rhs: &PlFixedSizeListArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let (length, width) = (lhs.len(), lhs.width());

    if width != rhs.width() || lhs.values().array_type() != rhs.values().array_type() {
        return repeated(mismatch, length);
    }
    // A list of no values is the same list on both sides, whatever is under it.
    if width == 0 {
        return repeated(!mismatch, length);
    }

    match (
        lhs.scalar_value_ignore_validity(),
        rhs.scalar_value_ignore_validity(),
    ) {
        // Each side repeats one list, so comparing those two lists once — `width` values, not
        // `length * width` of them — answers for every element.
        (Some(lhs), Some(rhs)) => {
            let bit = condense(inner(lhs, rhs), 1, width, how);
            repeated(bit.get(0), length)
        },
        // At least one side holds every element's values, so both are read that way.
        _ => {
            let (lhs, rhs) = (lhs.to_flat(), rhs.to_flat());
            let values = inner(lhs.as_array().values(), rhs.as_array().values());
            condense(values, length, width, how)
        },
    }
}

/// Compares the lists of `lhs` against the single list `rhs`.
fn fsl_compare_scalar(
    lhs: &PlFixedSizeListArray,
    rhs: &dyn PlArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let (length, width) = (lhs.len(), lhs.width());

    if width != rhs.len() || lhs.values().array_type() != rhs.array_type() {
        return repeated(mismatch, length);
    }
    if width == 0 || length == 0 {
        return repeated(!mismatch, length);
    }

    // The scalar is one list, so a side that repeats one list too is a single comparison.
    if let Some(lhs) = lhs.scalar_value_ignore_validity() {
        let bit = condense(inner(lhs, rhs), 1, width, how);
        return repeated(bit.get(0), length);
    }

    // A chunk that repeats the scalar's list holds it once and reads it for every element, which
    // is what makes the comparison against it the one over a pair of chunks: a single kernel call
    // over `length * width` values, rather than one call — and a bitmap of its own — per element.
    let rhs = PlFixedSizeListArray::new_broadcast(rhs.to_boxed(), width, length, None);
    fsl_compare_values(lhs, &rhs, how, inner, mismatch)
}
