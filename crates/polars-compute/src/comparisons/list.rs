//! The equality kernels over a [`PlListArray`], whose lengths often settle them.

use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlListArray};

use super::dyn_array::with_array_pair;
use super::{PlTotalEqKernel, repeated};

/// Compares the lists of `$lhs` against `$rhs`'s, element for element.
macro_rules! compare_values {
    ($lhs:expr, $rhs:expr, $mismatch:expr $(,)?) => {{
        let (lhs, rhs, mismatch) = ($lhs, $rhs, $mismatch);
        let length = lhs.len();

        if lhs.values().array_type() != rhs.values().array_type() {
            repeated(mismatch, length)
        } else {
            with_array_pair!(lhs.values(), rhs.values(), |lhs_values, rhs_values| {
                let element = |l: std::ops::Range<usize>, r: std::ops::Range<usize>| {
                    if l.len() != r.len() {
                        return mismatch;
                    }
                    if l.is_empty() {
                        return !mismatch;
                    }

                    // SAFETY: the offsets of a list array are ordered and bounded by the length
                    // of its values, so every range they hold is in bounds of them.
                    let (l, r) = unsafe {
                        (
                            lhs_values.sliced_unchecked(l.start, l.len()),
                            rhs_values.sliced_unchecked(r.start, r.len()),
                        )
                    };

                    PlTotalEqKernel::tot_eq_missing_all(&l, &r) != mismatch
                };

                if let (Some(l), Some(r)) = (lhs.scalar_offsets(), rhs.scalar_offsets())
                    && lhs.null_count() == 0
                    && rhs.null_count() == 0
                {
                    return repeated(element(l, r), length);
                }

                PlBitmap::from_iter((0..length).map(|i| {
                    if lhs.is_null(i) || rhs.is_null(i) {
                        return !mismatch;
                    }

                    // SAFETY: `i` is below the length both arrays share.
                    let (l, r) =
                        unsafe { (lhs.value_range_unchecked(i), rhs.value_range_unchecked(i)) };

                    element(l, r)
                }))
            })
        }
    }};
}

/// Compares the lists of `$lhs` against the single list `$rhs`, per [`compare_values`].
macro_rules! compare_scalar {
    ($lhs:expr, $rhs:expr, $mismatch:expr $(,)?) => {{
        let (lhs, rhs, mismatch) = ($lhs, $rhs, $mismatch);
        let length = lhs.len();

        if lhs.values().array_type() != rhs.array_type() {
            repeated(mismatch, length)
        } else {
            with_array_pair!(lhs.values(), rhs, |lhs_values, rhs_values| {
                let width = rhs_values.len();

                let element = |l: std::ops::Range<usize>| {
                    if l.len() != width {
                        return mismatch;
                    }
                    if l.is_empty() {
                        return !mismatch;
                    }

                    // SAFETY: as in `compare_values`.
                    let l = unsafe { lhs_values.sliced_unchecked(l.start, l.len()) };

                    PlTotalEqKernel::tot_eq_missing_all(&l, rhs_values) != mismatch
                };

                if let Some(l) = lhs.scalar_offsets()
                    && lhs.null_count() == 0
                {
                    return repeated(element(l), length);
                }

                PlBitmap::from_iter((0..length).map(|i| {
                    if lhs.is_null(i) {
                        return !mismatch;
                    }

                    // SAFETY: `i` is below the length of `lhs`.
                    element(unsafe { lhs.value_range_unchecked(i) })
                }))
            })
        }
    }};
}

impl PlTotalEqKernel for PlListArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        compare_values!(self, other, false)
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        compare_values!(self, other, true)
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        compare_scalar!(self, &**other, false)
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        compare_scalar!(self, &**other, true)
    }
}
