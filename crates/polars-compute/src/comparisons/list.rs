//! The equality kernels over a [`PlListArray`], whose lengths often settle them.

use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlListArray};

use super::dyn_array::with_array_pair;
use super::{PlTotalEqKernel, repeated};

/// Compares the lists of `$lhs` against `$rhs`'s, element for element.
///
/// `$mismatch` is the bit an element gets when its two lists hold no pair of values to compare at
/// all — differing in length, or in the type under them — and it is what turns equality into
/// inequality: the answer for a pair that *does* compare is flipped by it in turn.
///
/// The values of both sides are downcast once, ahead of the walk over the elements: an element is
/// then a slice of a concrete array, which is a clone of its buffers and nothing more. Reading it
/// as a `&dyn PlArray` instead costs a box and a dispatch of its own, once per element.
macro_rules! compare_values {
    ($lhs:expr, $rhs:expr, $mismatch:expr $(,)?) => {{
        let (lhs, rhs, mismatch) = ($lhs, $rhs, $mismatch);
        let length = lhs.len();

        // Lists of different value types hold no pair of values to compare.
        if lhs.values().array_type() != rhs.values().array_type() {
            repeated(mismatch, length)
        } else {
            with_array_pair!(lhs.values(), rhs.values(), |lhs_values, rhs_values| {
                // The bit an element gets from the ranges of the values its two lists cover.
                let element = |l: std::ops::Range<usize>, r: std::ops::Range<usize>| {
                    // Lists of different lengths hold no pair of values to compare.
                    if l.len() != r.len() {
                        return mismatch;
                    }
                    // Two empty lists are the same list, whatever the value type under them.
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

                    // The two lists answer with the one bit the caller wants, so nothing is
                    // written out per element — see `PlTotalEqKernel::tot_eq_missing_all`.
                    PlTotalEqKernel::tot_eq_missing_all(&l, &r) != mismatch
                };

                // Both sides hold the one range every element of them covers, and neither is null
                // anywhere, so comparing those two lists once answers for every element: a single
                // bit stands for all of them and none is written out.
                if let (Some(l), Some(r)) = (lhs.scalar_offsets(), rhs.scalar_offsets())
                    && lhs.null_count() == 0
                    && rhs.null_count() == 0
                {
                    return repeated(element(l, r), length);
                }

                PlBitmap::from_iter((0..length).map(|i| {
                    // A null element has no list to read; the missing-aware kernel answers for it.
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

        // Lists of different value types hold no pair of values to compare.
        if lhs.values().array_type() != rhs.array_type() {
            repeated(mismatch, length)
        } else {
            with_array_pair!(lhs.values(), rhs, |lhs_values, rhs_values| {
                let width = rhs_values.len();

                // The bit an element gets from the range of the values its list covers, against
                // the single list on the other side.
                let element = |l: std::ops::Range<usize>| {
                    if l.len() != width {
                        return mismatch;
                    }
                    if l.is_empty() {
                        return !mismatch;
                    }

                    // SAFETY: as in `compare_values`.
                    let l = unsafe { lhs_values.sliced_unchecked(l.start, l.len()) };

                    // As in `compare_values`: the one bit, not a mask to read it off.
                    PlTotalEqKernel::tot_eq_missing_all(&l, rhs_values) != mismatch
                };

                // Every element covers the one range the offsets hold and none of them is null,
                // so the one comparison against the scalar answers for all of them.
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
