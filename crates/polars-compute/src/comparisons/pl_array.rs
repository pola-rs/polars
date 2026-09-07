//! The comparison kernels over the arrays of `polars-array` whose kernel is still the Arrow one.

use arrow::bitmap::Bitmap;
use polars_array::arrow::bridge::ToArrow;
use polars_array::{
    Flat, PlArray, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlUtf8ViewArray,
};
use polars_utils::total_ord::TotalEq;

use super::{PlTotalEqKernel, TotalEqKernel, TotalOrdKernel};

/// The validity mask of a flat array, which holds one bit per element like its every other buffer.
fn flat_validity<A: PlArray>(array: &Flat<A>) -> Option<&Bitmap> {
    array
        .as_array()
        .validity()
        .map(|validity| validity.into_inner().0)
}

macro_rules! impl_total_eq_kernel {
    ($($A:ty),* $(,)?) => {
        $(
            impl TotalEqKernel for Flat<$A> {
                type Scalar = <<$A as ToArrow>::Arrow as TotalEqKernel>::Scalar;

                fn validity_mask(&self) -> Option<&Bitmap> {
                    flat_validity(self)
                }

                fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
                    <$A>::to_arrow(self).tot_eq_kernel(&<$A>::to_arrow(other))
                }

                fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
                    <$A>::to_arrow(self).tot_ne_kernel(&<$A>::to_arrow(other))
                }

                fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                    <$A>::to_arrow(self).tot_eq_kernel_broadcast(other)
                }

                fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                    <$A>::to_arrow(self).tot_ne_kernel_broadcast(other)
                }
            }
        )*
    };
}

macro_rules! impl_total_ord_kernel {
    ($($A:ty),* $(,)?) => {
        $(
            impl TotalOrdKernel for Flat<$A> {
                type Scalar = <<$A as ToArrow>::Arrow as TotalOrdKernel>::Scalar;

                fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
                    <$A>::to_arrow(self).tot_lt_kernel(&<$A>::to_arrow(other))
                }

                fn tot_le_kernel(&self, other: &Self) -> Bitmap {
                    <$A>::to_arrow(self).tot_le_kernel(&<$A>::to_arrow(other))
                }

                fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                    <$A>::to_arrow(self).tot_lt_kernel_broadcast(other)
                }

                fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                    <$A>::to_arrow(self).tot_le_kernel_broadcast(other)
                }

                fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                    <$A>::to_arrow(self).tot_gt_kernel_broadcast(other)
                }

                fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                    <$A>::to_arrow(self).tot_ge_kernel_broadcast(other)
                }
            }
        )*
    };
}

impl_total_eq_kernel!(PlBinaryArray, PlBinaryViewArray, PlUtf8ViewArray);

impl_total_ord_kernel!(PlBinaryArray, PlBinaryViewArray, PlUtf8ViewArray);

/// The body of a [`PlTotalEqKernel`] that dispatches on the values representation of its operands.
///
/// A values buffer that holds a single value says the same of every element, which turns the
/// comparison against it into the broadcast kernel — and, where both sides hold one, into the
/// single comparison of those two values, whose answer is the one bit every element shares.
macro_rules! pl_eq_kernel_body {
    ($lhs:expr, $rhs:expr, $scalar:path, $flat:path, $broadcast:path $(,)?) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        assert_eq!(lhs.len(), rhs.len());
        let length = lhs.len();

        match (lhs.scalar_values(), rhs.scalar_values()) {
            // Neither side is written out: the one comparison answers for every element.
            (Some(lhs), Some(rhs)) => PlBitmap::new_scalar($scalar(&lhs, &rhs), length),
            // One side holds the value the other is compared against element by element, which is
            // exactly what the broadcast kernel takes.
            (Some(value), None) => PlBitmap::new($broadcast(&*rhs.to_flat(), value), length),
            (None, Some(value)) => PlBitmap::new($broadcast(&*lhs.to_flat(), value), length),
            // Both sides hold one value per element, which is the layout the kernel reads.
            (None, None) => PlBitmap::new($flat(&*lhs.to_flat(), &*rhs.to_flat()), length),
        }
    }};
}

/// The equality kernels of the arrays whose values the Arrow kernel reads, over the array in
/// whichever representation it comes in.
///
/// What the [`PlBitmap`] buys here is that a caller (a nested array recursing into its children,
/// say) does not have to know which arrays those are.
macro_rules! impl_pl_total_eq_kernel {
    ($($A:ty),* $(,)?) => {
        $(
            impl PlTotalEqKernel for $A {
                type Scalar = <Flat<$A> as TotalEqKernel>::Scalar;

                fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
                    PlArray::validity(self)
                }

                fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
                    pl_eq_kernel_body!(
                        self,
                        other,
                        TotalEq::tot_eq,
                        TotalEqKernel::tot_eq_kernel,
                        TotalEqKernel::tot_eq_kernel_broadcast,
                    )
                }

                fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
                    pl_eq_kernel_body!(
                        self,
                        other,
                        TotalEq::tot_ne,
                        TotalEqKernel::tot_ne_kernel,
                        TotalEqKernel::tot_ne_kernel_broadcast,
                    )
                }

                fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
                    // A values buffer of one value is compared against the scalar once, and its
                    // answer is the bit every element of this array shares.
                    match self.scalar_values() {
                        Some(values) => {
                            PlBitmap::new_scalar(values.tot_eq(&other), self.len())
                        },
                        None => {
                            PlBitmap::new(self.to_flat().tot_eq_kernel_broadcast(other), self.len())
                        },
                    }
                }

                fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
                    // As above, with the answer the other way around.
                    match self.scalar_values() {
                        Some(values) => {
                            PlBitmap::new_scalar(values.tot_ne(&other), self.len())
                        },
                        None => {
                            PlBitmap::new(self.to_flat().tot_ne_kernel_broadcast(other), self.len())
                        },
                    }
                }
            }
        )*
    };
}

impl_pl_total_eq_kernel!(PlBinaryArray, PlBinaryViewArray, PlUtf8ViewArray);

#[cfg(test)]
mod tests {
    use polars_array::PlBitmap;

    use super::*;

    /// Two sides that each repeat one value are compared once, and the one bit that answers for
    /// every element is what comes back — a mask of a billion elements in `O(1)` memory.
    #[test]
    fn two_scalar_sides_compare_to_a_single_bit() {
        let length = 1_000_000_000;
        let foo = PlUtf8ViewArray::new_scalar("foo", length);
        let bar = PlUtf8ViewArray::new_scalar("bar", length);

        for (lhs, rhs, equal) in [(&foo, &foo, true), (&foo, &bar, false)] {
            let eq = lhs.tot_eq_kernel(rhs);
            assert_eq!(eq.len(), length);
            assert!(eq.is_scalar());
            assert_eq!(eq.scalar_value(), Some(equal));

            let ne = lhs.tot_ne_kernel(rhs);
            assert!(ne.is_scalar());
            assert_eq!(ne.scalar_value(), Some(!equal));
        }
    }

    /// One side repeating a value is compared against the other element by element, which is the
    /// broadcast kernel: the answer holds one bit per element and the repeated side is not
    /// written out to reach it.
    #[test]
    fn one_scalar_side_is_broadcast_over_the_other() {
        let flat = PlUtf8ViewArray::from_iter([Some("foo"), Some("bar"), None]);
        let scalar = PlUtf8ViewArray::new_scalar("foo", 3);

        for eq in [flat.tot_eq_kernel(&scalar), scalar.tot_eq_kernel(&flat)] {
            assert_eq!(eq.len(), 3);
            assert!(eq.is_flat());
            // The kernels answer over the values alone; the null element's bytes are its own.
            assert!(eq.get(0));
            assert!(!eq.get(1));
        }

        // A scalar values buffer answers a scalar comparison with the one bit it settles.
        let against_scalar = scalar.tot_eq_kernel_broadcast("foo");
        assert!(against_scalar.is_scalar());
        assert_eq!(against_scalar.scalar_value(), Some(true));
        assert_eq!(
            scalar.tot_ne_kernel_broadcast("foo").scalar_value(),
            Some(false)
        );
    }

    /// The shortcut the scalar arms take has to answer exactly what walking the two sides
    /// element by element would: the kernel is a *total* equality one, so the comparison of the
    /// two repeated values is `TotalEq`, not `PartialEq`. For bytes and strings the two agree —
    /// this pins them together so a values type where they do not cannot slip in unnoticed.
    #[test]
    fn every_scalar_arm_agrees_with_the_flat_kernel() {
        let values = ["", "foo", "bar", "a value that is too long to inline"];

        for lhs in values {
            for rhs in values {
                let flat_lhs = PlUtf8ViewArray::from_iter([Some(lhs), Some(lhs), Some(lhs)]);
                let flat_rhs = PlUtf8ViewArray::from_iter([Some(rhs), Some(rhs), Some(rhs)]);
                let scalar_lhs = PlUtf8ViewArray::new_scalar(lhs, 3);
                let scalar_rhs = PlUtf8ViewArray::new_scalar(rhs, 3);

                // Walking both sides is the answer the three shortcuts have to reproduce.
                let expected = flat_lhs.tot_eq_kernel(&flat_rhs);
                assert!(expected.is_flat(), "{lhs:?} {rhs:?}");

                for shortcut in [
                    scalar_lhs.tot_eq_kernel(&scalar_rhs),
                    scalar_lhs.tot_eq_kernel(&flat_rhs),
                    flat_lhs.tot_eq_kernel(&scalar_rhs),
                    scalar_lhs.tot_eq_kernel_broadcast(rhs),
                ] {
                    assert_eq!(shortcut.as_ref(), expected.as_ref(), "{lhs:?} {rhs:?}");
                }

                let expected = flat_lhs.tot_ne_kernel(&flat_rhs);
                for shortcut in [
                    scalar_lhs.tot_ne_kernel(&scalar_rhs),
                    scalar_lhs.tot_ne_kernel(&flat_rhs),
                    flat_lhs.tot_ne_kernel(&scalar_rhs),
                    scalar_lhs.tot_ne_kernel_broadcast(rhs),
                ] {
                    assert_eq!(shortcut.as_ref(), expected.as_ref(), "{lhs:?} {rhs:?}");
                }
            }
        }
    }

    /// A side null throughout is null throughout however its values are laid out, which the
    /// missing-aware kernel answers with one bit for all of them.
    #[test]
    fn a_scalar_side_that_is_null_throughout_still_reads_as_null() {
        let length = 1_000_000;
        let nulls = PlUtf8ViewArray::new_scalar("foo", length)
            .with_validity(Some(PlBitmap::new_scalar(false, length)));
        let values = PlUtf8ViewArray::new_scalar("foo", length);

        assert_eq!(
            nulls.tot_eq_missing_kernel(&nulls).scalar_value(),
            Some(true)
        );
        assert_eq!(
            nulls.tot_eq_missing_kernel(&values).scalar_value(),
            Some(false)
        );
    }
}
