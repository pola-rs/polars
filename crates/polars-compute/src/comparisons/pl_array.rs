//! The comparison kernels over the arrays of `polars-array` whose kernel is still the Arrow one.

use arrow::bitmap::Bitmap;
use polars_array::arrow::bridge::ToArrow;
use polars_array::{
    Flat, PlArray, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlUtf8ViewArray,
};
use polars_utils::total_ord::TotalEq;

use super::{IN_PLACE_COMPARISON_LIMIT, PlTotalEqKernel, TotalEqKernel, TotalOrdKernel};

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
macro_rules! pl_eq_kernel_body {
    ($lhs:expr, $rhs:expr, $scalar:path, $flat:path, $broadcast:path $(,)?) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        assert_eq!(lhs.len(), rhs.len());
        let length = lhs.len();

        match (
            lhs.scalar_value_ignore_validity(),
            rhs.scalar_value_ignore_validity(),
        ) {
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

/// The equality kernels of the arrays whose values the Arrow kernel reads, in any representation.
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
                    match self.scalar_value_ignore_validity() {
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
                    match self.scalar_value_ignore_validity() {
                        Some(values) => {
                            PlBitmap::new_scalar(values.tot_ne(&other), self.len())
                        },
                        None => {
                            PlBitmap::new(self.to_flat().tot_ne_kernel_broadcast(other), self.len())
                        },
                    }
                }
                /// The values are compared where they lie: see [`PlTotalEqKernel::tot_eq_missing_all`].
                fn tot_eq_missing_all(&self, other: &Self) -> bool {
                    assert_eq!(self.len(), other.len());

                    // Past this many values the written-out comparison earns its allocation back,
                    // and it is the vectorised kernel that reads them.
                    if self.len() > IN_PLACE_COMPARISON_LIMIT {
                        return self.tot_eq_missing_kernel(other).unset_bits() == 0;
                    }

                    self.iter().zip(other.iter()).all(|(lhs, rhs)| lhs.tot_eq(&rhs))
                }

            }
        )*
    };
}

impl_pl_total_eq_kernel!(PlBinaryArray, PlBinaryViewArray, PlUtf8ViewArray);
