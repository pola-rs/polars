//! The comparison kernels over the arrays of `polars-array` whose kernel is still the Arrow one.

use arrow::bitmap::Bitmap;
use polars_array::arrow::bridge::ToArrow;
use polars_array::{
    Flat, PlArray, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlUtf8ViewArray,
};

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

/// The equality kernels of the arrays whose values the Arrow kernel reads, over the array in
/// whichever representation it comes in.
///
/// The values of these arrays are compared one element at a time, so a scalar one is written out
/// first — a mask that repeats a single bit comes only from the shapes settling the answer, which
/// for these arrays they never do. What the [`PlBitmap`] buys here is that a caller (a nested
/// array recursing into its children, say) does not have to know which arrays those are.
macro_rules! impl_pl_total_eq_kernel {
    ($($A:ty),* $(,)?) => {
        $(
            impl PlTotalEqKernel for $A {
                type Scalar = <Flat<$A> as TotalEqKernel>::Scalar;

                fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
                    PlArray::validity(self)
                }

                fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
                    let (lhs, rhs) = (self.to_flat(), other.to_flat());
                    PlBitmap::new(lhs.tot_eq_kernel(&rhs), self.len())
                }

                fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
                    let (lhs, rhs) = (self.to_flat(), other.to_flat());
                    PlBitmap::new(lhs.tot_ne_kernel(&rhs), self.len())
                }

                fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
                    PlBitmap::new(self.to_flat().tot_eq_kernel_broadcast(other), self.len())
                }

                fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
                    PlBitmap::new(self.to_flat().tot_ne_kernel_broadcast(other), self.len())
                }
            }
        )*
    };
}

impl_pl_total_eq_kernel!(PlBinaryArray, PlBinaryViewArray, PlUtf8ViewArray);
