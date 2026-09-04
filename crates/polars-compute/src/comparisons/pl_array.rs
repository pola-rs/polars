//! The comparison kernels over the arrays of `polars-array`, for the element types whose kernel
//! is still the Arrow one: the chunk crosses over through [`ToArrow`] and the mask comes straight
//! back. A [`PlPrimitiveArray`](polars_array::PlPrimitiveArray) has its own kernel instead — see
//! `simd` and `scalar`.
//!
//! Every kernel here takes a [`Flat`] array, whose buffers the export hands over as they are.

use arrow::bitmap::Bitmap;
#[cfg(feature = "dtype-array")]
use polars_array::PlFixedSizeListArray;
use polars_array::arrow::bridge::ToArrow;
use polars_array::{
    Flat, PlArray, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlListArray, PlStructArray,
    PlUtf8ViewArray,
};

use super::{TotalEqKernel, TotalOrdKernel};

/// The validity mask of a flat array, which holds one bit per element like its every other buffer.
fn flat_validity<A: PlArray>(array: &Flat<A>) -> Option<&Bitmap> {
    array.as_array().validity().map(|validity| {
        validity
            .flat_bitmap()
            .expect("the mask of a flat array is flat")
    })
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

impl_total_eq_kernel!(
    PlBooleanArray,
    PlBinaryArray,
    PlBinaryViewArray,
    PlUtf8ViewArray,
    PlListArray,
    PlStructArray,
);

#[cfg(feature = "dtype-array")]
impl_total_eq_kernel!(PlFixedSizeListArray);

impl_total_ord_kernel!(
    PlBooleanArray,
    PlBinaryArray,
    PlBinaryViewArray,
    PlUtf8ViewArray,
);
