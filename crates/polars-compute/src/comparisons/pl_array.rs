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
///
/// The bitmap is taken as it stands rather than through `flat_bitmap`: a single bit over a single
/// element is both the flat and the scalar reading of that mask, and `flat_bitmap` resolves it as
/// the scalar one — so it answers `None` for an array of one element that is perfectly flat.
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

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::PlBitmap;

    use super::*;

    /// A mask of one bit over one element is both flat and scalar, and `PlBitmapRef::repr` resolves
    /// such a buffer as the scalar one. The mask of a flat array is therefore read as it stands:
    /// resolving it first would answer `None` for an array that is perfectly flat.
    #[test]
    fn a_flat_array_of_one_element_has_a_mask_to_read() {
        for bit in [false, true] {
            let array = PlBinaryViewArray::from_values_iter([b"foo".as_slice()])
                .with_validity(Some(PlBitmap::from_bitmap(Bitmap::new_with_value(bit, 1))));

            let flat = array.to_flat().into_owned();
            assert!(flat.as_array().validity().unwrap().is_scalar());

            assert_eq!(flat_validity(&flat).map(Bitmap::len), Some(1));

            // Which is what the missing-aware kernels read: one element equals itself when it is
            // there, and a null equals a null, so either way the two agree.
            assert!(flat.tot_eq_missing_kernel(&flat).get_bit(0));
            assert!(!flat.tot_ne_missing_kernel(&flat).get_bit(0));
        }
    }
}
