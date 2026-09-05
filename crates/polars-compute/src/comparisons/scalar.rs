use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use polars_array::{Flat, PlPrimitiveArray};
use polars_utils::total_ord::TotalOrd;

use super::{TotalEqKernel, TotalOrdKernel};
use crate::NotSimdPrimitive;

// The element-at-a-time kernels, for the types the SIMD ones do not cover. `$A` is the array they
// read the values of, in either the Arrow layout or the flat one of `polars-array`.
macro_rules! impl_scalar_total_ord_kernel {
    ($A: ty) => {
        impl<T: NotSimdPrimitive + TotalOrd> TotalEqKernel for $A {
            type Scalar = T;

            fn validity_mask(&self) -> Option<&Bitmap> {
                self.validity()
            }

            fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
                assert!(self.len() == other.len());
                self.values()
                    .iter()
                    .zip(other.values().iter())
                    .map(|(l, r)| l.tot_eq(r))
                    .collect()
            }

            fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
                assert!(self.len() == other.len());
                self.values()
                    .iter()
                    .zip(other.values().iter())
                    .map(|(l, r)| l.tot_ne(r))
                    .collect()
            }

            fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                self.values().iter().map(|l| l.tot_eq(other)).collect()
            }

            fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                self.values().iter().map(|l| l.tot_ne(other)).collect()
            }
        }

        impl<T: NotSimdPrimitive + TotalOrd> TotalOrdKernel for $A {
            type Scalar = T;

            fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
                assert!(self.len() == other.len());
                self.values()
                    .iter()
                    .zip(other.values().iter())
                    .map(|(l, r)| l.tot_lt(r))
                    .collect()
            }

            fn tot_le_kernel(&self, other: &Self) -> Bitmap {
                assert!(self.len() == other.len());
                self.values()
                    .iter()
                    .zip(other.values().iter())
                    .map(|(l, r)| l.tot_le(r))
                    .collect()
            }

            fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                self.values().iter().map(|l| l.tot_lt(other)).collect()
            }

            fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                self.values().iter().map(|l| l.tot_le(other)).collect()
            }

            fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                self.values().iter().map(|l| l.tot_gt(other)).collect()
            }

            fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                self.values().iter().map(|l| l.tot_ge(other)).collect()
            }
        }
    };
}

impl_scalar_total_ord_kernel!(PrimitiveArray<T>);
impl_scalar_total_ord_kernel!(Flat<PlPrimitiveArray<T>>);
