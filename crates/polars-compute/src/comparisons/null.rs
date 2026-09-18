//! The equality kernels over a [`PlNullArray`], every element of which is the same null.

use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlNullArray};

use super::{PlTotalEqKernel, repeated};

impl PlTotalEqKernel for PlNullArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        Some(self.validity())
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        repeated(true, self.len())
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        repeated(false, self.len())
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a null array against a scalar")
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a null array against a scalar")
    }
}
