//! The equality kernels over a [`PlStructArray`], which are the ones of its fields combined.

use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlStructArray};

use super::dyn_array::{pl_array_tot_eq_missing_kernel, pl_array_tot_ne_missing_kernel};
use super::{PlTotalEqKernel, repeated};

impl PlTotalEqKernel for PlStructArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());

        if self.num_fields() != other.num_fields() {
            return repeated(false, self.len());
        }

        let mut out = repeated(true, self.len());
        for (lhs, rhs) in self.fields().iter().zip(other.fields()) {
            if lhs.array_type() != rhs.array_type() || lhs.len() != rhs.len() {
                return repeated(false, self.len());
            }
            out = out.and(&pl_array_tot_eq_missing_kernel(&**lhs, &**rhs));

            if out.scalar_value() == Some(false) {
                break;
            }
        }
        out
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());

        if self.num_fields() != other.num_fields() {
            return repeated(true, self.len());
        }

        let mut out = repeated(false, self.len());
        for (lhs, rhs) in self.fields().iter().zip(other.fields()) {
            if lhs.array_type() != rhs.array_type() || lhs.len() != rhs.len() {
                return repeated(true, self.len());
            }
            out = out.or(&pl_array_tot_ne_missing_kernel(&**lhs, &**rhs));

            if out.scalar_value() == Some(true) {
                break;
            }
        }
        out
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a struct array against a scalar")
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a struct array against a scalar")
    }
}
