use polars_arrow::array::{Array, StructArray};
use polars_arrow::bitmap::Bitmap;

use super::TotalEqKernel;
use crate::comparisons::dyn_array::{array_tot_eq_missing_kernel, array_tot_ne_missing_kernel};

impl TotalEqKernel for StructArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        let lhs = self;
        let rhs = other;
        assert_eq!(lhs.len(), rhs.len());

        if lhs.fields() != rhs.fields() {
            return Bitmap::new_zeroed(lhs.len());
        }

        // Compare field by field; the `*_missing` callers apply the struct validity.
        lhs.values()
            .iter()
            .zip(rhs.values())
            .map(|(l, r)| array_tot_eq_missing_kernel(l.as_ref(), r.as_ref()))
            .reduce(|acc, eq| &acc & &eq)
            .unwrap_or_else(|| Bitmap::new_with_value(true, lhs.len()))
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        let lhs = self;
        let rhs = other;
        assert_eq!(lhs.len(), rhs.len());

        if lhs.fields() != rhs.fields() {
            return Bitmap::new_with_value(true, lhs.len());
        }

        // Compare field by field; the `*_missing` callers apply the struct validity.
        lhs.values()
            .iter()
            .zip(rhs.values())
            .map(|(l, r)| array_tot_ne_missing_kernel(l.as_ref(), r.as_ref()))
            .reduce(|acc, ne| &acc | &ne)
            .unwrap_or_else(|| Bitmap::new_zeroed(lhs.len()))
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }
}
