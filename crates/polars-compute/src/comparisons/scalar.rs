use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use polars_utils::total_ord::TotalOrd;

use super::{TotalEqKernel, TotalOrdKernel};
use crate::NotSimdPrimitive;

impl<T: NotSimdPrimitive + TotalOrd> TotalEqKernel for PrimitiveArray<T> {
    type Scalar = T;

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

impl<T: NotSimdPrimitive + TotalOrd> TotalOrdKernel for PrimitiveArray<T> {
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
