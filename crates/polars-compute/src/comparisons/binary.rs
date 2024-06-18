use arrow::array::{BinaryArray, FixedSizeBinaryArray};
use arrow::bitmap::Bitmap;
use arrow::types::Offset;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::{TotalEqKernel, TotalOrdKernel};

impl<O: Offset> TotalEqKernel for BinaryArray<O> {
    type Scalar = [u8];

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values_iter()
            .zip(other.values_iter())
            .map(|(l, r)| l.tot_eq(&r))
            .collect()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values_iter()
            .zip(other.values_iter())
            .map(|(l, r)| l.tot_ne(&r))
            .collect()
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.values_iter().map(|l| l.tot_eq(&other)).collect()
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.values_iter().map(|l| l.tot_ne(&other)).collect()
    }
}

impl<O: Offset> TotalOrdKernel for BinaryArray<O> {
    type Scalar = [u8];

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values_iter()
            .zip(other.values_iter())
            .map(|(l, r)| l.tot_lt(&r))
            .collect()
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values_iter()
            .zip(other.values_iter())
            .map(|(l, r)| l.tot_le(&r))
            .collect()
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.values_iter().map(|l| l.tot_lt(&other)).collect()
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.values_iter().map(|l| l.tot_le(&other)).collect()
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.values_iter().map(|l| l.tot_gt(&other)).collect()
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.values_iter().map(|l| l.tot_ge(&other)).collect()
    }
}

impl TotalEqKernel for FixedSizeBinaryArray {
    type Scalar = [u8];

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());

        if self.size() != other.size() {
            return Bitmap::new_zeroed(self.len());
        }

        (0..self.len())
            .map(|i| self.value(i) == other.value(i))
            .collect()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());

        if self.size() != other.size() {
            return Bitmap::new_with_value(true, self.len());
        }

        (0..self.len())
            .map(|i| self.value(i) != other.value(i))
            .collect()
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if self.size() != other.len() {
            return Bitmap::new_zeroed(self.len());
        }

        (0..self.len()).map(|i| self.value(i) == other).collect()
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if self.size() != other.len() {
            return Bitmap::new_with_value(true, self.len());
        }

        (0..self.len()).map(|i| self.value(i) != other).collect()
    }
}
