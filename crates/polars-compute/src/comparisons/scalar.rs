use arrow::array::{BinaryArray, PrimitiveArray, Utf8Array};
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::{NotSimdPrimitive, TotalOrdKernel};

impl<T: NativeType + NotSimdPrimitive + TotalOrd> TotalOrdKernel for PrimitiveArray<T> {
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

impl TotalOrdKernel for BinaryArray<i64> {
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

impl TotalOrdKernel for Utf8Array<i64> {
    type Scalar = str;

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_lt_kernel(&other.to_binary())
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_le_kernel(&other.to_binary())
    }

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_eq_kernel(&other.to_binary())
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_ne_kernel(&other.to_binary())
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_eq_kernel_broadcast(&other.as_bytes())
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_ne_kernel_broadcast(&other.as_bytes())
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_lt_kernel_broadcast(&other.as_bytes())
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_le_kernel_broadcast(&other.as_bytes())
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_gt_kernel_broadcast(&other.as_bytes())
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_ge_kernel_broadcast(&other.as_bytes())
    }
}
