use arrow::array::Utf8Array;
use arrow::bitmap::Bitmap;
use arrow::types::Offset;

use super::{TotalEqKernel, TotalOrdKernel};

impl<O: Offset> TotalEqKernel for Utf8Array<O> {
    type Scalar = str;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_eq_kernel(&other.to_binary())
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_ne_kernel(&other.to_binary())
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_eq_kernel_broadcast(other.as_bytes())
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_ne_kernel_broadcast(other.as_bytes())
    }
}

impl<O: Offset> TotalOrdKernel for Utf8Array<O> {
    type Scalar = str;

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_lt_kernel(&other.to_binary())
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        self.to_binary().tot_le_kernel(&other.to_binary())
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_lt_kernel_broadcast(other.as_bytes())
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_le_kernel_broadcast(other.as_bytes())
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_gt_kernel_broadcast(other.as_bytes())
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binary().tot_ge_kernel_broadcast(other.as_bytes())
    }
}
