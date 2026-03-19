use arrow::array::BooleanArray;
use arrow::bitmap::{self, Bitmap};

use super::{TotalEqKernel, TotalOrdKernel};

impl TotalEqKernel for BooleanArray {
    type Scalar = bool;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        bitmap::binary(self.values(), other.values(), |l, r| !(l ^ r))
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        self.values() ^ other.values()
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if *other {
            self.values().clone()
        } else {
            !self.values()
        }
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.tot_eq_kernel_broadcast(&!*other)
    }
}

impl TotalOrdKernel for BooleanArray {
    type Scalar = bool;

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        bitmap::binary(self.values(), other.values(), |l, r| !l & r)
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        bitmap::binary(self.values(), other.values(), |l, r| !l | r)
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if *other {
            !self.values()
        } else {
            Bitmap::new_zeroed(self.len())
        }
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if *other {
            Bitmap::new_with_value(true, self.len())
        } else {
            !self.values()
        }
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if *other {
            Bitmap::new_zeroed(self.len())
        } else {
            self.values().clone()
        }
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if *other {
            self.values().clone()
        } else {
            Bitmap::new_with_value(true, self.len())
        }
    }
}
