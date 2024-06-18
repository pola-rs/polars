use arrow::array::{Array, NullArray};
use arrow::bitmap::Bitmap;

use super::{TotalEqKernel, TotalOrdKernel};

impl TotalEqKernel for NullArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        Bitmap::new_with_value(true, self.len())
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        Bitmap::new_zeroed(self.len())
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }
}

impl TotalOrdKernel for NullArray {
    type Scalar = Box<dyn Array>;

    fn tot_lt_kernel(&self, _other: &Self) -> Bitmap {
        unimplemented!()
    }

    fn tot_le_kernel(&self, _other: &Self) -> Bitmap {
        unimplemented!()
    }

    fn tot_lt_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_le_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_gt_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_ge_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }
}
