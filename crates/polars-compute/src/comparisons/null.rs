use arrow::array::{Array, NullArray};
use arrow::bitmap::Bitmap;

use super::TotalEqKernel;

impl TotalEqKernel for NullArray {
    type Scalar = Box<dyn Array>;

    fn validity_mask(&self) -> Option<&Bitmap> {
        self.validity()
    }

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
