use arrow::array::{Array, FixedSizeListArray};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use super::TotalEqKernel;
use crate::comparisons::dyn_array::{array_tot_eq_missing_kernel, array_tot_ne_missing_kernel};

/// Condenses a bitmap of n * width elements into one with n elements.
///
/// For each block of width bits a zero count is done. The block of bits is then
/// replaced with a single bit: the result of true_zero_count(zero_count).
fn agg_array_bitmap<F>(bm: Bitmap, width: usize, true_zero_count: F) -> Bitmap
where
    F: Fn(usize) -> bool,
{
    if bm.len() == 1 {
        bm
    } else {
        assert!(width > 0 && bm.len() % width == 0);

        let (slice, offset, _len) = bm.as_slice();
        (0..bm.len() / width)
            .map(|i| true_zero_count(count_zeros(slice, offset + i * width, width)))
            .collect()
    }
}

impl TotalEqKernel for FixedSizeListArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        // Nested comparison always done with eq_missing, propagating doesn't
        // make any sense.

        assert_eq!(self.len(), other.len());
        let ArrowDataType::FixedSizeList(self_type, self_width) =
            self.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::FixedSizeList(other_type, other_width) =
            other.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(self_type.data_type(), other_type.data_type());

        if self_width != other_width {
            return Bitmap::new_with_value(false, self.len());
        }

        let inner = array_tot_eq_missing_kernel(self.values().as_ref(), other.values().as_ref());

        agg_array_bitmap(inner, self.size(), |zeroes| zeroes == 0)
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert_eq!(self.len(), other.len());
        let ArrowDataType::FixedSizeList(self_type, self_width) =
            self.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::FixedSizeList(other_type, other_width) =
            other.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(self_type.data_type(), other_type.data_type());

        if self_width != other_width {
            return Bitmap::new_with_value(true, self.len());
        }

        let inner = array_tot_ne_missing_kernel(self.values().as_ref(), other.values().as_ref());

        agg_array_bitmap(inner, self.size(), |zeroes| zeroes < self.size())
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }
}
