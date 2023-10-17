// fn to_bitmap(

use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;

use super::{TotalOrdKernel, NotSimd};

impl<T: NativeType + NotSimd> TotalOrdKernel for PrimitiveArray<T> {
    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values()
            .iter()
            .zip(other.values().iter())
            .map(|(a, b)| a.tot_lt(b))
            .collect()
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values()
            .iter()
            .zip(other.values().iter())
            .map(|(a, b)| a.tot_le(b))
            .collect()
    }

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values()
            .iter()
            .zip(other.values().iter())
            .map(|(a, b)| a.tot_eq(b))
            .collect()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert!(self.len() == other.len());
        self.values()
            .iter()
            .zip(other.values().iter())
            .map(|(a, b)| a.tot_ne(b))
            .collect()
    }
}