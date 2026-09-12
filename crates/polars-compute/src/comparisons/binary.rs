use arrow::array::BinaryArray;
use arrow::bitmap::Bitmap;
use arrow::types::Offset;
use polars_array::{PlBitmap, PlBitmapRef, PlFixedSizeBinaryArray};
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::{PlTotalEqKernel, TotalEqKernel, TotalOrdKernel, repeated};

impl<O: Offset> TotalEqKernel for BinaryArray<O> {
    type Scalar = [u8];

    fn validity_mask(&self) -> Option<&Bitmap> {
        self.validity()
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

impl PlTotalEqKernel for PlFixedSizeBinaryArray {
    type Scalar = [u8];

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());

        // Byte strings of different widths are never equal, and ones of no bytes always are:
        // either way the widths settle it for every element without a byte being read.
        if self.width() != other.width() {
            return repeated(false, self.len());
        }
        if self.width() == 0 {
            return repeated(true, self.len());
        }

        match (
            self.scalar_value_ignore_validity(),
            other.scalar_value_ignore_validity(),
        ) {
            // Each side repeats one byte string, so the one comparison answers for all of them.
            (Some(l), Some(r)) => repeated(l == r, self.len()),
            _ => PlBitmap::from_iter((0..self.len()).map(|i| self.value(i) == other.value(i))),
        }
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        self.tot_eq_kernel(other).not()
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        if self.width() != other.len() {
            return repeated(false, self.len());
        }

        match self.scalar_value_ignore_validity() {
            Some(l) => repeated(l == other, self.len()),
            None => PlBitmap::from_iter((0..self.len()).map(|i| self.value(i) == other)),
        }
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        self.tot_eq_kernel_broadcast(other).not()
    }
}
