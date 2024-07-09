use arrow::array::ListArray;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::Offset;

use super::TotalEqKernel;
use crate::comparisons::dyn_array::{array_tot_eq_missing_kernel, array_tot_ne_missing_kernel};

impl<O: Offset> TotalEqKernel for ListArray<O> {
    type Scalar = ();

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        assert_eq!(self.len(), other.len());

        let mut bitmap = MutableBitmap::with_capacity(self.len());

        for i in 0..self.len() {
            let lval = self.validity().map_or(true, |v| v.get(i).unwrap());
            let rval = other.validity().map_or(true, |v| v.get(i).unwrap());

            if !lval || !rval {
                bitmap.push(true);
                continue;
            }

            let (lstart, lend) = self.offsets().start_end(i);
            let (rstart, rend) = other.offsets().start_end(i);

            if lend - lstart != rend - rstart {
                bitmap.push(false);
                continue;
            }

            let mut lhs_values = self.values().clone();
            lhs_values.slice(lstart, lend - lstart);
            let mut rhs_values = self.values().clone();
            rhs_values.slice(rstart, rend - rstart);

            let result = array_tot_eq_missing_kernel(lhs_values.as_ref(), rhs_values.as_ref());
            bitmap.push(result.unset_bits() == 0);
        }

        bitmap.freeze()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert_eq!(self.len(), other.len());

        let mut bitmap = MutableBitmap::with_capacity(self.len());

        for i in 0..self.len() {
            let (lstart, lend) = self.offsets().start_end(i);
            let (rstart, rend) = other.offsets().start_end(i);

            let lval = self.validity().map_or(true, |v| v.get(i).unwrap());
            let rval = other.validity().map_or(true, |v| v.get(i).unwrap());

            if !lval || !rval {
                bitmap.push(false);
                continue;
            }

            if lend - lstart != rend - rstart {
                bitmap.push(true);
                continue;
            }

            let mut lhs_values = self.values().clone();
            lhs_values.slice(lstart, lend - lstart);
            let mut rhs_values = self.values().clone();
            rhs_values.slice(rstart, rend - rstart);

            let result = array_tot_ne_missing_kernel(lhs_values.as_ref(), rhs_values.as_ref());
            bitmap.push(result.set_bits() > 0);
        }

        bitmap.freeze()
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }
}
