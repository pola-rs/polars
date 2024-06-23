use arrow::array::{Array, DictionaryArray, DictionaryKey};
use arrow::bitmap::{Bitmap, MutableBitmap};

use super::TotalEqKernel;
use crate::comparisons::dyn_array::{array_tot_eq_missing_kernel, array_tot_ne_missing_kernel};

impl<K: DictionaryKey> TotalEqKernel for DictionaryArray<K> {
    type Scalar = Box<dyn Array>;

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

            let lkey = self.key_value(i);
            let rkey = other.key_value(i);

            let mut lhs_value = self.values().clone();
            lhs_value.slice(lkey, 1);
            let mut rhs_value = other.values().clone();
            rhs_value.slice(rkey, 1);

            let result = array_tot_eq_missing_kernel(lhs_value.as_ref(), rhs_value.as_ref());
            bitmap.push(result.unset_bits() == 0);
        }

        bitmap.freeze()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert_eq!(self.len(), other.len());

        let mut bitmap = MutableBitmap::with_capacity(self.len());

        for i in 0..self.len() {
            let lval = self.validity().map_or(true, |v| v.get(i).unwrap());
            let rval = other.validity().map_or(true, |v| v.get(i).unwrap());

            if !lval || !rval {
                bitmap.push(false);
                continue;
            }

            let lkey = self.key_value(i);
            let rkey = other.key_value(i);

            let mut lhs_value = self.values().clone();
            lhs_value.slice(lkey, 1);
            let mut rhs_value = other.values().clone();
            rhs_value.slice(rkey, 1);

            let result = array_tot_ne_missing_kernel(lhs_value.as_ref(), rhs_value.as_ref());
            bitmap.push(result.set_bits() > 0);
        }

        bitmap.freeze()
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> arrow::bitmap::Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> arrow::bitmap::Bitmap {
        todo!()
    }
}
