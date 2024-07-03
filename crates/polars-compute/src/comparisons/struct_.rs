use arrow::array::{Array, StructArray};
use arrow::bitmap::{Bitmap, MutableBitmap};

use super::TotalEqKernel;
use crate::comparisons::dyn_array::array_tot_eq_missing_kernel;

impl TotalEqKernel for StructArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        let lhs = self;
        let rhs = other;
        assert_eq!(lhs.len(), rhs.len());

        if lhs.fields() != rhs.fields() {
            return Bitmap::new_zeroed(lhs.len());
        }

        let ln = lhs.validity();
        let rn = rhs.validity();

        let lv = lhs.values();
        let rv = rhs.values();

        let mut bitmap = MutableBitmap::with_capacity(lhs.len());

        for i in 0..lhs.len() {
            let mut is_equal = true;

            if !ln.map_or(true, |v| v.get(i).unwrap()) || !rn.map_or(true, |v| v.get(i).unwrap()) {
                bitmap.push(true);
                continue;
            }

            for j in 0..lhs.values().len() {
                if lv[j].len() != rv[j].len() {
                    is_equal = false;
                    break;
                }

                let result = array_tot_eq_missing_kernel(lv[j].as_ref(), rv[j].as_ref());
                if result.unset_bits() != 0 {
                    is_equal = false;
                    break;
                }
            }

            bitmap.push(is_equal);
        }

        bitmap.freeze()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        let lhs = self;
        let rhs = other;

        if lhs.fields() != rhs.fields() {
            return Bitmap::new_with_value(true, lhs.len());
        }

        if lhs.values().len() != rhs.values().len() {
            return Bitmap::new_with_value(true, lhs.len());
        }

        let ln = lhs.validity();
        let rn = rhs.validity();

        let lv = lhs.values();
        let rv = rhs.values();

        let mut bitmap = MutableBitmap::with_capacity(lhs.len());

        for i in 0..lhs.len() {
            let mut is_equal = true;

            if !ln.map_or(true, |v| v.get(i).unwrap()) || !rn.map_or(true, |v| v.get(i).unwrap()) {
                bitmap.push(false);
                continue;
            }

            for j in 0..lhs.values().len() {
                if lv[j].len() != rv[j].len() {
                    is_equal = false;
                    break;
                }

                let result = array_tot_eq_missing_kernel(lv[j].as_ref(), rv[j].as_ref());
                if result.unset_bits() != 0 {
                    is_equal = false;
                    break;
                }
            }

            bitmap.push(!is_equal);
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
