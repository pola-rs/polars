use arrow::array::{Array, BooleanArray};
use arrow::bitmap::MutableBitmap;

use super::UniqueKernel;

fn bool_unique_fold<'a>(
    fst: &'a BooleanArray,
    arrs: impl Iterator<Item = &'a BooleanArray>,
) -> BooleanArray {
    // can be None, Some(true), Some(false)
    //
    // We assign values to each value
    // None        = 1
    // Some(false) = 2
    // Some(true)  = 3
    //
    // And keep track of 2 things
    // - `found_set`: which values have already appeared
    // - `order`:     in which order did the values appear

    #[inline(always)]
    fn append_arr(arr: &BooleanArray, found_set: &mut u32, order: &mut u32) {
        for v in arr {
            let value = v.map_or(1, |v| 2 + u32::from(v));
            let nulled_value = if *found_set & (1 << value) != 0 {
                0
            } else {
                value
            };

            *order |= nulled_value << (found_set.count_ones() * 2);
            *found_set |= 1 << value;

            if *found_set == 0b1110 {
                break;
            }
        }
    }

    let mut found_set = 0u32;
    let mut order = 0u32;

    append_arr(fst, &mut found_set, &mut order);
    for arr in arrs {
        append_arr(arr, &mut found_set, &mut order);
    }

    let mut values = MutableBitmap::with_capacity(3);
    let validity = if found_set & 0b10 != 0 {
        let mut validity = MutableBitmap::with_capacity(3);
        while order != 0 {
            values.push(order & 0b11 > 2);
            validity.push(order & 0b11 > 1);
            order >>= 2;
        }
        Some(validity.freeze())
    } else {
        while order != 0 {
            values.push(order & 0b11 > 2);
            order >>= 2;
        }
        None
    };

    let values = values.freeze();

    BooleanArray::new(fst.data_type().clone(), values, validity)
}

impl UniqueKernel for BooleanArray {
    fn unique_fold<'a>(fst: &'a Self, others: impl Iterator<Item = &'a Self>) -> Self {
        bool_unique_fold(fst, others)
    }

    fn unique(&self) -> Self {
        Self::unique_fold(self, [].iter())
    }

    fn unique_sorted(&self) -> Self {
        Self::unique_fold(self, [].iter())
    }

    fn n_unique(&self) -> usize {
        if self.len() == 0 {
            return 0;
        }

        let null_count = self.null_count();

        if self.len() == null_count {
            return 1;
        }

        let values = self.values();

        if null_count == 0 {
            let unset_bits = values.unset_bits();
            let is_uniform = unset_bits == 0 || unset_bits == values.len();
            return 2 - usize::from(is_uniform);
        }

        let validity = self.validity().unwrap();
        let set_bits = values.num_intersections_with(validity);
        let is_uniform = set_bits == 0 || set_bits == validity.set_bits();
        2 + usize::from(!is_uniform)
    }

    #[inline]
    fn n_unique_non_null(&self) -> usize {
        self.n_unique() - usize::from(self.null_count() > 0)
    }
}

#[test]
fn test_boolean_distinct_count() {
    use arrow::bitmap::Bitmap;
    use arrow::datatypes::ArrowDataType;

    macro_rules! assert_bool_dc {
        ($values:expr, $validity:expr => $dc:expr) => {
            let validity: Option<Bitmap> =
                <Option<Vec<bool>>>::map($validity, |v| Bitmap::from_iter(v));
            let arr =
                BooleanArray::new(ArrowDataType::Boolean, Bitmap::from_iter($values), validity);
            assert_eq!(arr.n_unique(), $dc);
        };
    }

    assert_bool_dc!(vec![], None => 0);
    assert_bool_dc!(vec![], Some(vec![]) => 0);
    assert_bool_dc!(vec![true], None => 1);
    assert_bool_dc!(vec![true], Some(vec![true]) => 1);
    assert_bool_dc!(vec![true], Some(vec![false]) => 1);
    assert_bool_dc!(vec![true, false], None => 2);
    assert_bool_dc!(vec![true, false, false], None => 2);
    assert_bool_dc!(vec![true, false, false], Some(vec![true, true, false]) => 3);

    // Copied from https://github.com/pola-rs/polars/pull/16765#discussion_r1629426159
    assert_bool_dc!(vec![true, true, true, true, true], Some(vec![true, false, true, false, false]) => 2);
    assert_bool_dc!(vec![false, true, false, true, true], Some(vec![true, false, true, false, false]) => 2);
    assert_bool_dc!(vec![true, false, true, false, true, true], Some(vec![true, true, false, true, false, false]) => 3);
}
