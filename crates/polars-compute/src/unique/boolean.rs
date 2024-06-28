use arrow::array::{Array, BooleanArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;

use super::{GenericUniqueKernel, RangedUniqueKernel};

pub struct BooleanUniqueKernelState {
    seen: u32,
    has_null: bool,
    data_type: ArrowDataType,
}

const fn to_value(scalar: Option<bool>) -> u8 {
    match scalar {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

impl BooleanUniqueKernelState {
    pub fn new(has_null: bool, data_type: ArrowDataType) -> Self {
        Self {
            seen: 0,
            has_null,
            data_type,
        }
    }

    fn has_seen_null(&self) -> bool {
        self.has_null && self.seen & (1 << to_value(None)) != 0
    }
}

impl RangedUniqueKernel for BooleanUniqueKernelState {
    type Array = BooleanArray;

    fn has_seen_all(&self) -> bool {
        self.seen == 0b111
    }

    fn append(&mut self, array: &Self::Array) {
        if array.len() == 0 {
            return;
        }

        let null_count = array.null_count();
        let values = array.values();

        if !self.has_null || null_count == 0 {
            let set_bits = values.set_bits();
            self.seen |= u32::from(set_bits != 0) << to_value(Some(true));
            self.seen |= u32::from(set_bits != values.len()) << to_value(Some(false));

            return;
        }

        self.seen |= u32::from(null_count > 0) << to_value(None);

        if array.len() != null_count {
            let validity = array.validity().unwrap();

            let set_bits = values.num_intersections_with(validity);
            self.seen |= u32::from(set_bits != 0) << to_value(Some(true));
            self.seen |= u32::from(set_bits != values.len() - null_count) << to_value(Some(false));
        }
    }

    fn finalize_unique(self) -> Self::Array {
        let mut values = MutableBitmap::with_capacity(3);
        let validity = if self.has_seen_null() {
            let mut validity = MutableBitmap::with_capacity(3);

            for i in 0..3 {
                if self.seen & (1 << i) != 0 {
                    values.push(i > 1);
                    validity.push(i > 0);
                }
            }

            Some(validity.freeze())
        } else {
            for i in 1..3 {
                if self.seen & (1 << i) != 0 {
                    values.push(i > 1);
                }
            }

            None
        };

        let values = values.freeze();

        BooleanArray::new(self.data_type, values, validity)
    }

    fn finalize_n_unique(self) -> usize {
        self.seen.count_ones() as usize
    }

    fn finalize_n_unique_non_null(self) -> usize {
        (self.seen & !1).count_ones() as usize
    }
}

impl GenericUniqueKernel for BooleanArray {
    fn unique(&self) -> Self {
        let mut state =
            BooleanUniqueKernelState::new(self.null_count() > 0, self.data_type().clone());
        state.append(self);
        state.finalize_unique()
    }

    fn n_unique(&self) -> usize {
        let mut state =
            BooleanUniqueKernelState::new(self.null_count() > 0, self.data_type().clone());
        state.append(self);
        state.finalize_n_unique()
    }

    fn n_unique_non_null(&self) -> usize {
        let mut state =
            BooleanUniqueKernelState::new(self.null_count() > 0, self.data_type().clone());
        state.append(self);
        state.finalize_n_unique_non_null()
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
