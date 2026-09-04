//! The unique kernel over booleans, of which there are only ever three: `false`, `true` and null.
//!
//! Which of the three a chunk holds is answered by counting its set bits, so a chunk that repeats
//! one bit — and one whose validity mask does — is read in `O(1)`: the run stands for its every
//! element, and there is nothing to count.

use arrow::array::{Array, BooleanArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use polars_array::{PlBitmapRef, PlBooleanArray};

use super::{GenericUniqueKernel, RangedUniqueKernel};

#[derive(Default, Clone)]
pub struct BooleanUniqueKernelState {
    seen: u32,
}

impl BooleanUniqueKernelState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records what an array of `len` elements holds, `null_count` of which are null and
    /// `num_trues` of which are a non-null `true`.
    fn see(&mut self, len: usize, null_count: usize, num_trues: usize) {
        if len == 0 {
            return;
        }

        self.seen |= u32::from(null_count > 0) << 2;
        self.seen |= u32::from(num_trues != len - null_count);
        self.seen |= u32::from(num_trues != 0) << 1;
    }

    /// The values and validity of an array holding one element per value `seen` says was seen.
    fn seen_to_bitmaps(seen: u32) -> (Bitmap, Option<Bitmap>) {
        let mut values = BitmapBuilder::with_capacity(seen.count_ones() as usize);

        if seen & 0b001 != 0 {
            values.push(false);
        }
        if seen & 0b010 != 0 {
            values.push(true);
        }
        let validity = if seen & 0b100 != 0 {
            let mut validity = BitmapBuilder::with_capacity(values.len() + 1);
            validity.extend_constant(values.len(), true);
            validity.push(false);
            values.push(false);
            Some(validity.freeze())
        } else {
            None
        };

        (values.freeze(), validity)
    }
}

/// The number of elements at which both `values` and `validity` are set.
fn num_valid_trues(values: PlBitmapRef<'_>, validity: Option<PlBitmapRef<'_>>) -> usize {
    let Some(validity) = validity else {
        // A mask that repeats one bit counts its elements without reading them one at a time.
        return values.set_bits();
    };

    match (values.scalar_value(), validity.scalar_value()) {
        // One bit on either side says the same of every element, so the count is all of them or
        // none — and where only one side repeats a set bit, the other side's count is the answer.
        (Some(value), Some(valid)) => {
            if value && valid {
                values.len()
            } else {
                0
            }
        },
        (Some(false), None) | (None, Some(false)) => 0,
        (Some(true), None) => validity.set_bits(),
        (None, Some(true)) => values.set_bits(),
        (None, None) => values
            .flat_bitmap()
            .expect("a mask that is not scalar is flat")
            .num_intersections_with(
                validity
                    .flat_bitmap()
                    .expect("a mask that is not scalar is flat"),
            ),
    }
}

impl RangedUniqueKernel for BooleanUniqueKernelState {
    type Array = PlBooleanArray;

    fn has_seen_all(&self) -> bool {
        self.seen == 0b111
    }

    fn append(&mut self, array: &Self::Array) {
        let null_count = array.null_count();
        // A mask with nothing unset says nothing the values do not already say.
        let validity = array.validity().filter(|_| null_count > 0);
        self.see(
            array.len(),
            null_count,
            num_valid_trues(array.values(), validity),
        );
    }

    fn append_state(&mut self, other: &Self) {
        self.seen |= other.seen;
    }

    fn finalize_unique(self) -> Self::Array {
        let (values, validity) = Self::seen_to_bitmaps(self.seen);
        PlBooleanArray::new(values.clone(), values.len(), validity)
    }

    fn finalize_n_unique(&self) -> usize {
        self.seen.count_ones() as usize
    }

    fn finalize_n_unique_non_null(&self) -> usize {
        (self.seen & 0b011).count_ones() as usize
    }
}

/// The state an Arrow array leaves behind, which holds one slot per element throughout.
fn arrow_state(array: &BooleanArray) -> BooleanUniqueKernelState {
    let mut state = BooleanUniqueKernelState::new();
    let num_trues = match array.validity().filter(|_| array.null_count() > 0) {
        None => array.values().set_bits(),
        Some(validity) => array.values().num_intersections_with(validity),
    };
    state.see(array.len(), array.null_count(), num_trues);
    state
}

impl GenericUniqueKernel for BooleanArray {
    fn unique(&self) -> Self {
        let (values, validity) = BooleanUniqueKernelState::seen_to_bitmaps(arrow_state(self).seen);
        BooleanArray::new(ArrowDataType::Boolean, values, validity)
    }

    fn n_unique(&self) -> usize {
        arrow_state(self).finalize_n_unique()
    }

    fn n_unique_non_null(&self) -> usize {
        arrow_state(self).finalize_n_unique_non_null()
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{BooleanArray, MutableBooleanArray, boolean_array};
    use polars_array::arrow::bridge::chunk_from_arrow;
    use proptest::prelude::*;

    use super::*;

    /// A chunk that repeats one bit holds that one value, and a mask that repeats one bit either
    /// leaves every element as it is or makes every one of them null. None of it is counted bit by
    /// bit.
    #[test]
    fn a_repeated_bit_holds_one_value() {
        let unique = |array: &PlBooleanArray| {
            let mut state = BooleanUniqueKernelState::new();
            state.append(array);
            state.finalize_unique()
        };

        for bit in [false, true] {
            let scalar = PlBooleanArray::new_scalar(bit, 100);
            assert_eq!(unique(&scalar), PlBooleanArray::from_vec(vec![bit]));

            // Under a mask that is unset everywhere the chunk holds nothing but nulls.
            let all_null = scalar
                .clone()
                .with_validity_broadcast(Some(Bitmap::new_with_value(false, 1)));
            assert_eq!(unique(&all_null), PlBooleanArray::from_iter([None]));

            // Under one that is set everywhere it holds what it held before.
            let none_null = scalar
                .clone()
                .with_validity_broadcast(Some(Bitmap::new_with_value(true, 1)));
            assert_eq!(unique(&none_null), PlBooleanArray::from_vec(vec![bit]));

            // A mask laid out one bit per element leaves both the value and a null behind.
            let some_null =
                scalar.with_validity(Some((0..100).map(|i| i % 2 == 0).collect::<Bitmap>()));
            assert_eq!(
                unique(&some_null),
                PlBooleanArray::from_iter([Some(bit), None]),
            );
        }
    }

    /// A chunk of no elements holds no value at all, in either representation.
    #[test]
    fn an_empty_chunk_holds_nothing() {
        let mut state = BooleanUniqueKernelState::new();
        state.append(&PlBooleanArray::new_empty());
        assert_eq!(state.finalize_n_unique(), 0);
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

    proptest! {
        #[test]
        fn test_proptest(array in boolean_array(0..100)) {
            let mut state = BooleanUniqueKernelState::new();
            state.append(&chunk_from_arrow::<PlBooleanArray>(&array));

            let mut has_none = false;
            let mut has_false = false;
            let mut has_true = false;
            for v in array.iter() {
                match v {
                    None => has_none |= true,
                    Some(false) => has_false |= true,
                    Some(true) => has_true |= true,
                }
            }

            let mut unique = MutableBooleanArray::new();
            if has_false {
                unique.push(Some(false));
            }
            if has_true {
                unique.push(Some(true));
            }
            if has_none {
                unique.push(None);
            }
            let unique = unique.freeze();

            assert_eq!(state.clone().finalize_unique(), chunk_from_arrow::<PlBooleanArray>(&unique));
            assert_eq!(state.clone().finalize_n_unique(), unique.len());
            assert_eq!(state.clone().finalize_n_unique_non_null(), unique.len() - usize::from(has_none));
        }
    }
}
