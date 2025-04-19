use arrow::array::{Array, DictionaryArray};
use arrow::datatypes::ArrowDataType;

use super::{PrimitiveRangedUniqueState, RangedUniqueKernel};

/// A specialized unique kernel for [`DictionaryArray`] for when all values are in a small known
/// range.
pub struct DictionaryRangedUniqueState {
    key_state: PrimitiveRangedUniqueState<u32>,
    values: Box<dyn Array>,
}

impl DictionaryRangedUniqueState {
    pub fn new(values: Box<dyn Array>) -> Self {
        Self {
            key_state: PrimitiveRangedUniqueState::new(0, values.len() as u32 + 1),
            values,
        }
    }

    pub fn key_state(&mut self) -> &mut PrimitiveRangedUniqueState<u32> {
        &mut self.key_state
    }
}

impl RangedUniqueKernel for DictionaryRangedUniqueState {
    type Array = DictionaryArray<u32>;

    fn has_seen_all(&self) -> bool {
        self.key_state.has_seen_all()
    }

    fn append(&mut self, array: &Self::Array) {
        self.key_state.append(array.keys());
    }

    fn append_state(&mut self, other: &Self) {
        debug_assert_eq!(self.values, other.values);
        self.key_state.append_state(&other.key_state);
    }

    fn finalize_unique(self) -> Self::Array {
        let keys = self.key_state.finalize_unique();
        DictionaryArray::<u32>::try_new(
            ArrowDataType::Dictionary(
                arrow::datatypes::IntegerType::UInt32,
                Box::new(self.values.dtype().clone()),
                false,
            ),
            keys,
            self.values,
        )
        .unwrap()
    }

    fn finalize_n_unique(&self) -> usize {
        self.key_state.finalize_n_unique()
    }

    fn finalize_n_unique_non_null(&self) -> usize {
        self.key_state.finalize_n_unique_non_null()
    }
}
