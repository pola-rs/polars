use arrow::array::{Array, DictionaryArray};
use arrow::datatypes::ArrowDataType;
use arrow::scalar::Scalar;

use super::{PrimitiveRangedUniqueState, RangedUniqueKernel};
use crate::filter::filter_with_bitmap;
use crate::min_max::dyn_array_min_max_propagate_nan;

/// A specialized unique kernel for [`DictionaryArray`] for when all values are in a small known
/// range.
pub struct DictionaryRangedUniqueState {
    key_state: PrimitiveRangedUniqueState<u32>,
    values: Box<dyn Array>,
}

impl DictionaryRangedUniqueState {
    pub fn new(values: Box<dyn Array>) -> Self {
        Self {
            key_state: PrimitiveRangedUniqueState::new(0, values.len().saturating_sub(1) as u32),
            values,
        }
    }

    pub fn key_state(&mut self) -> &mut PrimitiveRangedUniqueState<u32> {
        &mut self.key_state
    }

    pub fn values(&self) -> &dyn Array {
        self.values.as_ref()
    }

    pub fn seen_values(&self) -> Box<dyn Array> {
        if self.has_seen_all_ignore_null() {
            return self.values().to_boxed();
        }

        filter_with_bitmap(self.values(), &self.key_state.to_bitmap())
    }

    pub fn value_min_max(&self) -> Option<(Box<dyn Scalar>, Box<dyn Scalar>)> {
        if self.has_seen_all_ignore_null() {
            return dyn_array_min_max_propagate_nan(self.values());
        }

        let validity = match self.values().validity() {
            None => self.key_state.to_bitmap(),
            Some(v) => v & &self.key_state.to_bitmap(),
        };
        let values = self.values().with_validity(Some(validity));
        dyn_array_min_max_propagate_nan(values.as_ref())
    }

    pub fn key_min_max(&self) -> Option<(u32, u32)> {
        if self.values.is_empty() {
            return None;
        }

        self.key_state.min_max()
    }
}

impl RangedUniqueKernel for DictionaryRangedUniqueState {
    type Array = DictionaryArray<u32>;

    fn has_seen_all(&self) -> bool {
        if self.values().is_empty() {
            self.has_seen_null()
        } else {
            self.key_state.has_seen_all()
        }
    }

    fn has_seen_all_ignore_null(&self) -> bool {
        self.values().is_empty() || self.key_state.has_seen_all_ignore_null()
    }

    fn has_seen_null(&self) -> bool {
        self.key_state.has_seen_null()
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
