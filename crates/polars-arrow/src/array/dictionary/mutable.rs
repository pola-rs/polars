use std::hash::Hash;
use std::sync::Arc;

use polars_error::PolarsResult;

use super::value_map::ValueMap;
use super::{DictionaryArray, DictionaryKey};
use crate::array::indexable::{AsIndexed, Indexable};
use crate::array::primitive::MutablePrimitiveArray;
use crate::array::{Array, MutableArray, TryExtend, TryPush};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;

#[derive(Debug)]
pub struct MutableDictionaryArray<K: DictionaryKey, M: MutableArray> {
    data_type: ArrowDataType,
    map: ValueMap<K, M>,
    // invariant: `max(keys) < map.values().len()`
    keys: MutablePrimitiveArray<K>,
}

impl<K: DictionaryKey, M: MutableArray> From<MutableDictionaryArray<K, M>> for DictionaryArray<K> {
    fn from(other: MutableDictionaryArray<K, M>) -> Self {
        // SAFETY: the invariant of this struct ensures that this is up-held
        unsafe {
            DictionaryArray::<K>::try_new_unchecked(
                other.data_type,
                other.keys.into(),
                other.map.into_values().as_box(),
            )
            .unwrap()
        }
    }
}

impl<K: DictionaryKey, M: MutableArray + Default> MutableDictionaryArray<K, M> {
    /// Creates an empty [`MutableDictionaryArray`].
    pub fn new() -> Self {
        Self::try_empty(M::default()).unwrap()
    }
}

impl<K: DictionaryKey, M: MutableArray + Default> Default for MutableDictionaryArray<K, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: DictionaryKey, M: MutableArray> MutableDictionaryArray<K, M> {
    /// Creates an empty [`MutableDictionaryArray`] from a given empty values array.
    /// # Errors
    /// Errors if the array is non-empty.
    pub fn try_empty(values: M) -> PolarsResult<Self> {
        Ok(Self::from_value_map(ValueMap::<K, M>::try_empty(values)?))
    }

    /// Creates an empty [`MutableDictionaryArray`] preloaded with a given dictionary of values.
    /// Indices associated with those values are automatically assigned based on the order of
    /// the values.
    /// # Errors
    /// Errors if there's more values than the maximum value of `K` or if values are not unique.
    pub fn from_values(values: M) -> PolarsResult<Self>
    where
        M: Indexable,
        M::Type: Eq + Hash,
    {
        Ok(Self::from_value_map(ValueMap::<K, M>::from_values(values)?))
    }

    fn from_value_map(value_map: ValueMap<K, M>) -> Self {
        let keys = MutablePrimitiveArray::<K>::new();
        let data_type =
            ArrowDataType::Dictionary(K::KEY_TYPE, Box::new(value_map.data_type().clone()), false);
        Self {
            data_type,
            map: value_map,
            keys,
        }
    }

    /// Creates an empty [`MutableDictionaryArray`] retaining the same dictionary as the current
    /// mutable dictionary array, but with no data. This may come useful when serializing the
    /// array into multiple chunks, where there's a requirement that the dictionary is the same.
    /// No copying is performed, the value map is moved over to the new array.
    pub fn into_empty(self) -> Self {
        Self::from_value_map(self.map)
    }

    /// Same as `into_empty` but clones the inner value map instead of taking full ownership.
    pub fn to_empty(&self) -> Self
    where
        M: Clone,
    {
        Self::from_value_map(self.map.clone())
    }

    /// pushes a null value
    pub fn push_null(&mut self) {
        self.keys.push(None)
    }

    /// returns a reference to the inner values.
    pub fn values(&self) -> &M {
        self.map.values()
    }

    /// converts itself into [`Arc<dyn Array>`]
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: DictionaryArray<K> = self.into();
        Arc::new(a)
    }

    /// converts itself into [`Box<dyn Array>`]
    pub fn into_box(self) -> Box<dyn Array> {
        let a: DictionaryArray<K> = self.into();
        Box::new(a)
    }

    /// Reserves `additional` slots.
    pub fn reserve(&mut self, additional: usize) {
        self.keys.reserve(additional);
    }

    /// Shrinks the capacity of the [`MutableDictionaryArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
        self.keys.shrink_to_fit();
    }

    /// Returns the dictionary keys
    pub fn keys(&self) -> &MutablePrimitiveArray<K> {
        &self.keys
    }

    fn take_into(&mut self) -> DictionaryArray<K> {
        DictionaryArray::<K>::try_new(
            self.data_type.clone(),
            std::mem::take(&mut self.keys).into(),
            self.map.take_into(),
        )
        .unwrap()
    }
}

impl<K: DictionaryKey, M: 'static + MutableArray> MutableArray for MutableDictionaryArray<K, M> {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.keys.validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.take_into())
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(self.take_into())
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn push_null(&mut self) {
        self.keys.push(None)
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl<K, M, T> TryExtend<Option<T>> for MutableDictionaryArray<K, M>
where
    K: DictionaryKey,
    M: MutableArray + Indexable + TryExtend<Option<T>> + TryPush<Option<T>>,
    T: AsIndexed<M>,
    M::Type: Eq + Hash,
{
    fn try_extend<II: IntoIterator<Item = Option<T>>>(&mut self, iter: II) -> PolarsResult<()> {
        for value in iter {
            if let Some(value) = value {
                let key = self
                    .map
                    .try_push_valid(value, |arr, v| arr.try_push(Some(v)))?;
                self.keys.try_push(Some(key))?;
            } else {
                self.push_null();
            }
        }
        Ok(())
    }
}

impl<K, M, T> TryPush<Option<T>> for MutableDictionaryArray<K, M>
where
    K: DictionaryKey,
    M: MutableArray + Indexable + TryPush<Option<T>>,
    T: AsIndexed<M>,
    M::Type: Eq + Hash,
{
    fn try_push(&mut self, item: Option<T>) -> PolarsResult<()> {
        if let Some(value) = item {
            let key = self
                .map
                .try_push_valid(value, |arr, v| arr.try_push(Some(v)))?;
            self.keys.try_push(Some(key))?;
        } else {
            self.push_null();
        }
        Ok(())
    }
}
