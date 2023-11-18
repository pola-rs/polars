use std::any::Any;

use super::Scalar;
use crate::array::*;
use crate::datatypes::ArrowDataType;

/// The [`DictionaryArray`] equivalent of [`Array`] for [`Scalar`].
#[derive(Debug, Clone)]
pub struct DictionaryScalar<K: DictionaryKey> {
    value: Option<Box<dyn Scalar>>,
    phantom: std::marker::PhantomData<K>,
    data_type: ArrowDataType,
}

impl<K: DictionaryKey> PartialEq for DictionaryScalar<K> {
    fn eq(&self, other: &Self) -> bool {
        (self.data_type == other.data_type) && (self.value.as_ref() == other.value.as_ref())
    }
}

impl<K: DictionaryKey> DictionaryScalar<K> {
    /// returns a new [`DictionaryScalar`]
    /// # Panics
    /// iff
    /// * the `data_type` is not `List` or `LargeList` (depending on this scalar's offset `O`)
    /// * the child of the `data_type` is not equal to the `values`
    #[inline]
    pub fn new(data_type: ArrowDataType, value: Option<Box<dyn Scalar>>) -> Self {
        Self {
            value,
            phantom: std::marker::PhantomData,
            data_type,
        }
    }

    /// The values of the [`DictionaryScalar`]
    pub fn value(&self) -> Option<&Box<dyn Scalar>> {
        self.value.as_ref()
    }
}

impl<K: DictionaryKey> Scalar for DictionaryScalar<K> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_valid(&self) -> bool {
        self.value.is_some()
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }
}
