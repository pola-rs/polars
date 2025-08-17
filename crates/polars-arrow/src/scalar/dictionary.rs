use std::any::Any;

use super::Scalar;
use crate::array::*;
use crate::datatypes::ArrowDataType;

/// The [`DictionaryArray`] equivalent of [`Array`] for [`Scalar`].
#[derive(Debug, Clone)]
pub struct DictionaryScalar<K: DictionaryKey> {
    value: Option<Box<dyn Scalar>>,
    phantom: std::marker::PhantomData<K>,
    dtype: ArrowDataType,
}

impl<K: DictionaryKey> PartialEq for DictionaryScalar<K> {
    fn eq(&self, other: &Self) -> bool {
        (self.dtype == other.dtype) && (self.value.as_ref() == other.value.as_ref())
    }
}

impl<K: DictionaryKey> DictionaryScalar<K> {
    /// returns a new [`DictionaryScalar`]
    /// # Panics
    /// iff
    /// * the `dtype` is not `List` or `LargeList` (depending on this scalar's offset `O`)
    /// * the child of the `dtype` is not equal to the `values`
    #[inline]
    pub fn new(dtype: ArrowDataType, value: Option<Box<dyn Scalar>>) -> Self {
        Self {
            value,
            phantom: std::marker::PhantomData,
            dtype,
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

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }
}
