use crate::array::{Array, PrimitiveArray, Utf8Array};
use crate::error::{Error, Result};
use crate::trusted_len::TrustedLen;
use crate::types::Offset;

use super::DictionaryKey;

pub trait DictValue {
    type IterValue<'this>
    where
        Self: 'this;

    /// # Safety
    /// Will not do any bound checks but must check validity.
    unsafe fn get_unchecked(&self, item: usize) -> Self::IterValue<'_>;

    /// Take a [`dyn Array`] an try to downcast it to the type of `DictValue`.
    fn downcast_values(array: &dyn Array) -> Result<&Self>
    where
        Self: Sized;
}

impl<O: Offset> DictValue for Utf8Array<O> {
    type IterValue<'a> = &'a str;

    unsafe fn get_unchecked(&self, item: usize) -> Self::IterValue<'_> {
        self.value_unchecked(item)
    }

    fn downcast_values(array: &dyn Array) -> Result<&Self>
    where
        Self: Sized,
    {
        array
            .as_any()
            .downcast_ref::<Self>()
            .ok_or(Error::InvalidArgumentError(
                "could not convert array to dictionary value".into(),
            ))
            .map(|arr| {
                assert_eq!(
                    arr.null_count(),
                    0,
                    "null values in values not supported in iteration"
                );
                arr
            })
    }
}

/// Iterator of values of an `ListArray`.
pub struct DictionaryValuesIterTyped<'a, K: DictionaryKey, V: DictValue> {
    keys: &'a PrimitiveArray<K>,
    values: &'a V,
    index: usize,
    end: usize,
}

impl<'a, K: DictionaryKey, V: DictValue> DictionaryValuesIterTyped<'a, K, V> {
    pub(super) unsafe fn new(keys: &'a PrimitiveArray<K>, values: &'a V) -> Self {
        Self {
            keys,
            values,
            index: 0,
            end: keys.len(),
        }
    }
}

impl<'a, K: DictionaryKey, V: DictValue> Iterator for DictionaryValuesIterTyped<'a, K, V> {
    type Item = V::IterValue<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        unsafe {
            let key = self.keys.value_unchecked(old);
            let idx = key.as_usize();
            Some(self.values.get_unchecked(idx))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }
}

unsafe impl<'a, K: DictionaryKey, V: DictValue> TrustedLen for DictionaryValuesIterTyped<'a, K, V> {}

impl<'a, K: DictionaryKey, V: DictValue> DoubleEndedIterator
    for DictionaryValuesIterTyped<'a, K, V>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            unsafe {
                let key = self.keys.value_unchecked(self.end);
                let idx = key.as_usize();
                Some(self.values.get_unchecked(idx))
            }
        }
    }
}
