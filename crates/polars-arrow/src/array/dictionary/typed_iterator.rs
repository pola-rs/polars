use polars_error::{polars_err, PolarsResult};

use super::DictionaryKey;
use crate::array::{Array, PrimitiveArray, StaticArray, Utf8Array, Utf8ViewArray};
use crate::trusted_len::TrustedLen;
use crate::types::Offset;

pub trait DictValue {
    type IterValue<'this>
    where
        Self: 'this;

    /// # Safety
    /// Will not do any bound checks but must check validity.
    unsafe fn get_unchecked(&self, item: usize) -> Self::IterValue<'_>;

    /// Take a [`dyn Array`] an try to downcast it to the type of `DictValue`.
    fn downcast_values(array: &dyn Array) -> PolarsResult<&Self>
    where
        Self: Sized;
}

impl<O: Offset> DictValue for Utf8Array<O> {
    type IterValue<'a> = &'a str;

    unsafe fn get_unchecked(&self, item: usize) -> Self::IterValue<'_> {
        self.value_unchecked(item)
    }

    fn downcast_values(array: &dyn Array) -> PolarsResult<&Self>
    where
        Self: Sized,
    {
        array
            .as_any()
            .downcast_ref::<Self>()
            .ok_or_else(
                || polars_err!(InvalidOperation: "could not convert array to dictionary value"),
            )
            .inspect(|arr| {
                assert_eq!(
                    arr.null_count(),
                    0,
                    "null values in values not supported in iteration"
                );
            })
    }
}

impl DictValue for Utf8ViewArray {
    type IterValue<'a> = &'a str;

    unsafe fn get_unchecked(&self, item: usize) -> Self::IterValue<'_> {
        self.value_unchecked(item)
    }

    fn downcast_values(array: &dyn Array) -> PolarsResult<&Self>
    where
        Self: Sized,
    {
        array
            .as_any()
            .downcast_ref::<Self>()
            .ok_or_else(
                || polars_err!(InvalidOperation: "could not convert array to dictionary value"),
            )
            .inspect(|arr| {
                assert_eq!(
                    arr.null_count(),
                    0,
                    "null values in values not supported in iteration"
                );
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
    pub(super) fn new(keys: &'a PrimitiveArray<K>, values: &'a V) -> Self {
        assert_eq!(keys.null_count(), 0);
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

pub struct DictionaryIterTyped<'a, K: DictionaryKey, V: DictValue> {
    keys: &'a PrimitiveArray<K>,
    values: &'a V,
    index: usize,
    end: usize,
}

impl<'a, K: DictionaryKey, V: DictValue> DictionaryIterTyped<'a, K, V> {
    pub(super) fn new(keys: &'a PrimitiveArray<K>, values: &'a V) -> Self {
        Self {
            keys,
            values,
            index: 0,
            end: keys.len(),
        }
    }
}

impl<'a, K: DictionaryKey, V: DictValue> Iterator for DictionaryIterTyped<'a, K, V> {
    type Item = Option<V::IterValue<'a>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        unsafe {
            if let Some(key) = self.keys.get_unchecked(old) {
                let idx = key.as_usize();
                Some(Some(self.values.get_unchecked(idx)))
            } else {
                Some(None)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }
}

unsafe impl<'a, K: DictionaryKey, V: DictValue> TrustedLen for DictionaryIterTyped<'a, K, V> {}

impl<'a, K: DictionaryKey, V: DictValue> DoubleEndedIterator for DictionaryIterTyped<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            unsafe {
                if let Some(key) = self.keys.get_unchecked(self.end) {
                    let idx = key.as_usize();
                    Some(Some(self.values.get_unchecked(idx)))
                } else {
                    Some(None)
                }
            }
        }
    }
}
