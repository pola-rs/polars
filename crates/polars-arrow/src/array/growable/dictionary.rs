use std::sync::Arc;

use polars_utils::slice::GetSaferUnchecked;

use super::{make_growable, Growable};
use crate::array::growable::utils::{extend_validity, prepare_validity};
use crate::array::{Array, DictionaryArray, DictionaryKey, PrimitiveArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;

/// Concrete [`Growable`] for the [`DictionaryArray`].
/// # Implementation
/// This growable does not perform collision checks and instead concatenates
/// the values of each [`DictionaryArray`] one after the other.
pub struct GrowableDictionary<'a, K: DictionaryKey> {
    data_type: ArrowDataType,
    keys: Vec<&'a PrimitiveArray<K>>,
    key_values: Vec<K>,
    validity: Option<MutableBitmap>,
    offsets: Vec<usize>,
    values: Box<dyn Array>,
}

fn concatenate_values<K: DictionaryKey>(
    arrays_keys: &[&PrimitiveArray<K>],
    arrays_values: &[&dyn Array],
    capacity: usize,
) -> (Box<dyn Array>, Vec<usize>) {
    let mut mutable = make_growable(arrays_values, false, capacity);
    let mut offsets = Vec::with_capacity(arrays_keys.len() + 1);
    offsets.push(0);
    for (i, values) in arrays_values.iter().enumerate() {
        unsafe { mutable.extend(i, 0, values.len()) };
        offsets.push(offsets[i] + values.len());
    }
    (mutable.as_box(), offsets)
}

impl<'a, T: DictionaryKey> GrowableDictionary<'a, T> {
    /// Creates a new [`GrowableDictionary`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: &[&'a DictionaryArray<T>], mut use_validity: bool, capacity: usize) -> Self {
        let data_type = arrays[0].data_type().clone();

        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let arrays_keys = arrays.iter().map(|array| array.keys()).collect::<Vec<_>>();
        let arrays_values = arrays
            .iter()
            .map(|array| array.values().as_ref())
            .collect::<Vec<_>>();

        let (values, offsets) = concatenate_values(&arrays_keys, &arrays_values, capacity);

        Self {
            data_type,
            offsets,
            values,
            keys: arrays_keys,
            key_values: Vec::with_capacity(capacity),
            validity: prepare_validity(use_validity, capacity),
        }
    }

    #[inline]
    fn to(&mut self) -> DictionaryArray<T> {
        let validity = self.validity.take();
        let key_values = std::mem::take(&mut self.key_values);

        #[cfg(debug_assertions)]
        {
            crate::array::specification::check_indexes(&key_values, self.values.len()).unwrap();
        }
        let keys = PrimitiveArray::<T>::new(
            T::PRIMITIVE.into(),
            key_values.into(),
            validity.map(|v| v.into()),
        );

        // SAFETY: the invariant of this struct ensures that this is up-held
        unsafe {
            DictionaryArray::<T>::try_new_unchecked(
                self.data_type.clone(),
                keys,
                self.values.clone(),
            )
            .unwrap()
        }
    }
}

impl<'a, T: DictionaryKey> Growable<'a> for GrowableDictionary<'a, T> {
    #[inline]
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let keys_array = *self.keys.get_unchecked_release(index);
        extend_validity(&mut self.validity, keys_array, start, len);

        let values = &keys_array
            .values()
            .get_unchecked_release(start..start + len);
        let offset = self.offsets.get_unchecked_release(index);
        self.key_values.extend(
            values
                .iter()
                // `.unwrap_or(0)` because this operation does not check for null values, which may contain any key.
                .map(|x| {
                    let x: usize = offset + (*x).try_into().unwrap_or(0);
                    let x: T = match x.try_into() {
                        Ok(key) => key,
                        // todo: convert this to an error.
                        Err(_) => {
                            panic!("The maximum key is too small")
                        },
                    };
                    x
                }),
        );
    }

    #[inline]
    fn len(&self) -> usize {
        self.key_values.len()
    }

    #[inline]
    fn extend_validity(&mut self, additional: usize) {
        self.key_values
            .resize(self.key_values.len() + additional, T::default());
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
    }

    #[inline]
    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(self.to())
    }

    #[inline]
    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.to())
    }
}

impl<'a, T: DictionaryKey> From<GrowableDictionary<'a, T>> for DictionaryArray<T> {
    #[inline]
    fn from(mut val: GrowableDictionary<'a, T>) -> Self {
        val.to()
    }
}
