use std::sync::Arc;

use polars_utils::slice::GetSaferUnchecked;

use super::Growable;
use crate::array::growable::utils::{extend_validity, extend_validity_copies, prepare_validity};
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::types::NativeType;

/// Concrete [`Growable`] for the [`PrimitiveArray`].
pub struct GrowablePrimitive<'a, T: NativeType> {
    data_type: ArrowDataType,
    arrays: Vec<&'a PrimitiveArray<T>>,
    validity: Option<MutableBitmap>,
    values: Vec<T>,
}

impl<'a, T: NativeType> GrowablePrimitive<'a, T> {
    /// Creates a new [`GrowablePrimitive`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(
        arrays: Vec<&'a PrimitiveArray<T>>,
        mut use_validity: bool,
        capacity: usize,
    ) -> Self {
        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let data_type = arrays[0].data_type().clone();

        Self {
            data_type,
            arrays,
            values: Vec::with_capacity(capacity),
            validity: prepare_validity(use_validity, capacity),
        }
    }

    #[inline]
    fn to(&mut self) -> PrimitiveArray<T> {
        let validity = std::mem::take(&mut self.validity);
        let values = std::mem::take(&mut self.values);

        PrimitiveArray::<T>::new(
            self.data_type.clone(),
            values.into(),
            validity.map(|v| v.into()),
        )
    }
}

impl<'a, T: NativeType> Growable<'a> for GrowablePrimitive<'a, T> {
    #[inline]
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked_release(index);
        extend_validity(&mut self.validity, array, start, len);

        let values = array.values().as_slice();
        self.values
            .extend_from_slice(values.get_unchecked_release(start..start + len));
    }

    #[inline]
    unsafe fn extend_copies(&mut self, index: usize, start: usize, len: usize, copies: usize) {
        let array = *self.arrays.get_unchecked_release(index);
        extend_validity_copies(&mut self.validity, array, start, len, copies);

        let values = array.values().as_slice();
        self.values.reserve(len * copies);
        for _ in 0..copies {
            self.values
                .extend_from_slice(values.get_unchecked_release(start..start + len));
        }
    }

    #[inline]
    fn extend_validity(&mut self, additional: usize) {
        self.values
            .resize(self.values.len() + additional, T::default());
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
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

impl<'a, T: NativeType> From<GrowablePrimitive<'a, T>> for PrimitiveArray<T> {
    #[inline]
    fn from(val: GrowablePrimitive<'a, T>) -> Self {
        PrimitiveArray::<T>::new(
            val.data_type,
            val.values.into(),
            val.validity.map(|v| v.into()),
        )
    }
}
