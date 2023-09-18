use std::sync::Arc;

use crate::{
    array::{Array, PrimitiveArray},
    bitmap::MutableBitmap,
    datatypes::DataType,
    types::NativeType,
};

use super::{
    utils::{build_extend_null_bits, ExtendNullBits},
    Growable,
};

/// Concrete [`Growable`] for the [`PrimitiveArray`].
pub struct GrowablePrimitive<'a, T: NativeType> {
    data_type: DataType,
    arrays: Vec<&'a [T]>,
    validity: MutableBitmap,
    values: Vec<T>,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
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

        let extend_null_bits = arrays
            .iter()
            .map(|array| build_extend_null_bits(*array, use_validity))
            .collect();

        let arrays = arrays
            .iter()
            .map(|array| array.values().as_slice())
            .collect::<Vec<_>>();

        Self {
            data_type,
            arrays,
            values: Vec::with_capacity(capacity),
            validity: MutableBitmap::with_capacity(capacity),
            extend_null_bits,
        }
    }

    #[inline]
    fn to(&mut self) -> PrimitiveArray<T> {
        let validity = std::mem::take(&mut self.validity);
        let values = std::mem::take(&mut self.values);

        PrimitiveArray::<T>::new(self.data_type.clone(), values.into(), validity.into())
    }
}

impl<'a, T: NativeType> Growable<'a> for GrowablePrimitive<'a, T> {
    #[inline]
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);

        let values = self.arrays[index];
        self.values.extend_from_slice(&values[start..start + len]);
    }

    #[inline]
    fn extend_validity(&mut self, additional: usize) {
        self.values
            .resize(self.values.len() + additional, T::default());
        self.validity.extend_constant(additional, false);
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
        PrimitiveArray::<T>::new(val.data_type, val.values.into(), val.validity.into())
    }
}
