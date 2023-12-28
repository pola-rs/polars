use std::sync::Arc;

use super::utils::{build_extend_null_bits, extend_offset_values, ExtendNullBits};
use super::Growable;
use crate::array::binview::{BinaryViewArrayGeneric, MutableBinaryViewArray, ViewType};
use crate::array::{Array, BinaryArray};
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets};

/// Concrete [`Growable`] for the [`BinaryArray`].
pub struct GrowableBinaryViewArray<'a, T: ViewType + ?Sized> {
    arrays: Vec<&'a BinaryViewArrayGeneric<T>>,
    data_type: ArrowDataType,
    validity: MutableBitmap::with_capacity(capacity),
    views: Vec<u128>,
    buffers: Vec<Buffer<u8>>,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
}

impl<'a, T: ViewType + ?Sized> GrowableBinaryViewArray<'a, T> {
    /// Creates a new [`GrowableBinaryViewArray`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: Vec<&'a BinaryViewArrayGeneric<T>>, mut use_validity: bool, capacity: usize) -> Self {
        let data_type = arrays[0].data_type().clone();

        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let extend_null_bits = arrays
            .iter()
            .map(|array| build_extend_null_bits(*array, use_validity))
            .collect();

        let n_buffers = arrays.iter().map(|binview| binview.buffers().len()).sum::<usize>();

        Self {
            arrays,
            data_type,
            validity: MutableBitmap::with_capacity(capacity),
            views: Vec::with_capacity(capacity),
            buffers: Vec::with_capacity(n_buffers),
            extend_null_bits,
        }
    }

    fn to(&mut self) -> BinaryViewArrayGeneric<T> {
        // let mutable = std::mem::take(&mut self.mutable);
        // let out = mutable.into();
        // debug_assert!(out.data_type() == &self.data_type);
        // out
        todo!()
    }
}

impl<'a, T: ViewType + ?Sized> Growable<'a> for GrowableBinaryViewArray<'a, T> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);

        let array = self.arrays[index];

        let buffer_offset: u32 = self.buffers.len().try_into().expect("unsupported");
        let buffer_offset = (buffer_offset as u128) << 64;

        let range = start..start + len;
        self.buffers.extend_from_slice(&array.buffers()[range]);
        self.views.extend(array.views()[range.clone()].iter().map(|&view| {
            // If null the buffer index is ignored because the length is 0,
            // so we can just do this
            view + buffer_offset
        }));
    }

    fn extend_validity(&mut self, additional: usize) {
        self.views.extend(std::iter::repeat(0).take(additional));
        self.validity.extend_constant(additional, false);
    }

    #[inline]
    fn len(&self) -> usize {
        self.views.len()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        self.to().arced()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        self.to().boxed()
    }
}

impl<'a, T: ViewType + ?Sized> From<GrowableBinaryViewArray<'a, T>> for BinaryViewArrayGeneric<T> {
    fn from(val: GrowableBinaryViewArray<'a, T>) -> Self {
        BinaryViewArrayGeneric::<T>::new_unchecked(
            val.data_type,
            val.views.into(),
            val.buffers
            val.validity.into(),
        )
    }
}
