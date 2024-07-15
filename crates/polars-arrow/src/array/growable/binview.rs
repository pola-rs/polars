use std::ops::Deref;
use std::sync::Arc;

use super::Growable;
use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::array::growable::utils::{extend_validity, extend_validity_copies, prepare_validity};
use crate::array::{Array, MutableBinaryViewArray, View};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;

/// Concrete [`Growable`] for the [`BinaryArray`].
pub struct GrowableBinaryViewArray<'a, T: ViewType + ?Sized> {
    arrays: Vec<&'a BinaryViewArrayGeneric<T>>,
    data_type: ArrowDataType,
    validity: Option<MutableBitmap>,
    inner: MutableBinaryViewArray<T>,
}

impl<'a, T: ViewType + ?Sized> GrowableBinaryViewArray<'a, T> {
    /// Creates a new [`GrowableBinaryViewArray`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(
        arrays: Vec<&'a BinaryViewArrayGeneric<T>>,
        mut use_validity: bool,
        capacity: usize,
    ) -> Self {
        let data_type = arrays[0].data_type().clone();

        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        Self {
            arrays,
            data_type,
            validity: prepare_validity(use_validity, capacity),
            inner: MutableBinaryViewArray::<T>::with_capacity(capacity),
        }
    }

    fn to(&mut self) -> BinaryViewArrayGeneric<T> {
        let arr = std::mem::take(&mut self.inner);
        arr.freeze_with_dtype(self.data_type.clone())
            .with_validity(self.validity.take().map(Bitmap::from))
    }
}

impl<'a, T: ViewType + ?Sized> Growable<'a> for GrowableBinaryViewArray<'a, T> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked(index);
        let local_buffers = array.data_buffers();

        extend_validity(&mut self.validity, array, start, len);

        let range = start..start + len;

        self.inner.extend_non_null_views_trusted_len_unchecked(
            array.views().get_unchecked(range).iter().cloned(),
            local_buffers.deref(),
        );
    }

    unsafe fn extend_copies(&mut self, index: usize, start: usize, len: usize, copies: usize) {
        let orig_view_start = self.inner.views.len();
        if copies > 0 {
            self.extend(index, start, len);
        }
        if copies > 1 {
            let array = *self.arrays.get_unchecked(index);
            extend_validity_copies(&mut self.validity, array, start, len, copies - 1);
            let extended_view_end = self.inner.views.len();
            for _ in 0..copies - 1 {
                self.inner
                    .views
                    .extend_from_within(orig_view_start..extended_view_end)
            }
        }
    }

    fn extend_validity(&mut self, additional: usize) {
        self.inner
            .views
            .extend(std::iter::repeat(View::default()).take(additional));
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        self.to().arced()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        self.to().boxed()
    }
}

impl<'a, T: ViewType + ?Sized> From<GrowableBinaryViewArray<'a, T>> for BinaryViewArrayGeneric<T> {
    fn from(mut val: GrowableBinaryViewArray<'a, T>) -> Self {
        val.to()
    }
}
