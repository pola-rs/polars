use std::ops::Deref;
use std::sync::Arc;

use polars_utils::aliases::{InitHashMaps, PlHashSet};
use polars_utils::itertools::Itertools;

use super::Growable;
use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::array::growable::utils::{extend_validity, extend_validity_copies, prepare_validity};
use crate::array::{Array, MutableBinaryViewArray, View};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

/// Concrete [`Growable`] for the [`BinaryArray`].
pub struct GrowableBinaryViewArray<'a, T: ViewType + ?Sized> {
    arrays: Vec<&'a BinaryViewArrayGeneric<T>>,
    data_type: ArrowDataType,
    validity: Option<MutableBitmap>,
    inner: MutableBinaryViewArray<T>,
    same_buffers: Option<&'a Arc<[Buffer<u8>]>>,
    total_same_buffers_len: usize, // Only valid if same_buffers is Some.
    has_duplicate_buffers: bool,
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

        // Fast case.
        // This happens in group-by's
        // And prevents us to push `M` buffers insert in the buffers
        // #15615
        let all_same_buffer = arrays
            .iter()
            .map(|array| array.data_buffers().as_ptr())
            .all_equal()
            && !arrays.is_empty();
        let same_buffers = all_same_buffer.then(|| arrays[0].data_buffers());
        let total_same_buffers_len = all_same_buffer
            .then(|| arrays[0].total_buffer_len())
            .unwrap_or_default();

        let mut duplicates = PlHashSet::new();
        let mut has_duplicate_buffers = false;
        for arr in arrays.iter() {
            if !duplicates.insert(arr.data_buffers().as_ptr()) {
                has_duplicate_buffers = true;
                break;
            }
        }
        Self {
            arrays,
            data_type,
            validity: prepare_validity(use_validity, capacity),
            inner: MutableBinaryViewArray::<T>::with_capacity(capacity),
            same_buffers,
            total_same_buffers_len,
            has_duplicate_buffers,
        }
    }

    fn to(&mut self) -> BinaryViewArrayGeneric<T> {
        let arr = std::mem::take(&mut self.inner);
        if let Some(buffers) = self.same_buffers {
            unsafe {
                BinaryViewArrayGeneric::<T>::new_unchecked(
                    self.data_type.clone(),
                    arr.views.into(),
                    buffers.clone(),
                    self.validity.take().map(Bitmap::from),
                    arr.total_bytes_len,
                    self.total_same_buffers_len,
                )
            }
        } else {
            arr.freeze_with_dtype(self.data_type.clone())
                .with_validity(self.validity.take().map(Bitmap::from))
        }
    }
}

impl<'a, T: ViewType + ?Sized> Growable<'a> for GrowableBinaryViewArray<'a, T> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked(index);
        let local_buffers = array.data_buffers();

        extend_validity(&mut self.validity, array, start, len);

        let range = start..start + len;

        let views_iter = array.views().get_unchecked(range).iter().cloned();

        if self.same_buffers.is_some() {
            let mut total_len = 0;
            self.inner
                .views
                .extend(views_iter.inspect(|v| total_len += v.length as usize));
            self.inner.total_bytes_len += total_len;
        } else if self.has_duplicate_buffers {
            self.inner
                .extend_non_null_views_unchecked_dedupe(views_iter, local_buffers.deref());
        } else {
            self.inner
                .extend_non_null_views_unchecked(views_iter, local_buffers.deref());
        }
    }

    unsafe fn extend_copies(&mut self, index: usize, start: usize, len: usize, copies: usize) {
        let orig_view_start = self.inner.views.len();
        let orig_total_bytes_len = self.inner.total_bytes_len;
        if copies > 0 {
            self.extend(index, start, len);
        }
        if copies > 1 {
            let array = *self.arrays.get_unchecked(index);
            extend_validity_copies(&mut self.validity, array, start, len, copies - 1);
            let extended_view_end = self.inner.views.len();
            let total_bytes_len_end = self.inner.total_bytes_len;
            for _ in 0..copies - 1 {
                self.inner
                    .views
                    .extend_from_within(orig_view_start..extended_view_end);
                self.inner.total_bytes_len += total_bytes_len_end - orig_total_bytes_len;
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
