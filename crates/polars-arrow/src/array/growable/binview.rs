use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_utils::aliases::PlIndexSet;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::Growable;
use crate::array::binview::{BinaryViewArrayGeneric, View, ViewType};
use crate::array::growable::utils::{extend_validity, prepare_validity};
use crate::array::Array;
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

struct BufferKey<'a> {
    inner: &'a Buffer<u8>,
}

impl Hash for BufferKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.inner.as_ptr() as u64)
    }
}

impl PartialEq for BufferKey<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.as_ptr() == other.inner.as_ptr()
    }
}

impl Eq for BufferKey<'_> {}

/// Concrete [`Growable`] for the [`BinaryArray`].
pub struct GrowableBinaryViewArray<'a, T: ViewType + ?Sized> {
    arrays: Vec<&'a BinaryViewArrayGeneric<T>>,
    data_type: ArrowDataType,
    validity: Option<MutableBitmap>,
    views: Vec<View>,
    // We need to use a set/hashmap to deduplicate
    // A growable can be called with many chunks from self.
    // See: #14201
    buffers: PlIndexSet<BufferKey<'a>>,
    total_bytes_len: usize,
    total_buffer_len: usize,
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

        let buffers = arrays
            .iter()
            .flat_map(|array| {
                array
                    .data_buffers()
                    .as_ref()
                    .iter()
                    .map(|buf| BufferKey { inner: buf })
            })
            .collect::<PlIndexSet<_>>();
        let total_buffer_len = arrays
            .iter()
            .map(|arr| arr.data_buffers().len())
            .sum::<usize>();

        Self {
            arrays,
            data_type,
            validity: prepare_validity(use_validity, capacity),
            views: Vec::with_capacity(capacity),
            buffers,
            total_bytes_len: 0,
            total_buffer_len,
        }
    }

    fn to(&mut self) -> BinaryViewArrayGeneric<T> {
        let views = std::mem::take(&mut self.views);
        let buffers = std::mem::take(&mut self.buffers);
        let validity = self.validity.take();
        unsafe {
            BinaryViewArrayGeneric::<T>::new_unchecked(
                self.data_type.clone(),
                views.into(),
                Arc::from(
                    buffers
                        .into_iter()
                        .map(|buf| buf.inner.clone())
                        .collect::<Vec<_>>(),
                ),
                validity.map(|v| v.into()),
                self.total_bytes_len,
                self.total_buffer_len,
            )
            .maybe_gc()
        }
    }

    /// # Safety
    /// doesn't check bounds
    pub unsafe fn extend_unchecked(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked(index);
        let local_buffers = array.data_buffers();

        extend_validity(&mut self.validity, array, start, len);

        let range = start..start + len;

        self.views
            .extend(array.views().get_unchecked(range).iter().map(|view| {
                let mut view = *view;
                let len = view.length as usize;
                self.total_bytes_len += len;

                if len > 12 {
                    let buffer = local_buffers.get_unchecked_release(view.buffer_idx as usize);
                    let key = BufferKey { inner: buffer };
                    let idx = self.buffers.get_full(&key).unwrap_unchecked_release().0;

                    view.buffer_idx = idx as u32;
                }
                view
            }));
    }

    #[inline]
    /// Ignores the buffers and doesn't update the view. This is only correct in a filter.
    /// # Safety
    /// doesn't check bounds
    pub unsafe fn extend_unchecked_no_buffers(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked(index);

        extend_validity(&mut self.validity, array, start, len);

        let range = start..start + len;

        self.views
            .extend(array.views().get_unchecked(range).iter().map(|view| {
                let len = view.length as usize;
                self.total_bytes_len += len;

                *view
            }))
    }
}

impl<'a, T: ViewType + ?Sized> Growable<'a> for GrowableBinaryViewArray<'a, T> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        unsafe { self.extend_unchecked(index, start, len) }
    }

    fn extend_validity(&mut self, additional: usize) {
        self.views
            .extend(std::iter::repeat(View::default()).take(additional));
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
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
        unsafe {
            BinaryViewArrayGeneric::<T>::new_unchecked(
                val.data_type,
                val.views.into(),
                Arc::from(
                    val.buffers
                        .into_iter()
                        .map(|buf| buf.inner.clone())
                        .collect::<Vec<_>>(),
                ),
                val.validity.map(|v| v.into()),
                val.total_bytes_len,
                val.total_buffer_len,
            )
            .maybe_gc()
        }
    }
}
