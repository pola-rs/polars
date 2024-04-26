use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_utils::aliases::{InitHashMaps, PlHashSet, PlIndexSet};
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::Growable;
use crate::array::binview::{BinaryViewArrayGeneric, View, ViewType};
use crate::array::growable::utils::{extend_validity, extend_validity_copies, prepare_validity};
use crate::array::Array;
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::legacy::utils::CustomIterTools;

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
    // Borrow as this can happen `N` times.
    same_buffers: Option<&'a Arc<[Buffer<u8>]>>,
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
        if all_same_buffer {
            let buffers = arrays[0].data_buffers();
            Self {
                arrays,
                data_type,
                validity: prepare_validity(use_validity, capacity),
                views: Vec::with_capacity(capacity),
                buffers: Default::default(),
                total_bytes_len: 0,
                same_buffers: Some(buffers),
            }
        } else {
            // We deduplicate the individual buffers in `buffers`.
            // and the `data_buffers` in processed. As a `data_buffer` can hold M buffers, we  prevent
            // having N * M complexity. #15615
            let mut processed_buffer_groups = PlHashSet::new();
            let mut buffers = PlIndexSet::new();
            for array in arrays.iter() {
                let data_buffers = array.data_buffers();
                if processed_buffer_groups.insert(data_buffers.as_ptr() as usize) {
                    buffers.extend(data_buffers.iter().map(|buf| BufferKey { inner: buf }))
                }
            }

            Self {
                arrays,
                data_type,
                validity: prepare_validity(use_validity, capacity),
                views: Vec::with_capacity(capacity),
                buffers,
                total_bytes_len: 0,
                same_buffers: None,
            }
        }
    }

    fn to(&mut self) -> BinaryViewArrayGeneric<T> {
        let views = std::mem::take(&mut self.views);
        let buffers = std::mem::take(&mut self.buffers);
        let mut total_buffer_len = 0;

        let buffers = if let Some(buffers) = self.same_buffers {
            buffers.clone()
        } else {
            Arc::from(
                buffers
                    .into_iter()
                    .map(|buf| {
                        let buf = buf.inner;
                        total_buffer_len += buf.len();
                        buf.clone()
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let validity = self.validity.take();

        unsafe {
            BinaryViewArrayGeneric::<T>::new_unchecked(
                self.data_type.clone(),
                views.into(),
                buffers,
                validity.map(|v| v.into()),
                self.total_bytes_len,
                total_buffer_len,
            )
            .maybe_gc()
        }
    }

    /// # Safety
    /// doesn't check bounds
    pub unsafe fn extend_unchecked(&mut self, index: usize, start: usize, len: usize) {
        if self.same_buffers.is_none() {
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
        } else {
            self.extend_unchecked_no_buffers(index, start, len)
        }
    }

    #[inline]
    /// Ignores the buffers and doesn't update the view. This is only correct in a filter.
    ///
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

    unsafe fn extend_copies(&mut self, index: usize, start: usize, len: usize, copies: usize) {
        let orig_view_start = self.views.len();
        if copies > 0 {
            unsafe { self.extend_unchecked(index, start, len) }
        }
        if copies > 1 {
            let array = *self.arrays.get_unchecked(index);
            extend_validity_copies(&mut self.validity, array, start, len, copies - 1);
            let extended_view_end = self.views.len();
            for _ in 0..copies - 1 {
                self.views
                    .extend_from_within(orig_view_start..extended_view_end)
            }
        }
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
    fn from(mut val: GrowableBinaryViewArray<'a, T>) -> Self {
        val.to()
    }
}
