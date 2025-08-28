use std::marker::PhantomData;
use std::sync::{Arc, LazyLock};

use hashbrown::hash_map::Entry;
use polars_utils::IdxSize;
use polars_utils::aliases::{InitHashMaps, PlHashMap};

use crate::array::binview::{DEFAULT_BLOCK_SIZE, MAX_EXP_BLOCK_SIZE};
use crate::array::builder::{ShareStrategy, StaticArrayBuilder};
use crate::array::{Array, BinaryViewArrayGeneric, View, ViewType};
use crate::bitmap::OptBitmapBuilder;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::pushable::Pushable;

static PLACEHOLDER_BUFFER: LazyLock<Buffer<u8>> = LazyLock::new(|| Buffer::from_static(&[]));

pub struct BinaryViewArrayGenericBuilder<V: ViewType + ?Sized> {
    dtype: ArrowDataType,
    views: Vec<View>,
    active_buffer: Vec<u8>,
    active_buffer_idx: u32,
    buffer_set: Vec<Buffer<u8>>,
    stolen_buffers: PlHashMap<usize, u32>,

    // With these we can amortize buffer set translation costs if repeatedly
    // stealing from the same set of buffers.
    last_buffer_set_stolen_from: Option<Arc<[Buffer<u8>]>>,
    buffer_set_translation_idxs: Vec<(u32, u32)>, // (idx, generation)
    buffer_set_translation_generation: u32,

    validity: OptBitmapBuilder,
    /// Total bytes length if we would concatenate them all.
    total_bytes_len: usize,
    /// Total bytes in the buffer set (excluding remaining capacity).
    total_buffer_len: usize,
    view_type: PhantomData<V>,
}

impl<V: ViewType + ?Sized> BinaryViewArrayGenericBuilder<V> {
    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            dtype,
            views: Vec::new(),
            active_buffer: Vec::new(),
            active_buffer_idx: 0,
            buffer_set: Vec::new(),
            stolen_buffers: PlHashMap::new(),
            last_buffer_set_stolen_from: None,
            buffer_set_translation_idxs: Vec::new(),
            buffer_set_translation_generation: 0,
            validity: OptBitmapBuilder::default(),
            total_bytes_len: 0,
            total_buffer_len: 0,
            view_type: PhantomData,
        }
    }

    #[inline]
    fn reserve_active_buffer(&mut self, additional: usize) {
        let len = self.active_buffer.len();
        let cap = self.active_buffer.capacity();
        if additional > cap - len || len + additional >= (u32::MAX - 1) as usize {
            self.reserve_active_buffer_slow(additional);
        }
    }

    #[cold]
    fn reserve_active_buffer_slow(&mut self, additional: usize) {
        assert!(
            additional <= (u32::MAX - 1) as usize,
            "strings longer than 2^32 - 2 are not supported"
        );

        // Allocate a new buffer and flush the old buffer.
        let new_capacity = (self.active_buffer.capacity() * 2)
            .clamp(DEFAULT_BLOCK_SIZE, MAX_EXP_BLOCK_SIZE)
            .max(additional);

        let old_buffer =
            core::mem::replace(&mut self.active_buffer, Vec::with_capacity(new_capacity));
        if !old_buffer.is_empty() {
            //  Replace dummy with real buffer.
            self.buffer_set[self.active_buffer_idx as usize] = Buffer::from(old_buffer);
        }
        self.active_buffer_idx = self.buffer_set.len().try_into().unwrap();
        self.buffer_set.push(PLACEHOLDER_BUFFER.clone()) // Push placeholder so active_buffer_idx stays valid.
    }

    pub fn push_value_ignore_validity(&mut self, bytes: &V) {
        let bytes = bytes.to_bytes();
        self.total_bytes_len += bytes.len();
        unsafe {
            let view = if bytes.len() > View::MAX_INLINE_SIZE as usize {
                self.reserve_active_buffer(bytes.len());

                let offset = self.active_buffer.len() as u32; // Ensured no overflow by reserve_active_buffer.
                self.active_buffer.extend_from_slice(bytes);
                self.total_buffer_len += bytes.len();
                View::new_noninline_unchecked(bytes, self.active_buffer_idx, offset)
            } else {
                View::new_inline_unchecked(bytes)
            };
            self.views.push(view);
        }
    }

    /// # Safety
    /// The view must be inline.
    pub unsafe fn push_inline_view_ignore_validity(&mut self, view: View) {
        debug_assert!(view.is_inline());
        self.total_bytes_len += view.length as usize;
        self.views.push(view);
    }

    fn switch_active_stealing_bufferset_to(&mut self, buffer_set: &Arc<[Buffer<u8>]>) {
        // Fat pointer equality, checks both start and length.
        if self
            .last_buffer_set_stolen_from
            .as_ref()
            .is_some_and(|stolen_bs| std::ptr::eq(Arc::as_ptr(stolen_bs), Arc::as_ptr(buffer_set)))
        {
            return; // Already active.
        }

        // Switch to new generation (invalidating all old translation indices),
        // and resizing the buffer with invalid indices if necessary.
        let old_gen = self.buffer_set_translation_generation;
        self.buffer_set_translation_generation = old_gen.wrapping_add(1);
        if self.buffer_set_translation_idxs.len() < buffer_set.len() {
            self.buffer_set_translation_idxs
                .resize(buffer_set.len(), (0, old_gen));
        }
    }

    unsafe fn translate_view(
        &mut self,
        mut view: View,
        other_bufferset: &Arc<[Buffer<u8>]>,
    ) -> View {
        // Translate from old array-local buffer idx to global stolen buffer idx.
        let (mut new_buffer_idx, gen_) = *self
            .buffer_set_translation_idxs
            .get_unchecked(view.buffer_idx as usize);
        if gen_ != self.buffer_set_translation_generation {
            // This buffer index wasn't seen before for this array, do a dedup lookup.
            // Since we map by starting pointer and different subslices may have different lengths, we expand
            // the buffer to the maximum it could be.
            let buffer = other_bufferset
                .get_unchecked(view.buffer_idx as usize)
                .clone()
                .expand_end_to_storage();
            let buf_id = buffer.as_slice().as_ptr().addr();
            let idx = match self.stolen_buffers.entry(buf_id) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let idx = self.buffer_set.len() as u32;
                    self.total_buffer_len += buffer.len();
                    self.buffer_set.push(buffer);
                    v.insert(idx);
                    idx
                },
            };

            // Cache result for future lookups.
            *self
                .buffer_set_translation_idxs
                .get_unchecked_mut(view.buffer_idx as usize) =
                (idx, self.buffer_set_translation_generation);
            new_buffer_idx = idx;
        }
        view.buffer_idx = new_buffer_idx;
        view
    }

    unsafe fn extend_views_dedup_ignore_validity(
        &mut self,
        views: impl IntoIterator<Item = View>,
        other_bufferset: &Arc<[Buffer<u8>]>,
    ) {
        // TODO: if there are way more buffers than length translate per-view
        // rather than all at once.
        self.switch_active_stealing_bufferset_to(other_bufferset);

        for mut view in views {
            if view.length > View::MAX_INLINE_SIZE {
                view = self.translate_view(view, other_bufferset);
            }
            self.total_bytes_len += view.length as usize;
            self.views.push(view);
        }
    }

    unsafe fn extend_views_each_repeated_dedup_ignore_validity(
        &mut self,
        views: impl IntoIterator<Item = View>,
        repeats: usize,
        other_bufferset: &Arc<[Buffer<u8>]>,
    ) {
        // TODO: if there are way more buffers than length translate per-view
        // rather than all at once.
        self.switch_active_stealing_bufferset_to(other_bufferset);

        for mut view in views {
            if view.length > View::MAX_INLINE_SIZE {
                view = self.translate_view(view, other_bufferset);
            }
            self.total_bytes_len += repeats * view.length as usize;
            for _ in 0..repeats {
                self.views.push(view);
            }
        }
    }
}

impl<V: ViewType + ?Sized> StaticArrayBuilder for BinaryViewArrayGenericBuilder<V> {
    type Array = BinaryViewArrayGeneric<V>;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(mut self) -> Self::Array {
        // Flush active buffer and/or remove extra placeholder buffer.
        if !self.active_buffer.is_empty() {
            self.buffer_set[self.active_buffer_idx as usize] = Buffer::from(self.active_buffer);
        } else if self.buffer_set.last().is_some_and(|b| b.is_empty()) {
            self.buffer_set.pop();
        }

        unsafe {
            BinaryViewArrayGeneric::new_unchecked(
                self.dtype,
                Buffer::from(self.views),
                Arc::from(self.buffer_set),
                self.validity.into_opt_validity(),
                self.total_bytes_len,
                self.total_buffer_len,
            )
        }
    }

    fn freeze_reset(&mut self) -> Self::Array {
        // Flush active buffer and/or remove extra placeholder buffer.
        if !self.active_buffer.is_empty() {
            self.buffer_set[self.active_buffer_idx as usize] =
                Buffer::from(core::mem::take(&mut self.active_buffer));
        } else if self.buffer_set.last().is_some_and(|b| b.is_empty()) {
            self.buffer_set.pop();
        }

        let out = unsafe {
            BinaryViewArrayGeneric::new_unchecked(
                self.dtype.clone(),
                Buffer::from(core::mem::take(&mut self.views)),
                Arc::from(core::mem::take(&mut self.buffer_set)),
                core::mem::take(&mut self.validity).into_opt_validity(),
                self.total_bytes_len,
                self.total_buffer_len,
            )
        };

        self.total_buffer_len = 0;
        self.total_bytes_len = 0;
        self.active_buffer_idx = 0;
        self.stolen_buffers.clear();
        self.last_buffer_set_stolen_from = None;
        out
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn extend_nulls(&mut self, length: usize) {
        self.views.extend_constant(length, View::default());
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.views.reserve(length);

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for i in start..start + length {
                            if v.get_bit_unchecked(i) {
                                self.push_value_ignore_validity(other.value_unchecked(i));
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    } else {
                        for i in start..start + length {
                            self.push_value_ignore_validity(other.value_unchecked(i));
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_views = &other.views()[start..start + length];
                    self.extend_views_dedup_ignore_validity(
                        other_views.iter().copied(),
                        other.data_buffers(),
                    );
                },
            }
        }

        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.views.reserve(length * repeats);

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for i in start..start + length {
                            if v.get_bit_unchecked(i) {
                                for _ in 0..repeats {
                                    self.push_value_ignore_validity(other.value_unchecked(i));
                                }
                            } else {
                                for _ in 0..repeats {
                                    self.views.push(View::default())
                                }
                            }
                        }
                    } else {
                        for i in start..start + length {
                            for _ in 0..repeats {
                                self.push_value_ignore_validity(other.value_unchecked(i));
                            }
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_views = &other.views()[start..start + length];
                    self.extend_views_each_repeated_dedup_ignore_validity(
                        other_views.iter().copied(),
                        repeats,
                        other.data_buffers(),
                    );
                },
            }
        }

        self.validity
            .subslice_extend_each_repeated_from_opt_validity(
                other.validity(),
                start,
                length,
                repeats,
            );
    }

    unsafe fn gather_extend(
        &mut self,
        other: &Self::Array,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.views.reserve(idxs.len());

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for idx in idxs {
                            if v.get_bit_unchecked(*idx as usize) {
                                self.push_value_ignore_validity(
                                    other.value_unchecked(*idx as usize),
                                );
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    } else {
                        for idx in idxs {
                            self.push_value_ignore_validity(other.value_unchecked(*idx as usize));
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_view_slice = other.views().as_slice();
                    let other_views = idxs
                        .iter()
                        .map(|idx| *other_view_slice.get_unchecked(*idx as usize));
                    self.extend_views_dedup_ignore_validity(other_views, other.data_buffers());
                },
            }
        }

        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }

    fn opt_gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy) {
        self.views.reserve(idxs.len());

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for idx in idxs {
                            if (*idx as usize) < v.len() && v.get_bit_unchecked(*idx as usize) {
                                self.push_value_ignore_validity(
                                    other.value_unchecked(*idx as usize),
                                );
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    } else {
                        for idx in idxs {
                            if (*idx as usize) < other.len() {
                                self.push_value_ignore_validity(
                                    other.value_unchecked(*idx as usize),
                                );
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_view_slice = other.views().as_slice();
                    let other_views = idxs.iter().map(|idx| {
                        other_view_slice
                            .get(*idx as usize)
                            .copied()
                            .unwrap_or_default()
                    });
                    self.extend_views_dedup_ignore_validity(other_views, other.data_buffers());
                },
            }
        }

        self.validity
            .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
    }
}
