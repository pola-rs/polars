use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;

use hashbrown::hash_map::Entry;
use polars_error::PolarsResult;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::slice::GetSaferUnchecked;

use crate::array::binview::iterator::MutableBinaryViewValueIter;
use crate::array::binview::view::validate_utf8_only;
use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::array::{Array, MutableArray, TryExtend, TryPush, View};
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::legacy::trusted_len::TrustedLenPush;
use crate::trusted_len::TrustedLen;

const DEFAULT_BLOCK_SIZE: usize = 8 * 1024;
const MAX_EXP_BLOCK_SIZE: usize = 16 * 1024 * 1024;

// Invariants:
//
// - Each view must point to a valid slice of a buffer
// - `total_buffer_len` must be equal to `completed_buffers.iter().map(Vec::len).sum()`
// - `total_bytes_len` must be equal to `views.iter().map(View::len).sum()`
pub struct MutableBinaryViewArray<T: ViewType + ?Sized> {
    pub(crate) views: Vec<View>,
    pub(crate) completed_buffers: Vec<Buffer<u8>>,
    pub(crate) in_progress_buffer: Vec<u8>,
    pub(crate) validity: Option<MutableBitmap>,
    pub(crate) phantom: std::marker::PhantomData<T>,
    /// Total bytes length if we would concatenate them all.
    pub(crate) total_bytes_len: usize,
    /// Total bytes in the buffer (excluding remaining capacity)
    pub(crate) total_buffer_len: usize,
    /// Mapping from `Buffer::deref()` to index in `completed_buffers`.
    /// Used in `push_view()`.
    pub(crate) stolen_buffers: PlHashMap<usize, u32>,
}

impl<T: ViewType + ?Sized> Clone for MutableBinaryViewArray<T> {
    fn clone(&self) -> Self {
        Self {
            views: self.views.clone(),
            completed_buffers: self.completed_buffers.clone(),
            in_progress_buffer: self.in_progress_buffer.clone(),
            validity: self.validity.clone(),
            phantom: Default::default(),
            total_bytes_len: self.total_bytes_len,
            total_buffer_len: self.total_buffer_len,
            stolen_buffers: PlHashMap::new(),
        }
    }
}

impl<T: ViewType + ?Sized> Debug for MutableBinaryViewArray<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "mutable-binview{:?}", T::DATA_TYPE)
    }
}

impl<T: ViewType + ?Sized> Default for MutableBinaryViewArray<T> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<T: ViewType + ?Sized> From<MutableBinaryViewArray<T>> for BinaryViewArrayGeneric<T> {
    fn from(mut value: MutableBinaryViewArray<T>) -> Self {
        value.finish_in_progress();
        unsafe {
            Self::new_unchecked(
                T::DATA_TYPE,
                value.views.into(),
                Arc::from(value.completed_buffers),
                value.validity.map(|b| b.into()),
                value.total_bytes_len,
                value.total_buffer_len,
            )
        }
    }
}

impl<T: ViewType + ?Sized> MutableBinaryViewArray<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            views: Vec::with_capacity(capacity),
            completed_buffers: vec![],
            in_progress_buffer: vec![],
            validity: None,
            phantom: Default::default(),
            total_buffer_len: 0,
            total_bytes_len: 0,
            stolen_buffers: PlHashMap::new(),
        }
    }

    /// Get a mutable reference to the [`Vec`] of [`View`]s in this [`MutableBinaryViewArray`].
    ///
    /// # Safety
    ///
    /// This is safe as long as any mutation of the [`Vec`] does not break any invariants of the
    /// [`MutableBinaryViewArray`] before it is read again.
    #[inline]
    pub unsafe fn views_mut(&mut self) -> &mut Vec<View> {
        &mut self.views
    }

    /// Set the `total_bytes_len` of the [`MutableBinaryViewArray`]
    ///
    /// # Safety
    ///
    /// This should not break invariants of the [`MutableBinaryViewArray`]
    #[inline]
    pub unsafe fn set_total_bytes_len(&mut self, value: usize) {
        #[cfg(debug_assertions)]
        {
            let actual_length: usize = self.views().iter().map(|v| v.length as usize).sum();
            assert_eq!(value, actual_length);
        }

        self.total_bytes_len = value;
    }

    pub fn total_bytes_len(&self) -> usize {
        self.total_bytes_len
    }

    pub fn total_buffer_len(&self) -> usize {
        self.total_buffer_len
    }

    #[inline]
    pub fn views(&self) -> &[View] {
        &self.views
    }

    #[inline]
    pub fn completed_buffers(&self) -> &[Buffer<u8>] {
        &self.completed_buffers
    }

    pub fn validity(&mut self) -> Option<&mut MutableBitmap> {
        self.validity.as_mut()
    }

    /// Reserves `additional` elements and `additional_buffer` on the buffer.
    pub fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.views.capacity()
    }

    fn init_validity(&mut self, unset_last: bool) {
        let mut validity = MutableBitmap::with_capacity(self.views.capacity());
        validity.extend_constant(self.len(), true);
        if unset_last {
            validity.set(self.len() - 1, false);
        }
        self.validity = Some(validity);
    }

    /// # Safety
    /// - caller must allocate enough capacity
    /// - caller must ensure the view and buffers match.
    /// - The array must not have validity.
    pub(crate) unsafe fn push_view_unchecked(&mut self, v: View, buffers: &[Buffer<u8>]) {
        let len = v.length;
        self.total_bytes_len += len as usize;
        if len <= 12 {
            debug_assert!(self.views.capacity() > self.views.len());
            self.views.push_unchecked(v)
        } else {
            self.total_buffer_len += len as usize;
            let data = buffers.get_unchecked_release(v.buffer_idx as usize);
            let offset = v.offset as usize;
            let bytes = data.get_unchecked_release(offset..offset + len as usize);
            let t = T::from_bytes_unchecked(bytes);
            self.push_value_ignore_validity(t)
        }
    }

    /// # Safety
    /// - caller must allocate enough capacity
    /// - caller must ensure the view and buffers match.
    /// - The array must not have validity.
    /// - caller must not mix use this function with other push functions.
    pub unsafe fn push_view_unchecked_dedupe(&mut self, mut v: View, buffers: &[Buffer<u8>]) {
        let len = v.length;
        self.total_bytes_len += len as usize;
        if len <= 12 {
            self.views.push_unchecked(v);
        } else {
            let buffer = buffers.get_unchecked_release(v.buffer_idx as usize);
            let idx = match self.stolen_buffers.entry(buffer.deref().as_ptr() as usize) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let idx = self.completed_buffers.len() as u32;
                    entry.insert(idx);
                    self.completed_buffers.push(buffer.clone());
                    self.total_buffer_len += buffer.len();
                    idx
                },
            };
            v.buffer_idx = idx;
            self.views.push_unchecked(v);
        }
    }

    pub fn push_view(&mut self, mut v: View, buffers: &[Buffer<u8>]) {
        let len = v.length;
        self.total_bytes_len += len as usize;
        if len <= 12 {
            self.views.push(v);
        } else {
            // Do no mix use of push_view and push_value_ignore_validity -
            // it causes fragmentation.
            self.finish_in_progress();

            let buffer = &buffers[v.buffer_idx as usize];
            let idx = match self.stolen_buffers.entry(buffer.deref().as_ptr() as usize) {
                Entry::Occupied(entry) => {
                    let idx = *entry.get();
                    let target_buffer = &self.completed_buffers[idx as usize];
                    debug_assert_eq!(buffer, target_buffer);
                    idx
                },
                Entry::Vacant(entry) => {
                    let idx = self.completed_buffers.len() as u32;
                    entry.insert(idx);
                    self.completed_buffers.push(buffer.clone());
                    self.total_buffer_len += buffer.len();
                    idx
                },
            };
            v.buffer_idx = idx;
            self.views.push(v);
        }
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    #[inline]
    pub fn push_value_ignore_validity<V: AsRef<T>>(&mut self, value: V) {
        let bytes = value.as_ref().to_bytes();
        self.total_bytes_len += bytes.len();

        // A string can only be maximum of 4GB in size.
        let len = u32::try_from(bytes.len()).unwrap();

        let view = if len <= View::MAX_INLINE_SIZE {
            View::new_inline(bytes)
        } else {
            self.total_buffer_len += bytes.len();

            // We want to make sure that we never have to memcopy between buffers. So if the
            // current buffer is not large enough, create a new buffer that is large enough and try
            // to anticipate the larger size.
            let required_capacity = self.in_progress_buffer.len() + bytes.len();
            let does_not_fit_in_buffer = self.in_progress_buffer.capacity() < required_capacity;

            // We can only save offsets that are below u32::MAX
            let offset_will_not_fit = self.in_progress_buffer.len() > u32::MAX as usize;

            if does_not_fit_in_buffer || offset_will_not_fit {
                // Allocate a new buffer and flush the old buffer
                let new_capacity = (self.in_progress_buffer.capacity() * 2)
                    .clamp(DEFAULT_BLOCK_SIZE, MAX_EXP_BLOCK_SIZE)
                    .max(bytes.len());
                let in_progress = Vec::with_capacity(new_capacity);
                let flushed = std::mem::replace(&mut self.in_progress_buffer, in_progress);
                if !flushed.is_empty() {
                    self.completed_buffers.push(flushed.into())
                }
            }

            let offset = self.in_progress_buffer.len() as u32;
            self.in_progress_buffer.extend_from_slice(bytes);

            let buffer_idx = u32::try_from(self.completed_buffers.len()).unwrap();

            View::new_from_bytes(bytes, buffer_idx, offset)
        };

        self.views.push(view);
    }

    #[inline]
    pub fn push_buffer(&mut self, buffer: Buffer<u8>) -> u32 {
        self.finish_in_progress();

        let buffer_idx = self.completed_buffers.len();
        self.total_buffer_len += buffer.len();
        self.completed_buffers.push(buffer);
        buffer_idx as u32
    }

    #[inline]
    pub fn push_value<V: AsRef<T>>(&mut self, value: V) {
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        self.push_value_ignore_validity(value)
    }

    #[inline]
    pub fn push<V: AsRef<T>>(&mut self, value: Option<V>) {
        if let Some(value) = value {
            self.push_value(value)
        } else {
            self.push_null()
        }
    }

    #[inline]
    pub fn push_null(&mut self) {
        self.views.push(View::default());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(true),
        }
    }

    pub fn extend_null(&mut self, additional: usize) {
        if self.validity.is_none() && additional > 0 {
            self.init_validity(false);
        }
        self.views
            .extend(std::iter::repeat(View::default()).take(additional));
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
    }

    pub fn extend_constant<V: AsRef<T>>(&mut self, additional: usize, value: Option<V>) {
        if value.is_none() && self.validity.is_none() {
            self.init_validity(false);
        }

        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, value.is_some())
        }

        // Push and pop to get the properly encoded value.
        // For long string this leads to a dictionary encoding,
        // as we push the string only once in the buffers
        let view_value = value
            .map(|v| {
                self.push_value_ignore_validity(v);
                self.views.pop().unwrap()
            })
            .unwrap_or_default();
        self.views
            .extend(std::iter::repeat(view_value).take(additional));
    }

    impl_mutable_array_mut_validity!();

    #[inline]
    pub fn extend_values<I, P>(&mut self, iterator: I)
    where
        I: Iterator<Item = P>,
        P: AsRef<T>,
    {
        self.reserve(iterator.size_hint().0);
        for v in iterator {
            self.push_value(v)
        }
    }

    #[inline]
    pub fn extend_trusted_len_values<I, P>(&mut self, iterator: I)
    where
        I: TrustedLen<Item = P>,
        P: AsRef<T>,
    {
        self.extend_values(iterator)
    }

    #[inline]
    pub fn extend<I, P>(&mut self, iterator: I)
    where
        I: Iterator<Item = Option<P>>,
        P: AsRef<T>,
    {
        self.reserve(iterator.size_hint().0);
        for p in iterator {
            self.push(p)
        }
    }

    #[inline]
    pub fn extend_trusted_len<I, P>(&mut self, iterator: I)
    where
        I: TrustedLen<Item = Option<P>>,
        P: AsRef<T>,
    {
        self.extend(iterator)
    }

    #[inline]
    pub fn extend_views<I>(&mut self, iterator: I, buffers: &[Buffer<u8>])
    where
        I: Iterator<Item = Option<View>>,
    {
        self.reserve(iterator.size_hint().0);
        for p in iterator {
            match p {
                Some(v) => self.push_view(v, buffers),
                None => self.push_null(),
            }
        }
    }

    #[inline]
    pub fn extend_views_trusted_len<I>(&mut self, iterator: I, buffers: &[Buffer<u8>])
    where
        I: TrustedLen<Item = Option<View>>,
    {
        self.extend_views(iterator, buffers);
    }

    #[inline]
    pub fn extend_non_null_views<I>(&mut self, iterator: I, buffers: &[Buffer<u8>])
    where
        I: Iterator<Item = View>,
    {
        self.reserve(iterator.size_hint().0);
        for v in iterator {
            self.push_view(v, buffers);
        }
    }

    #[inline]
    pub fn extend_non_null_views_trusted_len<I>(&mut self, iterator: I, buffers: &[Buffer<u8>])
    where
        I: TrustedLen<Item = View>,
    {
        self.extend_non_null_views(iterator, buffers);
    }

    /// # Safety
    /// Same as `push_view_unchecked()`.
    #[inline]
    pub unsafe fn extend_non_null_views_unchecked<I>(&mut self, iterator: I, buffers: &[Buffer<u8>])
    where
        I: Iterator<Item = View>,
    {
        self.reserve(iterator.size_hint().0);
        for v in iterator {
            self.push_view_unchecked(v, buffers);
        }
    }

    /// # Safety
    /// Same as `push_view_unchecked()`.
    #[inline]
    pub unsafe fn extend_non_null_views_unchecked_dedupe<I>(
        &mut self,
        iterator: I,
        buffers: &[Buffer<u8>],
    ) where
        I: Iterator<Item = View>,
    {
        self.reserve(iterator.size_hint().0);
        for v in iterator {
            self.push_view_unchecked_dedupe(v, buffers);
        }
    }

    #[inline]
    pub fn from_iterator<I, P>(iterator: I) -> Self
    where
        I: Iterator<Item = Option<P>>,
        P: AsRef<T>,
    {
        let mut mutable = Self::with_capacity(iterator.size_hint().0);
        mutable.extend(iterator);
        mutable
    }

    pub fn from_values_iter<I, P>(iterator: I) -> Self
    where
        I: Iterator<Item = P>,
        P: AsRef<T>,
    {
        let mut mutable = Self::with_capacity(iterator.size_hint().0);
        mutable.extend_values(iterator);
        mutable
    }

    pub fn from<S: AsRef<T>, P: AsRef<[Option<S>]>>(slice: P) -> Self {
        Self::from_iterator(slice.as_ref().iter().map(|opt_v| opt_v.as_ref()))
    }

    fn finish_in_progress(&mut self) -> bool {
        if !self.in_progress_buffer.is_empty() {
            self.completed_buffers
                .push(std::mem::take(&mut self.in_progress_buffer).into());
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn freeze(self) -> BinaryViewArrayGeneric<T> {
        self.into()
    }

    #[inline]
    pub fn freeze_with_dtype(self, dtype: ArrowDataType) -> BinaryViewArrayGeneric<T> {
        let mut arr: BinaryViewArrayGeneric<T> = self.into();
        arr.data_type = dtype;
        arr
    }

    #[inline]
    pub fn value(&self, i: usize) -> &T {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i`
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &T {
        self.value_from_view_unchecked(self.views.get_unchecked(i))
    }

    /// Returns the element indicated by the given view.
    ///
    /// # Safety
    /// Assumes the View belongs to this MutableBinaryViewArray.
    pub unsafe fn value_from_view_unchecked<'a>(&'a self, view: &'a View) -> &'a T {
        // View layout:
        // length: 4 bytes
        // prefix: 4 bytes
        // buffer_index: 4 bytes
        // offset: 4 bytes

        // Inlined layout:
        // length: 4 bytes
        // data: 12 bytes
        let len = view.length;
        let bytes = if len <= 12 {
            let ptr = view as *const View as *const u8;
            std::slice::from_raw_parts(ptr.add(4), len as usize)
        } else {
            let buffer_idx = view.buffer_idx as usize;
            let offset = view.offset;

            let data = if buffer_idx == self.completed_buffers.len() {
                self.in_progress_buffer.as_slice()
            } else {
                self.completed_buffers.get_unchecked_release(buffer_idx)
            };

            let offset = offset as usize;
            data.get_unchecked(offset..offset + len as usize)
        };
        T::from_bytes_unchecked(bytes)
    }

    /// Returns an iterator of `&[u8]` over every element of this array, ignoring the validity
    pub fn values_iter(&self) -> MutableBinaryViewValueIter<T> {
        MutableBinaryViewValueIter::new(self)
    }
}

impl MutableBinaryViewArray<[u8]> {
    pub fn validate_utf8(&mut self, buffer_offset: usize, views_offset: usize) -> PolarsResult<()> {
        // Finish the in progress as it might be required for validation.
        let pushed = self.finish_in_progress();
        // views are correct
        unsafe {
            validate_utf8_only(
                &self.views[views_offset..],
                &self.completed_buffers[buffer_offset..],
                &self.completed_buffers,
            )?
        }
        // Restore in-progress buffer as we don't want to get too small buffers
        if pushed {
            if let Some(last) = self.completed_buffers.pop() {
                self.in_progress_buffer = last.into_mut().right().unwrap();
            }
        }
        Ok(())
    }

    /// Extend from a `buffer` and `length` of items given some statistics about the lengths.
    ///
    /// This will attempt to dispatch to several optimized implementations.
    ///
    /// # Safety
    ///
    /// This is safe if the statistics are correct.
    pub unsafe fn extend_from_lengths_with_stats(
        &mut self,
        buffer: &[u8],
        lengths_iterator: impl Clone + ExactSizeIterator<Item = usize>,
        min_length: usize,
        max_length: usize,
        sum_length: usize,
    ) {
        let num_items = lengths_iterator.len();

        if num_items == 0 {
            return;
        }

        #[cfg(debug_assertions)]
        {
            let (min, max, sum) = lengths_iterator.clone().map(|v| (v, v, v)).fold(
                (usize::MAX, usize::MIN, 0usize),
                |(cmin, cmax, csum), (emin, emax, esum)| {
                    (cmin.min(emin), cmax.max(emax), csum + esum)
                },
            );

            assert_eq!(min, min_length);
            assert_eq!(max, max_length);
            assert_eq!(sum, sum_length);
        }

        assert!(sum_length <= buffer.len());

        let mut buffer_offset = 0;
        if min_length > View::MAX_INLINE_SIZE as usize
            && (num_items == 1 || sum_length + self.in_progress_buffer.len() <= u32::MAX as usize)
        {
            let buffer_idx = self.completed_buffers().len() as u32;
            let in_progress_buffer_offset = self.in_progress_buffer.len();

            self.total_bytes_len += sum_length;
            self.total_buffer_len += sum_length;

            self.in_progress_buffer
                .extend_from_slice(&buffer[..sum_length]);
            self.views.extend(lengths_iterator.map(|length| {
                // SAFETY: We asserted before that the sum of all lengths is smaller or equal to
                // the buffer length.
                let view_buffer =
                    unsafe { buffer.get_unchecked(buffer_offset..buffer_offset + length) };

                // SAFETY: We know that the minimum length > View::MAX_INLINE_SIZE. Therefore, this
                // length is > View::MAX_INLINE_SIZE.
                let view = unsafe {
                    View::new_noninline_unchecked(
                        view_buffer,
                        buffer_idx,
                        (buffer_offset + in_progress_buffer_offset) as u32,
                    )
                };
                buffer_offset += length;
                view
            }));
        } else if max_length <= View::MAX_INLINE_SIZE as usize {
            self.total_bytes_len += sum_length;

            // If the min and max are the same, we can dispatch to the optimized SIMD
            // implementation.
            if min_length == max_length {
                let length = min_length;
                if length == 0 {
                    self.views
                        .resize(self.views.len() + num_items, View::new_inline(&[]));
                } else {
                    View::extend_with_inlinable_strided(
                        &mut self.views,
                        &buffer[..length * num_items],
                        length as u8,
                    );
                }
            } else {
                self.views.extend(lengths_iterator.map(|length| {
                    // SAFETY: We asserted before that the sum of all lengths is smaller or equal
                    // to the buffer length.
                    let view_buffer =
                        unsafe { buffer.get_unchecked(buffer_offset..buffer_offset + length) };

                    // SAFETY: We know that each view has a length <= View::MAX_INLINE_SIZE because
                    // the maximum length is <= View::MAX_INLINE_SIZE
                    let view = unsafe { View::new_inline_unchecked(view_buffer) };

                    buffer_offset += length;

                    view
                }));
            }
        } else {
            // If all fails, just fall back to a base implementation.
            self.reserve(num_items);
            for length in lengths_iterator {
                let value = &buffer[buffer_offset..buffer_offset + length];
                buffer_offset += length;
                self.push_value(value);
            }
        }
    }

    /// Extend from a `buffer` and `length` of items.
    ///
    /// This will attempt to dispatch to several optimized implementations.
    #[inline]
    pub fn extend_from_lengths(
        &mut self,
        buffer: &[u8],
        lengths_iterator: impl Clone + ExactSizeIterator<Item = usize>,
    ) {
        let (min, max, sum) = lengths_iterator.clone().map(|v| (v, v, v)).fold(
            (usize::MAX, usize::MIN, 0usize),
            |(cmin, cmax, csum), (emin, emax, esum)| (cmin.min(emin), cmax.max(emax), csum + esum),
        );

        // SAFETY: We just collected the right stats.
        unsafe { self.extend_from_lengths_with_stats(buffer, lengths_iterator, min, max, sum) }
    }
}

impl<T: ViewType + ?Sized, P: AsRef<T>> Extend<Option<P>> for MutableBinaryViewArray<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = Option<P>>>(&mut self, iter: I) {
        Self::extend(self, iter.into_iter())
    }
}

impl<T: ViewType + ?Sized, P: AsRef<T>> FromIterator<Option<P>> for MutableBinaryViewArray<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<P>>>(iter: I) -> Self {
        Self::from_iterator(iter.into_iter())
    }
}

impl<T: ViewType + ?Sized> MutableArray for MutableBinaryViewArray<T> {
    fn data_type(&self) -> &ArrowDataType {
        T::dtype()
    }

    fn len(&self) -> usize {
        MutableBinaryViewArray::len(self)
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let mutable = std::mem::take(self);
        let arr: BinaryViewArrayGeneric<T> = mutable.into();
        arr.boxed()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn push_null(&mut self) {
        MutableBinaryViewArray::push_null(self)
    }

    fn reserve(&mut self, additional: usize) {
        MutableBinaryViewArray::reserve(self, additional)
    }

    fn shrink_to_fit(&mut self) {
        self.views.shrink_to_fit()
    }
}

impl<T: ViewType + ?Sized, P: AsRef<T>> TryExtend<Option<P>> for MutableBinaryViewArray<T> {
    /// This is infallible and is implemented for consistency with all other types
    #[inline]
    fn try_extend<I: IntoIterator<Item = Option<P>>>(&mut self, iter: I) -> PolarsResult<()> {
        self.extend(iter.into_iter());
        Ok(())
    }
}

impl<T: ViewType + ?Sized, P: AsRef<T>> TryPush<Option<P>> for MutableBinaryViewArray<T> {
    /// This is infallible and is implemented for consistency with all other types
    #[inline(always)]
    fn try_push(&mut self, item: Option<P>) -> PolarsResult<()> {
        self.push(item.as_ref().map(|p| p.as_ref()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[&[u8]]) -> bool {
        let buffer = values
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect::<Vec<u8>>();
        let lengths = values.iter().map(|v| v.len()).collect::<Vec<usize>>();
        let mut bv = MutableBinaryViewArray::<[u8]>::with_capacity(values.len());

        bv.extend_from_lengths(&buffer[..], lengths.into_iter());

        &bv.values_iter().collect::<Vec<&[u8]>>()[..] == values
    }

    #[test]
    fn extend_with_lengths_basic() {
        assert!(roundtrip(&[]));
        assert!(roundtrip(&[b"abc"]));
        assert!(roundtrip(&[
            b"a_very_very_long_string_that_is_not_inlinable"
        ]));
        assert!(roundtrip(&[
            b"abc",
            b"a_very_very_long_string_that_is_not_inlinable"
        ]));
    }

    #[test]
    fn extend_with_inlinable_fastpath() {
        assert!(roundtrip(&[b"abc", b"defg", b"hix"]));
        assert!(roundtrip(&[b"abc", b"defg", b"hix", b"xyza1234abcd"]));
    }

    #[test]
    fn extend_with_inlinable_eq_len_fastpath() {
        assert!(roundtrip(&[b"abc", b"def", b"hix"]));
        assert!(roundtrip(&[b"abc", b"def", b"hix", b"xyz"]));
    }

    #[test]
    fn extend_with_not_inlinable_fastpath() {
        assert!(roundtrip(&[
            b"a_very_long_string123",
            b"a_longer_string_than_the_previous"
        ]));
    }
}
