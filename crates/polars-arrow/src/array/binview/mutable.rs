use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::slice::GetSaferUnchecked;

use crate::array::binview::view::validate_utf8_only_view;
use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;

const DEFAULT_BLOCK_SIZE: usize = 8 * 1024;

#[derive(Debug, Clone)]
pub struct MutableBinaryViewArray<T: ViewType + ?Sized> {
    views: Vec<u128>,
    completed_buffers: Vec<Buffer<u8>>,
    in_progress_buffer: Vec<u8>,
    validity: Option<MutableBitmap>,
    phantom: std::marker::PhantomData<T>,
    /// Total bytes length if we would concatenate them all.
    total_bytes_len: usize,
    /// Total bytes in the buffer (excluding remaining capacity)
    total_buffer_len: usize,
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
        }
    }

    pub fn views(&mut self) -> &mut Vec<u128> {
        &mut self.views
    }

    pub fn validity(&mut self) -> Option<&mut MutableBitmap> {
        self.validity.as_mut()
    }

    /// Reserves `additional` elements and `additional_buffer` on the buffer.
    pub fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.views.capacity());
        validity.extend_constant(self.len(), true);
        validity.set(self.len() - 1, false);
        self.validity = Some(validity);
    }

    pub fn push_value_ignore_validity<V: AsRef<T>>(&mut self, value: V) {
        let value = value.as_ref();
        let bytes = value.to_bytes();
        self.total_bytes_len += bytes.len();
        let len: u32 = bytes.len().try_into().unwrap();
        let mut payload = [0; 16];
        payload[0..4].copy_from_slice(&len.to_le_bytes());

        if len <= 12 {
            payload[4..4 + bytes.len()].copy_from_slice(bytes);
        } else {
            self.total_buffer_len += bytes.len();
            let required_cap = self.in_progress_buffer.len() + bytes.len();
            if self.in_progress_buffer.capacity() < required_cap {
                let new_capacity = (self.in_progress_buffer.capacity() * 2)
                    .clamp(DEFAULT_BLOCK_SIZE, 16 * 1024 * 1024)
                    .max(bytes.len());
                let in_progress = Vec::with_capacity(new_capacity);
                let flushed = std::mem::replace(&mut self.in_progress_buffer, in_progress);
                if !flushed.is_empty() {
                    self.completed_buffers.push(flushed.into())
                }
            }
            let offset = self.in_progress_buffer.len() as u32;
            self.in_progress_buffer.extend_from_slice(bytes);

            unsafe { payload[4..8].copy_from_slice(bytes.get_unchecked_release(0..4)) };
            let buffer_idx: u32 = self.completed_buffers.len().try_into().unwrap();
            payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
            payload[12..16].copy_from_slice(&offset.to_le_bytes());
        }
        let value = u128::from_le_bytes(payload);
        self.views.push(value);
    }

    pub fn push_value<V: AsRef<T>>(&mut self, value: V) {
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        self.push_value_ignore_validity(value)
    }

    pub fn push<V: AsRef<T>>(&mut self, value: Option<V>) {
        if let Some(value) = value {
            self.push_value(value)
        } else {
            self.push_null()
        }
    }

    pub fn push_null(&mut self) {
        self.views.push(0);
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
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

    pub fn from_iter<I, P>(iterator: I) -> Self
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
        Self::from_iter(slice.as_ref().iter().map(|opt_v| opt_v.as_ref()))
    }

    fn finish_in_progress(&mut self) {
        if !self.in_progress_buffer.is_empty() {
            self.completed_buffers
                .push(std::mem::take(&mut self.in_progress_buffer).into());
        }
    }
}

impl MutableBinaryViewArray<[u8]> {
    pub fn validate_utf8(&mut self) -> PolarsResult<()> {
        validate_utf8_only_view(&self.views, &self.completed_buffers)
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
        Self::from_iter(iter.into_iter())
    }
}
