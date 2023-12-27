use std::intrinsics::offset;
use arrow_data::layout;
use polars_utils::slice::GetSaferUnchecked;
use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::array::binview::view::View;
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;

const DEFAULT_BLOCK_SIZE: u32 = 8 * 1024;

#[derive(Debug, Clone)]
pub struct MutableBinaryViewArray<T: ViewType + ?Sized> {
    views: Vec<u128>,
    completed_buffers: Vec<Buffer<u8>>,
    in_progress_buffer: Vec<u8>,
    validity: Option<MutableBitmap>,
}


impl<T: ViewType + ?Sized> Default for MutableBinaryViewArray<T> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<T: ViewType + ?Sized> From<MutableBinaryViewArray<T>> for BinaryViewArrayGeneric<T> {
    fn from(value: MutableBinaryViewArray<T>) -> Self {
        unsafe { Self::new_unchecked(
            T::DATA_TYPE,
            value.views.into(),
            vec![value.buffers.into()],
            value.validity.map(|b|b.into())
        )}
    }
}


impl<T: ViewType + ?Sized> MutableBinaryViewArray<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    pub fn with_capacities(capacity: usize) -> Self {
        Self {
            views: Vec::with_capacity(capacity),
            completed_buffers: vec![],
            in_progress_buffer: vec![],
            validity: None
        }
    }

    /// Reserves `additional` elements and `additional_buffer` on the buffer.
    pub fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.values.capacity());
        validity.extend_constant(self.len(), true);
        validity.set(self.len() - 1, false);
        self.validity = Some(validity);
    }

    pub fn push_value<V: AsRef<T>>(&mut self, value: V) {
        if let Some(validity) = &mut validity {
            validity.push(true)
        }

        let value = value.as_ref();
        let bytes = value.to_bytes();
        let len: u32 = bytes.len().try_into().unwrap();
        let mut payload = [0; 16];
        payload[0..4].copy_from_slice(&len.to_le_bytes());

        if len <= 12 {
            payload[4..4 + bytes.len()].copy_from_slice(bytes);
        } else {
            let required_cap = self.in_progress_buffer.len() + bytes.len();
            if self.in_progress_buffer.capacity() < required_cap {
                let new_capacity = (self.in_progress_buffer.capacity() * 2).clamp(DEFAULT_BLOCK_SIZE, 16 * 1024 * 1024).max(v.len());
                let mut in_progress = Vec::with_capacity(new_capacity);
                let flushed = std::mem::replace(&mut self.in_progress_buffer, &mut in_progress);
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

    pub fn push<V: AsRef<T>>(&mut self, value: Option<V>) {
        if let Some(value) = value {
            self.push_value(value)
        }
        else {
            self.views.push(0);
            match &mut self.validity {
                Some(validity) => validity.push(false),
                None => self.init_validity(),
            }
        }



    }
}