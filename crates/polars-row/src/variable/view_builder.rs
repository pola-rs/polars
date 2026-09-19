//! Builds a [`BinaryViewArray`] from decoded values.
//!
//! Unlike [`MutableBinaryViewArray`](polars_arrow::array::MutableBinaryViewArray) this lets
//! decoders write value bytes straight into the data buffer.
use polars_arrow::array::{BinaryViewArray, View};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;

use crate::utils::BLOCK;

/// Maximum length of one data buffer. Values are limited to `u32::MAX` bytes.
const MAX_BUFFER_LEN: usize = i32::MAX as usize;

pub(crate) struct ViewBuilder {
    views: Vec<View>,
    buffer: Vec<u8>,
    completed_buffers: Vec<Buffer<u8>>,
    total_bytes_len: usize,
    total_buffer_len: usize,
}

impl ViewBuilder {
    pub fn with_capacity(num_rows: usize) -> Self {
        Self {
            views: Vec::with_capacity(num_rows),
            buffer: Vec::new(),
            completed_buffers: Vec::new(),
            total_bytes_len: 0,
            total_buffer_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    #[inline(always)]
    pub fn push_null(&mut self) {
        self.views.push(View::default());
    }

    /// Push a view that holds its bytes inline.
    #[inline(always)]
    pub fn push_inline(&mut self, view: View) {
        debug_assert!(view.length <= View::MAX_INLINE_SIZE);
        self.total_bytes_len += view.length as usize;
        self.views.push(view);
    }

    #[inline(always)]
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        if bytes.len() <= View::MAX_INLINE_SIZE as usize {
            self.push_inline(View::new_inline(bytes));
        } else {
            unsafe {
                let dst = self.start_value(bytes.len());
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                let prefix = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                self.finish_value(bytes.len(), prefix);
            }
        }
    }

    /// Start a value of at least `len` bytes. Returns where to write the bytes. Up to `len` +
    /// [`BLOCK`] bytes may be written. Use [`Self::grow_value`] if more room is needed and
    /// [`Self::finish_value`] once all bytes are written.
    #[inline(always)]
    pub unsafe fn start_value(&mut self, len: usize) -> *mut u8 {
        if self.buffer.len() > MAX_BUFFER_LEN - len.max(BLOCK) {
            let buffer = std::mem::take(&mut self.buffer);
            self.total_buffer_len += buffer.len();
            self.completed_buffers.push(Buffer::from(buffer));
        }
        self.buffer.reserve(len + BLOCK);
        self.buffer.as_mut_ptr().add(self.buffer.len())
    }

    /// Make room for `additional` more bytes in the value being written. Returns the new
    /// location of the start of the value.
    #[inline(always)]
    pub unsafe fn grow_value(&mut self, written: usize, additional: usize) -> *mut u8 {
        assert!(self.buffer.len() + written + additional <= MAX_BUFFER_LEN);
        self.buffer.reserve(written + additional);
        self.buffer.as_mut_ptr().add(self.buffer.len())
    }

    /// Finish the value started with [`Self::start_value`]. `prefix` holds the first 4 bytes
    /// of the value.
    ///
    /// # Safety
    /// `len` bytes must have been written and `len > View::MAX_INLINE_SIZE`.
    #[inline(always)]
    pub unsafe fn finish_value(&mut self, len: usize, prefix: u32) {
        debug_assert!(len > View::MAX_INLINE_SIZE as usize);
        let offset = self.buffer.len();
        self.buffer.set_len(offset + len);
        let view = View {
            length: len as u32,
            prefix,
            buffer_idx: self.completed_buffers.len() as u32,
            offset: offset as u32,
        };
        self.total_bytes_len += len;
        self.views.push(view);
    }

    /// Finish a value started with [`Self::start_value`] that turned out to fit inline.
    ///
    /// # Safety
    /// `len` bytes must have been written.
    #[inline(never)]
    pub unsafe fn finish_short_value(&mut self, len: usize) {
        let bytes = self
            .buffer
            .spare_capacity_mut()
            .get_unchecked(..len)
            .as_ptr()
            .cast::<u8>();
        let view = View::new_inline(std::slice::from_raw_parts(bytes, len));
        self.push_inline(view);
    }

    pub fn freeze(mut self, dtype: ArrowDataType, validity: Option<Bitmap>) -> BinaryViewArray {
        if !self.buffer.is_empty() {
            self.total_buffer_len += self.buffer.len();
            self.completed_buffers.push(Buffer::from(self.buffer));
        }
        unsafe {
            BinaryViewArray::new_unchecked(
                dtype,
                Buffer::from(self.views),
                Buffer::from(self.completed_buffers),
                validity,
                Some(self.total_bytes_len),
                self.total_buffer_len,
            )
        }
    }
}
