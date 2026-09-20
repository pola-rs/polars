#![allow(unsafe_op_in_unsafe_fn)]
//! Builds a [`BinaryViewArray`] from decoded values.
//!
//! Unlike [`MutableBinaryViewArray`](polars_arrow::array::MutableBinaryViewArray) this lets
//! decoders push ready made inline views.
use polars_arrow::array::{BinaryViewArray, View};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;

/// A data buffer is not grown past this length, except for a single value that is larger.
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

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        if bytes.len() <= View::MAX_INLINE_SIZE as usize {
            self.push_inline(View::new_inline(bytes));
            return;
        }
        let length = u32::try_from(bytes.len()).expect("value longer than u32::MAX bytes");
        // A single value may be larger than the limit and then gets its own buffer.
        if self.buffer.len() + bytes.len() > MAX_BUFFER_LEN && !self.buffer.is_empty() {
            self.flush_buffer();
        }
        let offset = self.buffer.len();
        self.buffer.extend_from_slice(bytes);
        let view = View {
            length,
            prefix: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            buffer_idx: self.completed_buffers.len() as u32,
            offset: offset as u32,
        };
        self.total_bytes_len += bytes.len();
        self.views.push(view);
    }

    fn flush_buffer(&mut self) {
        let buffer = std::mem::take(&mut self.buffer);
        self.total_buffer_len += buffer.len();
        self.completed_buffers.push(Buffer::from(buffer));
    }

    pub fn freeze(mut self, dtype: ArrowDataType, validity: Option<Bitmap>) -> BinaryViewArray {
        if !self.buffer.is_empty() {
            self.flush_buffer();
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
