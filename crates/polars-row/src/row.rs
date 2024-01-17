use std::sync::Arc;
use arrow::array::{BinaryArray, BinaryViewArray};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::ffi::mmap;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::vec::PushUnchecked;

#[derive(Clone, Default)]
pub struct SortField {
    /// Whether to sort in descending order
    pub descending: bool,
    /// Whether to sort nulls first
    pub nulls_last: bool,
}

#[derive(Default, Clone)]
pub struct RowsEncoded {
    pub(crate) values: Vec<u8>,
    pub(crate) offsets: Vec<usize>,
}

fn checks(offsets: &[usize]) {
    assert_eq!(
        std::mem::size_of::<usize>(),
        std::mem::size_of::<i64>(),
        "only supported on 64bit arch"
    );
    assert!(
        (*offsets.last().unwrap() as u64) < i64::MAX as u64,
        "overflow"
    );
}

unsafe fn rows_to_array(buf: Vec<u8>, offsets: Vec<usize>) -> BinaryArray<i64> {
    checks(&offsets);

    // Safety: we checked overflow
    let offsets = std::mem::transmute::<Vec<usize>, Vec<i64>>(offsets);

    // Safety: monotonically increasing
    let offsets = Offsets::new_unchecked(offsets);

    BinaryArray::new(ArrowDataType::LargeBinary, offsets.into(), buf.into(), None)
}

impl RowsEncoded {
    pub(crate) fn new(values: Vec<u8>, offsets: Vec<usize>) -> Self {
        RowsEncoded { values, offsets }
    }

    pub fn iter(&self) -> RowsEncodedIter {
        let iter = self.offsets[1..].iter();
        let offset = self.offsets[0];
        RowsEncodedIter {
            offset,
            end: iter,
            values: &self.values,
        }
    }

    /// Borrows the buffers and returns a [`BinaryArray`].
    ///
    /// # Safety
    /// The lifetime of that `BinaryArray` is tight to the lifetime of
    /// `Self`. The caller must ensure that both stay alive for the same time.
    pub unsafe fn borrow_array(&self) -> BinaryArray<i64> {
        checks(&self.offsets);

        unsafe {
            let (_, values, _) = mmap::slice(&self.values).into_inner();
            let offsets = std::mem::transmute::<&[usize], &[i64]>(self.offsets.as_slice());
            let (_, offsets, _) = mmap::slice(offsets).into_inner();
            let offsets = OffsetsBuffer::new_unchecked(offsets);

            BinaryArray::new(ArrowDataType::LargeBinary, offsets, values, None)
        }
    }

    pub fn into_array(self) -> BinaryArray<i64> {
        unsafe { rows_to_array(self.values, self.offsets) }
    }

    pub fn into_binview(self) -> BinaryViewArray {
        let buffer_idx = 0 as u32;
        let base_ptr = self.values.as_ptr() as usize;

        let mut views = Vec::with_capacity(self.offsets.len() - 1);
        for bytes in self.iter() {
            let len: u32 = bytes.len().try_into().unwrap();
            
            let mut payload = [0; 16];

            if len <= 12 {
                payload[4..4 + bytes.len()].copy_from_slice(bytes);
            } else {
                unsafe { payload[4..8].copy_from_slice(bytes.get_unchecked_release(0..4)) };
                let offset = (bytes.as_ptr() as usize - base_ptr) as u32;
                payload[0..4].copy_from_slice(&len.to_le_bytes());
                payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
                payload[12..16].copy_from_slice(&offset.to_le_bytes());
            }

            let value = u128::from_le_bytes(payload);
            unsafe { views.push_unchecked(value) };
        }
        unsafe {
            let buffer: Buffer<u8> = self.values.into();
            BinaryViewArray::new_unchecked_unknown_md(ArrowDataType::BinaryView,
                views.into(),
                Arc::from([buffer]),
                None,
            )
        }
    }

    #[cfg(test)]
    pub fn get(&self, i: usize) -> &[u8] {
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        &self.values[start..end]
    }
}

pub struct RowsEncodedIter<'a> {
    offset: usize,
    end: std::slice::Iter<'a, usize>,
    values: &'a [u8],
}

impl<'a> Iterator for RowsEncodedIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let new_offset = *self.end.next()?;
        let payload = unsafe { self.values.get_unchecked(self.offset..new_offset) };
        self.offset = new_offset;
        Some(payload)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.end.size_hint()
    }
}
