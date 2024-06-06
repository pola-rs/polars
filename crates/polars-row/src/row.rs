use arrow::array::{BinaryArray, BinaryViewArray};
use arrow::compute::cast::binary_to_binview;
use arrow::datatypes::ArrowDataType;
use arrow::ffi::mmap;
use arrow::offset::{Offsets, OffsetsBuffer};

#[derive(Clone, Default, Copy)]
pub struct EncodingField {
    /// Whether to sort in descending order
    pub descending: bool,
    /// Whether to sort nulls first
    pub nulls_last: bool,
    /// Ignore all order-related flags and don't encode order-preserving.
    /// This is faster for variable encoding as we can just memcopy all the bytes.
    pub no_order: bool,
}

impl EncodingField {
    pub fn new_sorted(descending: bool, nulls_last: bool) -> Self {
        EncodingField {
            descending,
            nulls_last,
            no_order: false,
        }
    }

    pub fn new_unsorted() -> Self {
        EncodingField {
            no_order: true,
            ..Default::default()
        }
    }
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

    // SAFETY: we checked overflow
    let offsets = bytemuck::cast_vec::<usize, i64>(offsets);

    // SAFETY: monotonically increasing
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
    /// The lifetime of that `BinaryArray` is tied to the lifetime of
    /// `Self`. The caller must ensure that both stay alive for the same time.
    pub unsafe fn borrow_array(&self) -> BinaryArray<i64> {
        checks(&self.offsets);

        unsafe {
            let (_, values, _) = mmap::slice(&self.values).into_inner();
            let offsets = bytemuck::cast_slice::<usize, i64>(self.offsets.as_slice());
            let (_, offsets, _) = mmap::slice(offsets).into_inner();
            let offsets = OffsetsBuffer::new_unchecked(offsets);

            BinaryArray::new(ArrowDataType::LargeBinary, offsets, values, None)
        }
    }

    /// This conversion is free.
    pub fn into_array(self) -> BinaryArray<i64> {
        unsafe { rows_to_array(self.values, self.offsets) }
    }

    /// This does allocate views.
    pub fn into_binview(self) -> BinaryViewArray {
        binary_to_binview(&self.into_array())
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
