use arrow::array::{BinaryArray, BinaryViewArray};
use arrow::datatypes::ArrowDataType;
use arrow::ffi::mmap;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_compute::cast::binary_to_binview;

const BOOLEAN_TRUE_SENTINEL: u8 = 0x03;
const BOOLEAN_FALSE_SENTINEL: u8 = 0x02;

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

const LIST_CONTINUATION_TOKEN: u8 = 0xFE;
const EMPTY_STR_TOKEN: u8 = 0x01;
const NON_EMPTY_STR_TOKEN: u8 = 0x02;

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

    pub fn null_sentinel(self) -> u8 {
        if self.nulls_last {
            0xFF
        } else {
            0x00
        }
    }

    pub(crate) fn bool_true_sentinel(self) -> u8 {
        if self.descending {
            !BOOLEAN_TRUE_SENTINEL
        } else {
            BOOLEAN_TRUE_SENTINEL
        }
    }

    pub(crate) fn bool_false_sentinel(self) -> u8 {
        if self.descending {
            !BOOLEAN_FALSE_SENTINEL
        } else {
            BOOLEAN_FALSE_SENTINEL
        }
    }

    pub fn list_null_sentinel(self) -> u8 {
        self.null_sentinel()
    }

    pub fn list_continuation_token(self) -> u8 {
        if self.descending {
            !LIST_CONTINUATION_TOKEN
        } else {
            LIST_CONTINUATION_TOKEN
        }
    }

    pub fn list_termination_token(self) -> u8 {
        !self.list_continuation_token()
    }

    pub fn empty_str_token(self) -> u8 {
        if self.descending {
            !EMPTY_STR_TOKEN
        } else {
            EMPTY_STR_TOKEN
        }
    }
    pub fn non_empty_str_token(self) -> u8 {
        if self.descending {
            !NON_EMPTY_STR_TOKEN
        } else {
            NON_EMPTY_STR_TOKEN
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
        size_of::<usize>(),
        size_of::<i64>(),
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
