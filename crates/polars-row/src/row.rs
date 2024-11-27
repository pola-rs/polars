use arrow::array::{BinaryArray, BinaryViewArray, Utf8ViewArray};
use arrow::datatypes::ArrowDataType;
use arrow::ffi::mmap;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_compute::cast::binary_to_binview;

const BOOLEAN_TRUE_SENTINEL: u8 = 0x03;
const BOOLEAN_FALSE_SENTINEL: u8 = 0x02;

/// The Row Encoding ordering used for Categorical types.
///
/// This includes both `Enum` and `Categorical`.
#[derive(Debug, Clone)]
pub enum RowEncodingCatOrder {
    Struct(Vec<Option<RowEncodingCatOrder>>),
    Physical(usize),
    Lexical(Box<Utf8ViewArray>),
}

bitflags::bitflags! {
    /// Options for the Polars Row Encoding.
    ///
    /// The row encoding provides a method to combine several columns into one binary column which
    /// has the same sort-order as the original columns.test
    ///
    /// By default, the row encoding provides the ascending, nulls first sort-order of the columns.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct RowEncodingOptions: u8 {
        /// Sort in descending order instead of ascending order
        const DESCENDING               = 0x01;
        /// Sort such that nulls / missing values are last
        const NULLS_LAST               = 0x02;

        /// Ignore all order-related flags and don't encode order-preserving. This will keep
        /// uniqueness.
        ///
        /// This is faster for several encodings
        const NO_ORDER                 = 0x04;
    }
}

const LIST_CONTINUATION_TOKEN: u8 = 0xFE;
const EMPTY_STR_TOKEN: u8 = 0x01;

impl RowEncodingOptions {
    pub fn new_sorted(descending: bool, nulls_last: bool) -> Self {
        let mut slf = Self::default();
        slf.set(Self::DESCENDING, descending);
        slf.set(Self::NULLS_LAST, nulls_last);
        slf
    }

    pub fn new_unsorted() -> Self {
        Self::NO_ORDER
    }

    pub fn null_sentinel(self) -> u8 {
        if self.contains(Self::NULLS_LAST) {
            0xFF
        } else {
            0x00
        }
    }

    pub(crate) fn bool_true_sentinel(self) -> u8 {
        if self.contains(Self::DESCENDING) {
            !BOOLEAN_TRUE_SENTINEL
        } else {
            BOOLEAN_TRUE_SENTINEL
        }
    }

    pub(crate) fn bool_false_sentinel(self) -> u8 {
        if self.contains(Self::DESCENDING) {
            !BOOLEAN_FALSE_SENTINEL
        } else {
            BOOLEAN_FALSE_SENTINEL
        }
    }

    pub fn list_null_sentinel(self) -> u8 {
        self.null_sentinel()
    }

    pub fn list_continuation_token(self) -> u8 {
        if self.contains(Self::DESCENDING) {
            !LIST_CONTINUATION_TOKEN
        } else {
            LIST_CONTINUATION_TOKEN
        }
    }

    pub fn list_termination_token(self) -> u8 {
        !self.list_continuation_token()
    }

    pub fn empty_str_token(self) -> u8 {
        if self.contains(Self::DESCENDING) {
            !EMPTY_STR_TOKEN
        } else {
            EMPTY_STR_TOKEN
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
