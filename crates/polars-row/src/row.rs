#![allow(unsafe_op_in_unsafe_fn)]
use std::sync::Arc;

use polars_array::PlBinaryArray;
use polars_buffer::Buffer;
use polars_dtype::categorical::CategoricalMapping;

const BOOLEAN_TRUE_SENTINEL: u8 = 0x03;
const BOOLEAN_FALSE_SENTINEL: u8 = 0x02;

/// Additional context provided to row encoding regarding a column.
///
/// This allows communication based on the Polars datatype instead on the Arrow datatype. Since
/// polars-row is used under polars-core, we don't have access to the actual datatypes.
#[derive(Debug, Clone)]
pub enum RowEncodingContext {
    Struct(Vec<Option<RowEncodingContext>>),
    /// Categorical / Enum
    Categorical(RowEncodingCategoricalContext),
    /// Decimal with given precision
    Decimal(usize),
}

#[derive(Debug, Clone)]
pub struct RowEncodingCategoricalContext {
    pub is_enum: bool,
    pub mapping: Arc<CategoricalMapping>,
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

    pub fn is_ordered(self) -> bool {
        !self.contains(Self::NO_ORDER)
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

    pub fn into_nested(mut self) -> RowEncodingOptions {
        // Correct nested ordering (see #22557)
        self.set(
            RowEncodingOptions::NULLS_LAST,
            self.contains(RowEncodingOptions::DESCENDING),
        );
        self
    }
}

#[derive(Default, Clone)]
pub struct RowsEncoded {
    pub(crate) values: Vec<u8>,
    pub(crate) offsets: Vec<usize>,
}

/// The offsets a row encoding wrote as the offsets of a [`PlBinaryArray`].
///
/// The cap is `i64::MAX` rather than `u64::MAX` because these offsets cross over to Arrow's
/// `i64`-offset [`BinaryArray`](arrow::array::BinaryArray) at the boundaries of the crates that
/// still hold one.
fn rows_to_offsets(offsets: Vec<usize>) -> Buffer<u64> {
    assert!(
        (*offsets.last().unwrap() as u64) < i64::MAX as u64,
        "row encoding output overflowed"
    );

    if size_of::<usize>() == size_of::<u64>() {
        // SAFETY: the sizes match, so this is a reinterpretation of the same bytes.
        Buffer::from(bytemuck::cast_vec::<usize, u64>(offsets))
    } else {
        offsets.into_iter().map(|v| v as u64).collect()
    }
}

impl RowsEncoded {
    pub(crate) fn new(values: Vec<u8>, offsets: Vec<usize>) -> Self {
        RowsEncoded { values, offsets }
    }

    pub fn iter(&self) -> RowsEncodedIter<'_> {
        let iter = self.offsets[1..].iter();
        let offset = self.offsets[0];
        RowsEncodedIter {
            offset,
            end: iter,
            values: &self.values,
        }
    }

    /// The encoded rows as an array, in `O(1)`: the buffers are handed over as they are.
    pub fn into_array(self) -> PlBinaryArray {
        let length = self.offsets.len() - 1;
        let offsets = rows_to_offsets(self.offsets);

        // SAFETY: the offsets a row encoding wrote are monotonically non-decreasing, hold the end
        // of the last row, and `values` holds every byte they cover.
        unsafe { PlBinaryArray::new_unchecked(Buffer::from(self.values), offsets, length, None) }
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
