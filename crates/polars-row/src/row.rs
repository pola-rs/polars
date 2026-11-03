use std::sync::Arc;

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

        /// Don't encode nulls: encoding a null panics unless nulls are propagated, and
        /// fixed-width values, structs and fixed-size lists take no validity byte. Not passed
        /// on to nested values.
        const SKIP_NULL_ENCODING       = 0x08;
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

    pub fn encode_nulls(self) -> bool {
        !self.contains(Self::SKIP_NULL_ENCODING)
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
        self.remove(RowEncodingOptions::SKIP_NULL_ENCODING);
        self
    }
}
