use polars_error::PolarsResult;
use polars_row::RowEncodingOptions;

use crate::prelude::{BinaryOffsetChunked, Field};

#[cfg(feature = "dtype-struct")]
impl BinaryOffsetChunked {
    pub fn row_decode_ordered(
        &self,
        fields: &[Field],
        descending: &[bool],
        nulls_last: &[bool],
    ) -> PolarsResult<super::StructChunked> {
        assert_eq!(fields.len(), descending.len());
        assert_eq!(fields.len(), nulls_last.len());

        let mut opts = Vec::with_capacity(fields.len());
        opts.extend(
            descending
                .iter()
                .zip(nulls_last)
                .map(|(d, n)| RowEncodingOptions::new_sorted(*d, *n)),
        );
        crate::prelude::row_encode::row_encoding_decode(self, fields, &opts)
    }

    pub fn row_decode_unordered(&self, fields: &[Field]) -> PolarsResult<super::StructChunked> {
        let mut opts = Vec::with_capacity(fields.len());
        opts.extend(std::iter::repeat_n(
            RowEncodingOptions::new_unsorted(),
            fields.len(),
        ));
        super::ops::row_encode::row_encoding_decode(self, fields, &opts)
    }
}
