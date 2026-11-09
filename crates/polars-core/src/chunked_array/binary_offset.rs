use polars_error::PolarsResult;
use polars_row::{RowEncodingOptions, RowEncodingOutput};

use crate::prelude::{BinaryOffsetChunked, DataType, Field, IntoSeries};

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

        use super::ops::row_encode::ordered_opts;

        self.row_decode(fields, &ordered_opts(descending, nulls_last))
    }

    pub fn row_decode_unordered(&self, fields: &[Field]) -> PolarsResult<super::StructChunked> {
        let mut opts = Vec::with_capacity(fields.len());
        opts.extend(std::iter::repeat_n(
            RowEncodingOptions::new_unsorted(),
            fields.len(),
        ));
        self.row_decode(fields, &opts)
    }

    fn row_decode(
        &self,
        fields: &[Field],
        opts: &[RowEncodingOptions],
    ) -> PolarsResult<super::StructChunked> {
        use super::ops::row_encode::{get_row_encoding_layout, row_encoding_decode};

        let dtypes: Vec<DataType> = fields.iter().map(|f| f.dtype().clone()).collect();
        let layout = get_row_encoding_layout(&dtypes, opts, RowEncodingOutput::AlwaysBinaryOffset);
        row_encoding_decode(&self.clone().into_series(), fields, &layout)
    }
}
