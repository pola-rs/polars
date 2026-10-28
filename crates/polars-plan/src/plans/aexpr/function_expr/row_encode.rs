use polars_core::chunked_array::ops::row_encode::get_row_encoding_layout;
use polars_core::prelude::{DataType, PolarsResult, RowEncodingOptions};
use polars_error::polars_ensure;
use polars_row::{RowEncodingLayout, RowEncodingOutput};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RowEncodingVariant {
    Unordered,
    Ordered {
        descending: Option<Vec<bool>>,
        nulls_last: Option<Vec<bool>>,
    },
}

/// The type of the row-encoded column.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RowEncodingOutputType {
    AlwaysBinaryOffset,
    AlwaysBinary,
    Optimal,
}

impl RowEncodingOutputType {
    /// The layout for encoding columns of these dtypes. `encode_nulls` says per column whether its
    /// nulls are encoded, `None` means for all columns.
    pub fn layout(
        &self,
        dtypes: &[DataType],
        variant: &RowEncodingVariant,
        encode_nulls: Option<&[bool]>,
    ) -> PolarsResult<RowEncodingLayout> {
        let n = dtypes.len();
        let opts = match variant {
            RowEncodingVariant::Unordered => vec![RowEncodingOptions::new_unsorted(); n],
            RowEncodingVariant::Ordered {
                descending,
                nulls_last,
            } => {
                let descending = descending.clone().unwrap_or_else(|| vec![false; n]);
                let nulls_last = nulls_last.clone().unwrap_or_else(|| vec![false; n]);
                polars_ensure!(
                    descending.len() == n && nulls_last.len() == n,
                    ShapeMismatch: "row encoding got {} columns but {} descending and {} nulls_last options",
                    n, descending.len(), nulls_last.len()
                );
                descending
                    .into_iter()
                    .zip(nulls_last)
                    .map(|(d, nl)| RowEncodingOptions::new_sorted(d, nl))
                    .collect()
            },
        };

        let output = match self {
            Self::AlwaysBinaryOffset => RowEncodingOutput::AlwaysBinaryOffset,
            Self::AlwaysBinary => RowEncodingOutput::AlwaysBinary,
            Self::Optimal => RowEncodingOutput::Optimal,
        };
        if let Some(encode_nulls) = encode_nulls {
            polars_ensure!(
                encode_nulls.len() == n,
                ShapeMismatch: "row encoding got {} columns but {} encode_nulls options", n, encode_nulls.len()
            );
        }
        Ok(get_row_encoding_layout(dtypes, &opts, encode_nulls, output))
    }
}
