use polars_core::prelude::row_encode::{_get_rows_encoded_ca, _get_rows_encoded_ca_unordered};
use polars_core::prelude::{Column, DataType, Field, IntoColumn, RowEncodingOptions};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RowEncodingVariant {
    Unordered,
    Ordered {
        descending: Option<Vec<bool>>,
        nulls_last: Option<Vec<bool>>,
    },
}

pub fn encode(
    c: &mut [Column],
    dts: Vec<DataType>,
    variant: RowEncodingVariant,
) -> PolarsResult<Column> {
    assert_eq!(c.len(), dts.len());

    // We need to make sure that the output types are correct or we will get wrong results or even
    // segfaults when decoding.
    for (dt, c) in dts.iter().zip(c.iter_mut()) {
        if c.dtype().matches_schema_type(dt)? {
            *c = c.cast(dt)?;
        }
    }

    let name = PlSmallStr::from_static("row_encoded");
    match variant {
        RowEncodingVariant::Unordered => _get_rows_encoded_ca_unordered(name, c),
        RowEncodingVariant::Ordered {
            descending,
            nulls_last,
        } => {
            let descending = descending.unwrap_or_else(|| vec![false; c.len()]);
            let nulls_last = nulls_last.unwrap_or_else(|| vec![false; c.len()]);

            assert_eq!(c.len(), descending.len());
            assert_eq!(c.len(), nulls_last.len());

            _get_rows_encoded_ca(name, c, &descending, &nulls_last)
        },
    }
    .map(IntoColumn::into_column)
}

#[cfg(feature = "dtype-struct")]
pub fn decode(
    c: &mut [Column],
    fields: Vec<Field>,
    variant: RowEncodingVariant,
) -> PolarsResult<Column> {
    use polars_core::prelude::row_encode::row_encoding_decode;

    assert_eq!(c.len(), 1);
    let ca = c[0].binary_offset()?;

    let mut opts = Vec::with_capacity(fields.len());
    match variant {
        RowEncodingVariant::Unordered => opts.extend(std::iter::repeat_n(
            RowEncodingOptions::new_unsorted(),
            fields.len(),
        )),
        RowEncodingVariant::Ordered {
            descending,
            nulls_last,
        } => {
            let descending = descending.unwrap_or_else(|| vec![false; fields.len()]);
            let nulls_last = nulls_last.unwrap_or_else(|| vec![false; fields.len()]);

            assert_eq!(fields.len(), descending.len());
            assert_eq!(fields.len(), nulls_last.len());

            opts.extend(
                descending
                    .into_iter()
                    .zip(nulls_last)
                    .map(|(d, n)| RowEncodingOptions::new_sorted(d, n)),
            )
        },
    }

    row_encoding_decode(ca, &fields, &opts).map(IntoColumn::into_column)
}
