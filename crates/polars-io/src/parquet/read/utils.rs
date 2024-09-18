use std::borrow::Cow;

use polars_core::prelude::{ArrowSchema, DataFrame, DataType, Series, IDX_DTYPE};
use polars_error::{polars_bail, PolarsResult};

use crate::hive::materialize_hive_partitions;
use crate::utils::apply_projection;
use crate::RowIndex;

pub fn materialize_empty_df(
    projection: Option<&[usize]>,
    reader_schema: &ArrowSchema,
    hive_partition_columns: Option<&[Series]>,
    row_index: Option<&RowIndex>,
) -> DataFrame {
    let schema = if let Some(projection) = projection {
        Cow::Owned(apply_projection(reader_schema, projection))
    } else {
        Cow::Borrowed(reader_schema)
    };
    let mut df = DataFrame::empty_with_arrow_schema(&schema);

    if let Some(row_index) = row_index {
        df.insert_column(0, Series::new_empty(row_index.name.clone(), &IDX_DTYPE))
            .unwrap();
    }

    materialize_hive_partitions(&mut df, reader_schema, hive_partition_columns, 0);

    df
}

pub(super) fn projected_arrow_schema_to_projection_indices(
    schema: &ArrowSchema,
    projected_arrow_schema: &ArrowSchema,
) -> PolarsResult<Option<Vec<usize>>> {
    let mut projection_indices = Vec::with_capacity(projected_arrow_schema.len());
    let mut is_full_ordered_projection = projected_arrow_schema.len() == schema.len();

    for (i, field) in projected_arrow_schema.iter_values().enumerate() {
        let dtype = {
            let Some((idx, _, field)) = schema.get_full(&field.name) else {
                polars_bail!(SchemaMismatch: "did not find column in file: {}", field.name)
            };

            projection_indices.push(idx);
            is_full_ordered_projection &= idx == i;

            DataType::from_arrow(&field.dtype, true)
        };
        let expected_dtype = DataType::from_arrow(&field.dtype, true);

        if dtype.clone() != expected_dtype {
            polars_bail!(SchemaMismatch: "data type mismatch for column {}: found: {}, expected: {}",
                &field.name, dtype, expected_dtype
            )
        }
    }

    Ok((!is_full_ordered_projection).then_some(projection_indices))
}
