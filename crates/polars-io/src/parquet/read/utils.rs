use std::borrow::Cow;

use polars_core::prelude::{ArrowSchema, DataFrame, DataType, PlHashMap, Series, IDX_DTYPE};
use polars_error::{polars_bail, PolarsResult};
use polars_utils::pl_str::PlSmallStr;

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

/// Ensures that a parquet file has all the necessary columns for a projection with the correct
/// dtype. There are no ordering requirements and extra columns are permitted.
pub fn ensure_schema_has_projected_fields(
    schema: &ArrowSchema,
    projected_arrow_schema: &ArrowSchema,
) -> PolarsResult<()> {
    // Note: We convert to Polars-native dtypes for timezone normalization.
    let mut schema = schema
        .iter_values()
        .map(|x| {
            let dtype = DataType::from_arrow(&x.dtype, true);
            (x.name.clone(), dtype)
        })
        .collect::<PlHashMap<PlSmallStr, DataType>>();

    for field in projected_arrow_schema.iter_values() {
        let Some(dtype) = schema.remove(&field.name) else {
            polars_bail!(SchemaMismatch: "did not find column: {}", field.name)
        };

        let expected_dtype = DataType::from_arrow(&field.dtype, true);

        if dtype != expected_dtype {
            polars_bail!(SchemaMismatch: "data type mismatch for column {}: found: {}, expected: {}",
                &field.name, dtype, expected_dtype
            )
        }
    }

    Ok(())
}
