use std::borrow::Cow;

use polars_core::prelude::{ArrowSchema, DataFrame, Series, IDX_DTYPE};

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
        df.insert_column(0, Series::new_empty(&row_index.name, &IDX_DTYPE))
            .unwrap();
    }

    materialize_hive_partitions(&mut df, reader_schema, hive_partition_columns, 0);

    df
}
