use std::sync::Arc;

use polars_core::schema::{SchemaExt, SchemaRef};
use polars_error::PolarsResult;
use polars_io::prelude::parse_ndjson;
use polars_plan::dsl::NDJsonReadOptions;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;

/// NDJSON chunk reader.
#[derive(Default)]
pub(super) struct ChunkReader {
    projected_schema: SchemaRef,
    #[cfg(feature = "dtype-categorical")]
    _cat_lock: Option<polars_core::StringCacheHolder>,
    ignore_errors: bool,
}

impl ChunkReader {
    pub(super) fn try_new(
        options: &NDJsonReadOptions,
        reader_schema: &SchemaRef,
        with_columns: Option<&[PlSmallStr]>,
    ) -> PolarsResult<Self> {
        let projected_schema: SchemaRef = if let Some(cols) = with_columns {
            Arc::new(
                cols.iter()
                    .map(|x| reader_schema.try_get_field(x))
                    .collect::<PolarsResult<_>>()?,
            )
        } else {
            reader_schema.clone()
        };

        #[cfg(feature = "dtype-categorical")]
        let _cat_lock = projected_schema
            .iter_values()
            .any(|x| x.is_categorical())
            .then(polars_core::StringCacheHolder::hold);

        Ok(Self {
            projected_schema,
            #[cfg(feature = "dtype-categorical")]
            _cat_lock,
            ignore_errors: options.ignore_errors,
        })
    }

    pub(super) fn read_chunk(&self, chunk: &[u8]) -> PolarsResult<DataFrame> {
        parse_ndjson(chunk, None, &self.projected_schema, self.ignore_errors)
    }
}
