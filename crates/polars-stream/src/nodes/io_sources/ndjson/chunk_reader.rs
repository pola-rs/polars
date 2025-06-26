use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::ndjson;
use polars_io::prelude::parse_ndjson;
use polars_plan::dsl::NDJsonReadOptions;

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
        projected_schema: &SchemaRef,
    ) -> PolarsResult<Self> {
        let projected_schema = projected_schema.clone();

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
        if self.projected_schema.is_empty() {
            Ok(DataFrame::empty_with_height(ndjson::count_rows(chunk)))
        } else {
            parse_ndjson(chunk, None, &self.projected_schema, self.ignore_errors)
        }
    }
}
