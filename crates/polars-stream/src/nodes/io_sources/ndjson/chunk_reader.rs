use std::sync::Arc;

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
    sub_json_path: Option<Arc<[String]>>,
    ignore_errors: bool,
}

impl ChunkReader {
    pub(super) fn try_new(
        options: &NDJsonReadOptions,
        projected_schema: &SchemaRef,
    ) -> PolarsResult<Self> {
        let projected_schema = projected_schema.clone();

        Ok(Self {
            projected_schema,
            sub_json_path: options.sub_json_path.clone(),
            ignore_errors: options.ignore_errors,
        })
    }

    pub(super) fn read_chunk(&self, chunk: &[u8]) -> PolarsResult<DataFrame> {
        if self.projected_schema.is_empty() {
            Ok(DataFrame::empty_with_height(ndjson::count_rows(chunk)))
        } else {
            parse_ndjson(
                chunk,
                None,
                &self.projected_schema,
                self.sub_json_path
                    .as_ref()
                    .map_or(&[] as &[String], |p| &**p),
                self.ignore_errors,
            )
        }
    }
}
