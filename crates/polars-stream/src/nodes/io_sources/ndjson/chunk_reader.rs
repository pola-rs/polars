use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::ndjson;
use polars_io::prelude::{ExtraKeysCheck, ExtraKeysHint, is_json_line, parse_ndjson};
use polars_plan::dsl::ExtraColumnsPolicy;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;

#[derive(Clone)]
pub enum ChunkReaderBuilder {
    NDJson {
        ignore_errors: bool,
        /// The full file schema.
        file_schema: Option<SchemaRef>,
        source: Option<PlSmallStr>,
        hint: ExtraKeysHint,
        /// Whether top-level keys outside `file_schema` raise. Taken from the scan, since
        /// internal row-count reads pass default per-read args.
        extra_columns_policy: ExtraColumnsPolicy,
        /// Whether struct fields outside a column's dtype raise.
        extra_struct_fields_policy: ExtraColumnsPolicy,
    },
    #[cfg(feature = "scan_lines")]
    Lines,
}

#[derive(Clone)]
pub enum ChunkReader {
    /// NDJSON chunk reader.
    NDJson {
        projected_schema: SchemaRef,
        ignore_errors: bool,
        extra_keys: Arc<ExtraKeysCheck>,
    },
    #[cfg(feature = "scan_lines")]
    Lines {
        /// If this is `None` we are projecting 0-width morsels.
        projection: Option<PlSmallStr>,
    },
}

impl ChunkReaderBuilder {
    pub(super) fn build(&self, projected_schema: SchemaRef) -> ChunkReader {
        match self {
            Self::NDJson {
                ignore_errors,
                file_schema,
                source,
                hint,
                extra_columns_policy,
                extra_struct_fields_policy,
            } => {
                let raise = |policy| matches!(policy, ExtraColumnsPolicy::Raise);
                let file_schema = file_schema.as_ref().unwrap_or(&projected_schema);
                // Unread columns aren't parsed: only reads of every column check the keys.
                let all_columns = projected_schema.len() == file_schema.len()
                    && projected_schema
                        .iter_names()
                        .all(|n| file_schema.contains(n));
                let top_level = raise(*extra_columns_policy) && all_columns;
                // Pruned struct columns aren't checked.
                let nested = if raise(*extra_struct_fields_policy) {
                    (projected_schema.iter())
                        .map(|(name, dtype)| file_schema.get(name) == Some(dtype))
                        .collect()
                } else {
                    Vec::new()
                };
                ChunkReader::NDJson {
                    projected_schema,
                    ignore_errors: *ignore_errors,
                    extra_keys: Arc::new(ExtraKeysCheck {
                        top_level,
                        nested,
                        source: source.clone(),
                        hint: *hint,
                    }),
                }
            },
            #[cfg(feature = "scan_lines")]
            Self::Lines => {
                use polars_core::prelude::DataType;

                assert!(projected_schema.len() <= 1);

                let projection = projected_schema
                    .get_at_index(0)
                    .map(|(projected_name, dtype)| {
                        assert!(matches!(dtype, DataType::String));
                        projected_name.clone()
                    });

                ChunkReader::Lines { projection }
            },
        }
    }

    pub(super) fn is_line_fn(&self) -> fn(&[u8]) -> bool {
        match self {
            Self::NDJson { .. } => is_json_line,
            #[cfg(feature = "scan_lines")]
            Self::Lines { .. } => |_: &[u8]| true,
        }
    }
}

impl ChunkReader {
    pub(super) fn read_chunk(&self, chunk: Buffer<u8>) -> PolarsResult<DataFrame> {
        match self {
            Self::NDJson {
                projected_schema,
                ignore_errors,
                extra_keys,
            } => {
                if projected_schema.is_empty() && !extra_keys.top_level {
                    Ok(DataFrame::empty_with_height(ndjson::count_rows(&chunk)))
                } else {
                    parse_ndjson(&chunk, None, projected_schema, *ignore_errors, extra_keys)
                }
            },
            #[cfg(feature = "scan_lines")]
            Self::Lines { projection } => {
                use polars_core::prelude::IntoColumn;
                use polars_core::series::Series;
                use polars_io::scan_lines;

                let Some(name) = projection else {
                    return Ok(DataFrame::empty_with_height(scan_lines::count_lines(
                        &chunk,
                    )));
                };

                let out: Series = scan_lines::split_lines_to_rows(chunk)?.with_name(name.clone());
                let out = unsafe { DataFrame::new_unchecked(out.len(), vec![out.into_column()]) };

                Ok(out)
            },
        }
    }
}
