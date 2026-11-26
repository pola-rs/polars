use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::ndjson;
use polars_io::prelude::{ExtraKeysCheck, is_json_line, parse_ndjson};
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;

#[derive(Clone)]
pub enum ChunkReaderBuilder {
    NDJson {
        ignore_errors: bool,
        /// The full file schema.
        file_schema: SchemaRef,
        /// Whether keys outside `file_schema`, at any level, raise.
        raise_on_extra_keys: bool,
        /// The file named in errors.
        source: Option<PlSmallStr>,
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
    pub(super) fn build(&self, projected_schema: SchemaRef, sliced: bool) -> ChunkReader {
        match self {
            Self::NDJson {
                ignore_errors,
                file_schema,
                raise_on_extra_keys,
                source,
            } => {
                // A sliced read also parses the rows next to the slice, so it doesn't check.
                let raise = *raise_on_extra_keys && !sliced;
                // Unread columns aren't parsed: only reads of every column check the keys.
                let top_level = raise
                    && projected_schema.len() == file_schema.len()
                    && (projected_schema.iter_names()).all(|n| file_schema.contains(n));
                // Pruned struct columns aren't checked.
                let nested = (projected_schema.iter())
                    .map(|(name, dtype)| raise && file_schema.get(name) == Some(dtype))
                    .collect();
                let extra_keys = ExtraKeysCheck {
                    top_level,
                    nested,
                    source: source.clone(),
                };
                ChunkReader::NDJson {
                    projected_schema,
                    ignore_errors: *ignore_errors,
                    extra_keys: Arc::new(extra_keys),
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
