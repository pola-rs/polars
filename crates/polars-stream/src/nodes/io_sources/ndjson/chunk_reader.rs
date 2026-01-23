use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::ndjson;
use polars_io::prelude::{is_json_line, parse_ndjson};
#[cfg(feature = "scan_lines")]
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;

#[derive(Clone)]
pub enum ChunkReaderBuilder {
    NDJson {
        ignore_errors: bool,
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
            Self::NDJson { ignore_errors } => ChunkReader::NDJson {
                projected_schema,
                ignore_errors: *ignore_errors,
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
    pub(super) fn read_chunk(&self, chunk: &[u8]) -> PolarsResult<DataFrame> {
        match self {
            Self::NDJson {
                projected_schema,
                ignore_errors,
            } => {
                if projected_schema.is_empty() {
                    Ok(DataFrame::empty_with_height(ndjson::count_rows(chunk)))
                } else {
                    parse_ndjson(chunk, None, projected_schema, *ignore_errors)
                }
            },
            #[cfg(feature = "scan_lines")]
            Self::Lines { projection } => {
                use polars_core::prelude::IntoColumn;
                use polars_core::series::Series;
                use polars_io::scan_lines;

                let Some(name) = projection else {
                    return Ok(DataFrame::empty_with_height(scan_lines::count_lines(chunk)));
                };

                let out: Series = scan_lines::split_lines_to_rows(chunk)?.with_name(name.clone());
                let out = unsafe { DataFrame::new_unchecked(out.len(), vec![out.into_column()]) };

                Ok(out)
            },
        }
    }
}
