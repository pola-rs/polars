use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::ndjson;
use polars_io::prelude::parse_ndjson;
#[cfg(feature = "scan_lines")]
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;

#[derive(Clone)]
pub enum ChunkReaderBuilder {
    NDJson {
        ignore_errors: bool,
    },
    #[cfg(feature = "scan_lines")]
    #[expect(unused)]
    Lines {
        name: PlSmallStr,
    },
}

#[derive(Clone)]
pub enum ChunkReader {
    /// NDJSON chunk reader.
    NDJson {
        projected_schema: SchemaRef,
        ignore_errors: bool,
    },
    #[cfg(feature = "scan_lines")]
    #[expect(unused)]
    Lines {
        /// If this is `None` we are projecting 0-width morsels.
        projection: Option<PlSmallStr>,
    },
}

impl ChunkReaderBuilder {
    pub(super) fn max_chunk_size(&self) -> usize {
        match self {
            Self::NDJson { .. } => usize::MAX,
            #[cfg(feature = "scan_lines")]
            Self::Lines { .. } => u32::MAX.try_into().unwrap(),
        }
    }

    pub(super) fn build(&self, projected_schema: SchemaRef) -> ChunkReader {
        match self {
            Self::NDJson { ignore_errors } => ChunkReader::NDJson {
                projected_schema,
                ignore_errors: *ignore_errors,
            },
            #[cfg(feature = "scan_lines")]
            Self::Lines { name } => {
                use polars_core::prelude::DataType;

                assert!(projected_schema.len() <= 1);

                let projection = projected_schema
                    .get_at_index(0)
                    .map(|(projected_name, dtype)| {
                        assert_eq!(projected_name, name);
                        assert!(matches!(dtype, DataType::String));

                        name.clone()
                    });

                ChunkReader::Lines { projection }
            },
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
            Self::Lines { projection: _ } => {
                todo!()
            },
        }
    }
}
