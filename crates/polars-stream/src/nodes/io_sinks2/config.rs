use std::sync::Arc;

use polars_core::prelude::SortMultipleOptions;
use polars_core::schema::SchemaRef;
use polars_plan::dsl::sink2::FileProviderType;
use polars_plan::dsl::{FileType, SinkTarget, UnifiedSinkArgs};
use polars_utils::plpath::{CloudScheme, PlPath};

use crate::expression::StreamExpr;
use crate::nodes::io_sinks2::components::hstack_columns::HStackColumns;
use crate::nodes::io_sinks2::components::partitioner::Partitioner;
use crate::nodes::io_sinks2::components::size::RowCountAndSize;

pub struct IOSinkNodeConfig {
    pub file_format: Arc<FileType>,
    pub target: IOSinkTarget,
    pub unified_sink_args: UnifiedSinkArgs,
    pub input_schema: SchemaRef,
    pub num_pipelines: usize,
}

impl IOSinkNodeConfig {
    pub fn per_sink_pipeline_depth(&self) -> usize {
        self.inflight_morsel_limit().min(self.num_pipelines)
    }

    pub fn inflight_morsel_limit(&self) -> usize {
        if let Ok(v) = std::env::var("POLARS_INFLIGHT_SINK_MORSEL_LIMIT").map(|x| {
            x.parse::<usize>()
                .ok()
                .filter(|x| *x > 0)
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_INFLIGHT_SINK_MORSEL_LIMIT: {x}")
                })
        }) {
            return v;
        };

        self.num_pipelines.saturating_add(
            // Additional buffer to accommodate head-of-line blocking
            4,
        )
    }

    pub fn max_open_sinks(&self) -> usize {
        if let Ok(v) = std::env::var("POLARS_MAX_OPEN_SINKS").map(|x| {
            x.parse::<usize>()
                .ok()
                .filter(|x| *x > 0)
                .unwrap_or_else(|| panic!("invalid value for POLARS_MAX_OPEN_SINKS: {x}"))
        }) {
            return v;
        }

        if self.target.is_cloud_location() {
            512
        } else {
            128
        }
    }

    pub fn cloud_upload_chunk_size(&self) -> usize {
        polars_io::get_upload_chunk_size()
    }

    pub fn partitioned_cloud_upload_chunk_size(&self) -> usize {
        if let Ok(v) = std::env::var("POLARS_PARTITIONED_UPLOAD_CHUNK_SIZE").map(|x| {
            x.parse::<usize>()
                .ok()
                .filter(|x| *x > 0)
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_PARTITIONED_UPLOAD_CHUNK_SIZE: {x}")
                })
        }) {
            return v;
        }

        6 * 1024 * 1024
    }
}

pub enum IOSinkTarget {
    File(SinkTarget),
    Partitioned(Box<PartitionedTarget>),
}

impl IOSinkTarget {
    pub fn is_cloud_location(&self) -> bool {
        match self {
            Self::File(v) => v.cloud_scheme(),
            Self::Partitioned(v) => v.base_path.cloud_scheme(),
        }
        .is_some_and(|x| !matches!(x, CloudScheme::File | CloudScheme::FileNoHostname))
    }
}

pub struct PartitionedTarget {
    pub base_path: PlPath,
    pub file_path_provider: FileProviderType,
    pub partitioner: Partitioner,
    /// How to hstack the keys back into the dataframe (with_columns)
    pub hstack_keys: Option<HStackColumns>,
    pub include_keys_in_file: bool,
    pub file_schema: SchemaRef,
    pub file_size_limit: Option<RowCountAndSize>,
    pub per_partition_sort: Option<(Arc<[StreamExpr]>, SortMultipleOptions)>,
}
