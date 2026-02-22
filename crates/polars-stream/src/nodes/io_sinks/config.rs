use std::num::NonZeroUsize;

use polars_core::schema::SchemaRef;
use polars_plan::dsl::file_provider::FileProviderType;
use polars_plan::dsl::{FileWriteFormat, SinkTarget, UnifiedSinkArgs};
use polars_utils::pl_path::{CloudScheme, PlRefPath};

use crate::nodes::io_sinks::components::hstack_columns::HStackColumns;
use crate::nodes::io_sinks::components::partitioner::Partitioner;
use crate::nodes::io_sinks::components::size::NonZeroRowCountAndSize;

pub struct IOSinkNodeConfig {
    pub file_format: FileWriteFormat,
    pub target: IOSinkTarget,
    pub unified_sink_args: UnifiedSinkArgs,
    pub input_schema: SchemaRef,
}

impl IOSinkNodeConfig {
    pub fn num_pipelines_per_sink(&self, num_pipelines: NonZeroUsize) -> NonZeroUsize {
        NonZeroUsize::min(num_pipelines, self.inflight_morsel_limit(num_pipelines))
    }

    pub fn inflight_morsel_limit(&self, num_pipelines: NonZeroUsize) -> NonZeroUsize {
        if let Ok(v) = std::env::var("POLARS_INFLIGHT_SINK_MORSEL_LIMIT").map(|x| {
            x.parse::<NonZeroUsize>().unwrap_or_else(|_| {
                panic!("invalid value for POLARS_INFLIGHT_SINK_MORSEL_LIMIT: {x}")
            })
        }) {
            return v;
        };

        NonZeroUsize::saturating_add(
            num_pipelines,
            // Additional buffer to accommodate head-of-line blocking
            4,
        )
    }

    pub fn max_open_sinks(&self) -> NonZeroUsize {
        if let Ok(v) = std::env::var("POLARS_MAX_OPEN_SINKS").map(|x| {
            x.parse::<NonZeroUsize>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_MAX_OPEN_SINKS: {x}"))
        }) {
            return v;
        }

        if self.target.is_cloud_location() {
            const { NonZeroUsize::new(512).unwrap() }
        } else {
            const { NonZeroUsize::new(128).unwrap() }
        }
    }

    pub fn cloud_upload_chunk_size(&self) -> usize {
        polars_io::configs::upload_chunk_size()
    }

    pub fn partitioned_upload_chunk_size(&self) -> usize {
        polars_io::configs::partitioned_upload_chunk_size()
    }

    pub fn upload_concurrency(&self) -> NonZeroUsize {
        polars_io::configs::upload_concurrency()
    }

    pub fn partitioned_upload_concurrency(&self) -> NonZeroUsize {
        polars_io::configs::partitioned_upload_concurrency()
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
            Self::Partitioned(v) => v.base_path.scheme(),
        }
        .is_some_and(|x| !matches!(x, CloudScheme::File | CloudScheme::FileNoHostname))
    }
}

pub struct PartitionedTarget {
    pub base_path: PlRefPath,
    pub file_path_provider: FileProviderType,
    pub partitioner: Partitioner,
    /// How to hstack the keys back into the dataframe (with_columns)
    pub hstack_keys: Option<HStackColumns>,
    pub include_keys_in_file: bool,
    pub file_schema: SchemaRef,
    pub file_size_limit: Option<NonZeroRowCountAndSize>,
}
