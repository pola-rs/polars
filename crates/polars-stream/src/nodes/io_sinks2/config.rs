use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_plan::dsl::{FileType, UnifiedSinkArgs};

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

        128
    }
}

pub enum IOSinkTarget {
    Partitioned {
        // TODO
    },
}
