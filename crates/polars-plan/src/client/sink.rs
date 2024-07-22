use std::sync::Arc;

use polars_io::parquet::write::ParquetWriteOptions;

use crate::plans::options::{FileType, SinkType};
use crate::plans::DslPlan;

/// Add a `Sink` node to the [`DslPlan`].
///
/// Currently only supports Parquet with default options.
pub(super) fn add_sink(dsl: DslPlan, uri: String) -> DslPlan {
    let sink_type = SinkType::Cloud {
        uri: Arc::new(uri),
        file_type: FileType::Parquet(ParquetWriteOptions::default()),
        cloud_options: None,
    };
    DslPlan::Sink {
        input: Arc::new(dsl),
        payload: sink_type,
    }
}
