mod check;

use std::sync::Arc;

use polars_core::error::{polars_ensure, polars_err, PolarsResult};
use polars_io::parquet::write::ParquetWriteOptions;
use polars_io::path_utils::is_cloud_url;

use crate::plans::options::{FileType, SinkType};
use crate::plans::DslPlan;

/// Prepare the given [`DslPlan`] for execution on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan, uri: String) -> PolarsResult<Vec<u8>> {
    // Check the plan for cloud eligibility.
    check::assert_cloud_eligible(&dsl)?;

    // Add Sink node.
    polars_ensure!(
        is_cloud_url(&uri),
        InvalidOperation: "non-cloud paths not supported: {uri}"
    );
    let sink_type = SinkType::Cloud {
        uri: Arc::new(uri),
        file_type: FileType::Parquet(ParquetWriteOptions::default()),
        cloud_options: None,
    };
    let dsl = DslPlan::Sink {
        input: Arc::new(dsl),
        payload: sink_type,
    };

    // Serialize the plan.
    let mut writer = Vec::new();
    ciborium::into_writer(&dsl, &mut writer)
        .map_err(|err| polars_err!(ComputeError: err.to_string()))?;

    Ok(writer)
}
