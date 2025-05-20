use std::sync::Arc;

use polars_core::error::PolarsResult;

use crate::dsl::{DslPlan, IcebergWriteMode, SinkType};

pub fn iceberg_dataset_to_dsl(
    input: DslPlan,
    dataset: &crate::dsl::python_dataset::PythonDatasetProvider,
    cloud_options: Option<polars_io::cloud::CloudOptions>,
    mode: IcebergWriteMode,
) -> PolarsResult<(Arc<DslPlan>, SinkType)> {
    match dataset.to_dataset_sink(input, mode)? {
        DslPlan::Sink { input, mut payload } => {
            match &mut payload {
                SinkType::Partition(f) => f.cloud_options = cloud_options,
                _ => unreachable!(),
            }

            Ok((input, payload))
        },
        _ => unreachable!(),
    }
}
