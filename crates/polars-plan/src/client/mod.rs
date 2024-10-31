mod check;

use arrow::legacy::error::to_compute_err;
use polars_core::error::PolarsResult;

use crate::plans::DslPlan;

/// Prepare the given [`DslPlan`] for execution on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan) -> PolarsResult<Vec<u8>> {
    // Check the plan for cloud eligibility.
    check::assert_cloud_eligible(&dsl)?;

    // Serialize the plan.
    let mut writer = Vec::new();
    ciborium::into_writer(&dsl, &mut writer).map_err(to_compute_err)?;

    Ok(writer)
}
