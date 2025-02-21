mod check;

use polars_core::error::PolarsResult;

use crate::dsl::DslPlan;

/// Prepare the given [`DslPlan`] for execution on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan) -> PolarsResult<Vec<u8>> {
    // Check the plan for cloud eligibility.
    check::assert_cloud_eligible(&dsl)?;

    // Serialize the plan.
    let mut writer = Vec::new();
    dsl.serialize_versioned(&mut writer)?;

    Ok(writer)
}
