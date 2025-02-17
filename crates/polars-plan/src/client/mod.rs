mod check;

use polars_core::error::PolarsResult;
use polars_utils::pl_serialize;

use crate::plans::DslPlan;

/// Prepare the given [`DslPlan`] for execution on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan) -> PolarsResult<Vec<u8>> {
    // Check the plan for cloud eligibility.
    check::assert_cloud_eligible(&dsl)?;

    // Serialize the plan.
    let mut writer = Vec::new();
    pl_serialize::SerializeOptions::default().serialize_into_writer(&mut writer, &dsl)?;

    Ok(writer)
}
