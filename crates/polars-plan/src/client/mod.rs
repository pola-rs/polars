mod check;

use polars_core::error::PolarsResult;

use crate::dsl::{DslPlan, PlanSerializationContext};

/// Prepare the given [`DslPlan`] for execution on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan, allow_local_scans: bool) -> PolarsResult<Vec<u8>> {
    // Check the plan for cloud eligibility.
    check::assert_cloud_eligible(&dsl, allow_local_scans)?;

    // Serialize the plan.
    let mut writer = Vec::new();
    dsl.serialize_versioned(
        &mut writer,
        PlanSerializationContext {
            use_cloudpickle: true,
        },
    )?;

    Ok(writer)
}
