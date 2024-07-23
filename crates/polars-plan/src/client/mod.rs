mod check;
mod sink;

use polars_core::error::PolarsResult;

use crate::plans::DslPlan;

/// Prepare the given [`DslPlan`] f executed on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan, uri: String) -> PolarsResult<DslPlan> {
    let dsl = sink::add_sink(dsl, uri);
    check::assert_cloud_eligible(&dsl)?;

    // TODO: Serialize to binary.

    Ok(dsl)
}
