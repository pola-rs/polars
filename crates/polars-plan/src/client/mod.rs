mod check;
mod sink;

use polars_core::error::{polars_err, PolarsResult};

use crate::plans::DslPlan;

/// Prepare the given [`DslPlan`] for execution on Polars Cloud.
pub fn prepare_cloud_plan(dsl: DslPlan, uri: String) -> PolarsResult<Vec<u8>> {
    let dsl = sink::add_sink(dsl, uri);
    check::assert_cloud_eligible(&dsl)?;

    let mut writer = Vec::new();
    ciborium::into_writer(&dsl, &mut writer)
        .map_err(|err| polars_err!(ComputeError: err.to_string()))?;

    Ok(writer)
}
