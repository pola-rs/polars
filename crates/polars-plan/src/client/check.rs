use polars_core::error::{polars_err, PolarsResult};
use polars_io::path_utils::is_cloud_url;

use crate::plans::{DslPlan, FileScan, ScanSources};

/// Assert that the given [`DslPlan`] is eligible to be executed on Polars Cloud.
pub(super) fn assert_cloud_eligible(dsl: &DslPlan) -> PolarsResult<()> {
    if std::env::var("POLARS_SKIP_CLIENT_CHECK").as_deref() == Ok("1") {
        return Ok(());
    }
    for plan_node in dsl.into_iter() {
        match plan_node {
            #[cfg(feature = "python")]
            DslPlan::PythonScan { .. } => return ineligible_error("contains Python scan"),
            DslPlan::GroupBy { apply, .. } if apply.is_some() => {
                return ineligible_error("contains map groups")
            },
            DslPlan::Scan {
                sources, scan_type, ..
            } => {
                match sources {
                    ScanSources::Paths(paths) => {
                        if paths.iter().any(|p| !is_cloud_url(p)) {
                            return ineligible_error("contains scan of local file system");
                        }
                    },
                    ScanSources::Files(_) => {
                        return ineligible_error("contains scan of opened files");
                    },
                    ScanSources::Buffers(_) => {
                        return ineligible_error("contains scan of in-memory buffer");
                    },
                }

                if matches!(scan_type, FileScan::Anonymous { .. }) {
                    return ineligible_error("contains anonymous scan");
                }
            },
            DslPlan::Sink { payload, .. } => {
                if !payload.is_cloud_destination() {
                    return ineligible_error("contains sink to non-cloud location");
                }
            },
            _ => (),
        }
    }
    Ok(())
}

fn ineligible_error(message: &str) -> PolarsResult<()> {
    Err(polars_err!(
        InvalidOperation:
        "logical plan ineligible for execution on Polars Cloud: {message}"
    ))
}

impl DslPlan {
    fn inputs<'a>(&'a self, scratch: &mut Vec<&'a DslPlan>) {
        use DslPlan::*;
        match self {
            Select { input, .. }
            | GroupBy { input, .. }
            | Filter { input, .. }
            | Distinct { input, .. }
            | Sort { input, .. }
            | Slice { input, .. }
            | HStack { input, .. }
            | MapFunction { input, .. }
            | Sink { input, .. }
            | Cache { input, .. } => scratch.push(input),
            Union { inputs, .. } | HConcat { inputs, .. } => scratch.extend(inputs),
            Join {
                input_left,
                input_right,
                ..
            } => {
                scratch.push(input_left);
                scratch.push(input_right);
            },
            ExtContext { input, contexts } => {
                scratch.push(input);
                scratch.extend(contexts);
            },
            IR { dsl, .. } => scratch.push(dsl),
            Scan { .. } | DataFrameScan { .. } => (),
            #[cfg(feature = "python")]
            PythonScan { .. } => (),
        }
    }
}

pub struct DslPlanIter<'a> {
    stack: Vec<&'a DslPlan>,
}

impl<'a> Iterator for DslPlanIter<'a> {
    type Item = &'a DslPlan;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack
            .pop()
            .inspect(|next| next.inputs(&mut self.stack))
    }
}

impl<'a> IntoIterator for &'a DslPlan {
    type Item = &'a DslPlan;
    type IntoIter = DslPlanIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DslPlanIter { stack: vec![self] }
    }
}
