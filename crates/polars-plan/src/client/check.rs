use polars_core::error::{PolarsResult, polars_err};
use polars_io::path_utils::is_cloud_url;

use crate::constants::POLARS_PLACEHOLDER;
use crate::dsl::{DslPlan, FileScan, ScanSources, SinkType};

/// Assert that the given [`DslPlan`] is eligible to be executed on Polars Cloud.
pub(super) fn assert_cloud_eligible(dsl: &DslPlan) -> PolarsResult<()> {
    if std::env::var("POLARS_SKIP_CLIENT_CHECK").as_deref() == Ok("1") {
        return Ok(());
    }

    // Check that the plan ends with a sink.
    if !matches!(dsl, DslPlan::Sink { .. }) {
        return ineligible_error("does not contain a sink");
    }

    for plan_node in dsl.into_iter() {
        match plan_node {
            #[cfg(feature = "python")]
            DslPlan::PythonScan { .. } => (),
            DslPlan::GroupBy { apply, .. } if apply.is_some() => {
                return ineligible_error("contains map groups");
            },
            DslPlan::Scan {
                sources, scan_type, ..
            } => {
                match sources {
                    ScanSources::Paths(paths) => {
                        if paths
                            .iter()
                            .any(|p| !is_cloud_url(p) && p.to_str() != Some(POLARS_PLACEHOLDER))
                        {
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

                if matches!(&**scan_type, FileScan::Anonymous { .. }) {
                    return ineligible_error("contains anonymous scan");
                }
            },
            DslPlan::Sink { payload, .. } => {
                match payload {
                    SinkType::Memory => {
                        return ineligible_error("contains memory sink");
                    },
                    SinkType::File(_) => {
                        // The sink destination is passed around separately, can't check the
                        // eligibility here.
                    },
                    SinkType::Partition(_) => {
                        return ineligible_error("contains partition sink");
                    },
                }
            },
            DslPlan::SinkMultiple { .. } => {
                return ineligible_error("contains sink multiple");
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
            | MatchToSchema { input, .. }
            | MapFunction { input, .. }
            | Sink { input, .. }
            | Cache { input, .. } => scratch.push(input),
            Union { inputs, .. } | HConcat { inputs, .. } | SinkMultiple { inputs } => {
                scratch.extend(inputs)
            },
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
            #[cfg(feature = "merge_sorted")]
            MergeSorted {
                input_left,
                input_right,
                ..
            } => {
                scratch.push(input_left);
                scratch.push(input_right);
            },
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
