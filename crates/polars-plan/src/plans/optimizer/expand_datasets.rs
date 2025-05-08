use std::fmt::Debug;
use std::sync::Arc;

use polars_core::config;
use polars_core::error::{PolarsResult, polars_bail};
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
use polars_utils::slice_enum::Slice;

use super::OptimizationRule;
#[cfg(feature = "python")]
use crate::dsl::python_dsl::PythonScanSource;
use crate::dsl::{DslPlan, FileScan, UnifiedScanArgs};
use crate::plans::IR;

/// Note: Currently only used for iceberg. This is so that we can call iceberg to fetch the files
/// list with a potential row limit from slice pushdown.
///
/// In the future this can also apply to hive path expansion with predicates.
pub(super) struct ExpandDatasets;

impl OptimizationRule for ExpandDatasets {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        _expr_arena: &mut Arena<crate::prelude::AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let ir = lp_arena.get(node);

        if let IR::Scan {
            scan_type,
            unified_scan_args,
            ..
        } = ir
        {
            let projection = unified_scan_args.projection.clone();
            let limit = match unified_scan_args.pre_slice.clone() {
                Some(v @ Slice::Positive { .. }) => Some(v.end_position()),
                _ => None,
            };

            match scan_type.as_ref() {
                #[cfg(feature = "python")]
                FileScan::PythonDataset {
                    dataset_object,
                    cached_ir,
                } => {
                    let cached_ir = cached_ir.clone();

                    let mut guard = cached_ir.lock().unwrap();

                    // Note: We always get called twice in succession from the stack optimizer,
                    // as it was designed to optimize until fixed point. Ensure we return
                    // Ok(None) if the mutex contains the initialized state.
                    if match guard.as_ref() {
                        // Reject cached if limit or projection does not match. This can happen if a scan is reused.
                        Some(resolved) => {
                            let ExpandedDataset {
                                limit: cached_limit,
                                projection: cached_projection,
                                resolved_ir: _,
                                python_scan: _,
                            } = resolved;

                            cached_limit == &limit && cached_projection == &projection
                        },

                        None => false,
                    } {
                        return Ok(None);
                    }

                    if config::verbose() {
                        eprintln!(
                            "expand_datasets(): python[{}]: limit: {:?}, project: {}",
                            dataset_object.reader_name(),
                            limit,
                            projection.as_ref().map_or(
                                PlSmallStr::from_static("all"),
                                |x| format_pl_smallstr!("{}", x.len())
                            )
                        )
                    }

                    let plan = dataset_object.to_dataset_scan(limit, projection.as_deref())?;

                    let (resolved_ir, python_scan) = match plan {
                        DslPlan::Scan {
                            sources: resolved_sources,
                            file_info: _,
                            unified_scan_args: resolved_unified_scan_args,
                            scan_type: resolved_scan_type,
                            cached_ir: _,
                        } => {
                            let mut ir = ir.clone();

                            let IR::Scan {
                                sources,
                                scan_type,
                                unified_scan_args,

                                file_info: _,
                                hive_parts: _,
                                predicate: _,
                                output_schema: _,
                                id: _,
                            } = &mut ir
                            else {
                                unreachable!()
                            };

                            // We only want a few configuration flags from here (e.g. column casting config).
                            // The rest we either expect to be None (e.g. projection / row_index), or ignore.
                            let UnifiedScanArgs {
                                schema: _,
                                cloud_options,
                                hive_options: _,
                                rechunk,
                                cache,
                                glob: _,
                                projection: _projection @ None,
                                row_index: _row_index @ None,
                                pre_slice: _pre_slice @ None,
                                cast_columns_policy,
                                missing_columns_policy,
                                include_file_paths: _include_file_paths @ None,
                            } = *resolved_unified_scan_args
                            else {
                                panic!(
                                    "invalid scan args from python dataset resolve: {:?}",
                                    &resolved_unified_scan_args
                                )
                            };

                            unified_scan_args.cloud_options = cloud_options;
                            unified_scan_args.rechunk = rechunk;
                            unified_scan_args.cache = cache;
                            unified_scan_args.cast_columns_policy = cast_columns_policy;
                            unified_scan_args.missing_columns_policy = missing_columns_policy;

                            *sources = resolved_sources;
                            *scan_type = resolved_scan_type;

                            (ir, None)
                        },

                        DslPlan::PythonScan { options } => {
                            assert!(options.scan_fn.is_some());

                            (
                                ir.clone(),
                                Some((
                                    dataset_object.reader_name(),
                                    options.scan_fn.expect("scan_fn is required"),
                                    options.python_source,
                                )),
                            )
                        },

                        dsl => {
                            polars_bail!(
                                ComputeError:
                                "unknown DSL when resolving python dataset scan: {}",
                                dsl.display()?
                            )
                        },
                    };

                    let resolved = ExpandedDataset {
                        limit,
                        projection,
                        resolved_ir,
                        python_scan,
                    };

                    *guard = Some(resolved);

                    let resolved_ir = guard.as_ref().map(|x| x.resolved_ir.clone()).unwrap();

                    return Ok(Some(resolved_ir));
                },

                _ => {},
            }
        }
        Ok(None)
    }
}

#[derive(Clone)]
pub struct ExpandedDataset {
    limit: Option<usize>,
    projection: Option<Arc<[PlSmallStr]>>,
    resolved_ir: IR,

    /// Fallback python scan
    #[cfg(feature = "python")]
    python_scan: Option<(PlSmallStr, PythonObject, PythonScanSource)>,
}

impl ExpandedDataset {
    #[cfg(feature = "python")]
    pub fn python_scan(&self) -> Option<(&PlSmallStr, &PythonObject, &PythonScanSource)> {
        self.python_scan.as_ref().map(|(a, b, c)| (a, b, c))
    }
}

impl Debug for ExpandedDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ExpandedDataset {
            limit,
            projection,
            resolved_ir,

            #[cfg(feature = "python")]
            python_scan,
        } = self;

        return display::ExpandedDataset {
            limit,
            projection,
            resolved_ir,

            #[cfg(feature = "python")]
            python_scan: python_scan.as_ref().map(|(name, _, scan_type)| {
                format_pl_smallstr!("python-scan[{} @ {:?}]", name, scan_type)
            }),
        }
        .fmt(f);

        mod display {
            use std::sync::Arc;

            use polars_utils::pl_str::PlSmallStr;

            use crate::prelude::IR;

            #[derive(Debug)]
            #[expect(unused)]
            pub struct ExpandedDataset<'a> {
                pub limit: &'a Option<usize>,
                pub projection: &'a Option<Arc<[PlSmallStr]>>,
                pub resolved_ir: &'a IR,

                #[cfg(feature = "python")]
                pub python_scan: Option<PlSmallStr>,
            }
        }
    }
}
