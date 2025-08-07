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
use crate::dsl::{DslPlan, FileScanIR, UnifiedScanArgs};
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
        // # Note
        // This function mutates the IR node in-place rather than returning the new IR - the
        // StackOptimizer will re-call this function otherwise.
        if let IR::Scan {
            sources,
            scan_type,
            unified_scan_args,

            file_info: _,
            hive_parts: _,
            predicate: _,
            output_schema: _,
        } = lp_arena.get_mut(node)
        {
            let projection = unified_scan_args.projection.clone();
            let limit = match unified_scan_args.pre_slice.clone() {
                Some(v @ Slice::Positive { .. }) => Some(v.end_position()),
                _ => None,
            };

            match scan_type.as_mut() {
                #[cfg(feature = "python")]
                FileScanIR::PythonDataset {
                    dataset_object,
                    cached_ir,
                } => {
                    let cached_ir = cached_ir.clone();
                    let mut guard = cached_ir.lock().unwrap();

                    if config::verbose() {
                        eprintln!(
                            "expand_datasets(): python[{}]: limit: {:?}, project: {}",
                            dataset_object.name(),
                            limit,
                            projection.as_ref().map_or(
                                PlSmallStr::from_static("all"),
                                |x| format_pl_smallstr!("{}", x.len())
                            )
                        )
                    }

                    let existing_resolved_version_key = match guard.as_ref() {
                        Some(resolved) => {
                            let ExpandedDataset {
                                version,
                                limit: cached_limit,
                                projection: cached_projection,
                                expanded_dsl: _,
                                python_scan: _,
                            } = resolved;

                            (cached_limit == &limit && cached_projection == &projection)
                                .then_some(version.as_str())
                        },

                        None => None,
                    };

                    if let Some((expanded_dsl, version)) = dataset_object.to_dataset_scan(
                        existing_resolved_version_key,
                        limit,
                        projection.as_deref(),
                    )? {
                        *guard = Some(ExpandedDataset {
                            version,
                            limit,
                            projection,
                            expanded_dsl,
                            python_scan: None,
                        })
                    }

                    let ExpandedDataset {
                        version: _,
                        limit: _,
                        projection: _,
                        expanded_dsl,
                        python_scan,
                    } = guard.as_mut().unwrap();

                    match expanded_dsl {
                        DslPlan::Scan {
                            sources: resolved_sources,
                            unified_scan_args: resolved_unified_scan_args,
                            scan_type: resolved_scan_type,
                            cached_ir: _,
                        } => {
                            use crate::dsl::FileScanDsl;

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
                                column_mapping,
                                default_values,
                                row_index: _row_index @ None,
                                pre_slice: _pre_slice @ None,
                                cast_columns_policy,
                                missing_columns_policy,
                                extra_columns_policy,
                                include_file_paths: _include_file_paths @ None,
                                deletion_files,
                            } = resolved_unified_scan_args.as_ref()
                            else {
                                panic!(
                                    "invalid scan args from python dataset resolve: {:?}",
                                    &resolved_unified_scan_args
                                )
                            };

                            unified_scan_args.cloud_options = cloud_options.clone();
                            unified_scan_args.rechunk = *rechunk;
                            unified_scan_args.cache = *cache;
                            unified_scan_args.cast_columns_policy = cast_columns_policy.clone();
                            unified_scan_args.missing_columns_policy = *missing_columns_policy;
                            unified_scan_args.extra_columns_policy = *extra_columns_policy;
                            unified_scan_args.column_mapping = column_mapping.clone();
                            unified_scan_args.default_values = default_values.clone();
                            unified_scan_args.deletion_files = deletion_files.clone();

                            *sources = resolved_sources.clone();

                            *scan_type = Box::new(match *resolved_scan_type.clone() {
                                #[cfg(feature = "csv")]
                                FileScanDsl::Csv { options } => FileScanIR::Csv { options },

                                #[cfg(feature = "ipc")]
                                FileScanDsl::Ipc { options } => FileScanIR::Ipc {
                                    options,
                                    metadata: None,
                                },

                                #[cfg(feature = "parquet")]
                                FileScanDsl::Parquet { options } => FileScanIR::Parquet {
                                    options,
                                    metadata: None,
                                },

                                #[cfg(feature = "json")]
                                FileScanDsl::NDJson { options } => FileScanIR::NDJson { options },

                                #[cfg(feature = "python")]
                                FileScanDsl::PythonDataset { dataset_object } => {
                                    FileScanIR::PythonDataset {
                                        dataset_object,
                                        cached_ir: Default::default(),
                                    }
                                },

                                FileScanDsl::Anonymous {
                                    options,
                                    function,
                                    file_info: _,
                                } => FileScanIR::Anonymous { options, function },
                            });
                        },

                        DslPlan::PythonScan { options } => {
                            *python_scan = Some(ExpandedPythonScan {
                                name: dataset_object.name(),
                                scan_fn: options.scan_fn.clone().unwrap(),
                                variant: options.python_source.clone(),
                            })
                        },

                        dsl => {
                            polars_bail!(
                                ComputeError:
                                "unknown DSL when resolving python dataset scan: {}",
                                dsl.display()?
                            )
                        },
                    };
                },

                _ => {},
            }
        }

        Ok(None)
    }
}

#[derive(Clone)]
pub struct ExpandedDataset {
    version: PlSmallStr,
    limit: Option<usize>,
    projection: Option<Arc<[PlSmallStr]>>,
    expanded_dsl: DslPlan,

    /// Fallback python scan
    #[cfg(feature = "python")]
    python_scan: Option<ExpandedPythonScan>,
}

#[cfg(feature = "python")]
#[derive(Clone)]
pub struct ExpandedPythonScan {
    pub name: PlSmallStr,
    pub scan_fn: PythonObject,
    pub variant: PythonScanSource,
}

impl ExpandedDataset {
    #[cfg(feature = "python")]
    pub fn python_scan(&self) -> Option<&ExpandedPythonScan> {
        self.python_scan.as_ref()
    }
}

impl Debug for ExpandedDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ExpandedDataset {
            version,
            limit,
            projection,
            expanded_dsl,

            #[cfg(feature = "python")]
            python_scan,
        } = self;

        return display::ExpandedDataset {
            version,
            limit,
            projection,
            expanded_dsl: &match expanded_dsl.display() {
                Ok(v) => v.to_string(),
                Err(e) => e.to_string(),
            },
            #[cfg(feature = "python")]
            python_scan: python_scan.as_ref().map(
                |ExpandedPythonScan {
                     name,
                     scan_fn: _,
                     variant,
                 }| {
                    format_pl_smallstr!("python-scan[{} @ {:?}]", name, variant)
                },
            ),
        }
        .fmt(f);

        mod display {
            use std::fmt::Debug;
            use std::sync::Arc;

            use polars_utils::pl_str::PlSmallStr;

            #[derive(Debug)]
            #[expect(unused)]
            pub struct ExpandedDataset<'a> {
                pub version: &'a str,
                pub limit: &'a Option<usize>,
                pub projection: &'a Option<Arc<[PlSmallStr]>>,
                pub expanded_dsl: &'a str,

                #[cfg(feature = "python")]
                pub python_scan: Option<PlSmallStr>,
            }
        }
    }
}
