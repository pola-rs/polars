use std::fmt::Debug;
use std::sync::Arc;

use polars_core::config;
use polars_core::error::{PolarsResult, polars_bail};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
use polars_utils::slice_enum::Slice;
use polars_utils::{format_pl_smallstr, unitvec};

#[cfg(feature = "python")]
use crate::dsl::python_dsl::PythonScanSource;
use crate::dsl::{DslPlan, FileScanIR, UnifiedScanArgs};
use crate::plans::{AExpr, IR};

pub(super) fn expand_datasets(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<()> {
    let mut stack = unitvec![root];

    while let Some(node) = stack.pop() {
        lp_arena.get(node).copy_inputs(&mut stack);

        let IR::Scan {
            sources,
            scan_type,
            unified_scan_args,

            file_info: _,
            hive_parts: _,
            predicate,
            output_schema: _,
        } = lp_arena.get_mut(node)
        else {
            continue;
        };

        let mut projection = unified_scan_args.projection.clone();

        if let Some(row_index) = &unified_scan_args.row_index
            && let Some(projection) = projection.as_mut()
        {
            *projection = projection
                .iter()
                .filter(|x| *x != &row_index.name)
                .cloned()
                .collect();
        }

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

                // Note
                // row_index is removed from projection/live_columns set, and is therefore not
                // considered when comparing cached expansion equality. This is safe as the
                // `row_index_in_live_filter` variable does not depend on the cached values.

                let mut row_index_in_live_filter = false;

                let live_filter_columns: Option<Arc<[PlSmallStr]>> = predicate.as_ref().map(|x| {
                    use polars_core::prelude::PlHashSet;

                    use crate::utils::aexpr_to_leaf_names_iter;

                    let mut out: Arc<[PlSmallStr]> =
                        PlHashSet::from_iter(aexpr_to_leaf_names_iter(x.node(), expr_arena))
                            .into_iter()
                            .filter(|live_col| {
                                if unified_scan_args
                                    .row_index
                                    .as_ref()
                                    .is_some_and(|ri| live_col == &ri.name)
                                {
                                    row_index_in_live_filter = true;
                                    false
                                } else {
                                    true
                                }
                            })
                            .collect();

                    Arc::get_mut(&mut out).unwrap().sort_unstable();

                    out
                });

                let existing_resolved_version_key = match guard.as_ref() {
                    Some(resolved) => {
                        let ExpandedDataset {
                            version,
                            limit: cached_limit,
                            projection: cached_projection,
                            live_filter_columns: cached_live_filter_columns,
                            expanded_dsl: _,
                            python_scan: _,
                        } = resolved;

                        (&limit == cached_limit
                            && &projection == cached_projection
                            && &live_filter_columns == cached_live_filter_columns)
                            .then_some(version.as_str())
                    },

                    None => None,
                };

                if let Some((expanded_dsl, version)) = dataset_object.to_dataset_scan(
                    existing_resolved_version_key,
                    limit,
                    projection.as_deref(),
                    live_filter_columns.as_deref(),
                )? {
                    *guard = Some(ExpandedDataset {
                        version,
                        limit,
                        projection,
                        live_filter_columns,
                        expanded_dsl,
                        python_scan: None,
                    })
                }

                let ExpandedDataset {
                    version: _,
                    limit: _,
                    projection: _,
                    live_filter_columns: _,
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
                            hidden_file_prefix: _hidden_file_prefix @ None,
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
                            table_statistics,
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
                        unified_scan_args.table_statistics = table_statistics.clone();

                        if row_index_in_live_filter {
                            use polars_core::prelude::{Column, DataType, IdxCa, IntoColumn};
                            use polars_core::series::IntoSeries;

                            let row_index_name =
                                &unified_scan_args.row_index.as_ref().unwrap().name;
                            let table_statistics =
                                unified_scan_args.table_statistics.as_mut().unwrap();

                            let statistics_df = Arc::make_mut(&mut table_statistics.0);
                            assert!(
                                !statistics_df
                                    .schema()
                                    .contains(&format_pl_smallstr!("{}_nc", row_index_name))
                            );

                            statistics_df.clear_schema();

                            unsafe { statistics_df.get_columns_mut() }.extend([
                                IdxCa::from_vec(
                                    format_pl_smallstr!("{}_nc", row_index_name),
                                    vec![0],
                                )
                                .into_series()
                                .into_column()
                                .new_from_index(0, sources.len()),
                                Column::full_null(
                                    format_pl_smallstr!("{}_min", row_index_name),
                                    sources.len(),
                                    &DataType::IDX_DTYPE,
                                ),
                                Column::full_null(
                                    format_pl_smallstr!("{}_max", row_index_name),
                                    sources.len(),
                                    &DataType::IDX_DTYPE,
                                ),
                            ]);
                        }

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

    Ok(())
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ExpandedDataset {
    version: PlSmallStr,
    limit: Option<usize>,
    projection: Option<Arc<[PlSmallStr]>>,
    live_filter_columns: Option<Arc<[PlSmallStr]>>,
    expanded_dsl: DslPlan,

    /// Fallback python scan
    #[cfg(feature = "python")]
    python_scan: Option<ExpandedPythonScan>,
}

#[cfg(feature = "python")]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
            live_filter_columns,
            expanded_dsl,

            #[cfg(feature = "python")]
            python_scan,
        } = self;

        return display::ExpandedDataset {
            version,
            limit,
            projection,
            live_filter_columns,
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
                pub live_filter_columns: &'a Option<Arc<[PlSmallStr]>>,
                pub expanded_dsl: &'a str,

                #[cfg(feature = "python")]
                pub python_scan: Option<PlSmallStr>,
            }
        }
    }
}
