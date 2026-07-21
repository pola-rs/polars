use std::fmt::Debug;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::error::{PolarsResult, polars_bail};
use polars_core::runtime::ASYNC;
use polars_utils::arena::{Arena, Node};
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
use polars_utils::slice_enum::Slice;
use polars_utils::{format_pl_smallstr, unitvec};

#[cfg(feature = "python")]
use crate::dsl::python_dsl::PythonScanSource;
use crate::dsl::{DslPlan, FileScanIR, UnifiedScanArgs};
use crate::plans::optimizer::ir_traversal::ir_graph_traversal;
use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::plans::{AExpr, IR};
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

pub(super) fn expand_datasets(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    apply_scan_predicate_to_scan_ir: fn(
        Node,
        &mut Arena<IR>,
        &mut Arena<AExpr>,
    ) -> PolarsResult<()>,
) -> PolarsResult<()> {
    #[expect(clippy::type_complexity)]
    let mut expansion_tasks: FuturesUnordered<_> = FuturesUnordered::new();

    #[cfg(feature = "python")]
    let mut py_scan_resolve_threadpool: Option<Arc<PyScanResolveThreadPool>> = None;

    match ir_graph_traversal(
        root,
        &mut FnVisitors::new(
            || (),
            |key, storage: &mut IRTraversalStorage, _| {
                match (|| {
                    let IR::Scan {
                        sources: _,
                        scan_type,
                        unified_scan_args,

                        file_info: _,
                        hive_parts: _,
                        predicate,
                        predicate_file_skip_applied: _,
                        output_schema: _,
                    } = storage.get_mut(key)
                    else {
                        return Ok(());
                    };

                    match scan_type.as_mut() {
                        #[cfg(feature = "python")]
                        FileScanIR::PythonDataset { .. } => {
                            use polars_core::runtime::ASYNC;

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

                            // Note
                            // row_index is removed from projection/live_columns set, and is therefore not
                            // considered when comparing cached expansion equality. This is safe as the
                            // `row_index_in_live_filter` variable does not depend on the cached values.

                            let mut row_index_in_live_filter = false;

                            let live_filter_columns: Option<Arc<[PlSmallStr]>> =
                                predicate.as_ref().map(|x| {
                                    use polars_core::prelude::PlIndexSet;

                                    use crate::utils::aexpr_to_leaf_names_iter;

                                    let mut out: Arc<[PlSmallStr]> = PlIndexSet::from_iter(
                                        aexpr_to_leaf_names_iter(x.node(), expr_arena),
                                    )
                                    .into_iter()
                                    .filter(|&live_col| {
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
                                    .cloned()
                                    .collect();

                                    Arc::get_mut(&mut out).unwrap().sort_unstable();

                                    out
                                });

                            let pyarrow_predicate: Option<String> = if !unified_scan_args
                                .has_row_index_or_slice()
                                && let Some(predicate) = &predicate
                            {
                                use crate::plans::aexpr::MintermIter;
                                use crate::plans::python::pyarrow::predicate_to_pa;

                                // Convert minterms independently, can allow conversion to partially succeed if there are unsupported expressions
                                let parts: Vec<String> =
                                    MintermIter::new(predicate.node(), expr_arena)
                                        .filter_map(|node| predicate_to_pa(node, expr_arena))
                                        .collect();
                                match parts.len() {
                                    0 => None,
                                    1 => Some(parts.into_iter().next().unwrap()),
                                    _ => Some(format!("({})", parts.join(" & "))),
                                }
                            } else {
                                None
                            };

                            let ir = storage.take(key);

                            assert!(matches!(ir, IR::Scan { .. }));

                            let py_scan_resolve_threadpool =
                                Arc::clone(py_scan_resolve_threadpool.get_or_insert_with(|| {
                                    Arc::new(PyScanResolveThreadPool::new())
                                }));

                            let handle = AbortOnDropHandle(ASYNC.spawn_blocking(move || {
                                (
                                    key,
                                    expand_python_dataset(
                                        ir,
                                        projection,
                                        limit,
                                        live_filter_columns,
                                        row_index_in_live_filter,
                                        pyarrow_predicate,
                                        py_scan_resolve_threadpool.as_ref(),
                                    ),
                                )
                            }));

                            expansion_tasks.push(handle);
                        },

                        _ => apply_scan_predicate_to_scan_ir(key, storage, expr_arena)?,
                    };
                    PolarsResult::Ok(())
                })() {
                    Ok(()) => ControlFlow::Continue(SubtreeVisit::Visit),
                    Err(err) => ControlFlow::Break(err),
                }
            },
            |_, _, _| ControlFlow::Continue(()),
        ),
        &mut vec![],
        &mut vec![],
        IRTraversalStorage {
            arena: ir_arena,
            skip_subtree: |_| false,
        },
    ) {
        ControlFlow::Continue(()) => {},
        ControlFlow::Break(err) => return Err(err),
    }

    if !expansion_tasks.is_empty() {
        ASYNC.block_in_place_on(async {
            while let Some(v) = expansion_tasks.next().await {
                let (node, ir) = v.unwrap();
                let ir = ir?;
                ir_arena.replace(node, ir);
                apply_scan_predicate_to_scan_ir(node, ir_arena, expr_arena)?;
            }

            PolarsResult::Ok(())
        })?;
    }

    Ok(())
}

#[cfg(feature = "python")]
fn expand_python_dataset(
    mut scan_ir: IR,
    projection: Option<Arc<[PlSmallStr]>>,
    limit: Option<usize>,
    live_filter_columns: Option<Arc<[PlSmallStr]>>,
    row_index_in_live_filter: bool,
    pyarrow_predicate: Option<String>,
    py_scan_resolve_threadpool: &PyScanResolveThreadPool,
) -> PolarsResult<IR> {
    let IR::Scan {
        sources,
        scan_type,
        unified_scan_args,

        file_info,
        hive_parts,
        predicate: _,
        predicate_file_skip_applied: _,
        output_schema: _,
    } = &mut scan_ir
    else {
        unreachable!()
    };

    let FileScanIR::PythonDataset {
        dataset_object,
        cached_ir,
    } = scan_type.as_mut()
    else {
        unreachable!()
    };

    let cached_ir = cached_ir.clone();
    let mut guard = cached_ir.lock().unwrap();

    if config::verbose() {
        eprintln!(
            "expand_datasets(): python[{}]: limit: {:?}, project: {}",
            dataset_object.name(),
            limit,
            projection
                .as_ref()
                .map_or(PlSmallStr::from_static("all"), |x| format_pl_smallstr!(
                    "{}",
                    x.len()
                ))
        )
    }

    let existing_resolved_version_key = match guard.as_ref() {
        Some(resolved) => {
            let ExpandedDataset {
                version,
                limit: cached_limit,
                projection: cached_projection,
                live_filter_columns: cached_live_filter_columns,
                pyarrow_predicate: cached_pyarrow_predicate,
                expanded_dsl: _,
                python_scan: _,
            } = resolved;

            (&limit == cached_limit
                && &projection == cached_projection
                && &live_filter_columns == cached_live_filter_columns
                && &pyarrow_predicate == cached_pyarrow_predicate)
                .then_some(version.as_str())
        },

        None => None,
    };

    if let Some((expanded_dsl, version)) = dataset_object.to_dataset_scan(
        existing_resolved_version_key,
        limit,
        projection.as_deref(),
        live_filter_columns.as_deref(),
        pyarrow_predicate.as_deref(),
        py_scan_resolve_threadpool,
    )? {
        *guard = Some(ExpandedDataset {
            version,
            limit,
            projection,
            live_filter_columns,
            pyarrow_predicate,
            expanded_dsl,
            python_scan: None,
        })
    }

    let ExpandedDataset {
        version: _,
        limit: _,
        projection: _,
        live_filter_columns: _,
        pyarrow_predicate: _,
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
                hive_options,
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
                row_count,
            } = resolved_unified_scan_args.as_ref()
            else {
                panic!(
                    "invalid scan args from python dataset resolve: {:?}",
                    resolved_unified_scan_args
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
            unified_scan_args.row_count = *row_count;

            if row_index_in_live_filter {
                use polars_core::prelude::{Column, DataType, IdxCa, IntoColumn};
                use polars_core::series::IntoSeries;

                let row_index_name = &unified_scan_args.row_index.as_ref().unwrap().name;
                let table_statistics = unified_scan_args.table_statistics.as_mut().unwrap();

                let statistics_df = Arc::make_mut(&mut table_statistics.0);
                assert!(
                    !statistics_df
                        .schema()
                        .contains(&format_pl_smallstr!("{}_nc", row_index_name))
                );

                unsafe { statistics_df.columns_mut() }.extend([
                    IdxCa::from_vec(format_pl_smallstr!("{}_nc", row_index_name), vec![0])
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

            **scan_type = match *resolved_scan_type.clone() {
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
                    // Metadata is resolved later in `parquet_file_info`.
                    first_metadata: None,
                    metadata_per_source: None,
                },

                #[cfg(feature = "json")]
                FileScanDsl::NDJson { options } => FileScanIR::NDJson { options },

                #[cfg(feature = "python")]
                FileScanDsl::PythonDataset { dataset_object } => FileScanIR::PythonDataset {
                    dataset_object,
                    cached_ir: Default::default(),
                },

                #[cfg(feature = "scan_lines")]
                FileScanDsl::Lines { name } => FileScanIR::Lines { name },

                FileScanDsl::ExpandedPaths { name } => FileScanIR::ExpandedPaths { name },

                FileScanDsl::Anonymous {
                    options,
                    function,
                    file_info: _,
                } => FileScanIR::Anonymous { options, function },
            };

            if hive_options.enabled == Some(true)
                && let Some(paths) = sources.as_paths()
            {
                use arrow::Either;

                use crate::plans::hive::hive_partitions_from_paths;

                let owned;

                *hive_parts = hive_partitions_from_paths(
                    paths,
                    hive_options.hive_start_idx,
                    hive_options.schema.clone(),
                    match file_info.reader_schema.as_ref().unwrap() {
                        Either::Left(v) => {
                            use polars_core::schema::{Schema, SchemaExt as _};

                            owned = Some(Schema::from_arrow_schema(v.as_ref()));
                            owned.as_ref().unwrap()
                        },
                        Either::Right(v) => v.as_ref(),
                    },
                    hive_options.try_parse_dates,
                )?;
            }
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

    if let Some((physical, deleted)) = unified_scan_args.row_count {
        let row_count = u64::saturating_sub(physical, deleted) as usize;
        file_info.row_estimation = (Some(row_count), row_count);
    }

    Ok(scan_ir)
}

#[cfg(feature = "python")]
#[derive(pyo3::IntoPyObject)]
pub struct PyScanResolveThreadPool(
    /// polars._utils.threading.PyScanResolveThreadPool
    pyo3::Py<pyo3::PyAny>,
);

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &'py PyScanResolveThreadPool {
    type Output = <&'py pyo3::Py<pyo3::PyAny> as pyo3::IntoPyObject<'py>>::Output;
    type Target = <&'py pyo3::Py<pyo3::PyAny> as pyo3::IntoPyObject<'py>>::Target;
    type Error = <&'py pyo3::Py<pyo3::PyAny> as pyo3::IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: pyo3::prelude::Python<'py>) -> Result<Self::Output, Self::Error> {
        pyo3::IntoPyObject::into_pyobject(&self.0, py)
    }
}

#[cfg(feature = "python")]
impl PyScanResolveThreadPool {
    pub fn new() -> Self {
        use std::num::NonZeroUsize;
        use std::sync::LazyLock;

        use pyo3::types::PyAnyMethods;
        use pyo3::{Py, PyAny, PyResult, Python};

        Python::attach(|py| {
            let num_threads =
                std::env::var("POLARS_PYTHON_SCAN_RESOLVE_THREADS").map_or(128, |x| {
                    x.parse::<NonZeroUsize>()
                        .unwrap_or_else(|_| {
                            panic!("invalid value for POLARS_PYTHON_SCAN_RESOLVE_THREADS: {x}")
                        })
                        .get()
                });

            if polars_config::config().verbose() {
                eprintln!("python scan_resolve_threadpool threads: {num_threads}")
            }

            if LazyLock::get(&PY_SCAN_RESOLVE_THREADPOOL_CLS).is_none() {
                py.detach(|| LazyLock::force(&PY_SCAN_RESOLVE_THREADPOOL_CLS));
            }

            return Self(
                PY_SCAN_RESOLVE_THREADPOOL_CLS
                    .bind(py)
                    .call1((num_threads,))
                    .map(|x| x.unbind())
                    .unwrap(),
            );

            static PY_SCAN_RESOLVE_THREADPOOL_CLS: LazyLock<Py<PyAny>> = LazyLock::new(|| {
                Python::attach(|py| {
                    (|| {
                        PyResult::Ok(
                            py.import("polars._utils.threading")?
                                .getattr("PyScanResolveThreadPool")?
                                .unbind(),
                        )
                    })()
                    .unwrap()
                })
            });
        })
    }
}

#[cfg(feature = "python")]
impl Default for PyScanResolveThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ExpandedDataset {
    version: PlSmallStr,
    limit: Option<usize>,
    projection: Option<Arc<[PlSmallStr]>>,
    live_filter_columns: Option<Arc<[PlSmallStr]>>,
    pyarrow_predicate: Option<String>,
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
            pyarrow_predicate,
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
            pyarrow_predicate: if pyarrow_predicate.is_some() {
                "Some(<redacted>)"
            } else {
                "None"
            },
            #[cfg(feature = "python")]
            python_scan: python_scan.as_ref().map(
                |ExpandedPythonScan {
                     name,
                     scan_fn: _,
                     variant,
                 }| {
                    format_pl_smallstr!("streaming-python-scan[{} @ {:?}]", name, variant)
                },
            ),
        }
        .fmt(f);

        mod display {
            use std::fmt::Debug;
            use std::sync::Arc;

            use polars_utils::pl_str::PlSmallStr;

            #[allow(dead_code)]
            #[derive(Debug)]
            pub struct ExpandedDataset<'a> {
                pub version: &'a str,
                pub limit: &'a Option<usize>,
                pub projection: &'a Option<Arc<[PlSmallStr]>>,
                pub live_filter_columns: &'a Option<Arc<[PlSmallStr]>>,
                pub pyarrow_predicate: &'static str,
                pub expanded_dsl: &'a str,

                #[cfg(feature = "python")]
                pub python_scan: Option<PlSmallStr>,
            }
        }
    }
}
