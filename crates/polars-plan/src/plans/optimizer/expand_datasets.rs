#[cfg(feature = "python")]
use std::cell::LazyCell;
use std::fmt::Debug;
use std::ops::ControlFlow;
use std::sync::Arc;
#[cfg(feature = "python")]
use std::sync::Mutex;

use futures::StreamExt;
use futures::future::LocalBoxFuture;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::runtime::ASYNC;
use polars_utils::arena::{Arena, Node};
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
use polars_utils::slice_enum::Slice;

#[cfg(feature = "parquet")]
use crate::dsl::MetadataPerSource::Unresolved;
#[cfg(feature = "python")]
use crate::dsl::python_dsl::PythonScanSource;
use crate::dsl::{DslPlan, FileScanIR, UnifiedScanArgs};
use crate::plans::optimizer::ApplyScanPredicateFn;
use crate::plans::optimizer::ir_traversal::ir_graph_traversal;
use crate::plans::{AExpr, Card, IR};
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

#[cfg(feature = "python")]
pub type PyScanResolveThreadPool = polars_utils::python_thread_pool::PyThreadPool;

pub(super) fn expand_datasets(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    apply_scan_predicate_to_scan_ir: ApplyScanPredicateFn,
) -> PolarsResult<()> {
    // Polled locally by block_in_place_on; the continuations do not need Send.
    let mut expansion_tasks: FuturesUnordered<LocalBoxFuture<'static, (Node, PolarsResult<IR>)>> =
        FuturesUnordered::new();

    #[cfg(feature = "python")]
    let py_scan_resolve_threadpool: LazyCell<Arc<PyScanResolveThreadPool>> =
        LazyCell::new(|| Arc::new(PyScanResolveThreadPool::new_scan_resolve_thread_pool()));

    match ir_graph_traversal(
        root,
        &mut FnVisitors::new(
            || (),
            |key, storage: &mut Arena<IR>, _| {
                match (|| {
                    let IR::Scan {
                        sources: _,
                        scan_type,
                        unified_scan_args,

                        file_info,
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
                                        .filter_map(|node| {
                                            predicate_to_pa(node, expr_arena, &file_info.schema)
                                        })
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
                                Arc::clone(&py_scan_resolve_threadpool);

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

                            // Resolve before filtering, concurrently with other datasets.
                            expansion_tasks.push(Box::pin(resolve_after_expansion(handle)));
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
        ir_arena,
    ) {
        ControlFlow::Continue(()) => {},
        ControlFlow::Break(err) => return Err(err),
    }

    if !expansion_tasks.is_empty() {
        ASYNC.block_in_place_on(async {
            while let Some((node, ir)) = expansion_tasks.next().await {
                ir_arena.replace(node, ir?);
                apply_scan_predicate_to_scan_ir(node, ir_arena, expr_arena)?;
            }

            PolarsResult::Ok(())
        })?;
    }

    Ok(())
}

/// Await one dataset expansion, then read its heavy-source footers.
#[cfg(feature = "python")]
async fn resolve_after_expansion(
    handle: AbortOnDropHandle<(Node, PolarsResult<IR>)>,
) -> (Node, PolarsResult<IR>) {
    let (key, ir) = handle.await.unwrap();
    let ir = async {
        let mut ir = ir?;
        resolve_heavy_footers(&mut ir).await?;
        PolarsResult::Ok(ir)
    }
    .await;
    (key, ir)
}

/// Resolve heavy-source footers for distributed row-group splitting.
///
/// Dataset expansion runs after the usual DSL-to-IR footer resolution.
/// Enabled by [`UnifiedScanArgs::resolve_heavy_sources`].
#[cfg(feature = "parquet")]
async fn resolve_heavy_footers(scan_ir: &mut IR) -> PolarsResult<()> {
    use crate::dsl::MetadataPerSource;
    use crate::plans::parquet_footers::resolve_for_splitting;

    let IR::Scan {
        sources,
        scan_type,
        unified_scan_args,
        ..
    } = scan_ir
    else {
        return Ok(());
    };

    let Some(n_parts) = unified_scan_args.resolve_heavy_sources else {
        return Ok(());
    };
    let cloud_options = unified_scan_args.cloud_options.as_ref();

    let FileScanIR::Parquet {
        metadata_per_source,
        bytes_per_source,
        ..
    } = scan_type.as_mut()
    else {
        return Ok(());
    };
    if !matches!(metadata_per_source, MetadataPerSource::Unresolved) {
        return Ok(());
    }
    let Some(bytes) = bytes_per_source.as_deref() else {
        return Ok(());
    };

    *metadata_per_source = resolve_for_splitting(sources, bytes, n_parts, cloud_options).await;

    Ok(())
}

#[cfg(not(feature = "parquet"))]
async fn resolve_heavy_footers(_scan_ir: &mut IR) -> PolarsResult<()> {
    Ok(())
}

/// Rebuild the outer scan from a dataset expansion, leaving footers unresolved.
fn rebuild_scan_from_expanded(
    scan_ir: &mut IR,
    expanded_dsl: &DslPlan,
    row_index_in_live_filter: bool,
) -> PolarsResult<()> {
    use crate::dsl::FileScanDsl;

    let IR::Scan {
        sources,
        scan_type,
        unified_scan_args,

        file_info,
        hive_parts,
        predicate: _,
        predicate_file_skip_applied: _,
        output_schema: _,
    } = scan_ir
    else {
        unreachable!()
    };

    let DslPlan::Scan {
        sources: resolved_sources,
        unified_scan_args: resolved_unified_scan_args,
        scan_type: resolved_scan_type,
        cached_ir: _,
    } = expanded_dsl
    else {
        unreachable!()
    };

    // Copy provider-owned options; query-specific options stay on the outer scan.
    let UnifiedScanArgs {
        schema: _,
        cloud_options,
        hive_options,
        rechunk,
        cache,
        glob: _,
        expand_paths: _,
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
        source_sizes,
        resolve_heavy_sources: _,
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

        let height = statistics_df.height();

        unsafe { statistics_df.columns_mut() }.extend([
            IdxCa::from_vec(format_pl_smallstr!("{}_nc", row_index_name), vec![0])
                .into_series()
                .into_column()
                .new_from_index(0, height),
            Column::full_null(
                format_pl_smallstr!("{}_min", row_index_name),
                height,
                &DataType::IDX_DTYPE,
            ),
            Column::full_null(
                format_pl_smallstr!("{}_max", row_index_name),
                height,
                &DataType::IDX_DTYPE,
            ),
        ]);
    }

    if let Some(source_sizes) = source_sizes {
        polars_ensure!(
            source_sizes.len() == resolved_sources.len(),
            ShapeMismatch:
            "number of source sizes ({}) does not match number of scan sources ({})",
            source_sizes.len(),
            resolved_sources.len(),
        );
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
            // Heavy-source footers are resolved after expansion, if requested.
            metadata_per_source: Unresolved,
            bytes_per_source: source_sizes.clone(),
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

        FileScanDsl::ExternalReaderBuilder { external } => {
            FileScanIR::ExternalReaderBuilder { external }
        },

        FileScanDsl::Anonymous {
            options,
            function,
            file_info: _,
        } => FileScanIR::Anonymous { options, function },
    };

    if hive_options.enabled == Some(true)
        && let Some(paths) = sources.as_paths()
    {
        use polars_arrow::Either;

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
    let IR::Scan { scan_type, .. } = &mut scan_ir else {
        unreachable!()
    };

    let FileScanIR::PythonDataset {
        dataset_object,
        cached_ir,
    } = scan_type.as_mut()
    else {
        unreachable!()
    };

    let shared_cached_ir = Arc::clone(cached_ir);
    let mut guard = shared_cached_ir.lock().unwrap();

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

    // The `dataset_object` borrow of `scan_ir` must end before the rebuild below.
    let dataset_name = dataset_object.name();

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
        DslPlan::Scan { .. } => {
            rebuild_scan_from_expanded(&mut scan_ir, expanded_dsl, row_index_in_live_filter)?
        },

        DslPlan::PythonScan { options } => {
            *python_scan = Some(ExpandedPythonScan {
                name: dataset_name,
                scan_fn: options.scan_fn.clone().unwrap(),
                variant: options.python_source.clone(),
            });
            *cached_ir = Arc::new(Mutex::new((*guard).clone()));
        },

        dsl => {
            polars_bail!(
                ComputeError:
                "unknown DSL when resolving python dataset scan: {}",
                dsl.display()?
            )
        },
    };

    let IR::Scan {
        unified_scan_args,
        file_info,
        ..
    } = &mut scan_ir
    else {
        unreachable!()
    };

    if let Some((physical, deleted)) = unified_scan_args.row_count {
        file_info.stats.rows = Card::Exact(u64::saturating_sub(physical, deleted));
    }

    Ok(scan_ir)
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

#[cfg(all(test, feature = "parquet"))]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::Mutex;

    use polars_buffer::Buffer;
    use polars_core::prelude::*;
    use polars_io::prelude::ParquetOptions;
    use polars_utils::pl_path::PlRefPath;

    use super::*;
    use crate::dsl::{FileScanDsl, FileScanIR, MetadataPerSource, ScanSources, UnifiedScanArgs};
    use crate::plans::{FileInfo, ScanStats};

    /// Three files whose middle one holds nearly all the bytes.
    fn write_sources(dir: &std::path::Path) -> (ScanSources, Buffer<u64>) {
        use polars_io::prelude::ParquetWriter;

        let mut paths = Vec::new();
        let mut sizes = Vec::new();

        for (i, n) in [20i64, 4000, 20].into_iter().enumerate() {
            let path = dir.join(format!("{i}.parquet"));
            let mut df = df!("x" => (0..n).collect::<Vec<_>>()).unwrap();
            let file = std::fs::File::create(&path).unwrap();
            ParquetWriter::new(file)
                .with_row_group_size(Some(500))
                .finish(&mut df)
                .unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
            paths.push(PlRefPath::try_from_pathbuf(path).unwrap());
        }

        (
            ScanSources::Paths(Buffer::from_owner(paths)),
            Buffer::from_owner(sizes),
        )
    }

    fn expanded_scan_dsl(sources: ScanSources, sizes: Buffer<u64>) -> DslPlan {
        let mut args = UnifiedScanArgs::default();
        args.hive_options.enabled = Some(false);
        args.source_sizes = Some(sizes);

        DslPlan::Scan {
            sources,
            unified_scan_args: Box::new(args),
            scan_type: Box::new(FileScanDsl::Parquet {
                options: ParquetOptions::default(),
            }),
            cached_ir: Arc::new(Mutex::new(None)),
        }
    }

    /// Outer scan before dataset expansion.
    fn outer_scan_ir(resolve_heavy_sources: Option<NonZeroU32>) -> IR {
        let schema = Arc::new(Schema::from_iter([Field::new("x".into(), DataType::Int64)]));

        let mut args = UnifiedScanArgs::default();
        args.hive_options.enabled = Some(false);
        args.resolve_heavy_sources = resolve_heavy_sources;

        IR::Scan {
            sources: ScanSources::default(),
            file_info: FileInfo::new(schema.clone(), None, ScanStats::unknown()),
            hive_parts: None,
            predicate: None,
            predicate_file_skip_applied: None,
            output_schema: None,
            scan_type: Box::new(FileScanIR::Parquet {
                options: ParquetOptions::default(),
                metadata_per_source: MetadataPerSource::Unresolved,
                bytes_per_source: None,
            }),
            unified_scan_args: Box::new(args),
        }
    }

    fn retained(scan_ir: &IR) -> Vec<(usize, usize)> {
        let IR::Scan { scan_type, .. } = scan_ir else {
            unreachable!()
        };
        let FileScanIR::Parquet {
            metadata_per_source,
            ..
        } = scan_type.as_ref()
        else {
            unreachable!()
        };
        metadata_per_source
            .iter_resolved()
            .map(|(i, md)| (i, md.row_groups.len()))
            .collect()
    }

    /// Reusing a cached expansion must still honor the outer scan's resolution flag.
    #[test]
    fn cached_expansion_resolves_heavy_footers_when_the_flag_is_added() {
        let dir = tempfile::tempdir().unwrap();
        let (sources, sizes) = write_sources(dir.path());
        // Three sources fit the default footer budget.
        let expanded_dsl = expanded_scan_dsl(sources, sizes);

        let mut without = outer_scan_ir(None);
        let mut with = outer_scan_ir(NonZeroU32::new(4));

        ASYNC
            .block_on(async {
                for ir in [&mut without, &mut with] {
                    rebuild_scan_from_expanded(ir, &expanded_dsl, false)?;
                    resolve_heavy_footers(ir).await?;
                }
                PolarsResult::Ok(())
            })
            .unwrap();

        assert_eq!(retained(&without), []);
        // Partial metadata requires source 0; source 1 is heavy.
        assert_eq!(retained(&with), [(0, 1), (1, 8)]);
    }
}
