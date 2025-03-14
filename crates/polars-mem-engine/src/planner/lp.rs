use polars_core::POOL;
use polars_core::prelude::*;
use polars_expr::state::ExecutionState;
use polars_io::utils::file::Writeable;
use polars_io::utils::mkdir::mkdir_recursive;
use polars_plan::global::_set_n_rows_for_scan;
use polars_plan::plans::expr_ir::ExprIR;
use polars_utils::format_pl_smallstr;
use recursive::recursive;

use self::expr_ir::OutputName;
use self::predicates::{aexpr_to_column_predicates, aexpr_to_skip_batch_predicate};
#[cfg(feature = "python")]
use self::python_dsl::PythonScanSource;
use super::super::executors::{self, Executor};
use super::*;
use crate::ScanPredicate;
use crate::executors::{CachePrefiller, SinkExecutor};
use crate::predicate::PhysicalColumnPredicates;

fn partitionable_gb(
    keys: &[ExprIR],
    aggs: &[ExprIR],
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
    apply: &Option<Arc<dyn DataFrameUdf>>,
) -> bool {
    // checks:
    //      1. complex expressions in the group_by itself are also not partitionable
    //          in this case anything more than col("foo")
    //      2. a custom function cannot be partitioned
    //      3. we don't bother with more than 2 keys, as the cardinality likely explodes
    //         by the combinations
    if !keys.is_empty() && keys.len() < 3 && apply.is_none() {
        // complex expressions in the group_by itself are also not partitionable
        // in this case anything more than col("foo")
        for key in keys {
            if (expr_arena).iter(key.node()).count() > 1
                || has_aexpr(key.node(), expr_arena, |ae| match ae {
                    AExpr::Literal(lv) => !lv.is_scalar(),
                    _ => false,
                })
            {
                return false;
            }
        }

        can_pre_agg_exprs(aggs, expr_arena, input_schema)
    } else {
        false
    }
}

#[derive(Clone)]
struct ConversionState {
    expr_depth: u16,
    has_cache_child: bool,
    has_cache_parent: bool,
}

impl ConversionState {
    fn new() -> PolarsResult<Self> {
        Ok(ConversionState {
            expr_depth: get_expr_depth_limit()?,
            has_cache_child: false,
            has_cache_parent: false,
        })
    }

    fn with_new_branch<K, F: FnOnce(&mut Self) -> K>(&mut self, func: F) -> K {
        let mut new_state = self.clone();
        new_state.has_cache_child = false;
        let out = func(&mut new_state);
        self.has_cache_child = new_state.has_cache_child;
        out
    }
}

pub fn create_physical_plan(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    let mut state = ConversionState::new()?;
    let mut cache_nodes = Default::default();
    let plan = create_physical_plan_impl(root, lp_arena, expr_arena, &mut state, &mut cache_nodes)?;

    if cache_nodes.is_empty() {
        Ok(plan)
    } else {
        Ok(Box::new(CachePrefiller {
            caches: cache_nodes,
            phys_plan: plan,
        }))
    }
}

pub struct MultiplePhysicalPlans {
    pub cache_prefiller: Option<Box<dyn Executor>>,
    pub physical_plans: Vec<Box<dyn Executor>>,
}
pub fn create_multiple_physical_plans(
    roots: &[Node],
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<MultiplePhysicalPlans> {
    let mut state = ConversionState::new()?;
    let mut cache_nodes = Default::default();
    let plans = state.with_new_branch(|new_state| {
        roots
            .iter()
            .map(|&node| {
                create_physical_plan_impl(node, lp_arena, expr_arena, new_state, &mut cache_nodes)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let cache_prefiller = (!cache_nodes.is_empty()).then(|| {
        struct Empty;
        impl Executor for Empty {
            fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
                Ok(DataFrame::empty())
            }
        }
        Box::new(CachePrefiller {
            caches: cache_nodes,
            phys_plan: Box::new(Empty),
        }) as _
    });

    Ok(MultiplePhysicalPlans {
        cache_prefiller,
        physical_plans: plans,
    })
}

#[recursive]
fn create_physical_plan_impl(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    state: &mut ConversionState,
    // Cache nodes in order of discovery
    cache_nodes: &mut PlIndexMap<usize, Box<dyn Executor>>,
) -> PolarsResult<Box<dyn Executor>> {
    use IR::*;

    macro_rules! recurse {
        ($node:expr, $state: expr) => {
            create_physical_plan_impl($node, lp_arena, expr_arena, $state, cache_nodes)
        };
    }

    let logical_plan = if state.has_cache_parent {
        lp_arena.get(root).clone()
    } else {
        lp_arena.take(root)
    };

    match logical_plan {
        #[cfg(feature = "python")]
        PythonScan { mut options } => {
            let mut predicate_serialized = None;

            let predicate = if let PythonPredicate::Polars(e) = &options.predicate {
                let phys_expr = || {
                    let mut state = ExpressionConversionState::new(true, state.expr_depth);
                    create_physical_expr(
                        e,
                        Context::Default,
                        expr_arena,
                        &options.schema,
                        &mut state,
                    )
                };

                // Convert to a pyarrow eval string.
                if matches!(options.python_source, PythonScanSource::Pyarrow) {
                    if let Some(eval_str) = polars_plan::plans::python::pyarrow::predicate_to_pa(
                        e.node(),
                        expr_arena,
                        Default::default(),
                    ) {
                        options.predicate = PythonPredicate::PyArrow(eval_str);
                        // We don't have to use a physical expression as pyarrow deals with the filter.
                        None
                    } else {
                        Some(phys_expr()?)
                    }
                }
                // Convert to physical expression for the case the reader cannot consume the predicate.
                else {
                    let dsl_expr = e.to_expr(expr_arena);
                    predicate_serialized =
                        polars_plan::plans::python::predicate::serialize(&dsl_expr)?;

                    Some(phys_expr()?)
                }
            } else {
                None
            };
            Ok(Box::new(executors::PythonScanExec {
                options,
                predicate,
                predicate_serialized,
            }))
        },
        Sink { input, payload } => {
            let input = recurse!(input, state)?;
            match payload {
                SinkTypeIR::Memory => Ok(Box::new(SinkExecutor {
                    input,
                    name: "mem".to_string(),
                    f: Box::new(move |df, _state| Ok(Some(df))),
                })),
                SinkTypeIR::File(FileSinkType {
                    file_type,
                    path,
                    sink_options,
                    cloud_options,
                }) => match file_type {
                    #[cfg(feature = "parquet")]
                    FileType::Parquet(options) => Ok(Box::new(SinkExecutor {
                        input,
                        name: "parquet".to_string(),
                        f: Box::new(move |mut df, _state| {
                            use std::io::BufWriter;
                            use std::ops::DerefMut;

                            use polars_io::parquet::write::ParquetWriter;

                            if sink_options.mkdir {
                                mkdir_recursive(path.as_path())?;
                            }

                            let path = path.as_ref().display().to_string();
                            let mut file = polars_io::utils::file::Writeable::try_new(
                                &path,
                                cloud_options.as_ref(),
                            )?;
                            ParquetWriter::new(BufWriter::new(file.deref_mut()))
                                .with_compression(options.compression)
                                .with_statistics(options.statistics)
                                .with_row_group_size(options.row_group_size)
                                .with_data_page_size(options.data_page_size)
                                .finish(&mut df)?;

                            if let Writeable::Local(file) = &mut file {
                                polars_io::utils::sync_on_close::sync_on_close(
                                    sink_options.sync_on_close,
                                    file,
                                )?;
                            }
                            file.close()?;

                            Ok(None)
                        }),
                    })),
                    #[cfg(feature = "ipc")]
                    FileType::Ipc(options) => Ok(Box::new(SinkExecutor {
                        input,
                        name: "ipc".to_string(),
                        f: Box::new(move |mut df, _state| {
                            use std::io::BufWriter;
                            use std::ops::DerefMut;

                            use polars_io::SerWriter;
                            use polars_io::ipc::IpcWriter;

                            if sink_options.mkdir {
                                mkdir_recursive(path.as_path())?;
                            }

                            let path = path.as_ref().display().to_string();
                            let mut file = polars_io::utils::file::Writeable::try_new(
                                &path,
                                cloud_options.as_ref(),
                            )?;
                            IpcWriter::new(BufWriter::new(file.deref_mut()))
                                .with_compression(options.compression)
                                .with_compat_level(options.compat_level)
                                .finish(&mut df)?;

                            if let Writeable::Local(file) = &mut file {
                                polars_io::utils::sync_on_close::sync_on_close(
                                    sink_options.sync_on_close,
                                    file,
                                )?;
                            }

                            file.close()?;

                            Ok(None)
                        }),
                    })),
                    #[cfg(feature = "csv")]
                    FileType::Csv(options) => Ok(Box::new(SinkExecutor {
                        input,
                        name: "csv".to_string(),
                        f: Box::new(move |mut df, _state| {
                            use std::io::BufWriter;
                            use std::ops::DerefMut;

                            use polars_io::SerWriter;
                            use polars_io::csv::write::CsvWriter;

                            if sink_options.mkdir {
                                mkdir_recursive(path.as_path())?;
                            }

                            let path = path.as_ref().display().to_string();
                            let mut file = polars_io::utils::file::Writeable::try_new(
                                &path,
                                cloud_options.as_ref(),
                            )?;
                            CsvWriter::new(BufWriter::new(file.deref_mut()))
                                .include_bom(options.include_bom)
                                .include_header(options.include_header)
                                .with_separator(options.serialize_options.separator)
                                .with_line_terminator(
                                    options.serialize_options.line_terminator.clone(),
                                )
                                .with_quote_char(options.serialize_options.quote_char)
                                .with_batch_size(options.batch_size)
                                .with_datetime_format(
                                    options.serialize_options.datetime_format.clone(),
                                )
                                .with_date_format(options.serialize_options.date_format.clone())
                                .with_time_format(options.serialize_options.time_format.clone())
                                .with_float_scientific(options.serialize_options.float_scientific)
                                .with_float_precision(options.serialize_options.float_precision)
                                .with_null_value(options.serialize_options.null.clone())
                                .with_quote_style(options.serialize_options.quote_style)
                                .finish(&mut df)?;

                            if let Writeable::Local(file) = &mut file {
                                polars_io::utils::sync_on_close::sync_on_close(
                                    sink_options.sync_on_close,
                                    file,
                                )?;
                            }
                            file.close()?;

                            Ok(None)
                        }),
                    })),
                    #[cfg(feature = "json")]
                    FileType::Json(_) => Ok(Box::new(SinkExecutor {
                        input,
                        name: "ndjson".to_string(),
                        f: Box::new(move |mut df, _state| {
                            use std::io::BufWriter;
                            use std::ops::DerefMut;

                            use polars_io::SerWriter;
                            use polars_io::json::{JsonFormat, JsonWriter};

                            if sink_options.mkdir {
                                mkdir_recursive(path.as_path())?;
                            }

                            let path = path.as_ref().display().to_string();
                            let mut file = polars_io::utils::file::Writeable::try_new(
                                &path,
                                cloud_options.as_ref(),
                            )?;
                            JsonWriter::new(BufWriter::new(file.deref_mut()))
                                .with_json_format(JsonFormat::JsonLines)
                                .finish(&mut df)?;

                            if let Writeable::Local(file) = &mut file {
                                polars_io::utils::sync_on_close::sync_on_close(
                                    sink_options.sync_on_close,
                                    file,
                                )?;
                            }

                            file.close()?;

                            Ok(None)
                        }),
                    })),
                },
                SinkTypeIR::Partition { .. } => {
                    polars_bail!(InvalidOperation:
                        "partition sinks not yet supported in standard engine."
                    )
                },
            }
        },
        SinkMultiple { .. } => {
            unreachable!("should be handled with create_multiple_physical_plans")
        },
        Union { inputs, options } => {
            let inputs = state.with_new_branch(|new_state| {
                inputs
                    .into_iter()
                    .map(|node| recurse!(node, new_state))
                    .collect::<PolarsResult<Vec<_>>>()
            });
            let inputs = inputs?;
            Ok(Box::new(executors::UnionExec { inputs, options }))
        },
        HConcat {
            inputs, options, ..
        } => {
            let inputs = state.with_new_branch(|new_state| {
                inputs
                    .into_iter()
                    .map(|node| recurse!(node, new_state))
                    .collect::<PolarsResult<Vec<_>>>()
            });

            let inputs = inputs?;

            Ok(Box::new(executors::HConcatExec { inputs, options }))
        },
        Slice { input, offset, len } => {
            let input = recurse!(input, state)?;
            Ok(Box::new(executors::SliceExec { input, offset, len }))
        },
        Filter { input, predicate } => {
            let mut streamable =
                is_elementwise_rec_no_cat_cast(expr_arena.get(predicate.node()), expr_arena);
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            if streamable {
                // This can cause problems with string caches
                streamable = !input_schema
                    .iter_values()
                    .any(|dt| dt.contains_categoricals())
                    || {
                        #[cfg(feature = "dtype-categorical")]
                        {
                            polars_core::using_string_cache()
                        }

                        #[cfg(not(feature = "dtype-categorical"))]
                        {
                            false
                        }
                    }
            }
            let input = recurse!(input, state)?;
            let mut state = ExpressionConversionState::new(true, state.expr_depth);
            let predicate = create_physical_expr(
                &predicate,
                Context::Default,
                expr_arena,
                &input_schema,
                &mut state,
            )?;
            Ok(Box::new(executors::FilterExec::new(
                predicate,
                input,
                state.has_windows,
                streamable,
            )))
        },
        #[allow(unused_variables)]
        Scan {
            sources,
            file_info,
            hive_parts,
            output_schema,
            scan_type,
            predicate,
            mut file_options,
        } => {
            file_options.pre_slice = if let Some((offset, len)) = file_options.pre_slice {
                Some((offset, _set_n_rows_for_scan(Some(len)).unwrap()))
            } else {
                _set_n_rows_for_scan(None).map(|x| (0, x))
            };

            let mut state = ExpressionConversionState::new(true, state.expr_depth);
            let do_new_multifile = (sources.len() > 1 || hive_parts.is_some())
                && !matches!(&*scan_type, FileScan::Anonymous { .. })
                && std::env::var("POLARS_NEW_MULTIFILE").as_deref() == Ok("1");

            let mut create_skip_batch_predicate = false;
            create_skip_batch_predicate |= do_new_multifile;
            #[cfg(feature = "parquet")]
            {
                create_skip_batch_predicate |= matches!(
                    &*scan_type,
                    FileScan::Parquet {
                        options: polars_io::prelude::ParquetOptions {
                            use_statistics: true,
                            ..
                        },
                        ..
                    }
                );
            }

            let predicate = predicate
                .map(|predicate| {
                    create_scan_predicate(
                        &predicate,
                        expr_arena,
                        output_schema.as_ref().unwrap_or(&file_info.schema),
                        &mut state,
                        create_skip_batch_predicate,
                        false,
                    )
                })
                .transpose()?;

            if do_new_multifile {
                return Ok(Box::new(executors::MultiScanExec::new(
                    sources,
                    file_info,
                    hive_parts.map(|h| h.into_statistics()),
                    predicate,
                    file_options,
                    scan_type,
                )));
            }

            match *scan_type {
                #[cfg(feature = "csv")]
                FileScan::Csv { options, .. } => Ok(Box::new(executors::CsvExec {
                    sources,
                    file_info,
                    options,
                    predicate,
                    file_options,
                })),
                #[cfg(feature = "ipc")]
                FileScan::Ipc {
                    options,
                    cloud_options,
                    metadata,
                } => Ok(Box::new(executors::IpcExec {
                    sources,
                    file_info,
                    predicate,
                    options,
                    file_options: *file_options,
                    hive_parts: hive_parts.map(|h| h.into_statistics()),
                    cloud_options,
                    metadata,
                })),
                #[cfg(feature = "parquet")]
                FileScan::Parquet {
                    options,
                    cloud_options,
                    metadata,
                } => Ok(Box::new(executors::ParquetExec::new(
                    sources,
                    file_info,
                    hive_parts.map(|h| h.into_statistics()),
                    predicate,
                    options,
                    cloud_options,
                    file_options,
                    metadata,
                ))),
                #[cfg(feature = "json")]
                FileScan::NDJson { options, .. } => Ok(Box::new(executors::JsonExec::new(
                    sources,
                    options,
                    file_options,
                    file_info,
                    predicate,
                ))),
                FileScan::Anonymous { function, .. } => {
                    Ok(Box::new(executors::AnonymousScanExec {
                        function,
                        predicate,
                        file_options,
                        file_info,
                        output_schema,
                        predicate_has_windows: state.has_windows,
                    }))
                },
            }
        },
        Select {
            expr,
            input,
            schema: _schema,
            options,
            ..
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = recurse!(input, state)?;
            let mut state = ExpressionConversionState::new(
                POOL.current_num_threads() > expr.len(),
                state.expr_depth,
            );
            let phys_expr = create_physical_expressions_from_irs(
                &expr,
                Context::Default,
                expr_arena,
                &input_schema,
                &mut state,
            )?;

            let allow_vertical_parallelism = options.should_broadcast && expr.iter().all(|e| is_elementwise_rec_no_cat_cast(expr_arena.get(e.node()), expr_arena))
                // If all columns are literal we would get a 1 row per thread.
                && !phys_expr.iter().all(|p| {
                    p.is_literal()
                });

            Ok(Box::new(executors::ProjectionExec {
                input,
                expr: phys_expr,
                has_windows: state.has_windows,
                input_schema,
                #[cfg(test)]
                schema: _schema,
                options,
                allow_vertical_parallelism,
            }))
        },
        DataFrameScan {
            df, output_schema, ..
        } => Ok(Box::new(executors::DataFrameExec {
            df,
            projection: output_schema.map(|s| s.iter_names_cloned().collect()),
        })),
        Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena);
            let by_column = create_physical_expressions_from_irs(
                &by_column,
                Context::Default,
                expr_arena,
                input_schema.as_ref(),
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let input = recurse!(input, state)?;
            Ok(Box::new(executors::SortExec {
                input,
                by_column,
                slice,
                sort_options,
            }))
        },
        Cache {
            input,
            id,
            cache_hits,
        } => {
            state.has_cache_parent = true;
            state.has_cache_child = true;

            if !cache_nodes.contains_key(&id) {
                let input = recurse!(input, state)?;

                let cache = Box::new(executors::CacheExec {
                    id,
                    input: Some(input),
                    count: cache_hits,
                });

                cache_nodes.insert(id, cache);
            }

            Ok(Box::new(executors::CacheExec {
                id,
                input: None,
                count: cache_hits,
            }))
        },
        Distinct { input, options } => {
            let input = recurse!(input, state)?;
            Ok(Box::new(executors::UniqueExec { input, options }))
        },
        GroupBy {
            input,
            keys,
            aggs,
            apply,
            schema,
            maintain_order,
            options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let options = Arc::try_unwrap(options).unwrap_or_else(|options| (*options).clone());
            let phys_keys = create_physical_expressions_from_irs(
                &keys,
                Context::Default,
                expr_arena,
                &input_schema,
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let phys_aggs = create_physical_expressions_from_irs(
                &aggs,
                Context::Aggregation,
                expr_arena,
                &input_schema,
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;

            let _slice = options.slice;
            #[cfg(feature = "dynamic_group_by")]
            if let Some(options) = options.dynamic {
                let input = recurse!(input, state)?;
                return Ok(Box::new(executors::GroupByDynamicExec {
                    input,
                    keys: phys_keys,
                    aggs: phys_aggs,
                    options,
                    input_schema,
                    slice: _slice,
                    apply,
                }));
            }

            #[cfg(feature = "dynamic_group_by")]
            if let Some(options) = options.rolling {
                let input = recurse!(input, state)?;
                return Ok(Box::new(executors::GroupByRollingExec {
                    input,
                    keys: phys_keys,
                    aggs: phys_aggs,
                    options,
                    input_schema,
                    slice: _slice,
                    apply,
                }));
            }

            // We first check if we can partition the group_by on the latest moment.
            let partitionable = partitionable_gb(&keys, &aggs, &input_schema, expr_arena, &apply);
            if partitionable {
                let from_partitioned_ds = (&*lp_arena).iter(input).any(|(_, lp)| {
                    if let Union { options, .. } = lp {
                        options.from_partitioned_ds
                    } else {
                        false
                    }
                });
                let input = recurse!(input, state)?;
                let keys = keys
                    .iter()
                    .map(|e| e.to_expr(expr_arena))
                    .collect::<Vec<_>>();
                let aggs = aggs
                    .iter()
                    .map(|e| e.to_expr(expr_arena))
                    .collect::<Vec<_>>();
                Ok(Box::new(executors::PartitionGroupByExec::new(
                    input,
                    phys_keys,
                    phys_aggs,
                    maintain_order,
                    options.slice,
                    input_schema,
                    schema,
                    from_partitioned_ds,
                    keys,
                    aggs,
                )))
            } else {
                let input = recurse!(input, state)?;
                Ok(Box::new(executors::GroupByExec::new(
                    input,
                    phys_keys,
                    phys_aggs,
                    apply,
                    maintain_order,
                    input_schema,
                    options.slice,
                )))
            }
        },
        Join {
            input_left,
            input_right,
            left_on,
            right_on,
            options,
            schema,
            ..
        } => {
            let schema_left = lp_arena.get(input_left).schema(lp_arena).into_owned();
            let schema_right = lp_arena.get(input_right).schema(lp_arena).into_owned();

            let (input_left, input_right) = state.with_new_branch(|new_state| {
                (
                    recurse!(input_left, new_state),
                    recurse!(input_right, new_state),
                )
            });
            let input_left = input_left?;
            let input_right = input_right?;

            // Todo! remove the force option. It can deadlock.
            let parallel = if options.force_parallel {
                true
            } else {
                options.allow_parallel
            };

            let left_on = create_physical_expressions_from_irs(
                &left_on,
                Context::Default,
                expr_arena,
                &schema_left,
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let right_on = create_physical_expressions_from_irs(
                &right_on,
                Context::Default,
                expr_arena,
                &schema_right,
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let options = Arc::try_unwrap(options).unwrap_or_else(|options| (*options).clone());

            // Convert the join options, to the physical join options. This requires the physical
            // planner, so we do this last minute.
            let join_type_options = options
                .options
                .map(|o| {
                    o.compile(|e| {
                        let phys_expr = create_physical_expr(
                            e,
                            Context::Default,
                            expr_arena,
                            &schema,
                            &mut ExpressionConversionState::new(false, state.expr_depth),
                        )?;

                        let execution_state = ExecutionState::default();

                        Ok(Arc::new(move |df: DataFrame| {
                            let mask = phys_expr.evaluate(&df, &execution_state)?;
                            let mask = mask.as_materialized_series();
                            let mask = mask.bool()?;
                            df._filter_seq(mask)
                        }))
                    })
                })
                .transpose()?;

            Ok(Box::new(executors::JoinExec::new(
                input_left,
                input_right,
                left_on,
                right_on,
                parallel,
                options.args,
                join_type_options,
            )))
        },
        HStack {
            input,
            exprs,
            schema: output_schema,
            options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = recurse!(input, state)?;

            let allow_vertical_parallelism = options.should_broadcast
                && exprs
                    .iter()
                    .all(|e| is_elementwise_rec_no_cat_cast(expr_arena.get(e.node()), expr_arena));

            let mut state = ExpressionConversionState::new(
                POOL.current_num_threads() > exprs.len(),
                state.expr_depth,
            );

            let phys_exprs = create_physical_expressions_from_irs(
                &exprs,
                Context::Default,
                expr_arena,
                &input_schema,
                &mut state,
            )?;
            Ok(Box::new(executors::StackExec {
                input,
                has_windows: state.has_windows,
                exprs: phys_exprs,
                input_schema,
                output_schema,
                options,
                allow_vertical_parallelism,
            }))
        },
        MapFunction {
            input, function, ..
        } => {
            let input = recurse!(input, state)?;
            Ok(Box::new(executors::UdfExec { input, function }))
        },
        ExtContext {
            input, contexts, ..
        } => {
            let input = recurse!(input, state)?;
            let contexts = contexts
                .into_iter()
                .map(|node| recurse!(node, state))
                .collect::<PolarsResult<_>>()?;
            Ok(Box::new(executors::ExternalContext { input, contexts }))
        },
        SimpleProjection { input, columns } => {
            let input = recurse!(input, state)?;
            let exec = executors::ProjectionSimple { input, columns };
            Ok(Box::new(exec))
        },
        #[cfg(feature = "merge_sorted")]
        MergeSorted {
            input_left,
            input_right,
            key,
        } => {
            let (input_left, input_right) = state.with_new_branch(|new_state| {
                (
                    recurse!(input_left, new_state),
                    recurse!(input_right, new_state),
                )
            });
            let input_left = input_left?;
            let input_right = input_right?;

            let exec = executors::MergeSorted {
                input_left,
                input_right,
                key,
            };
            Ok(Box::new(exec))
        },
        Invalid => unreachable!(),
    }
}

pub fn create_scan_predicate(
    predicate: &ExprIR,
    expr_arena: &mut Arena<AExpr>,
    schema: &Arc<Schema>,
    state: &mut ExpressionConversionState,
    create_skip_batch_predicate: bool,
    create_column_predicates: bool,
) -> PolarsResult<ScanPredicate> {
    let phys_predicate =
        create_physical_expr(predicate, Context::Default, expr_arena, schema, state)?;
    let live_columns = Arc::new(PlIndexSet::from_iter(aexpr_to_leaf_names_iter(
        predicate.node(),
        expr_arena,
    )));

    let mut skip_batch_predicate = None;

    if create_skip_batch_predicate {
        if let Some(node) = aexpr_to_skip_batch_predicate(predicate.node(), expr_arena, schema) {
            let expr = ExprIR::new(node, predicate.output_name_inner().clone());

            if std::env::var("POLARS_OUTPUT_SKIP_BATCH_PRED").as_deref() == Ok("1") {
                eprintln!("predicate: {}", predicate.display(expr_arena));
                eprintln!("skip_batch_predicate: {}", expr.display(expr_arena));
            }

            let mut skip_batch_schema = Schema::with_capacity(1 + live_columns.len());

            skip_batch_schema.insert(PlSmallStr::from_static("len"), IDX_DTYPE);
            for (col, dtype) in schema.iter() {
                if !live_columns.contains(col) {
                    continue;
                }

                skip_batch_schema.insert(format_pl_smallstr!("{col}_min"), dtype.clone());
                skip_batch_schema.insert(format_pl_smallstr!("{col}_max"), dtype.clone());
                skip_batch_schema.insert(format_pl_smallstr!("{col}_nc"), IDX_DTYPE);
            }

            skip_batch_predicate = Some(create_physical_expr(
                &expr,
                Context::Default,
                expr_arena,
                &Arc::new(skip_batch_schema),
                state,
            )?);
        }
    }

    let column_predicates = if create_column_predicates {
        let column_predicates = aexpr_to_column_predicates(predicate.node(), expr_arena, schema);
        if std::env::var("POLARS_OUTPUT_COLUMN_PREDS").as_deref() == Ok("1") {
            eprintln!("column_predicates: {{");
            eprintln!("  [");
            for (pred, spec) in column_predicates.predicates.values() {
                eprintln!(
                    "    {} ({spec:?}),",
                    ExprIRDisplay::display_node(*pred, expr_arena)
                );
            }
            eprintln!("  ],");
            eprintln!(
                "  is_sumwise_complete: {}",
                column_predicates.is_sumwise_complete
            );
            eprintln!("}}");
        }
        PhysicalColumnPredicates {
            predicates: column_predicates
                .predicates
                .into_iter()
                .map(|(n, (p, s))| {
                    PolarsResult::Ok((
                        n,
                        (
                            create_physical_expr(
                                &ExprIR::new(p, OutputName::Alias(PlSmallStr::EMPTY)),
                                Context::Default,
                                expr_arena,
                                schema,
                                state,
                            )?,
                            s,
                        ),
                    ))
                })
                .collect::<PolarsResult<PlHashMap<_, _>>>()?,
            is_sumwise_complete: column_predicates.is_sumwise_complete,
        }
    } else {
        PhysicalColumnPredicates {
            predicates: PlHashMap::default(),
            is_sumwise_complete: false,
        }
    };

    PolarsResult::Ok(ScanPredicate {
        predicate: phys_predicate,
        live_columns,
        skip_batch_predicate,
        column_predicates,
    })
}
