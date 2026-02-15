use polars_core::POOL;
use polars_core::prelude::*;
use polars_expr::state::ExecutionState;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::prelude::sink::CallbackSinkType;
use polars_utils::unique_id::UniqueId;
use recursive::recursive;

#[cfg(feature = "python")]
use self::python_dsl::PythonScanSource;
use super::*;
use crate::executors::{self, CachePrefiller, Executor, GroupByStreamingExec, SinkExecutor};
use crate::scan_predicate::functions::create_scan_predicate;

pub type StreamingExecutorBuilder =
    fn(Node, &mut Arena<IR>, &mut Arena<AExpr>) -> PolarsResult<Box<dyn Executor>>;

fn partitionable_gb(
    keys: &[ExprIR],
    aggs: &[ExprIR],
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
    apply: &Option<PlanCallback<DataFrame, DataFrame>>,
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
    has_cache_child: bool,
    has_cache_parent: bool,
}

impl ConversionState {
    fn new() -> PolarsResult<Self> {
        Ok(ConversionState {
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
    build_streaming_executor: Option<StreamingExecutorBuilder>,
) -> PolarsResult<Box<dyn Executor>> {
    let mut state = ConversionState::new()?;
    let mut cache_nodes = Default::default();
    let plan = create_physical_plan_impl(
        root,
        lp_arena,
        expr_arena,
        &mut state,
        &mut cache_nodes,
        build_streaming_executor,
    )?;

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
    build_streaming_executor: Option<StreamingExecutorBuilder>,
) -> PolarsResult<MultiplePhysicalPlans> {
    let mut state = ConversionState::new()?;
    let mut cache_nodes = Default::default();
    let plans = state.with_new_branch(|new_state| {
        roots
            .iter()
            .map(|&node| {
                create_physical_plan_impl(
                    node,
                    lp_arena,
                    expr_arena,
                    new_state,
                    &mut cache_nodes,
                    build_streaming_executor,
                )
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

#[cfg(feature = "python")]
#[allow(clippy::type_complexity)]
pub fn python_scan_predicate(
    options: &mut PythonOptions,
    expr_arena: &mut Arena<AExpr>,
    state: &mut ExpressionConversionState,
) -> PolarsResult<(
    Option<Arc<dyn polars_expr::prelude::PhysicalExpr>>,
    Option<Vec<u8>>,
)> {
    let mut predicate_serialized = None;
    let predicate = if let PythonPredicate::Polars(e) = &options.predicate {
        // Convert to a pyarrow eval string.
        if matches!(options.python_source, PythonScanSource::Pyarrow) {
            use polars_core::config::verbose_print_sensitive;

            let predicate_pa = polars_plan::plans::python::pyarrow::predicate_to_pa(
                e.node(),
                expr_arena,
                Default::default(),
            );

            verbose_print_sensitive(|| {
                format!(
                    "python_scan_predicate: \
                    predicate node: {}, \
                    converted pyarrow predicate: {}",
                    ExprIRDisplay::display_node(e.node(), expr_arena),
                    &predicate_pa.as_deref().unwrap_or("<conversion failed>")
                )
            });

            if let Some(eval_str) = predicate_pa {
                options.predicate = PythonPredicate::PyArrow(eval_str);
                // We don't have to use a physical expression as pyarrow deals with the filter.
                None
            } else {
                Some(create_physical_expr(e, expr_arena, &options.schema, state)?)
            }
        }
        // Convert to physical expression for the case the reader cannot consume the predicate.
        else {
            let dsl_expr = e.to_expr(expr_arena);
            predicate_serialized = polars_plan::plans::python::predicate::serialize(&dsl_expr)?;

            Some(create_physical_expr(e, expr_arena, &options.schema, state)?)
        }
    } else {
        None
    };

    Ok((predicate, predicate_serialized))
}

#[recursive]
fn create_physical_plan_impl(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    state: &mut ConversionState,
    // Cache nodes in order of discovery
    cache_nodes: &mut PlIndexMap<UniqueId, executors::CachePrefill>,
    build_streaming_executor: Option<StreamingExecutorBuilder>,
) -> PolarsResult<Box<dyn Executor>> {
    use IR::*;

    let get_streaming_executor_builder = || {
        build_streaming_executor.expect(
            "get_streaming_executor_builder() failed (hint: missing feature new-streaming?)",
        )
    };

    macro_rules! recurse {
        ($node:expr, $state: expr) => {
            create_physical_plan_impl(
                $node,
                lp_arena,
                expr_arena,
                $state,
                cache_nodes,
                build_streaming_executor,
            )
        };
    }

    let logical_plan = if state.has_cache_parent
        || matches!(
            lp_arena.get(root),
            IR::Scan { .. } // Needed for the streaming impl
                | IR::Cache { .. } // Needed for plans branching from the same cache node
                | IR::GroupBy { .. } // Needed for the streaming impl
                | IR::Sink { // Needed for the streaming impl
                    payload:
                        SinkTypeIR::File(_) | SinkTypeIR::Partitioned { .. },
                    ..
                }
        ) {
        lp_arena.get(root).clone()
    } else {
        lp_arena.take(root)
    };

    match logical_plan {
        #[cfg(feature = "python")]
        PythonScan { mut options } => {
            let mut expr_conv_state = ExpressionConversionState::new(true);
            let (predicate, predicate_serialized) =
                python_scan_predicate(&mut options, expr_arena, &mut expr_conv_state)?;
            Ok(Box::new(executors::PythonScanExec {
                options,
                predicate,
                predicate_serialized,
            }))
        },
        Sink { input, payload } => match payload {
            SinkTypeIR::Memory => Ok(Box::new(SinkExecutor {
                input: recurse!(input, state)?,
                name: PlSmallStr::from_static("mem"),
                f: Box::new(move |df, _state| Ok(Some(df))),
            })),
            SinkTypeIR::Callback(CallbackSinkType {
                function,
                maintain_order: _,
                chunk_size,
            }) => {
                let chunk_size = chunk_size.map_or(usize::MAX, Into::into);

                Ok(Box::new(SinkExecutor {
                    input: recurse!(input, state)?,
                    name: PlSmallStr::from_static("batches"),
                    f: Box::new(move |mut buffer, _state| {
                        while buffer.height() > 0 {
                            let df;
                            (df, buffer) = buffer.split_at(buffer.height().min(chunk_size) as i64);
                            let should_stop = function.call(df)?;
                            if should_stop {
                                break;
                            }
                        }
                        Ok(Some(DataFrame::empty()))
                    }),
                }))
            },
            SinkTypeIR::File(_) | SinkTypeIR::Partitioned { .. } => {
                get_streaming_executor_builder()(root, lp_arena, expr_arena)
            },
        },
        SinkMultiple { .. } => {
            polars_bail!(InvalidOperation: "lazy multisinks only supported on streaming engine")
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
            let streamable = is_elementwise_rec(predicate.node(), expr_arena);
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = recurse!(input, state)?;
            let mut state = ExpressionConversionState::new(true);
            let predicate =
                create_physical_expr(&predicate, expr_arena, &input_schema, &mut state)?;
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
            predicate_file_skip_applied,
            unified_scan_args,
        } => {
            let mut expr_conversion_state = ExpressionConversionState::new(true);

            let mut create_skip_batch_predicate = unified_scan_args.table_statistics.is_some();
            #[cfg(feature = "parquet")]
            {
                if let FileScanIR::Parquet { options, .. } = scan_type.as_ref() {
                    create_skip_batch_predicate |= options.use_statistics;
                }
            }

            let predicate = predicate
                .map(|predicate| {
                    create_scan_predicate(
                        &predicate,
                        expr_arena,
                        output_schema.as_ref().unwrap_or(&file_info.schema),
                        None, // hive_schema
                        &mut expr_conversion_state,
                        create_skip_batch_predicate,
                        false,
                    )
                })
                .transpose()?;

            match *scan_type {
                FileScanIR::Anonymous { function, .. } => {
                    Ok(Box::new(executors::AnonymousScanExec {
                        function,
                        predicate,
                        unified_scan_args,
                        file_info,
                        output_schema,
                        predicate_has_windows: expr_conversion_state.has_windows,
                    }))
                },
                #[cfg_attr(
                    not(any(
                        feature = "parquet",
                        feature = "ipc",
                        feature = "csv",
                        feature = "json",
                        feature = "scan_lines"
                    )),
                    expect(unreachable_patterns)
                )]
                _ => get_streaming_executor_builder()(root, lp_arena, expr_arena),
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
            let mut state = ExpressionConversionState::new(POOL.current_num_threads() > expr.len());
            let phys_expr =
                create_physical_expressions_from_irs(&expr, expr_arena, &input_schema, &mut state)?;

            let allow_vertical_parallelism = options.should_broadcast && expr.iter().all(|e| is_elementwise_rec(e.node(), expr_arena))
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
            debug_assert!(!by_column.is_empty());
            let input_schema = lp_arena.get(input).schema(lp_arena);
            let by_column = create_physical_expressions_from_irs(
                &by_column,
                expr_arena,
                input_schema.as_ref(),
                &mut ExpressionConversionState::new(true),
            )?;
            let input = recurse!(input, state)?;
            Ok(Box::new(executors::SortExec {
                input,
                by_column,
                slice,
                sort_options,
            }))
        },
        Cache { input, id } => {
            state.has_cache_parent = true;
            state.has_cache_child = true;

            if let Some(cache) = cache_nodes.get_mut(&id) {
                Ok(Box::new(cache.make_exec()))
            } else {
                let input = recurse!(input, state)?;

                let mut prefill = executors::CachePrefill::new_cache(input, id);
                let exec = prefill.make_exec();

                cache_nodes.insert(id, prefill);

                Ok(Box::new(exec))
            }
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
            schema: output_schema,
            maintain_order,
            options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let options = Arc::try_unwrap(options).unwrap_or_else(|options| (*options).clone());
            let phys_keys = create_physical_expressions_from_irs(
                &keys,
                expr_arena,
                &input_schema,
                &mut ExpressionConversionState::new(true),
            )?;
            let phys_aggs = create_physical_expressions_from_irs(
                &aggs,
                expr_arena,
                &input_schema,
                &mut ExpressionConversionState::new(true),
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
                    output_schema,
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
                    output_schema,
                    slice: _slice,
                    apply,
                }));
            }

            // We first check if we can partition the group_by on the latest moment.
            let partitionable = partitionable_gb(&keys, &aggs, &input_schema, expr_arena, &apply);
            if partitionable {
                let from_partitioned_ds = lp_arena.iter(input).any(|(_, lp)| {
                    if let Union { options, .. } = lp {
                        options.from_partitioned_ds
                    } else {
                        false
                    }
                });
                let builder = get_streaming_executor_builder();

                let input = recurse!(input, state)?;

                let gb_root = if state.has_cache_parent {
                    lp_arena.add(lp_arena.get(root).clone())
                } else {
                    root
                };

                let executor = Box::new(GroupByStreamingExec::new(
                    input,
                    builder,
                    gb_root,
                    lp_arena,
                    expr_arena,
                    phys_keys,
                    phys_aggs,
                    maintain_order,
                    output_schema,
                    _slice,
                    from_partitioned_ds,
                ));

                Ok(executor)
            } else {
                let input = recurse!(input, state)?;
                Ok(Box::new(executors::GroupByExec::new(
                    input,
                    phys_keys,
                    phys_aggs,
                    apply,
                    maintain_order,
                    input_schema,
                    output_schema,
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
                expr_arena,
                &schema_left,
                &mut ExpressionConversionState::new(true),
            )?;
            let right_on = create_physical_expressions_from_irs(
                &right_on,
                expr_arena,
                &schema_right,
                &mut ExpressionConversionState::new(true),
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
                            expr_arena,
                            &schema,
                            &mut ExpressionConversionState::new(false),
                        )?;

                        let execution_state = ExecutionState::default();

                        Ok(Arc::new(move |df: DataFrame| {
                            let mask = phys_expr.evaluate(&df, &execution_state)?;
                            let mask = mask.as_materialized_series();
                            let mask = mask.bool()?;
                            df.filter_seq(mask)
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
                    .all(|e| is_elementwise_rec(e.node(), expr_arena));

            let mut state =
                ExpressionConversionState::new(POOL.current_num_threads() > exprs.len());

            let phys_exprs = create_physical_expressions_from_irs(
                &exprs,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_multiple_physical_plans_reused_cache() {
        // Check that reusing the same cache node doesn't panic.
        // CSE creates duplicate cache nodes with the same ID, but cloud reuses them.

        let mut ir = Arena::new();

        let schema = Schema::from_iter([(PlSmallStr::from_static("x"), DataType::Float32)]);
        let scan = ir.add(IR::DataFrameScan {
            df: Arc::new(DataFrame::empty_with_schema(&schema)),
            schema: Arc::new(schema),
            output_schema: None,
        });

        let cache = ir.add(IR::Cache {
            input: scan,
            id: UniqueId::new(),
        });

        let left_sink = ir.add(IR::Sink {
            input: cache,
            payload: SinkTypeIR::Memory,
        });
        let right_sink = ir.add(IR::Sink {
            input: cache,
            payload: SinkTypeIR::Memory,
        });

        let _multiplan = create_multiple_physical_plans(
            &[left_sink, right_sink],
            &mut ir,
            &mut Arena::new(),
            None,
        )
        .unwrap();
    }
}
