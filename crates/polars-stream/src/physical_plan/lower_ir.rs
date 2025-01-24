use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::frame::{DataFrame, UniqueKeepStrategy};
use polars_core::prelude::{DataType, InitHashMaps, PlHashMap, PlHashSet, PlIndexMap};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::RowIndex;
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, FileScan, FunctionIR, IRAggExpr, LiteralValue, IR};
use polars_plan::prelude::{FileType, GroupbyOptions, SinkType};
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream};
use crate::physical_plan::lower_expr::{
    build_select_stream, is_elementwise_rec_cached, lower_exprs, unique_column_name, ExprCache,
};
use crate::physical_plan::lower_group_by::build_group_by_stream;
use crate::utils::late_materialized_df::LateMaterializedDataFrame;

/// Creates a new PhysStream which outputs a slice of the input stream.
pub fn build_slice_stream(
    input: PhysStream,
    offset: i64,
    length: usize,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PhysStream {
    if offset >= 0 {
        let offset = offset as usize;
        PhysStream::first(phys_sm.insert(PhysNode::new(
            phys_sm[input.node].output_schema.clone(),
            PhysNodeKind::StreamingSlice {
                input,
                offset,
                length,
            },
        )))
    } else {
        todo!()
    }
}

/// Creates a new PhysStream which is filters the input stream.
fn build_filter_stream(
    input: PhysStream,
    predicate: ExprIR,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
) -> PolarsResult<PhysStream> {
    let predicate = predicate.clone();
    let cols_and_predicate = phys_sm[input.node]
        .output_schema
        .iter_names()
        .cloned()
        .map(|name| {
            ExprIR::new(
                expr_arena.add(AExpr::Column(name.clone())),
                OutputName::ColumnLhs(name),
            )
        })
        .chain([predicate])
        .collect_vec();
    let (trans_input, mut trans_cols_and_predicate) =
        lower_exprs(input, &cols_and_predicate, expr_arena, phys_sm, expr_cache)?;

    let filter_schema = phys_sm[trans_input.node].output_schema.clone();
    let filter = PhysNodeKind::Filter {
        input: trans_input,
        predicate: trans_cols_and_predicate.last().unwrap().clone(),
    };

    let post_filter = phys_sm.insert(PhysNode::new(filter_schema, filter));
    trans_cols_and_predicate.pop(); // Remove predicate.
    build_select_stream(
        PhysStream::first(post_filter),
        &trans_cols_and_predicate,
        expr_arena,
        phys_sm,
        expr_cache,
    )
}

#[recursive::recursive]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
    cache_nodes: &mut PlHashMap<usize, PhysStream>,
) -> PolarsResult<PhysStream> {
    // Helper macro to simplify recursive calls.
    macro_rules! lower_ir {
        ($input:expr) => {
            lower_ir(
                $input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
                cache_nodes,
            )
        };
    }

    let ir_node = ir_arena.get(node);
    let output_schema = IR::schema_with_cache(node, ir_arena, schema_cache);
    let node_kind = match ir_node {
        IR::SimpleProjection { input, columns } => {
            let columns = columns.iter_names_cloned().collect::<Vec<_>>();
            let phys_input = lower_ir!(*input)?;
            PhysNodeKind::SimpleProjection {
                input: phys_input,
                columns,
            }
        },

        IR::Select { input, expr, .. } => {
            let selectors = expr.clone();
            let phys_input = lower_ir!(*input)?;
            return build_select_stream(phys_input, &selectors, expr_arena, phys_sm, expr_cache);
        },

        IR::HStack { input, exprs, .. }
            if exprs
                .iter()
                .all(|e| is_elementwise_rec_cached(e.node(), expr_arena, expr_cache)) =>
        {
            // FIXME: constant literal columns should be broadcasted with hstack.
            let selectors = exprs.clone();
            let phys_input = lower_ir!(*input)?;
            PhysNodeKind::Select {
                input: phys_input,
                selectors,
                extend_original: true,
            }
        },

        IR::HStack { input, exprs, .. } => {
            // We already handled the all-streamable case above, so things get more complicated.
            // For simplicity we just do a normal select with all the original columns prepended.
            //
            // FIXME: constant literal columns should be broadcasted with hstack.
            let exprs = exprs.clone();
            let phys_input = lower_ir!(*input)?;
            let input_schema = &phys_sm[phys_input.node].output_schema;
            let mut selectors = PlIndexMap::with_capacity(input_schema.len() + exprs.len());
            for name in input_schema.iter_names() {
                let col_name = name.clone();
                let col_expr = expr_arena.add(AExpr::Column(col_name.clone()));
                selectors.insert(
                    name.clone(),
                    ExprIR::new(col_expr, OutputName::ColumnLhs(col_name)),
                );
            }
            for expr in exprs {
                selectors.insert(expr.output_name().clone(), expr);
            }
            let selectors = selectors.into_values().collect_vec();
            return build_select_stream(phys_input, &selectors, expr_arena, phys_sm, expr_cache);
        },

        IR::Slice { input, offset, len } => {
            let offset = *offset;
            let len = *len as usize;
            let phys_input = lower_ir!(*input)?;
            return Ok(build_slice_stream(phys_input, offset, len, phys_sm));
        },

        IR::Filter { input, predicate } => {
            let predicate = predicate.clone();
            let phys_input = lower_ir!(*input)?;
            return build_filter_stream(phys_input, predicate, expr_arena, phys_sm, expr_cache);
        },

        IR::DataFrameScan {
            df,
            output_schema: projection,
            schema,
            ..
        } => {
            let schema = schema.clone(); // This is initially the schema of df, but can change with the projection.
            let mut node_kind = PhysNodeKind::InMemorySource { df: df.clone() };

            // Do we need to apply a projection?
            if let Some(projection_schema) = projection {
                if projection_schema.len() != schema.len()
                    || projection_schema
                        .iter_names()
                        .zip(schema.iter_names())
                        .any(|(l, r)| l != r)
                {
                    let phys_input = phys_sm.insert(PhysNode::new(schema, node_kind));
                    node_kind = PhysNodeKind::SimpleProjection {
                        input: PhysStream::first(phys_input),
                        columns: projection_schema.iter_names_cloned().collect::<Vec<_>>(),
                    };
                }
            }

            node_kind
        },

        IR::Sink { input, payload } => match payload {
            SinkType::Memory => {
                let phys_input = lower_ir!(*input)?;
                PhysNodeKind::InMemorySink { input: phys_input }
            },
            SinkType::File {
                path,
                file_type,
                cloud_options: _,
            } => {
                let path = path.clone();
                let file_type = file_type.clone();

                match file_type {
                    #[cfg(feature = "ipc")]
                    FileType::Ipc(_) => {
                        let phys_input = lower_ir!(*input)?;
                        PhysNodeKind::FileSink {
                            path,
                            file_type,
                            input: phys_input,
                        }
                    },
                    #[cfg(feature = "parquet")]
                    FileType::Parquet(_) => {
                        let phys_input = lower_ir!(*input)?;
                        PhysNodeKind::FileSink {
                            path,
                            file_type,
                            input: phys_input,
                        }
                    },
                    #[cfg(feature = "csv")]
                    FileType::Csv(_) => {
                        let phys_input = lower_ir!(*input)?;
                        PhysNodeKind::FileSink {
                            path,
                            file_type,
                            input: phys_input,
                        }
                    },
                    #[cfg(feature = "json")]
                    FileType::Json(_) => {
                        let phys_input = lower_ir!(*input)?;
                        PhysNodeKind::FileSink {
                            path,
                            file_type,
                            input: phys_input,
                        }
                    },
                }
            },
        },

        IR::MapFunction { input, function } => {
            // MergeSorted uses a rechunk hack incompatible with the
            // streaming engine.
            #[cfg(feature = "merge_sorted")]
            if let FunctionIR::MergeSorted { .. } = function {
                todo!()
            }

            let function = function.clone();
            let phys_input = lower_ir!(*input)?;

            match function {
                FunctionIR::RowIndex {
                    name,
                    offset,
                    schema: _,
                } => PhysNodeKind::WithRowIndex {
                    input: phys_input,
                    name,
                    offset,
                },

                function if function.is_streamable() => {
                    let map = Arc::new(move |df| function.evaluate(df));
                    PhysNodeKind::Map {
                        input: phys_input,
                        map,
                    }
                },

                function => {
                    let map = Arc::new(move |df| function.evaluate(df));
                    PhysNodeKind::InMemoryMap {
                        input: phys_input,
                        map,
                    }
                },
            }
        },

        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => PhysNodeKind::Sort {
            by_column: by_column.clone(),
            slice: *slice,
            sort_options: sort_options.clone(),
            input: lower_ir!(*input)?,
        },

        IR::Union { inputs, options } => {
            let options = *options;
            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir!(input))
                .collect::<Result<_, _>>()?;

            let node = phys_sm.insert(PhysNode {
                output_schema,
                kind: PhysNodeKind::OrderedUnion { inputs },
            });
            let mut stream = PhysStream::first(node);
            if let Some((offset, length)) = options.slice {
                stream = build_slice_stream(stream, offset, length, phys_sm);
            }
            return Ok(stream);
        },

        IR::HConcat {
            inputs,
            schema: _,
            options: _,
        } => {
            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir!(input))
                .collect::<Result<_, _>>()?;
            PhysNodeKind::Zip {
                inputs,
                null_extend: true,
            }
        },

        v @ IR::Scan { .. } => {
            let IR::Scan {
                sources: scan_sources,
                file_info,
                hive_parts,
                output_schema: scan_output_schema,
                scan_type,
                mut predicate,
                mut file_options,
            } = v.clone()
            else {
                unreachable!();
            };

            if scan_sources.is_empty() {
                // If there are no sources, just provide an empty in-memory source with the right
                // schema.
                PhysNodeKind::InMemorySource {
                    df: Arc::new(DataFrame::empty_with_schema(output_schema.as_ref())),
                }
            } else if scan_sources.len() > 1 || hive_parts.is_some() {
                // @TODO: At the moment, we materialize for Hive. I would also not like to do this,
                // but at the moment it is this or panicking.
                //
                // This is very much a hack, forgive me please.
                let phys_node = phys_sm.insert(PhysNode {
                    output_schema: Arc::new(Schema::default()),
                    kind: PhysNodeKind::InputIndependentSelect {
                        selectors: Vec::new(),
                    },
                });
                let input = PhysStream::first(phys_node);
                let in_memory_physical_plan = create_physical_plan(node, ir_arena, expr_arena)?;
                let in_memory_physical_plan =
                    Arc::new(std::sync::Mutex::new(in_memory_physical_plan));

                PhysNodeKind::InMemoryMap {
                    input,
                    map: Arc::new(move |_| {
                        let mut in_memory_physical_plan = in_memory_physical_plan.lock().unwrap();
                        in_memory_physical_plan.execute(&mut Default::default())
                    }),
                }
            } else {
                #[cfg(feature = "ipc")]
                if matches!(scan_type, FileScan::Ipc { .. }) {
                    // @TODO: All the things the IPC source does not support yet.
                    if scan_sources.is_cloud_url() {
                        todo!();
                    }
                }

                // Operation ordering:
                // * with_row_index() -> slice() -> filter()

                // Some scans have built-in support for applying these operations in an optimized manner.
                let opt_rewrite_to_nodes: (Option<RowIndex>, Option<(i64, usize)>, Option<ExprIR>) =
                    match &scan_type {
                        #[cfg(feature = "parquet")]
                        FileScan::Parquet { .. } => (None, None, None),
                        #[cfg(feature = "ipc")]
                        FileScan::Ipc { .. } => (None, None, predicate.take()),
                        #[cfg(feature = "csv")]
                        FileScan::Csv { options, .. } => {
                            if options.parse_options.comment_prefix.is_none()
                                && std::env::var("POLARS_DISABLE_EXPERIMENTAL_CSV_SLICE").as_deref()
                                    != Ok("1")
                            {
                                // Note: This relies on `CountLines` being exact.
                                (None, None, predicate.take())
                            } else {
                                // There can be comments in the middle of the file, then `CountLines` won't
                                // return an accurate line count :'(.
                                (
                                    file_options.row_index.take(),
                                    file_options.slice.take(),
                                    predicate.take(),
                                )
                            }
                        },
                        _ => todo!(),
                    };

                let node_kind = PhysNodeKind::FileScan {
                    scan_sources,
                    file_info,
                    hive_parts,
                    output_schema: scan_output_schema,
                    scan_type,
                    predicate,
                    file_options,
                };

                let (row_index, slice, predicate) = opt_rewrite_to_nodes;

                let node_kind = if let Some(ri) = row_index {
                    let mut schema = Arc::unwrap_or_clone(output_schema.clone());

                    let v = schema.shift_remove_index(0).unwrap().0;
                    assert_eq!(v, ri.name);
                    let input_node = phys_sm.insert(PhysNode::new(Arc::new(schema), node_kind));

                    PhysNodeKind::WithRowIndex {
                        input: PhysStream::first(input_node),
                        name: ri.name,
                        offset: Some(ri.offset),
                    }
                } else {
                    node_kind
                };

                let node = phys_sm.insert(PhysNode {
                    output_schema,
                    kind: node_kind,
                });
                let mut stream = PhysStream::first(node);

                if let Some((offset, length)) = slice {
                    stream = build_slice_stream(stream, offset, length, phys_sm);
                }

                if let Some(predicate) = predicate {
                    stream =
                        build_filter_stream(stream, predicate, expr_arena, phys_sm, expr_cache)?;
                }

                return Ok(stream);
            }
        },

        #[cfg(feature = "python")]
        IR::PythonScan { .. } => todo!(),

        IR::Cache {
            input,
            id,
            cache_hits: _,
        } => {
            let id = *id;
            if let Some(cached) = cache_nodes.get(&id) {
                return Ok(*cached);
            }

            let phys_input = lower_ir!(*input)?;
            cache_nodes.insert(id, phys_input);
            return Ok(phys_input);
        },

        IR::GroupBy {
            input,
            keys,
            aggs,
            schema: output_schema,
            apply,
            maintain_order,
            options,
        } => {
            let input = *input;
            let keys = keys.clone();
            let aggs = aggs.clone();
            let output_schema = output_schema.clone();
            let apply = apply.clone();
            let maintain_order = *maintain_order;
            let options = options.clone();

            let phys_input = lower_ir!(input)?;
            return build_group_by_stream(
                phys_input,
                &keys,
                &aggs,
                output_schema,
                maintain_order,
                options.clone(),
                apply,
                expr_arena,
                phys_sm,
                expr_cache,
            );
        },
        IR::Join {
            input_left,
            input_right,
            schema: _,
            left_on,
            right_on,
            options,
        } => {
            let input_left = *input_left;
            let input_right = *input_right;
            let left_on = left_on.clone();
            let right_on = right_on.clone();
            let args = options.args.clone();
            let options = options.options.clone();
            let phys_left = lower_ir!(input_left)?;
            let phys_right = lower_ir!(input_right)?;
            if args.how.is_equi() && !args.validation.needs_checks() {
                // When lowering the expressions for the keys we need to ensure we keep around the
                // payload columns, otherwise the input nodes can get replaced by input-independent
                // nodes since the lowering code does not see we access any non-literal expressions.
                // So we add dummy expressions before lowering and remove them afterwards.
                let mut aug_left_on = left_on.clone();
                for name in phys_sm[phys_left.node].output_schema.iter_names() {
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    aug_left_on.push(ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone())));
                }
                let mut aug_right_on = right_on.clone();
                for name in phys_sm[phys_right.node].output_schema.iter_names() {
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    aug_right_on.push(ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone())));
                }
                let (trans_input_left, mut trans_left_on) =
                    lower_exprs(phys_left, &aug_left_on, expr_arena, phys_sm, expr_cache)?;
                let (trans_input_right, mut trans_right_on) =
                    lower_exprs(phys_right, &aug_right_on, expr_arena, phys_sm, expr_cache)?;
                trans_left_on.drain(left_on.len()..);
                trans_right_on.drain(right_on.len()..);

                let node = phys_sm.insert(PhysNode::new(
                    output_schema,
                    PhysNodeKind::EquiJoin {
                        input_left: trans_input_left,
                        input_right: trans_input_right,
                        left_on: trans_left_on,
                        right_on: trans_right_on,
                        args: args.clone(),
                    },
                ));
                let mut stream = PhysStream::first(node);
                if let Some((offset, len)) = args.slice {
                    stream = build_slice_stream(stream, offset, len, phys_sm);
                }
                return Ok(stream);
            } else {
                PhysNodeKind::InMemoryJoin {
                    input_left: phys_left,
                    input_right: phys_right,
                    left_on,
                    right_on,
                    args,
                    options,
                }
            }
        },

        IR::Distinct { input, options } => {
            let options = options.clone();
            let phys_input = lower_ir!(*input)?;

            // We don't have a dedicated distinct operator (yet), lower to group
            // by with an aggregate for each column.
            let input_schema = &phys_sm[phys_input.node].output_schema;
            if input_schema.is_empty() {
                // Can't group (or have duplicates) if dataframe has zero-width.
                return Ok(phys_input);
            }

            if options.maintain_order && options.keep_strategy == UniqueKeepStrategy::Last {
                // Unfortunately the order-preserving groupby always orders by the first occurrence
                // of the group so we can't lower this and have to fallback.
                let input_schema = phys_sm[phys_input.node].output_schema.clone();
                let lmdf = Arc::new(LateMaterializedDataFrame::default());
                let mut lp_arena = Arena::default();
                let input_lp_node = lp_arena.add(lmdf.clone().as_ir_node(input_schema.clone()));
                let distinct_lp_node = lp_arena.add(IR::Distinct {
                    input: input_lp_node,
                    options,
                });
                let executor = Mutex::new(create_physical_plan(
                    distinct_lp_node,
                    &mut lp_arena,
                    expr_arena,
                )?);

                let distinct_node = PhysNode {
                    output_schema,
                    kind: PhysNodeKind::InMemoryMap {
                        input: phys_input,
                        map: Arc::new(move |df| {
                            lmdf.set_materialized_dataframe(df);
                            let mut state = ExecutionState::new();
                            executor.lock().execute(&mut state)
                        }),
                    },
                };

                return Ok(PhysStream::first(phys_sm.insert(distinct_node)));
            }

            // Create the key and aggregate expressions.
            let all_col_names = input_schema.iter_names().cloned().collect_vec();
            let key_names = if let Some(subset) = options.subset {
                subset.to_vec()
            } else {
                all_col_names.clone()
            };
            let key_name_set: PlHashSet<_> = key_names.iter().cloned().collect();

            let mut group_by_output_schema = Schema::with_capacity(all_col_names.len() + 1);
            let keys = key_names
                .iter()
                .map(|name| {
                    group_by_output_schema
                        .insert(name.clone(), input_schema.get(name).unwrap().clone());
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone()))
                })
                .collect_vec();

            let mut aggs = all_col_names
                .iter()
                .filter(|name| !key_name_set.contains(*name))
                .map(|name| {
                    group_by_output_schema
                        .insert(name.clone(), input_schema.get(name).unwrap().clone());
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    use UniqueKeepStrategy::*;
                    let agg_expr = match options.keep_strategy {
                        First | None | Any => {
                            expr_arena.add(AExpr::Agg(IRAggExpr::First(col_expr)))
                        },
                        Last => expr_arena.add(AExpr::Agg(IRAggExpr::Last(col_expr))),
                    };
                    ExprIR::new(agg_expr, OutputName::ColumnLhs(name.clone()))
                })
                .collect_vec();

            if options.keep_strategy == UniqueKeepStrategy::None {
                // Track the length so we can filter out non-unique keys later.
                let name = unique_column_name();
                group_by_output_schema.insert(name.clone(), DataType::new_idxsize());
                aggs.push(ExprIR::new(
                    expr_arena.add(AExpr::Len),
                    OutputName::Alias(name),
                ));
            }

            let mut stream = build_group_by_stream(
                phys_input,
                &keys,
                &aggs,
                Arc::new(group_by_output_schema),
                options.maintain_order,
                Arc::new(GroupbyOptions::default()),
                None,
                expr_arena,
                phys_sm,
                expr_cache,
            )?;

            if options.keep_strategy == UniqueKeepStrategy::None {
                // Filter to keep only those groups with length 1.
                let unique_name = aggs.last().unwrap().output_name();
                let left = expr_arena.add(AExpr::Column(unique_name.clone()));
                let right = expr_arena.add(AExpr::Literal(LiteralValue::new_idxsize(1)));
                let predicate_aexpr = expr_arena.add(AExpr::BinaryExpr {
                    left,
                    op: polars_plan::dsl::Operator::Eq,
                    right,
                });
                let predicate =
                    ExprIR::new(predicate_aexpr, OutputName::ColumnLhs(unique_name.clone()));
                stream = build_filter_stream(stream, predicate, expr_arena, phys_sm, expr_cache)?;
            }

            // Restore column order and drop the temporary length column if any.
            let exprs = all_col_names
                .iter()
                .map(|name| {
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone()))
                })
                .collect_vec();
            stream = build_select_stream(stream, &exprs, expr_arena, phys_sm, expr_cache)?;

            // We didn't pass the slice earlier to build_group_by_stream because
            // we might have the intermediate keep = "none" filter.
            if let Some((offset, length)) = options.slice {
                stream = build_slice_stream(stream, offset, length, phys_sm);
            }

            return Ok(stream);
        },
        IR::ExtContext { .. } => todo!(),
        IR::Invalid => unreachable!(),
    };

    let node_key = phys_sm.insert(PhysNode::new(output_schema, node_kind));
    Ok(PhysStream::first(node_key))
}
