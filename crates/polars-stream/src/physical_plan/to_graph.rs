use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::prelude::PlRandomState;
use polars_core::schema::{Schema, SchemaExt};
use polars_error::PolarsResult;
use polars_expr::groups::new_hash_grouper;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_expr::reduce::into_reduction;
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_plan::dsl::JoinOptions;
use polars_plan::global::_set_n_rows_for_scan;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, ArenaExprIter, Context, IR};
use polars_plan::prelude::{FileType, FunctionFlags};
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use recursive::recursive;
use slotmap::{SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind};
use crate::expression::StreamExpr;
use crate::graph::{Graph, GraphNodeKey};
use crate::nodes;
use crate::physical_plan::lower_expr::compute_output_schema;
use crate::utils::late_materialized_df::LateMaterializedDataFrame;

fn has_potential_recurring_entrance(node: Node, arena: &Arena<AExpr>) -> bool {
    arena.iter(node).any(|(_n, ae)| match ae {
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            options.flags.contains(FunctionFlags::OPTIONAL_RE_ENTRANT)
        },
        _ => false,
    })
}

fn create_stream_expr(
    expr_ir: &ExprIR,
    ctx: &mut GraphConversionContext<'_>,
    schema: &Arc<Schema>,
) -> PolarsResult<StreamExpr> {
    let reentrant = has_potential_recurring_entrance(expr_ir.node(), ctx.expr_arena);
    let phys = create_physical_expr(
        expr_ir,
        Context::Default,
        ctx.expr_arena,
        schema,
        &mut ctx.expr_conversion_state,
    )?;
    Ok(StreamExpr::new(phys, reentrant))
}

struct GraphConversionContext<'a> {
    phys_sm: &'a SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &'a mut Arena<AExpr>,
    graph: Graph,
    phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
    expr_conversion_state: ExpressionConversionState,
}

pub fn physical_plan_to_graph(
    root: PhysNodeKey,
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<(Graph, SecondaryMap<PhysNodeKey, GraphNodeKey>)> {
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut ctx = GraphConversionContext {
        phys_sm,
        expr_arena,
        graph: Graph::with_capacity(phys_sm.len()),
        phys_to_graph: SecondaryMap::with_capacity(phys_sm.len()),
        expr_conversion_state: ExpressionConversionState::new(false, expr_depth_limit),
    };

    to_graph_rec(root, &mut ctx)?;

    Ok((ctx.graph, ctx.phys_to_graph))
}

#[recursive]
fn to_graph_rec<'a>(
    phys_node_key: PhysNodeKey,
    ctx: &mut GraphConversionContext<'a>,
) -> PolarsResult<GraphNodeKey> {
    // This will ensure we create a proper acyclic directed graph instead of a tree.
    if let Some(graph_key) = ctx.phys_to_graph.get(phys_node_key) {
        return Ok(*graph_key);
    }

    use PhysNodeKind::*;
    let node = &ctx.phys_sm[phys_node_key];
    let graph_key = match &node.kind {
        InMemorySource { df } => ctx.graph.add_node(
            nodes::in_memory_source::InMemorySourceNode::new(df.clone()),
            [],
        ),

        StreamingSlice {
            input,
            offset,
            length,
        } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::streaming_slice::StreamingSliceNode::new(*offset, *length),
                [input_key],
            )
        },

        Filter { predicate, input } => {
            let input_schema = &ctx.phys_sm[*input].output_schema;
            let phys_predicate_expr = create_stream_expr(predicate, ctx, input_schema)?;
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::filter::FilterNode::new(phys_predicate_expr),
                [input_key],
            )
        },

        Select {
            selectors,
            input,
            extend_original,
        } => {
            let input_schema = &ctx.phys_sm[*input].output_schema;
            let phys_selectors = selectors
                .iter()
                .map(|selector| create_stream_expr(selector, ctx, input_schema))
                .collect::<PolarsResult<_>>()?;
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::select::SelectNode::new(
                    phys_selectors,
                    node.output_schema.clone(),
                    *extend_original,
                ),
                [input_key],
            )
        },

        WithRowIndex {
            input,
            name,
            offset,
        } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::with_row_index::WithRowIndexNode::new(name.clone(), *offset),
                [input_key],
            )
        },

        InputIndependentSelect { selectors } => {
            let empty_schema = Default::default();
            let phys_selectors = selectors
                .iter()
                .map(|selector| create_stream_expr(selector, ctx, &empty_schema))
                .collect::<PolarsResult<_>>()?;
            ctx.graph.add_node(
                nodes::input_independent_select::InputIndependentSelectNode::new(phys_selectors),
                [],
            )
        },

        Reduce { input, exprs } => {
            let input_key = to_graph_rec(*input, ctx)?;
            let input_schema = &ctx.phys_sm[*input].output_schema;

            let mut reductions = Vec::with_capacity(exprs.len());
            let mut inputs = Vec::with_capacity(reductions.len());

            for e in exprs {
                let (red, input_node) = into_reduction(e.node(), ctx.expr_arena, input_schema)?;
                reductions.push(red);

                let input_phys = create_stream_expr(
                    &ExprIR::from_node(input_node, ctx.expr_arena),
                    ctx,
                    input_schema,
                )?;

                inputs.push(input_phys)
            }

            ctx.graph.add_node(
                nodes::reduce::ReduceNode::new(inputs, reductions, node.output_schema.clone()),
                [input_key],
            )
        },
        SimpleProjection { input, columns } => {
            let input_schema = ctx.phys_sm[*input].output_schema.clone();
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::simple_projection::SimpleProjectionNode::new(columns.clone(), input_schema),
                [input_key],
            )
        },

        InMemorySink { input } => {
            let input_schema = ctx.phys_sm[*input].output_schema.clone();
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_sink::InMemorySinkNode::new(input_schema),
                [input_key],
            )
        },

        FileSink {
            path,
            file_type,
            input,
        } => {
            let input_schema = ctx.phys_sm[*input].output_schema.clone();
            let input_key = to_graph_rec(*input, ctx)?;

            match file_type {
                #[cfg(feature = "ipc")]
                FileType::Ipc(ipc_writer_options) => ctx.graph.add_node(
                    nodes::io_sinks::ipc::IpcSinkNode::new(input_schema, path, ipc_writer_options)?,
                    [input_key],
                ),
                _ => todo!(),
            }
        },

        InMemoryMap { input, map } => {
            let input_schema = ctx.phys_sm[*input].output_schema.clone();
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_map::InMemoryMapNode::new(input_schema, map.clone()),
                [input_key],
            )
        },

        Map { input, map } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph
                .add_node(nodes::map::MapNode::new(map.clone()), [input_key])
        },

        Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input_schema = ctx.phys_sm[*input].output_schema.clone();
            let lmdf = Arc::new(LateMaterializedDataFrame::default());
            let mut lp_arena = Arena::default();
            let df_node = lp_arena.add(lmdf.clone().as_ir_node(input_schema.clone()));
            let sort_node = lp_arena.add(IR::Sort {
                input: df_node,
                by_column: by_column.clone(),
                slice: *slice,
                sort_options: sort_options.clone(),
            });
            let executor = Mutex::new(create_physical_plan(
                sort_node,
                &mut lp_arena,
                ctx.expr_arena,
            )?);

            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_map::InMemoryMapNode::new(
                    input_schema,
                    Arc::new(move |df| {
                        lmdf.set_materialized_dataframe(df);
                        let mut state = ExecutionState::new();
                        executor.lock().execute(&mut state)
                    }),
                ),
                [input_key],
            )
        },

        OrderedUnion { inputs } => {
            let input_keys = inputs
                .iter()
                .map(|i| to_graph_rec(*i, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            ctx.graph
                .add_node(nodes::ordered_union::OrderedUnionNode::new(), input_keys)
        },

        Zip {
            inputs,
            null_extend,
        } => {
            let input_schemas = inputs
                .iter()
                .map(|i| ctx.phys_sm[*i].output_schema.clone())
                .collect_vec();
            let input_keys = inputs
                .iter()
                .map(|i| to_graph_rec(*i, ctx))
                .try_collect_vec()?;
            ctx.graph.add_node(
                nodes::zip::ZipNode::new(*null_extend, input_schemas),
                input_keys,
            )
        },

        Multiplexer { input } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph
                .add_node(nodes::multiplexer::MultiplexerNode::new(), [input_key])
        },

        v @ FileScan { .. } => {
            let FileScan {
                scan_sources,
                file_info,
                hive_parts,
                output_schema,
                scan_type,
                predicate,
                mut file_options,
            } = v.clone()
            else {
                unreachable!()
            };

            file_options.slice = if let Some((offset, len)) = file_options.slice {
                Some((offset, _set_n_rows_for_scan(Some(len)).unwrap()))
            } else {
                _set_n_rows_for_scan(None).map(|x| (0, x))
            };

            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(
                        &pred,
                        Context::Default,
                        ctx.expr_arena,
                        output_schema.as_ref().unwrap_or(&file_info.schema),
                        &mut ctx.expr_conversion_state,
                    )
                })
                .map_or(Ok(None), |v| v.map(Some))?;

            {
                use polars_plan::prelude::FileScan;

                match scan_type {
                    #[cfg(feature = "parquet")]
                    FileScan::Parquet {
                        options,
                        cloud_options,
                        metadata: first_metadata,
                    } => {
                        if std::env::var("POLARS_DISABLE_PARQUET_SOURCE").as_deref() != Ok("1") {
                            ctx.graph.add_node(
                                nodes::parquet_source::ParquetSourceNode::new(
                                    scan_sources,
                                    file_info,
                                    hive_parts,
                                    predicate,
                                    options,
                                    cloud_options,
                                    file_options,
                                    first_metadata,
                                ),
                                [],
                            )
                        } else {
                            todo!()
                        }
                    },
                    FileScan::Ipc {
                        options,
                        cloud_options,
                        metadata: first_metadata,
                    } => ctx.graph.add_node(
                        nodes::io_sources::ipc::IpcSourceNode::new(
                            scan_sources,
                            file_info,
                            hive_parts,
                            predicate,
                            options,
                            cloud_options,
                            file_options,
                            first_metadata,
                        )?,
                        [],
                    ),
                    FileScan::Csv { options, .. } => {
                        assert!(predicate.is_none());

                        if options.parse_options.comment_prefix.is_some() {
                            // Should have been re-written to separate streaming nodes
                            assert!(file_options.row_index.is_none());
                            assert!(file_options.slice.is_none());
                        }

                        ctx.graph.add_node(
                            nodes::csv_source::CsvSourceNode::new(
                                scan_sources,
                                file_info,
                                file_options,
                                options,
                            ),
                            [],
                        )
                    },
                    _ => todo!(),
                }
            }
        },

        GroupBy { input, key, aggs } => {
            let input_key = to_graph_rec(*input, ctx)?;

            let input_schema = &ctx.phys_sm[*input].output_schema;
            let key_schema = compute_output_schema(input_schema, key, ctx.expr_arena)?
                .materialize_unknown_dtypes()?;
            let grouper = new_hash_grouper(Arc::new(key_schema));

            let key_selectors = key
                .iter()
                .map(|e| create_stream_expr(e, ctx, input_schema))
                .try_collect_vec()?;

            let mut grouped_reductions = Vec::new();
            let mut grouped_reduction_selectors = Vec::new();
            for agg in aggs {
                let (reduction, input_node) =
                    into_reduction(agg.node(), ctx.expr_arena, input_schema)?;
                let selector = create_stream_expr(
                    &ExprIR::from_node(input_node, ctx.expr_arena),
                    ctx,
                    input_schema,
                )?;
                grouped_reductions.push(reduction);
                grouped_reduction_selectors.push(selector);
            }

            ctx.graph.add_node(
                nodes::group_by::GroupByNode::new(
                    key_selectors,
                    grouped_reduction_selectors,
                    grouped_reductions,
                    grouper,
                    node.output_schema.clone(),
                    PlRandomState::new(),
                ),
                [input_key],
            )
        },

        InMemoryJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
        } => {
            let left_input_key = to_graph_rec(*input_left, ctx)?;
            let right_input_key = to_graph_rec(*input_right, ctx)?;
            let left_input_schema = ctx.phys_sm[*input_left].output_schema.clone();
            let right_input_schema = ctx.phys_sm[*input_right].output_schema.clone();

            let mut lp_arena = Arena::default();
            let left_lmdf = Arc::new(LateMaterializedDataFrame::default());
            let right_lmdf = Arc::new(LateMaterializedDataFrame::default());

            let left_node = lp_arena.add(left_lmdf.clone().as_ir_node(left_input_schema.clone()));
            let right_node =
                lp_arena.add(right_lmdf.clone().as_ir_node(right_input_schema.clone()));
            let join_node = lp_arena.add(IR::Join {
                input_left: left_node,
                input_right: right_node,
                schema: node.output_schema.clone(),
                left_on: left_on.clone(),
                right_on: right_on.clone(),
                options: Arc::new(JoinOptions {
                    allow_parallel: true,
                    force_parallel: false,
                    args: args.clone(),
                    rows_left: (None, 0),
                    rows_right: (None, 0),
                }),
            });

            let executor = Mutex::new(create_physical_plan(
                join_node,
                &mut lp_arena,
                ctx.expr_arena,
            )?);

            ctx.graph.add_node(
                nodes::joins::in_memory::InMemoryJoinNode::new(
                    left_input_schema,
                    right_input_schema,
                    Arc::new(move |left, right| {
                        left_lmdf.set_materialized_dataframe(left);
                        right_lmdf.set_materialized_dataframe(right);
                        let mut state = ExecutionState::new();
                        executor.lock().execute(&mut state)
                    }),
                ),
                [left_input_key, right_input_key],
            )
        },
    };

    ctx.phys_to_graph.insert(phys_node_key, graph_key);
    Ok(graph_key)
}
