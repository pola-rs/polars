use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::Mutex;
use polars_core::prelude::PlRandomState;
use polars_core::schema::Schema;
use polars_core::{POOL, config};
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_expr::groups::new_hash_grouper;
use polars_expr::planner::{ExpressionConversionState, create_physical_expr};
use polars_expr::reduce::into_reduction;
use polars_expr::state::ExecutionState;
use polars_mem_engine::{create_physical_plan, create_scan_predicate};
use polars_plan::dsl::{JoinOptions, PartitionVariantIR, ScanSources};
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, ArenaExprIter, Context, IR};
use polars_plan::prelude::{FileType, FunctionFlags};
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use recursive::recursive;
use slotmap::{SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind};
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::{Graph, GraphNodeKey};
use crate::morsel::{MorselSeq, get_ideal_morsel_size};
use crate::nodes;
use crate::nodes::io_sinks::SinkComputeNode;
use crate::nodes::io_sources::multi_file_reader::MultiFileReaderConfig;
use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;
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
    num_pipelines: usize,
}

pub fn physical_plan_to_graph(
    root: PhysNodeKey,
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<(Graph, SecondaryMap<PhysNodeKey, GraphNodeKey>)> {
    // Get the number of threads from the rayon thread-pool as that respects our config.
    let num_pipelines = POOL.current_num_threads();
    let mut ctx = GraphConversionContext {
        phys_sm,
        expr_arena,
        graph: Graph::with_capacity(phys_sm.len()),
        phys_to_graph: SecondaryMap::with_capacity(phys_sm.len()),
        expr_conversion_state: ExpressionConversionState::new(false),
        num_pipelines,
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
            nodes::in_memory_source::InMemorySourceNode::new(df.clone(), MorselSeq::default()),
            [],
        ),
        SinkMultiple { sinks } => {
            // @NOTE: This is always the root node and gets ignored by the physical_plan anyway so
            // we give one of the inputs back.
            let node = to_graph_rec(sinks[0], ctx)?;
            for sink in &sinks[1..] {
                to_graph_rec(*sink, ctx)?;
            }
            return Ok(node);
        },

        StreamingSlice {
            input,
            offset,
            length,
        } => {
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::streaming_slice::StreamingSliceNode::new(*offset, *length),
                [(input_key, input.port)],
            )
        },

        NegativeSlice {
            input,
            offset,
            length,
        } => {
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::negative_slice::NegativeSliceNode::new(*offset, *length),
                [(input_key, input.port)],
            )
        },

        Filter { predicate, input } => {
            let input_schema = &ctx.phys_sm[input.node].output_schema;
            let phys_predicate_expr = create_stream_expr(predicate, ctx, input_schema)?;
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::filter::FilterNode::new(phys_predicate_expr),
                [(input_key, input.port)],
            )
        },

        Select {
            selectors,
            input,
            extend_original,
        } => {
            let input_schema = &ctx.phys_sm[input.node].output_schema;
            let phys_selectors = selectors
                .iter()
                .map(|selector| create_stream_expr(selector, ctx, input_schema))
                .collect::<PolarsResult<_>>()?;
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::select::SelectNode::new(
                    phys_selectors,
                    node.output_schema.clone(),
                    *extend_original,
                ),
                [(input_key, input.port)],
            )
        },

        WithRowIndex {
            input,
            name,
            offset,
        } => {
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::with_row_index::WithRowIndexNode::new(name.clone(), *offset),
                [(input_key, input.port)],
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
            let input_key = to_graph_rec(input.node, ctx)?;
            let input_schema = &ctx.phys_sm[input.node].output_schema;

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
                [(input_key, input.port)],
            )
        },
        SimpleProjection { input, columns } => {
            let input_schema = ctx.phys_sm[input.node].output_schema.clone();
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::simple_projection::SimpleProjectionNode::new(columns.clone(), input_schema),
                [(input_key, input.port)],
            )
        },

        InMemorySink { input } => {
            let input_schema = ctx.phys_sm[input.node].output_schema.clone();
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_sink::InMemorySinkNode::new(input_schema),
                [(input_key, input.port)],
            )
        },

        FileSink {
            target,
            sink_options,
            file_type,
            input,
            cloud_options,
        } => {
            let sink_options = sink_options.clone();
            let input_schema = ctx.phys_sm[input.node].output_schema.clone();
            let input_key = to_graph_rec(input.node, ctx)?;

            match file_type {
                #[cfg(feature = "ipc")]
                FileType::Ipc(ipc_writer_options) => ctx.graph.add_node(
                    SinkComputeNode::from(nodes::io_sinks::ipc::IpcSinkNode::new(
                        input_schema,
                        target.clone(),
                        sink_options,
                        *ipc_writer_options,
                        cloud_options.clone(),
                    )),
                    [(input_key, input.port)],
                ),
                #[cfg(feature = "json")]
                FileType::Json(_) => ctx.graph.add_node(
                    SinkComputeNode::from(nodes::io_sinks::json::NDJsonSinkNode::new(
                        target.clone(),
                        sink_options,
                        cloud_options.clone(),
                    )),
                    [(input_key, input.port)],
                ),
                #[cfg(feature = "parquet")]
                FileType::Parquet(parquet_writer_options) => ctx.graph.add_node(
                    SinkComputeNode::from(nodes::io_sinks::parquet::ParquetSinkNode::new(
                        input_schema,
                        target.clone(),
                        sink_options,
                        parquet_writer_options,
                        cloud_options.clone(),
                    )?),
                    [(input_key, input.port)],
                ),
                #[cfg(feature = "csv")]
                FileType::Csv(csv_writer_options) => ctx.graph.add_node(
                    SinkComputeNode::from(nodes::io_sinks::csv::CsvSinkNode::new(
                        target.clone(),
                        input_schema,
                        sink_options,
                        csv_writer_options.clone(),
                        cloud_options.clone(),
                    )),
                    [(input_key, input.port)],
                ),
                #[cfg(not(any(
                    feature = "csv",
                    feature = "parquet",
                    feature = "json",
                    feature = "ipc"
                )))]
                _ => {
                    panic!("activate source feature")
                },
            }
        },

        PartitionSink {
            base_path,
            file_path_cb,
            sink_options,
            variant,
            file_type,
            input,
            cloud_options,
        } => {
            let input_schema = ctx.phys_sm[input.node].output_schema.clone();
            let input_key = to_graph_rec(input.node, ctx)?;

            let base_path = base_path.clone();
            let file_path_cb = file_path_cb.clone();
            let ext = PlSmallStr::from_static(file_type.extension());
            let create_new = nodes::io_sinks::partition::get_create_new_fn(
                file_type.clone(),
                sink_options.clone(),
                cloud_options.clone(),
            );

            match variant {
                PartitionVariantIR::MaxSize(max_size) => ctx.graph.add_node(
                    SinkComputeNode::from(
                        nodes::io_sinks::partition::max_size::MaxSizePartitionSinkNode::new(
                            input_schema,
                            *max_size,
                            base_path,
                            file_path_cb,
                            create_new,
                            ext,
                            sink_options.clone(),
                        ),
                    ),
                    [(input_key, input.port)],
                ),
                PartitionVariantIR::Parted {
                    key_exprs,
                    include_key,
                } => ctx.graph.add_node(
                    SinkComputeNode::from(
                        nodes::io_sinks::partition::parted::PartedPartitionSinkNode::new(
                            input_schema,
                            key_exprs.iter().map(|e| e.output_name().clone()).collect(),
                            base_path,
                            file_path_cb,
                            create_new,
                            ext,
                            sink_options.clone(),
                            *include_key,
                        ),
                    ),
                    [(input_key, input.port)],
                ),
                PartitionVariantIR::ByKey {
                    key_exprs,
                    include_key,
                } => ctx.graph.add_node(
                    SinkComputeNode::from(
                        nodes::io_sinks::partition::by_key::PartitionByKeySinkNode::new(
                            input_schema,
                            key_exprs.iter().map(|e| e.output_name().clone()).collect(),
                            base_path,
                            file_path_cb,
                            create_new,
                            ext,
                            sink_options.clone(),
                            *include_key,
                        ),
                    ),
                    [(input_key, input.port)],
                ),
            }
        },

        InMemoryMap {
            input,
            map,
            format_str: _,
        } => {
            let input_schema = ctx.phys_sm[input.node].output_schema.clone();
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_map::InMemoryMapNode::new(input_schema, map.clone()),
                [(input_key, input.port)],
            )
        },

        Map { input, map } => {
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::map::MapNode::new(map.clone()),
                [(input_key, input.port)],
            )
        },

        Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input_schema = ctx.phys_sm[input.node].output_schema.clone();
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
                None,
            )?);

            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_map::InMemoryMapNode::new(
                    input_schema,
                    Arc::new(move |df| {
                        lmdf.set_materialized_dataframe(df);
                        let mut state = ExecutionState::new();
                        executor.lock().execute(&mut state)
                    }),
                ),
                [(input_key, input.port)],
            )
        },

        OrderedUnion { inputs } => {
            let input_keys = inputs
                .iter()
                .map(|i| PolarsResult::Ok((to_graph_rec(i.node, ctx)?, i.port)))
                .try_collect_vec()?;
            ctx.graph
                .add_node(nodes::ordered_union::OrderedUnionNode::new(), input_keys)
        },

        Zip {
            inputs,
            null_extend,
        } => {
            let input_schemas = inputs
                .iter()
                .map(|i| ctx.phys_sm[i.node].output_schema.clone())
                .collect_vec();
            let input_keys = inputs
                .iter()
                .map(|i| PolarsResult::Ok((to_graph_rec(i.node, ctx)?, i.port)))
                .try_collect_vec()?;
            ctx.graph.add_node(
                nodes::zip::ZipNode::new(*null_extend, input_schemas),
                input_keys,
            )
        },

        Multiplexer { input } => {
            let input_key = to_graph_rec(input.node, ctx)?;
            ctx.graph.add_node(
                nodes::multiplexer::MultiplexerNode::new(),
                [(input_key, input.port)],
            )
        },

        MultiScan {
            scan_sources,
            file_reader_builder,
            cloud_options,
            projected_file_schema,
            output_schema,
            row_index,
            pre_slice,
            predicate,
            hive_parts,
            missing_columns_policy,
            extra_columns_policy,
            cast_columns_policy,
            include_file_paths,
            file_schema,
        } => {
            let hive_parts = hive_parts.clone();

            let predicate = predicate
                .as_ref()
                .map(|pred| {
                    create_scan_predicate(
                        pred,
                        ctx.expr_arena,
                        output_schema,
                        &mut ctx.expr_conversion_state,
                        true, // create_skip_batch_predicate
                        file_reader_builder
                            .reader_capabilities()
                            .contains(ReaderCapabilities::PARTIAL_FILTER), // create_column_predicates
                    )
                })
                .transpose()?
                .map(|p| p.to_io(None, file_schema.clone()));

            let sources = scan_sources.clone();
            let file_reader_builder = file_reader_builder.clone();
            let cloud_options = cloud_options.clone();

            let final_output_schema = output_schema.clone();
            let projected_file_schema = projected_file_schema.clone();
            let full_file_schema = file_schema.clone();

            let row_index = row_index.clone();
            let pre_slice = pre_slice.clone();
            let hive_parts = hive_parts.map(Arc::new);
            let include_file_paths = include_file_paths.clone();
            let missing_columns_policy = *missing_columns_policy;
            let extra_columns_policy = *extra_columns_policy;
            let cast_columns_policy = cast_columns_policy.clone();

            let verbose = config::verbose();

            ctx.graph.add_node(
                nodes::io_sources::multi_file_reader::MultiFileReader::new(Arc::new(
                    MultiFileReaderConfig {
                        sources,
                        file_reader_builder,
                        cloud_options,
                        final_output_schema,
                        projected_file_schema,
                        full_file_schema,
                        row_index,
                        pre_slice,
                        predicate,
                        hive_parts,
                        include_file_paths,
                        missing_columns_policy,
                        extra_columns_policy,
                        cast_columns_policy,
                        // Initialized later
                        num_pipelines: AtomicUsize::new(0),
                        n_readers_pre_init: AtomicUsize::new(0),
                        max_concurrent_scans: AtomicUsize::new(0),
                        verbose,
                    },
                )),
                [],
            )
        },

        GroupBy { input, key, aggs } => {
            let input_key = to_graph_rec(input.node, ctx)?;

            let input_schema = &ctx.phys_sm[input.node].output_schema;
            let key_schema = compute_output_schema(input_schema, key, ctx.expr_arena)?;
            let grouper = new_hash_grouper(key_schema.clone());

            let key_selectors = key
                .iter()
                .map(|e| create_stream_expr(e, ctx, input_schema))
                .try_collect_vec()?;

            let mut grouped_reductions = Vec::new();
            let mut grouped_reduction_cols = Vec::new();
            for agg in aggs {
                let (reduction, input_node) =
                    into_reduction(agg.node(), ctx.expr_arena, input_schema)?;
                let AExpr::Column(col) = ctx.expr_arena.get(input_node) else {
                    unreachable!()
                };
                grouped_reductions.push(reduction);
                grouped_reduction_cols.push(col.clone());
            }

            ctx.graph.add_node(
                nodes::group_by::GroupByNode::new(
                    key_schema,
                    key_selectors,
                    grouper,
                    grouped_reduction_cols,
                    grouped_reductions,
                    node.output_schema.clone(),
                    PlRandomState::default(),
                    ctx.num_pipelines,
                ),
                [(input_key, input.port)],
            )
        },

        InMemoryJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            options,
        } => {
            let left_input_key = to_graph_rec(input_left.node, ctx)?;
            let right_input_key = to_graph_rec(input_right.node, ctx)?;
            let left_input_schema = ctx.phys_sm[input_left.node].output_schema.clone();
            let right_input_schema = ctx.phys_sm[input_right.node].output_schema.clone();

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
                    options: options.clone(),
                    rows_left: (None, 0),
                    rows_right: (None, 0),
                }),
            });

            let executor = Mutex::new(create_physical_plan(
                join_node,
                &mut lp_arena,
                ctx.expr_arena,
                None,
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
                [
                    (left_input_key, input_left.port),
                    (right_input_key, input_right.port),
                ],
            )
        },

        EquiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
        }
        | SemiAntiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            output_bool: _,
        } => {
            let args = args.clone();
            let left_input_key = to_graph_rec(input_left.node, ctx)?;
            let right_input_key = to_graph_rec(input_right.node, ctx)?;
            let left_input_schema = ctx.phys_sm[input_left.node].output_schema.clone();
            let right_input_schema = ctx.phys_sm[input_right.node].output_schema.clone();

            let left_key_schema =
                compute_output_schema(&left_input_schema, left_on, ctx.expr_arena)?;
            let right_key_schema =
                compute_output_schema(&right_input_schema, right_on, ctx.expr_arena)?;

            // We want to make sure here that the key types match otherwise we get out garbage out
            // since the hashes will be calculated differently.
            polars_ensure!(
                left_on.len() == right_on.len() &&
                left_on.iter().zip(right_on.iter()).all(|(l, r)| {
                    let l_dtype = left_key_schema.get(l.output_name()).unwrap();
                    let r_dtype = right_key_schema.get(r.output_name()).unwrap();
                    l_dtype == r_dtype
                }),
                SchemaMismatch: "join received different key types on left and right side"
            );

            // We use key columns entirely by position, and allow duplicate names in key selectors,
            // so just assign arbitrary unique names for the selectors.
            let unique_left_on = left_on
                .iter()
                .enumerate()
                .map(|(i, expr)| expr.with_alias(format_pl_smallstr!("__POLARS_KEYCOL_{i}")))
                .collect_vec();
            let unique_right_on = right_on
                .iter()
                .enumerate()
                .map(|(i, expr)| expr.with_alias(format_pl_smallstr!("__POLARS_KEYCOL_{i}")))
                .collect_vec();

            let left_key_selectors = unique_left_on
                .iter()
                .map(|e| create_stream_expr(e, ctx, &left_input_schema))
                .try_collect_vec()?;
            let right_key_selectors = unique_right_on
                .iter()
                .map(|e| create_stream_expr(e, ctx, &right_input_schema))
                .try_collect_vec()?;

            let unique_key_schema =
                compute_output_schema(&right_input_schema, &unique_left_on, ctx.expr_arena)?;

            match node.kind {
                #[cfg(feature = "semi_anti_join")]
                SemiAntiJoin { output_bool, .. } => ctx.graph.add_node(
                    nodes::joins::semi_anti_join::SemiAntiJoinNode::new(
                        unique_key_schema,
                        left_key_selectors,
                        right_key_selectors,
                        args,
                        output_bool,
                        ctx.num_pipelines,
                    )?,
                    [
                        (left_input_key, input_left.port),
                        (right_input_key, input_right.port),
                    ],
                ),
                _ => ctx.graph.add_node(
                    nodes::joins::equi_join::EquiJoinNode::new(
                        left_input_schema,
                        right_input_schema,
                        left_key_schema,
                        right_key_schema,
                        unique_key_schema,
                        left_key_selectors,
                        right_key_selectors,
                        args,
                        ctx.num_pipelines,
                    )?,
                    [
                        (left_input_key, input_left.port),
                        (right_input_key, input_right.port),
                    ],
                ),
            }
        },

        CrossJoin {
            input_left,
            input_right,
            args,
        } => {
            let args = args.clone();
            let left_input_key = to_graph_rec(input_left.node, ctx)?;
            let right_input_key = to_graph_rec(input_right.node, ctx)?;
            let left_input_schema = ctx.phys_sm[input_left.node].output_schema.clone();
            let right_input_schema = ctx.phys_sm[input_right.node].output_schema.clone();

            ctx.graph.add_node(
                nodes::joins::cross_join::CrossJoinNode::new(
                    left_input_schema,
                    right_input_schema,
                    &args,
                ),
                [
                    (left_input_key, input_left.port),
                    (right_input_key, input_right.port),
                ],
            )
        },

        #[cfg(feature = "merge_sorted")]
        MergeSorted {
            input_left,
            input_right,
            key,
        } => {
            let left_input_key = to_graph_rec(input_left.node, ctx)?;
            let right_input_key = to_graph_rec(input_right.node, ctx)?;

            let input_schema = ctx.phys_sm[input_left.node].output_schema.clone();

            ctx.graph.add_node(
                nodes::merge_sorted::MergeSortedNode::new(input_schema, key.clone()),
                [
                    (left_input_key, input_left.port),
                    (right_input_key, input_right.port),
                ],
            )
        },

        #[cfg(feature = "python")]
        PythonScan { options } => {
            use polars_plan::dsl::python_dsl::PythonScanSource as S;
            use polars_plan::plans::PythonPredicate;
            use pyo3::exceptions::PyStopIteration;
            use pyo3::prelude::*;
            use pyo3::types::{PyBytes, PyNone};
            use pyo3::{IntoPyObjectExt, PyTypeInfo, intern};

            let mut options = options.clone();
            let with_columns = options.with_columns.take();
            let n_rows = options.n_rows.take();

            let python_scan_function = options.scan_fn.take().unwrap().0;

            let with_columns = with_columns.map(|cols| cols.iter().cloned().collect::<Vec<_>>());

            let (pl_predicate, predicate_serialized) = polars_mem_engine::python_scan_predicate(
                &mut options,
                ctx.expr_arena,
                &mut ctx.expr_conversion_state,
            )?;

            let output_schema = options.output_schema.unwrap_or(options.schema);
            let validate_schema = options.validate_schema;

            let (name, get_batch_fn) = match options.python_source {
                S::Pyarrow => todo!(),
                S::Cuda => todo!(),
                S::IOPlugin => {
                    let batch_size = Some(get_ideal_morsel_size());
                    let output_schema = output_schema.clone();

                    let with_columns = with_columns.map(|x| {
                        x.into_iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>()
                    });

                    // Setup the IO plugin generator.
                    let (generator, can_parse_predicate) = {
                        Python::with_gil(|py| {
                            let pl = PyModule::import(py, intern!(py, "polars")).unwrap();
                            let utils = pl.getattr(intern!(py, "_utils")).unwrap();
                            let callable =
                                utils.getattr(intern!(py, "_execute_from_rust")).unwrap();

                            let mut could_serialize_predicate = true;
                            let predicate = match &options.predicate {
                                PythonPredicate::PyArrow(s) => s.into_bound_py_any(py).unwrap(),
                                PythonPredicate::None => None::<()>.into_bound_py_any(py).unwrap(),
                                PythonPredicate::Polars(_) => {
                                    assert!(pl_predicate.is_some(), "should be set");
                                    match &predicate_serialized {
                                        None => {
                                            could_serialize_predicate = false;
                                            PyNone::get(py).to_owned().into_any()
                                        },
                                        Some(buf) => PyBytes::new(py, buf).into_any(),
                                    }
                                },
                            };

                            let args = (
                                python_scan_function,
                                with_columns,
                                predicate,
                                n_rows,
                                batch_size,
                            );

                            let generator_init = callable.call1(args)?;
                            let generator = generator_init.get_item(0).map_err(
                                |_| polars_err!(ComputeError: "expected tuple got {generator_init}"),
                            )?;
                            let can_parse_predicate = generator_init.get_item(1).map_err(
                                |_| polars_err!(ComputeError: "expected tuple got {generator}"),
                            )?;
                            let can_parse_predicate = can_parse_predicate.extract::<bool>().map_err(
                                |_| polars_err!(ComputeError: "expected bool got {can_parse_predicate}"),
                            )? && could_serialize_predicate;

                            let generator = generator.into_py_any(py).map_err(
                                |_| polars_err!(ComputeError: "unable to grab reference to IO plugin generator"),
                            )?;

                            PolarsResult::Ok((generator, can_parse_predicate))
                        })
                    }?;

                    let get_batch_fn = Box::new(move |state: &StreamingExecutionState| {
                        let df = Python::with_gil(|py| {
                            match generator.bind(py).call_method0(intern!(py, "__next__")) {
                                Ok(out) => polars_plan::plans::python_df_to_rust(py, out).map(Some),
                                Err(err)
                                    if err.matches(py, PyStopIteration::type_object(py))? =>
                                {
                                    Ok(None)
                                },
                                Err(err) => polars_bail!(
                                    ComputeError: "caught exception during execution of a Python source, exception: {err}"
                                ),
                            }
                        })?;

                        let Some(mut df) = df else { return Ok(None) };

                        if validate_schema {
                            polars_ensure!(
                                df.schema() == &output_schema,
                                SchemaMismatch: "user provided schema: {:?} doesn't match the DataFrame schema: {:?}",
                                output_schema, df.schema()
                            );
                        }

                        // TODO: Move this to a FilterNode so that it happens in parallel. We may need
                        // to move all of the enclosing code to `lower_ir` for this.
                        if let (Some(pred), false) = (&pl_predicate, can_parse_predicate) {
                            let mask = pred.evaluate(&df, &state.in_memory_exec_state)?;
                            df = df.filter(mask.bool()?)?;
                        }

                        Ok(Some(df))
                    }) as Box<_>;

                    (PlSmallStr::from_static("io_plugin"), get_batch_fn)
                },
            };

            use polars_plan::dsl::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy};

            use crate::nodes::io_sources::batch::builder::BatchFnReaderBuilder;
            use crate::nodes::io_sources::batch::{BatchFnReader, GetBatchState};

            let mut reader = BatchFnReader {
                name: name.clone(),
                // If validate_schema is false, the schema of the morsels may not match the
                // configured schema. In this case we set this to `None` and the reader will
                // retrieve the schema from the first morsel.
                output_schema: validate_schema.then(|| output_schema.clone()),
                get_batch_state: Some(GetBatchState::from(get_batch_fn)),
                verbose: config::verbose(),
            };

            // Note: This will potentially override the output schema if `validate_schema` is `false`.
            let output_schema = reader._file_schema()?;

            let file_reader_builder = Arc::new(BatchFnReaderBuilder {
                name,
                reader: std::sync::Mutex::new(Some(reader)),
            }) as Arc<dyn FileReaderBuilder>;

            // Give multiscan a single scan source. (It doesn't actually read from this).
            let sources = ScanSources::Paths(Arc::from([PathBuf::from("python-scan-0")]));
            let cloud_options = None;
            let final_output_schema = output_schema.clone();
            let projected_file_schema = output_schema.clone();
            let full_file_schema = output_schema.clone();
            let row_index = None;
            let pre_slice = None;
            let predicate = None;
            let hive_parts = None;
            let include_file_paths = None;
            let missing_columns_policy = MissingColumnsPolicy::Raise;
            let extra_columns_policy = ExtraColumnsPolicy::Ignore;
            let cast_columns_policy = CastColumnsPolicy::ERROR_ON_MISMATCH;
            let verbose = config::verbose();

            ctx.graph.add_node(
                nodes::io_sources::multi_file_reader::MultiFileReader::new(Arc::new(
                    MultiFileReaderConfig {
                        sources,
                        file_reader_builder,
                        cloud_options,
                        final_output_schema,
                        projected_file_schema,
                        full_file_schema,
                        row_index,
                        pre_slice,
                        predicate,
                        hive_parts,
                        include_file_paths,
                        missing_columns_policy,
                        extra_columns_policy,
                        cast_columns_policy,
                        // Initialized later
                        num_pipelines: AtomicUsize::new(0),
                        n_readers_pre_init: AtomicUsize::new(0),
                        max_concurrent_scans: AtomicUsize::new(0),
                        verbose,
                    },
                )),
                [],
            )
        },
    };

    ctx.phys_to_graph.insert(phys_node_key, graph_key);
    Ok(graph_key)
}
