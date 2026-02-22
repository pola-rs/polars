#![allow(unused)] // TODO: remove me
use std::cmp::Reverse;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_expr::planner::{ExpressionConversionState, create_physical_expr, get_expr_depth_limit};
use polars_plan::plans::{IR, IRPlan};
use polars_plan::prelude::AExpr;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_utils::arena::{Arena, Node};
use polars_utils::relaxed_cell::RelaxedCell;
use slotmap::{SecondaryMap, SlotMap};

use crate::graph::{Graph, GraphNodeKey};
use crate::metrics::GraphMetrics;
use crate::physical_plan::{PhysNode, PhysNodeKey, PhysNodeKind, StreamingLowerIRContext};

/// Executes the IR with the streaming engine.
///
/// Unsupported operations can fall back to the in-memory engine.
///
/// Returns:
/// - `Ok(QueryResult::Single(DataFrame))` when collecting to a single sink.
/// - `Ok(QueryResult::Multiple(Vec<DataFrame>))` when collecting to multiple sinks.
/// - `Err` if the IR can't be executed.
///
/// Returned `DataFrame`s contain data only for memory sinks,
/// `DataFrame`s corresponding to file sinks are empty.
pub fn run_query(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<QueryResult> {
    StreamingQuery::build(node, ir_arena, expr_arena)?.execute()
}

/// Visualizes the physical plan as a dot graph.
pub fn visualize_physical_plan(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<String> {
    let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());

    let ctx = StreamingLowerIRContext {
        prepare_visualization: true,
    };
    let root_phys_node =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, &mut phys_sm, ctx)?;

    let out = crate::physical_plan::visualize_plan(root_phys_node, &phys_sm, expr_arena);

    Ok(out)
}

pub struct StreamingQuery {
    top_ir: IR,
    pub graph: Graph,
    pub root_phys_node: PhysNodeKey,
    pub phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    pub phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
    pub metrics: Option<Arc<Mutex<GraphMetrics>>>,
}

/// Configures if IR lowering creates the `format_str` for `InMemoryMap`.
pub static PREPARE_VISUALIZATION_DATA: RelaxedCell<bool> = RelaxedCell::new_bool(false);

/// Sets config to ensure IR lowering always creates the `format_str` for `InMemoryMap`.
pub fn always_prepare_visualization_data() {
    PREPARE_VISUALIZATION_DATA.store(true);
}

fn cfg_prepare_visualization_data() -> bool {
    if !PREPARE_VISUALIZATION_DATA.load() {
        PREPARE_VISUALIZATION_DATA.fetch_or(
            std::env::var("POLARS_STREAM_ALWAYS_PREPARE_VISUALIZATION_DATA").as_deref() == Ok("1"),
        );
    }

    PREPARE_VISUALIZATION_DATA.load()
}

impl StreamingQuery {
    pub fn build(
        node: Node,
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Self> {
        if let Ok(visual_path) = std::env::var("POLARS_VISUALIZE_IR") {
            let plan = IRPlan {
                lp_top: node,
                lp_arena: ir_arena.clone(),
                expr_arena: expr_arena.clone(),
            };
            let visualization = plan.display_dot().to_string();
            std::fs::write(visual_path, visualization).unwrap();
        }
        let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
        let ctx = StreamingLowerIRContext {
            prepare_visualization: cfg_prepare_visualization_data(),
        };
        let root_phys_node = crate::physical_plan::build_physical_plan(
            node,
            ir_arena,
            expr_arena,
            &mut phys_sm,
            ctx,
        )?;
        if let Ok(visual_path) = std::env::var("POLARS_VISUALIZE_PHYSICAL_PLAN") {
            let visualization =
                crate::physical_plan::visualize_plan(root_phys_node, &phys_sm, expr_arena);
            std::fs::write(visual_path, visualization).unwrap();
        }

        let (mut graph, phys_to_graph) =
            crate::physical_plan::physical_plan_to_graph(root_phys_node, &phys_sm, expr_arena)?;

        let top_ir = ir_arena.get(node).clone();

        let metrics = if std::env::var("POLARS_TRACK_METRICS").as_deref() == Ok("1")
            || std::env::var("POLARS_LOG_METRICS").as_deref() == Ok("1")
        {
            crate::async_executor::track_task_metrics(true);
            Some(Arc::default())
        } else {
            None
        };

        let out = StreamingQuery {
            top_ir,
            graph,
            root_phys_node,
            phys_sm,
            phys_to_graph,
            metrics,
        };

        Ok(out)
    }

    pub fn execute(self) -> PolarsResult<QueryResult> {
        let StreamingQuery {
            top_ir,
            mut graph,
            root_phys_node,
            phys_sm,
            phys_to_graph,
            metrics,
        } = self;

        let query_start = Instant::now();
        let mut results = crate::execute::execute_graph(&mut graph, metrics.clone())?;
        let query_elapsed = query_start.elapsed();

        // Print metrics.
        if let Some(lock) = metrics
            && std::env::var("POLARS_LOG_METRICS").as_deref() == Ok("1")
        {
            let mut total_query_ns = 0;
            let mut lines = Vec::new();
            let m = lock.lock();
            for phys_node_key in phys_sm.keys() {
                let Some(graph_node_key) = phys_to_graph.get(phys_node_key) else {
                    continue;
                };
                let Some(node_metrics) = m.get(*graph_node_key) else {
                    continue;
                };
                let name = graph.nodes[*graph_node_key].compute.name();
                let total_ns =
                    node_metrics.total_poll_time_ns + node_metrics.total_state_update_time_ns;
                let total_time = Duration::from_nanos(total_ns);
                let poll_time = Duration::from_nanos(node_metrics.total_poll_time_ns);
                let update_time = Duration::from_nanos(node_metrics.total_state_update_time_ns);
                let max_poll_time = Duration::from_nanos(node_metrics.max_poll_time_ns);
                let max_update_time = Duration::from_nanos(node_metrics.max_state_update_time_ns);
                let total_polls = node_metrics.total_polls;
                let total_updates = node_metrics.total_state_updates;
                let perc_stolen = node_metrics.total_stolen_polls as f64
                    / node_metrics.total_polls as f64
                    * 100.0;

                let rows_received = node_metrics.rows_received;
                let morsels_received = node_metrics.morsels_received;
                let max_received = node_metrics.largest_morsel_received;
                let rows_sent = node_metrics.rows_sent;
                let morsels_sent = node_metrics.morsels_sent;
                let max_sent = node_metrics.largest_morsel_sent;

                let io_total_active_time = Duration::from_nanos(node_metrics.io_total_active_ns);
                let io_total_bytes_requested = node_metrics.io_total_bytes_requested;
                let io_total_bytes_received = node_metrics.io_total_bytes_received;
                let io_total_bytes_sent = node_metrics.io_total_bytes_sent;

                lines.push(
                    (total_time, format!(
                        "{name}: tot({total_time:.2?}), \
                                 poll({poll_time:.2?}, n={total_polls}, max={max_poll_time:.2?}, stolen={perc_stolen:.1}%), \
                                 update({update_time:.2?}, n={total_updates}, max={max_update_time:.2?}), \
                                 recv(row={rows_received}, morsel={morsels_received}, max={max_received}), \
                                 sent(row={rows_sent}, morsel={morsels_sent}, max={max_sent}), \
                                 io(\
                                    total_active_time={io_total_active_time:.2?}, \
                                    total_bytes_requested={io_total_bytes_requested}, \
                                    total_bytes_received={io_total_bytes_received}, \
                                    total_bytes_sent={io_total_bytes_sent})"))
                );

                total_query_ns += total_ns;
            }
            lines.sort_by_key(|(tot, _)| Reverse(*tot));

            let total_query_time = Duration::from_nanos(total_query_ns);
            eprintln!(
                "Streaming query took {query_elapsed:.2?} ({total_query_time:.2?} CPU), detailed breakdown:"
            );
            for (_tot, line) in lines {
                eprintln!("{line}");
            }
            eprintln!();
        }

        match top_ir {
            IR::SinkMultiple { inputs } => {
                let phys_node = &phys_sm[root_phys_node];
                let PhysNodeKind::SinkMultiple { sinks } = phys_node.kind() else {
                    unreachable!();
                };

                Ok(QueryResult::Multiple(
                    sinks
                        .iter()
                        .map(|phys_node_key| {
                            results
                                .remove(phys_to_graph[*phys_node_key])
                                .unwrap_or_else(DataFrame::empty)
                        })
                        .collect(),
                ))
            },
            _ => Ok(QueryResult::Single(
                results
                    .remove(phys_to_graph[root_phys_node])
                    .unwrap_or_else(DataFrame::empty),
            )),
        }
    }
}

pub enum QueryResult {
    Single(DataFrame),
    /// Collected to multiple in-memory sinks
    Multiple(Vec<DataFrame>),
}

impl QueryResult {
    pub fn unwrap_single(self) -> DataFrame {
        use QueryResult::*;
        match self {
            Single(df) => df,
            Multiple(_) => panic!(),
        }
    }

    pub fn unwrap_multiple(self) -> Vec<DataFrame> {
        use QueryResult::*;
        match self {
            Single(_) => panic!(),
            Multiple(dfs) => dfs,
        }
    }
}
