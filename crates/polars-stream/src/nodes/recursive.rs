use std::sync::Arc;

use polars_async::executor::TaskMetricAggregator;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_plan::dsl::SinkTypeIR;
use polars_plan::plans::{AExpr, IR};
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use super::in_memory_source::InMemorySourceNode;

/// The step of a recursive query, run as its own streaming query in every round.
pub struct RecursiveStep {
    pub id: UniqueId,
    pub root: Node,
    pub ir_arena: Arena<IR>,
    pub expr_arena: Arena<AExpr>,
    /// The caches in the step that are computed once for the whole query. Their values are
    /// inputs of the node, in this order.
    pub external_caches: Vec<UniqueId>,
}

impl std::fmt::Debug for RecursiveStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecursiveStep")
            .field("id", &self.id)
            .finish()
    }
}

impl RecursiveStep {
    fn run_round(
        &self,
        frontier: &Arc<DataFrame>,
        externals: &[Arc<DataFrame>],
    ) -> PolarsResult<DataFrame> {
        let mut ir_arena = self.ir_arena.clone();
        let mut expr_arena = self.expr_arena.clone();
        for i in 0..ir_arena.len() {
            let node = Node(i);
            let df = match ir_arena.get(node) {
                IR::WorkingTableScan { id, .. } if *id == self.id => frontier.clone(),
                IR::Cache { id, .. } => match self.external_caches.iter().position(|c| c == id) {
                    Some(idx) => externals[idx].clone(),
                    None => continue,
                },
                _ => continue,
            };
            let schema = df.schema().clone();
            ir_arena.replace(
                node,
                IR::DataFrameScan {
                    df,
                    schema,
                    output_schema: None,
                },
            );
        }
        let root = ir_arena.add(IR::Sink {
            input: self.root,
            payload: SinkTypeIR::Memory,
        });
        let query = crate::StreamingQuery::build(root, &mut ir_arena, &mut expr_arena, false)?;
        Ok(query.execute()?.unwrap_single())
    }

    /// The anchor rows followed by the rows of every round, until a round has no rows.
    fn run(&self, anchor: DataFrame, externals: &[Arc<DataFrame>]) -> PolarsResult<DataFrame> {
        let mut frontier = Arc::new(anchor);
        let mut out = vec![];
        while frontier.height() > 0 {
            // The loop has no bound, so let a keyboard interrupt stop it.
            polars_error::abort::try_raise_polars_abort();
            let next = Arc::new(self.run_round(&frontier, externals)?);
            out.push(Arc::unwrap_or_clone(std::mem::replace(&mut frontier, next)));
        }
        if out.is_empty() {
            return Ok(Arc::unwrap_or_clone(frontier));
        }
        Ok(accumulate_dataframes_vertical_unchecked(out))
    }
}

enum RecursiveState {
    Sink {
        anchor: InMemorySinkNode,
        externals: Vec<InMemorySinkNode>,
    },
    Source(InMemorySourceNode),
    Done,
}

/// Collects the anchor and the external caches, then runs the step once per round.
pub struct RecursiveNode {
    state: RecursiveState,
    step: Arc<RecursiveStep>,
}

impl RecursiveNode {
    pub fn new(
        anchor_schema: Arc<Schema>,
        external_schemas: Vec<Arc<Schema>>,
        step: Arc<RecursiveStep>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        Self {
            state: RecursiveState::Sink {
                anchor: InMemorySinkNode::new(anchor_schema, task_metrics.clone()),
                externals: external_schemas
                    .into_iter()
                    .map(|schema| InMemorySinkNode::new(schema, task_metrics.clone()))
                    .collect(),
            },
            step,
        }
    }
}

impl ComputeNode for RecursiveNode {
    fn name(&self) -> &str {
        "recursive"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(send.len() == 1);

        if send[0] == PortState::Done && !matches!(self.state, RecursiveState::Done) {
            self.state = RecursiveState::Done;
        }

        if let RecursiveState::Sink { anchor, externals } = &mut self.state
            && recv.iter().all(|r| *r == PortState::Done)
        {
            let anchor = anchor.get_output()?.unwrap();
            let externals = externals
                .iter_mut()
                .map(|sink| Ok(Arc::new(sink.get_output()?.unwrap())))
                .collect::<PolarsResult<Vec<_>>>()?;
            // The rounds run new queries that may need Rayon, so don't block a Rayon worker
            // while they run.
            let step = &*self.step;
            let df = RAYON.block_on(|| step.run(anchor, &externals))?;
            self.state =
                RecursiveState::Source(InMemorySourceNode::new(Arc::new(df), MorselSeq::default()));
        }

        match &mut self.state {
            RecursiveState::Sink { anchor, externals } => {
                anchor.update_state(&mut recv[0..1], &mut [], state)?;
                for (i, sink) in externals.iter_mut().enumerate() {
                    sink.update_state(&mut recv[i + 1..i + 2], &mut [], state)?;
                }
                send[0] = PortState::Blocked;
            },
            RecursiveState::Source(source) => {
                recv.fill(PortState::Done);
                source.update_state(&mut [], send, state)?;
            },
            RecursiveState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.state {
            RecursiveState::Sink { .. } => NodeMemoryUsage::Accumulating,
            RecursiveState::Source(source) => source.memory_usage(),
            RecursiveState::Done => NodeMemoryUsage::Bounded,
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send_ports.len() == 1);
        match &mut self.state {
            RecursiveState::Sink { anchor, externals } => {
                let sinks = std::iter::once(anchor).chain(externals.iter_mut());
                for (i, sink) in sinks.enumerate() {
                    if recv_ports[i].is_some() {
                        sink.spawn(
                            scope,
                            &mut recv_ports[i..i + 1],
                            &mut [],
                            state,
                            join_handles,
                        );
                    }
                }
            },
            RecursiveState::Source(source) => {
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            RecursiveState::Done => unreachable!(),
        }
    }
}
