use std::sync::Arc;

use polars_async::executor::TaskMetricAggregator;
use polars_core::prelude::PlHashMap;
use polars_core::schema::{Schema, SchemaRef};
use polars_ooc::{LeastRecentSpillContext, SpillFrame};
use polars_plan::dsl::SinkTypeIR;
use polars_plan::plans::{AExpr, IR};
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use super::spill_frame_source::SpillFrameSourceNode;
use crate::morsel::get_ideal_morsel_size;

/// The values of the working tables and caches that recursive queries hold, by id.
pub type RecursiveInputs = PlHashMap<UniqueId, Arc<[SpillFrame]>>;

/// The step of a recursive query, run as its own streaming query in every round.
pub struct RecursiveStep {
    pub id: UniqueId,
    pub root: Node,
    pub ir_arena: Arena<IR>,
    pub expr_arena: Arena<AExpr>,
    /// The caches in the step that are computed once for the whole query. Their values are
    /// inputs of the node, in this order.
    pub external_caches: Vec<UniqueId>,
    /// The inputs of the recursive queries around this one.
    pub outer_inputs: RecursiveInputs,
}

impl std::fmt::Debug for RecursiveStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecursiveStep")
            .field("id", &self.id)
            .finish()
    }
}

impl RecursiveStep {
    fn run_round(&self, inputs: &RecursiveInputs, num_pipelines: usize) -> PolarsResult<DataFrame> {
        let mut ir_arena = self.ir_arena.clone();
        let mut expr_arena = self.expr_arena.clone();
        let root = ir_arena.add(IR::Sink {
            input: self.root,
            payload: SinkTypeIR::Memory,
        });
        let query = crate::StreamingQuery::build(
            root,
            &mut ir_arena,
            &mut expr_arena,
            false,
            num_pipelines,
            Some(inputs),
        )?;
        Ok(query.execute()?.unwrap_single())
    }

    /// The number of rows a round reads from memory. `None` if it also reads other sources.
    fn in_memory_rows(&self, inputs: &RecursiveInputs) -> Option<usize> {
        let mut rows = 0;
        let mut stack = vec![self.root];
        while let Some(node) = stack.pop() {
            let ir = self.ir_arena.get(node);
            let frames = match ir {
                IR::WorkingTableScan { id, .. } | IR::Cache { id, .. } => inputs.get(id),
                _ => None,
            };
            if let Some(frames) = frames {
                rows += frames.iter().map(|sf| sf.height()).sum::<usize>();
                continue;
            }
            match ir {
                IR::DataFrameScan { df, .. } => rows += df.height(),
                ir => {
                    let n = stack.len();
                    stack.extend(ir.inputs());
                    if stack.len() == n {
                        return None;
                    }
                },
            }
        }
        Some(rows)
    }

    /// Runs rounds until a round has no rows. Returns the rows of the anchor and of every round.
    fn run(
        &self,
        anchor: Arc<[SpillFrame]>,
        externals: Vec<Arc<[SpillFrame]>>,
        spill_ctx: &LeastRecentSpillContext,
    ) -> PolarsResult<Vec<Arc<[SpillFrame]>>> {
        // Starting the pipelines of a round has a cost per pipeline, so small rounds use fewer.
        let max_pipelines = polars_config::config().max_threads();
        let rows_per_pipeline = get_ideal_morsel_size().min(1000);
        let mut inputs = self.outer_inputs.clone();
        inputs.extend(self.external_caches.iter().copied().zip(externals));
        let mut out = vec![anchor];
        loop {
            let frontier = out.last().unwrap().clone();
            if frontier.iter().all(|sf| sf.height() == 0) {
                break;
            }
            // The loop has no bound, so let a keyboard interrupt stop it.
            polars_error::abort::try_raise_polars_abort();
            inputs.insert(self.id, frontier);
            let num_pipelines = match self.in_memory_rows(&inputs) {
                Some(rows) => rows.div_ceil(rows_per_pipeline).clamp(1, max_pipelines),
                None => max_pipelines,
            };
            let df = self.run_round(&inputs, num_pipelines)?;
            out.push(Arc::from([SpillFrame::new_blocking(df, spill_ctx)]));
        }
        Ok(out)
    }
}

enum RecursiveState {
    Sink {
        anchor: InMemorySinkNode,
        externals: Vec<InMemorySinkNode>,
    },
    Source {
        source: SpillFrameSourceNode,
        /// Keeps the frames of the source spillable.
        _spill_ctx: LeastRecentSpillContext,
    },
    Done,
}

/// Collects the anchor and the external caches, then runs the step once per round.
pub struct RecursiveNode {
    state: RecursiveState,
    step: Arc<RecursiveStep>,
    schema: SchemaRef,
    task_metrics: Option<Arc<TaskMetricAggregator>>,
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
                anchor: InMemorySinkNode::new(anchor_schema.clone(), task_metrics.clone()),
                externals: external_schemas
                    .into_iter()
                    .map(|schema| InMemorySinkNode::new(schema, task_metrics.clone()))
                    .collect(),
            },
            step,
            schema: anchor_schema,
            task_metrics,
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
            let results_ctx =
                LeastRecentSpillContext::new("recursive-results".into(), self.task_metrics.clone());
            let inputs_ctx =
                LeastRecentSpillContext::new("recursive-inputs".into(), self.task_metrics.clone());
            let anchor = anchor.take_spill_frames(&results_ctx);
            let externals = externals
                .iter_mut()
                .map(|sink| sink.take_spill_frames(&inputs_ctx))
                .collect();
            let groups = self.step.run(anchor, externals, &results_ctx)?;
            self.state = RecursiveState::Source {
                source: SpillFrameSourceNode::new(groups, self.schema.clone()),
                _spill_ctx: results_ctx,
            };
        }

        match &mut self.state {
            RecursiveState::Sink { anchor, externals } => {
                anchor.update_state(&mut recv[0..1], &mut [], state)?;
                for (i, sink) in externals.iter_mut().enumerate() {
                    sink.update_state(&mut recv[i + 1..i + 2], &mut [], state)?;
                }
                send[0] = PortState::Blocked;
            },
            RecursiveState::Source { source, .. } => {
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
            RecursiveState::Source { source, .. } => source.memory_usage(),
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
            RecursiveState::Source { source, .. } => {
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            RecursiveState::Done => unreachable!(),
        }
    }
}
