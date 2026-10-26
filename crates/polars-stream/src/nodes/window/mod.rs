mod evaluate;
mod flush;
mod tuning;

use std::sync::Arc;

use polars_async::executor::TaskMetricAggregator;
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_expr::prelude::WindowExpr;
use polars_ooc::{LeastRecentSpillContext, ParameterFreeSpillContext, SpillFrame, memory_manager};
use polars_utils::pl_str::PlSmallStr;

use self::flush::Flush;
use self::tuning::WindowTuning;
use super::compute_node_prelude::*;

/// How the node outputs its rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WindowOutput {
    /// The input morsels in input order, with the window columns appended.
    OrderedMorsels,
    /// The input morsels in any order, with the window columns appended.
    UnorderedMorsels,
    /// The partitions sorted by partition and order key, with the window columns appended. Only
    /// used when the windows read every input column, so a partition holds whole rows.
    Partitions,
}

pub struct WindowParams {
    partition_by: Vec<PlSmallStr>,
    order_by: Option<(Vec<PlSmallStr>, SortMultipleOptions)>,
    /// The window expressions with their output names.
    exprs: Vec<(PlSmallStr, WindowExpr)>,
    /// The input columns the window expressions read, including the keys.
    read_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    /// Evaluate the rows of a partition in input order.
    ordered_eval: bool,
    output: WindowOutput,
}

impl WindowParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition_by: Vec<PlSmallStr>,
        order_by: Option<(Vec<PlSmallStr>, SortMultipleOptions)>,
        exprs: Vec<(PlSmallStr, WindowExpr)>,
        input_schema: &Schema,
        read_schema: Schema,
        output_schema: Arc<Schema>,
        ordered_eval: bool,
        maintain_order: bool,
    ) -> Self {
        let output = if maintain_order {
            WindowOutput::OrderedMorsels
        } else if read_schema.len() == input_schema.len() {
            WindowOutput::Partitions
        } else {
            WindowOutput::UnorderedMorsels
        };
        Self {
            partition_by,
            order_by,
            exprs,
            read_schema: Arc::new(read_schema),
            output_schema,
            ordered_eval,
            output,
        }
    }
}

/// Evaluates window expressions that share one partitioning.
///
/// The input morsels are stored as spillable frames. When the input is done, the columns the
/// windows read are hash partitioned on the partition keys, and the partitions are evaluated a
/// few at a time, as many as fit in the memory budget. The window columns are then put back on
/// the input morsels, which are output with fresh sequence numbers in input order. Without
/// `maintain_order` the rows are output in an unspecified order.
pub struct WindowNode {
    params: Arc<WindowParams>,
    tuning: WindowTuning,
    num_pipelines: usize,
    state: WindowState,
}

enum WindowState {
    Sink {
        /// The received morsels of each pipeline, in arrival order.
        morsels: Vec<Vec<StoredMorsel>>,
        spill_ctx: LeastRecentSpillContext,
    },
    Flush(Flush),
    Done,
}

struct StoredMorsel {
    seq: MorselSeq,
    sf: SpillFrame,
    /// The estimated size of the read columns.
    read_bytes: usize,
    /// The estimated size of the morsel.
    bytes: usize,
}

impl WindowNode {
    pub fn new(
        params: Arc<WindowParams>,
        num_pipelines: usize,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        Self {
            params,
            tuning: WindowTuning::from_env(),
            num_pipelines,
            state: WindowState::Sink {
                morsels: (0..num_pipelines).map(|_| Vec::new()).collect(),
                spill_ctx: LeastRecentSpillContext::new("window".into(), task_metrics),
            },
        }
    }
}

impl ComputeNode for WindowNode {
    fn name(&self) -> &str {
        "window"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done && !matches!(self.state, WindowState::Done) {
            self.state = WindowState::Done;
        }

        if matches!(self.state, WindowState::Sink { .. }) && recv[0] == PortState::Done {
            let WindowState::Sink { morsels, spill_ctx } =
                std::mem::replace(&mut self.state, WindowState::Done)
            else {
                unreachable!()
            };
            self.state = WindowState::Flush(evaluate::evaluate(
                &self.params,
                &self.tuning,
                self.num_pipelines,
                morsels,
                spill_ctx,
                &state.in_memory_exec_state,
            )?);
        }
        if let WindowState::Flush(flush) = &self.state
            && flush.is_finished()
        {
            self.state = WindowState::Done;
        }

        match &mut self.state {
            WindowState::Sink { .. } => {
                if recv[0] != PortState::Done {
                    recv[0] = PortState::Ready;
                }
                send[0] = PortState::Blocked;
            },
            WindowState::Flush(_) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            WindowState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.state {
            WindowState::Sink { .. } => NodeMemoryUsage::Accumulating,
            WindowState::Flush(_) => NodeMemoryUsage::Draining,
            WindowState::Done => NodeMemoryUsage::Bounded,
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        match &mut self.state {
            WindowState::Sink { morsels, spill_ctx } => {
                assert!(send_ports[0].is_none());
                let receivers = recv_ports[0].take().unwrap().parallel();
                let read_schema = &*self.params.read_schema;
                let spill_ctx = &*spill_ctx;

                for (local, mut recv) in morsels.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            local.push(store_morsel(morsel, read_schema, spill_ctx).await);
                        }
                        Ok(())
                    }));
                }
            },
            WindowState::Flush(flush) => {
                assert!(recv_ports[0].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                flush.spawn(scope, senders, join_handles);
            },
            WindowState::Done => unreachable!(),
        }
    }
}

/// Keeps `morsel` as a spillable frame, with its read columns in one chunk each.
async fn store_morsel(
    mut morsel: Morsel,
    read_schema: &Schema,
    spill_ctx: &LeastRecentSpillContext,
) -> StoredMorsel {
    morsel.take_consume_token();
    let (read_bytes, bytes) = {
        let mut df = morsel.df_mut().await;
        // SAFETY: the lengths and names of the columns do not change.
        let columns = unsafe { df.columns_mut_retain_schema() };
        let mut read_bytes = 0;
        for c in columns
            .iter_mut()
            .filter(|c| read_schema.contains(c.name()))
        {
            // Reading rows by index needs single-chunk columns.
            if c.n_chunks() > 1 {
                *c = c.rechunk();
            }
            read_bytes += c.estimated_size(false);
        }
        (read_bytes, df.estimated_size(false))
    };
    let seq = morsel.seq();
    let sf = morsel.into_sf();
    spill_ctx.register_no_spill_check(&sf);
    memory_manager().spill().await;
    StoredMorsel {
        seq,
        sf,
        read_bytes,
        bytes,
    }
}
