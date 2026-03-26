//! Gather node for streaming engine.
//!
//! This node collects both inputs (data and indices) to memory and then
//! performs the gather operation. It falls back to in-memory execution.

use std::sync::Arc;

use polars_core::prelude::IdxSize;
use polars_core::schema::Schema;
use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_utils::index::check_bounds;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use super::in_memory_source::InMemorySourceNode;
use crate::morsel::MorselSeq;

pub enum GatherNode {
    Sink {
        data_sink: InMemorySinkNode,
        indices_sink: InMemorySinkNode,
    },
    Source(InMemorySourceNode),
    Done,
}

impl GatherNode {
    pub fn new(data_schema: Arc<Schema>, indices_schema: Arc<Schema>) -> Self {
        Self::Sink {
            data_sink: InMemorySinkNode::new(data_schema),
            indices_sink: InMemorySinkNode::new(indices_schema),
        }
    }
}

fn perform_gather(data_df: DataFrame, indices_df: DataFrame) -> PolarsResult<DataFrame> {
    // Extract indices from the indices dataframe
    polars_ensure!(
        indices_df.width() == 1,
        ComputeError: "gather indices DataFrame must have exactly one column, got {}",
        indices_df.width()
    );

    let indices_col = indices_df.columns()[0].clone();
    let indices = indices_col.idx().map_err(|_| {
        polars_err!(
            ComputeError: "gather indices must be of type UInt32 (IdxSize), got {}",
            indices_col.dtype()
        )
    })?;

    // Rechunk to get contiguous memory, then get slice
    let indices = indices.rechunk();
    polars_ensure!(
        !indices.has_nulls(),
        ComputeError: "gather indices contain null values"
    );
    let indices_slice = indices.cont_slice().unwrap();

    check_bounds(indices_slice, data_df.height() as IdxSize)?;

    // SAFETY: bounds checked above
    Ok(unsafe { data_df.take_slice_unchecked(indices_slice) })
}

impl ComputeNode for GatherNode {
    fn name(&self) -> &str {
        "gather"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert_eq!(recv.len(), 2);
        assert_eq!(send.len(), 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self, Self::Done) {
            *self = Self::Done;
        }

        // If both inputs are done, transition to being a source.
        if let Self::Sink {
            data_sink,
            indices_sink,
        } = self
        {
            if recv[0] == PortState::Done && recv[1] == PortState::Done {
                let data_df = data_sink.get_output()?.unwrap();
                let indices_df = indices_sink.get_output()?.unwrap();
                let result = perform_gather(data_df, indices_df)?;
                let source_node = InMemorySourceNode::new(Arc::new(result), MorselSeq::default());
                *self = Self::Source(source_node);
            }
        }

        match self {
            Self::Sink {
                data_sink,
                indices_sink,
                ..
            } => {
                // Update data sink with first recv port
                let mut data_recv = [recv[0]];
                data_sink.update_state(&mut data_recv, &mut [], state)?;
                recv[0] = data_recv[0];

                // Update indices sink with second recv port
                let mut indices_recv = [recv[1]];
                indices_sink.update_state(&mut indices_recv, &mut [], state)?;
                recv[1] = indices_recv[0];

                // Output is blocked while we're still collecting
                send[0] = PortState::Blocked;
            },
            Self::Source(source_node) => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                source_node.update_state(&mut [], send, state)?;
            },
            Self::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self, Self::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 2);
        assert_eq!(send_ports.len(), 1);

        match self {
            Self::Sink {
                data_sink,
                indices_sink,
                ..
            } => {
                // Spawn task for data sink
                let mut data_recv = [recv_ports[0].take()];
                data_sink.spawn(scope, &mut data_recv, &mut [], state, join_handles);

                // Spawn task for indices sink
                let mut indices_recv = [recv_ports[1].take()];
                indices_sink.spawn(scope, &mut indices_recv, &mut [], state, join_handles);
            },
            Self::Source(source) => source.spawn(scope, &mut [], send_ports, state, join_handles),
            Self::Done => unreachable!(),
        }
    }
}
