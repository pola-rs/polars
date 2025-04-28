use std::sync::Arc;

use polars_core::schema::Schema;
use polars_plan::plans::DataFrameUdf;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use super::in_memory_source::InMemorySourceNode;

pub enum InMemoryMapNode {
    Sink {
        sink_node: InMemorySinkNode,
        map: Arc<dyn DataFrameUdf>,
    },
    Source(InMemorySourceNode),
    Done,
}

impl InMemoryMapNode {
    pub fn new(input_schema: Arc<Schema>, map: Arc<dyn DataFrameUdf>) -> Self {
        Self::Sink {
            sink_node: InMemorySinkNode::new(input_schema),
            map,
        }
    }
}

impl ComputeNode for InMemoryMapNode {
    fn name(&self) -> &str {
        "in-memory-map"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self, Self::Done) {
            *self = Self::Done;
        }

        // If the input is done, transition to being a source.
        if let Self::Sink { sink_node, map } = self {
            if recv[0] == PortState::Done {
                let df = sink_node.get_output()?;
                let source_node = InMemorySourceNode::new(
                    Arc::new(map.call_udf(df.unwrap())?),
                    MorselSeq::default(),
                );
                *self = Self::Source(source_node);
            }
        }

        match self {
            Self::Sink { sink_node, .. } => {
                sink_node.update_state(recv, &mut [], state)?;
                send[0] = PortState::Blocked;
            },
            Self::Source(source_node) => {
                recv[0] = PortState::Done;
                source_node.update_state(&mut [], send, state)?;
            },
            Self::Done => {
                recv[0] = PortState::Done;
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
        match self {
            Self::Sink { sink_node, .. } => {
                sink_node.spawn(scope, recv_ports, &mut [], state, join_handles)
            },
            Self::Source(source) => source.spawn(scope, &mut [], send_ports, state, join_handles),
            Self::Done => unreachable!(),
        }
    }
}
