use std::sync::Arc;

use polars_core::schema::Schema;

use super::compute_node_prelude::*;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::nodes::negative_slice::NegativeSliceNode;
use crate::nodes::streaming_slice::StreamingSliceNode;

/// A node that will dispatch either to StreamingSlice or NegativeSlice
/// depending on the offset which is dynamically dispatched.
pub enum DynamicSliceNode {
    GatheringParams {
        offset: InMemorySinkNode,
        length: InMemorySinkNode,
    },
    Streaming(StreamingSliceNode),
    Negative(NegativeSliceNode),
}

impl DynamicSliceNode {
    pub fn new(offset_schema: Arc<Schema>, length_schema: Arc<Schema>) -> Self {
        assert!(offset_schema.len() == 1);
        assert!(length_schema.len() == 1);
        Self::GatheringParams {
            offset: InMemorySinkNode::new(offset_schema),
            length: InMemorySinkNode::new(length_schema),
        }
    }
}

impl ComputeNode for DynamicSliceNode {
    fn name(&self) -> &str {
        "dynamic-slice"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 3 && send.len() == 1);

        if recv[1] == PortState::Done && recv[2] == PortState::Done {
            if let Self::GatheringParams { offset, length } = self {
                let offset = offset.get_output()?.unwrap();
                let length = length.get_output()?.unwrap();
                let offset_item = offset.get_columns()[0].get(0)?;
                let length_item = length.get_columns()[0].get(0)?;
                let offset = offset_item.extract::<i64>().unwrap_or(0);
                let length = length_item.extract::<usize>().unwrap_or(usize::MAX);
                if let Ok(non_neg_offset) = offset.try_into() {
                    *self = Self::Streaming(StreamingSliceNode::new(non_neg_offset, length));
                } else {
                    *self = Self::Negative(NegativeSliceNode::new(offset, length));
                }
            }
        }

        match self {
            Self::GatheringParams { offset, length } => {
                offset.update_state(&mut recv[1..2], &mut [], state)?;
                length.update_state(&mut recv[2..3], &mut [], state)?;
                recv[0] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            Self::Streaming(node) => {
                node.update_state(&mut recv[0..1], send, state)?;
                recv[1] = PortState::Done;
                recv[2] = PortState::Done;
            },
            Self::Negative(node) => {
                node.update_state(&mut recv[0..1], send, state)?;
                recv[1] = PortState::Done;
                recv[2] = PortState::Done;
            },
        }
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 3 && send_ports.len() == 1);
        match self {
            Self::GatheringParams { offset, length } => {
                assert!(recv_ports[0].is_none());
                assert!(send_ports[0].is_none());
                if recv_ports[1].is_some() {
                    offset.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
                if recv_ports[2].is_some() {
                    length.spawn(scope, &mut recv_ports[2..3], &mut [], state, join_handles);
                }
            },
            Self::Streaming(node) => {
                node.spawn(
                    scope,
                    &mut recv_ports[0..1],
                    send_ports,
                    state,
                    join_handles,
                );
                assert!(recv_ports[1].is_none());
                assert!(recv_ports[2].is_none());
            },
            Self::Negative(node) => {
                node.spawn(
                    scope,
                    &mut recv_ports[0..1],
                    send_ports,
                    state,
                    join_handles,
                );
                assert!(recv_ports[1].is_none());
                assert!(recv_ports[2].is_none());
            },
        }
    }
}
