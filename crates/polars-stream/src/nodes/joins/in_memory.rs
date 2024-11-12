use std::sync::Arc;

use polars_core::schema::Schema;

use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::nodes::in_memory_source::InMemorySourceNode;

enum InMemoryJoinState {
    Sink {
        left: InMemorySinkNode,
        right: InMemorySinkNode,
    },
    Source(InMemorySourceNode),
    Done,
}

pub struct InMemoryJoinNode {
    state: InMemoryJoinState,
    num_pipelines: usize,
    joiner: Arc<dyn Fn(DataFrame, DataFrame) -> PolarsResult<DataFrame> + Send + Sync>,
}

impl InMemoryJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        joiner: Arc<dyn Fn(DataFrame, DataFrame) -> PolarsResult<DataFrame> + Send + Sync>,
    ) -> Self {
        Self {
            state: InMemoryJoinState::Sink {
                left: InMemorySinkNode::new(left_input_schema),
                right: InMemorySinkNode::new(right_input_schema),
            },
            num_pipelines: 0,
            joiner,
        }
    }
}

impl ComputeNode for InMemoryJoinNode {
    fn name(&self) -> &str {
        "in_memory_join"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self.state, InMemoryJoinState::Done) {
            self.state = InMemoryJoinState::Done;
        }

        // If the input is done, transition to being a source.
        if let InMemoryJoinState::Sink { left, right } = &mut self.state {
            if recv[0] == PortState::Done && recv[1] == PortState::Done {
                let left_df = left.get_output()?.unwrap();
                let right_df = right.get_output()?.unwrap();
                let mut source_node =
                    InMemorySourceNode::new(Arc::new((self.joiner)(left_df, right_df)?));
                source_node.initialize(self.num_pipelines);
                self.state = InMemoryJoinState::Source(source_node);
            }
        }

        match &mut self.state {
            InMemoryJoinState::Sink { left, right, .. } => {
                left.update_state(&mut recv[0..1], &mut [])?;
                right.update_state(&mut recv[1..2], &mut [])?;
                send[0] = PortState::Blocked;
            },
            InMemoryJoinState::Source(source_node) => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                source_node.update_state(&mut [], send)?;
            },
            InMemoryJoinState::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, InMemoryJoinState::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);
        match &mut self.state {
            InMemoryJoinState::Sink { left, right, .. } => {
                if recv_ports[0].is_some() {
                    left.spawn(scope, &mut recv_ports[0..1], &mut [], state, join_handles);
                }
                if recv_ports[1].is_some() {
                    right.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
            },
            InMemoryJoinState::Source(source) => {
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            InMemoryJoinState::Done => unreachable!(),
        }
    }
}
