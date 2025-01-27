use std::sync::Arc;

use polars_core::schema::Schema;
use polars_ops::frame::_merge_sorted_dfs;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::nodes::in_memory_source::InMemorySourceNode;

enum MergeSortedState {
    Sink {
        left: InMemorySinkNode,
        right: InMemorySinkNode,
    },
    Source(InMemorySourceNode),
    Done,
}

pub struct MergeSortedNode {
    state: MergeSortedState,
    num_pipelines: usize,
    key: PlSmallStr,
}

impl MergeSortedNode {
    pub fn new(schema: Arc<Schema>, key: PlSmallStr) -> Self {
        assert!(schema.contains(key.as_str()));
        Self {
            state: MergeSortedState::Sink {
                left: InMemorySinkNode::new(schema.clone()),
                right: InMemorySinkNode::new(schema),
            },
            num_pipelines: 0,
            key,
        }
    }
}

impl ComputeNode for MergeSortedNode {
    fn name(&self) -> &str {
        "in_memory_merge_sorted"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self.state, MergeSortedState::Done) {
            self.state = MergeSortedState::Done;
        }

        // If the input is done, transition to being a source.
        if let MergeSortedState::Sink { left, right } = &mut self.state {
            if recv[0] == PortState::Done && recv[1] == PortState::Done {
                let left_df = left.get_output()?.unwrap();
                let right_df = right.get_output()?.unwrap();
                let left_s = left_df.column(self.key.as_str()).unwrap();
                let right_s = right_df.column(self.key.as_str()).unwrap();
                let df = _merge_sorted_dfs(
                    &left_df,
                    &right_df,
                    left_s.as_materialized_series(),
                    right_s.as_materialized_series(),
                    true,
                )?;
                let mut source_node = InMemorySourceNode::new(Arc::new(df), MorselSeq::default());
                source_node.initialize(self.num_pipelines);
                self.state = MergeSortedState::Source(source_node);
            }
        }

        match &mut self.state {
            MergeSortedState::Sink { left, right, .. } => {
                left.update_state(&mut recv[0..1], &mut [])?;
                right.update_state(&mut recv[1..2], &mut [])?;
                send[0] = PortState::Blocked;
            },
            MergeSortedState::Source(source_node) => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                source_node.update_state(&mut [], send)?;
            },
            MergeSortedState::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, MergeSortedState::Sink { .. })
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
            MergeSortedState::Sink { left, right, .. } => {
                if recv_ports[0].is_some() {
                    left.spawn(scope, &mut recv_ports[0..1], &mut [], state, join_handles);
                }
                if recv_ports[1].is_some() {
                    right.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
            },
            MergeSortedState::Source(source) => {
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            MergeSortedState::Done => unreachable!(),
        }
    }
}
