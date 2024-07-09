use std::sync::Arc;

use polars_core::schema::Schema;
use polars_plan::plans::DataFrameUdf;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use super::in_memory_source::InMemorySourceNode;

pub enum InMemoryMapNode {
    Sink {
        sink_node: InMemorySinkNode,
        num_pipelines: usize,
        map: Arc<dyn DataFrameUdf>,
    },
    Source(InMemorySourceNode),
    Done,
}

impl InMemoryMapNode {
    pub fn new(input_schema: Arc<Schema>, map: Arc<dyn DataFrameUdf>) -> Self {
        Self::Sink {
            sink_node: InMemorySinkNode::new(input_schema),
            num_pipelines: 0,
            map,
        }
    }
}

impl ComputeNode for InMemoryMapNode {
    fn name(&self) -> &str {
        "in_memory_map"
    }

    fn initialize(&mut self, num_pipelines_: usize) {
        match self {
            Self::Sink { num_pipelines, .. } => *num_pipelines = num_pipelines_,
            _ => unreachable!(),
        }
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(recv.len() == 1 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self, Self::Done) {
            *self = Self::Done;
        }

        // If the input is done, transition to being a source.
        if let Self::Sink {
            sink_node,
            num_pipelines,
            map,
        } = self
        {
            if recv[0] == PortState::Done {
                let df = sink_node.get_output().unwrap();
                let mut source_node =
                    InMemorySourceNode::new(Arc::new(map.call_udf(df.unwrap()).unwrap()));
                source_node.initialize(*num_pipelines);
                *self = Self::Source(source_node);
            }
        }

        match self {
            Self::Sink { sink_node, .. } => {
                sink_node.update_state(recv, &mut []);
                send[0] = PortState::Blocked;
            },
            Self::Source(source_node) => {
                recv[0] = PortState::Done;
                source_node.update_state(&mut [], send);
            },
            Self::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self, Self::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        match self {
            Self::Sink { sink_node, .. } => sink_node.spawn(scope, pipeline, recv, &mut [], state),
            Self::Source(source) => source.spawn(scope, pipeline, &mut [], send, state),
            Self::Done => unreachable!(),
        }
    }
}
