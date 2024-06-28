use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_plan::plans::DataFrameUdf;

use super::in_memory_sink::InMemorySinkNode;
use super::in_memory_source::InMemorySourceNode;
use super::ComputeNode;
use crate::async_executor::JoinHandle;
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::graph::PortState;
use crate::morsel::Morsel;

pub enum InMemoryMapNode {
    Sink(InMemorySinkNode, Arc<dyn DataFrameUdf>),
    Source(InMemorySourceNode),
}

impl InMemoryMapNode {
    pub fn new(input_schema: Arc<Schema>, map: Arc<dyn DataFrameUdf>) -> Self {
        Self::Sink(InMemorySinkNode::new(input_schema), map)
    }
}

impl ComputeNode for InMemoryMapNode {
    fn name(&self) -> &'static str {
        "in_memory_map"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(recv.len() == 1 && send.len() == 1);

        // If the output doesn't want any more data, we are always done.
        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            return;
        }

        match self {
            Self::Sink(sink, _) => {
                sink.update_state(recv, &mut []);
                send[0] = PortState::Blocked;
            },
            Self::Source(source) => {
                source.update_state(&mut [], send);
                recv[0] = PortState::Done;
            },
        }
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self, Self::Sink(_, _))
    }

    fn initialize(&mut self, num_pipelines: usize) {
        match self {
            Self::Sink(sink, _) => sink.initialize(num_pipelines),
            Self::Source(source) => source.initialize(num_pipelines),
        }
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s crate::async_executor::TaskScope<'s, 'env>,
        pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        match self {
            Self::Sink(sink, _) => sink.spawn(scope, pipeline, recv, &mut [], state),
            Self::Source(source) => source.spawn(scope, pipeline, &mut [], send, state),
        }
    }

    fn finalize(&mut self) -> PolarsResult<Option<DataFrame>> {
        match self {
            Self::Sink(sink, map) => {
                let df = sink.finalize()?.unwrap();
                *self = Self::Source(InMemorySourceNode::new(Arc::new(map.call_udf(df)?)));
            },
            Self::Source(source) => {
                source.finalize()?;
            },
        };
        Ok(None)
    }
}
