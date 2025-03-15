use std::sync::Arc;

use polars_core::POOL;
use polars_core::prelude::IntoColumn;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::nodes::in_memory_source::InMemorySourceNode;

pub enum InputIndependentSelectNode {
    ToSelect {
        selectors: Vec<StreamExpr>,
        num_pipelines: usize,
    },
    Source(InMemorySourceNode),
    Done,
}

impl InputIndependentSelectNode {
    pub fn new(selectors: Vec<StreamExpr>) -> Self {
        Self::ToSelect {
            selectors,
            num_pipelines: 0,
        }
    }
}

impl ComputeNode for InputIndependentSelectNode {
    fn name(&self) -> &str {
        "input_independent_select"
    }

    fn initialize(&mut self, num_pipelines_: usize) {
        match self {
            Self::ToSelect { num_pipelines, .. } => *num_pipelines = num_pipelines_,
            _ => unreachable!(),
        }
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.is_empty() && send.len() == 1);
        if send[0] == PortState::Done {
            *self = Self::Done;
            return Ok(());
        }

        POOL.install(|| {
            if let Self::ToSelect {
                selectors,
                num_pipelines,
            } = self
            {
                let empty_df = DataFrame::empty();
                let state = ExecutionState::new();
                let selected: Result<Vec<_>, _> = selectors
                    .par_iter()
                    .map(|selector| {
                        let s = selector.evaluate_blocking(&empty_df, &state)?;
                        PolarsResult::Ok(s.into_column())
                    })
                    .collect();
                let ret = DataFrame::new_with_broadcast(selected?)?;
                let mut src_node = InMemorySourceNode::new(Arc::new(ret), MorselSeq::default());
                src_node.initialize(*num_pipelines);
                *self = InputIndependentSelectNode::Source(src_node);
            }
            PolarsResult::Ok(())
        })?;

        match self {
            Self::ToSelect { .. } => unreachable!(),
            Self::Source(src) => src.update_state(recv, send),
            Self::Done => {
                send[0] = PortState::Done;
                Ok(())
            },
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);
        let Self::Source(src) = self else {
            unreachable!()
        };
        src.spawn(scope, recv_ports, send_ports, state, join_handles);
    }
}
