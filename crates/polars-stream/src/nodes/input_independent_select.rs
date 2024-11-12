use polars_core::prelude::IntoColumn;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::morsel::SourceToken;

pub struct InputIndependentSelectNode {
    selectors: Vec<StreamExpr>,
    done: bool,
}

impl InputIndependentSelectNode {
    pub fn new(selectors: Vec<StreamExpr>) -> Self {
        Self {
            selectors,
            done: false,
        }
    }
}

impl ComputeNode for InputIndependentSelectNode {
    fn name(&self) -> &str {
        "input_independent_select"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.is_empty() && send.len() == 1);
        send[0] = if send[0] == PortState::Done || self.done {
            PortState::Done
        } else {
            PortState::Ready
        };
        Ok(())
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
        let mut sender = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            let empty_df = DataFrame::empty();
            let mut selected = Vec::new();
            for selector in self.selectors.iter() {
                let s = selector.evaluate(&empty_df, state).await?;
                selected.push(s.into_column());
            }

            let ret = DataFrame::new_with_broadcast(selected)?;
            let seq = MorselSeq::default();
            let source_token = SourceToken::new();
            let morsel = Morsel::new(ret, seq, source_token);
            sender.send(morsel).await.ok();
            self.done = true;
            Ok(())
        }));
    }
}
