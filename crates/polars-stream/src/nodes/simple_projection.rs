use std::sync::Arc;

use polars_core::schema::Schema;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;

pub struct SimpleProjectionNode {
    columns: Vec<PlSmallStr>,
    input_schema: Arc<Schema>,
}

impl SimpleProjectionNode {
    pub fn new(columns: Vec<PlSmallStr>, input_schema: Arc<Schema>) -> Self {
        Self {
            columns,
            input_schema,
        }
    }
}

impl ComputeNode for SimpleProjectionNode {
    fn name(&self) -> &str {
        "simple-projection"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
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
        let receivers = recv_ports[0].take().unwrap().parallel();
        let senders = send_ports[0].take().unwrap().parallel();

        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {
                    let morsel = morsel.try_map(|df| {
                        // TODO: can this be unchecked?
                        let check_duplicates = true;
                        df._select_with_schema_impl(
                            slf.columns.as_slice(),
                            &slf.input_schema,
                            check_duplicates,
                        )
                    })?;

                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
