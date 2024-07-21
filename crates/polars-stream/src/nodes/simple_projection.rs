use std::sync::Arc;

use polars_core::schema::Schema;

use super::compute_node_prelude::*;

pub struct SimpleProjectionNode {
    schema: Arc<Schema>,
}

impl SimpleProjectionNode {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }
}

impl ComputeNode for SimpleProjectionNode {
    fn name(&self) -> &str {
        "simple_projection"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.len() == 1 && send.len() == 1);
        let receivers = recv[0].take().unwrap().parallel();
        let senders = send[0].take().unwrap().parallel();

        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {
                    let morsel = morsel.try_map(|df| {
                        // TODO: can this be unchecked?
                        df.select_with_schema(slf.schema.iter_names(), &slf.schema)
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
