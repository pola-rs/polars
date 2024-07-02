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
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(recv.len() == 1 && send.len() == 1);
        let mut recv = recv[0].take().unwrap();
        let mut send = send[0].take().unwrap();

        scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel.try_map(|df| {
                    // TODO: can this be unchecked?
                    df.select_with_schema(self.schema.iter_names(), &self.schema)
                })?;

                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        })
    }
}
