use std::sync::Arc;

use polars_plan::plans::DataFrameUdf;

use super::compute_node_prelude::*;

/// A simple mapping node. Assumes the given udf is elementwise.
pub struct MapNode {
    map: Arc<dyn DataFrameUdf>,
}

impl MapNode {
    pub fn new(map: Arc<dyn DataFrameUdf>) -> Self {
        Self { map }
    }
}

impl ComputeNode for MapNode {
    fn name(&self) -> &str {
        "map"
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
                let morsel = morsel.try_map(|df| self.map.call_udf(df))?;
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        })
    }
}
