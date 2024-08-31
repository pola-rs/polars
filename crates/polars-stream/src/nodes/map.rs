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

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
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
                    let morsel = morsel.try_map(|df| slf.map.call_udf(df))?;
                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
