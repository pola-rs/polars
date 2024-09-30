use polars_core::prelude::*;
use polars_core::utils::Container;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;

pub struct WithRowIndexNode {
    name: PlSmallStr,
    offset: IdxSize,
}

impl WithRowIndexNode {
    pub fn new(name: PlSmallStr, offset: Option<IdxSize>) -> Self {
        Self {
            name,
            offset: offset.unwrap_or(0),
        }
    }
}

impl ComputeNode for WithRowIndexNode {
    fn name(&self) -> &str {
        "with_row_index"
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
        let mut recv = recv[0].take().unwrap().serial();
        let mut send = send[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel.try_map(|df| {
                    let out = df.with_row_index(self.name.clone(), Some(self.offset));
                    self.offset = self.offset.checked_add(df.len().try_into().unwrap()).unwrap();
                    out
                })?;
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
