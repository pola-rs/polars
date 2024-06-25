use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;

pub struct SimpleProjectionNode {
    schema: SchemaRef,
}

impl SimpleProjectionNode {
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }
}

impl ComputeNode for SimpleProjectionNode {
    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        let [mut recv] = <[_; 1]>::try_from(recv).ok().unwrap();
        let [mut send] = <[_; 1]>::try_from(send).ok().unwrap();

        scope.spawn_task(true, async move {
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
