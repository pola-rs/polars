use std::sync::{Arc};
use parking_lot::Mutex;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::prelude::{ExecutionState, PhysicalExpr};
use polars_expr::reduce::Reduction;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::nodes::ComputeNode;

pub struct ReduceNode {
    inputs: Vec<Arc<dyn PhysicalExpr>>,
    reductions: Mutex<Vec<Box<dyn Reduction>>>,
    output_schema: SchemaRef,
}

impl ReduceNode {
    pub fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        reductions: Vec<Box<dyn Reduction>>,
        output_schema: SchemaRef,
    ) -> Self {
        Self {
            inputs,
            reductions,
            output_schema,
        }
    }
}

impl ComputeNode for ReduceNode {
    fn name(&self) -> &str {
        "reduce"
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
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(recv.len() == 1 && send.len() == 1);
        let mut recv = recv[0].take().unwrap();
        let mut send = send[0].take().unwrap();

        scope.spawn_task(TaskPriority::High, async move {

            self.reductions.clone()

            while let Ok(morsel) = recv.recv().await {
                let df = morsel.into_df();

                for (i, input) in self.inputs.iter().map(|s| s.evaluate(&df, state)).enumerate() {
                    let input = input?;
                    self.reductions[i].update(&input)?;
                }
            }

            let columns = self.reductions.into_iter().zip(self.output_schema.iter_names()).map(|(mut reduction, name)| reduction.finalize().map(|s| s.into_series(name))).collect::<PolarsResult<Vec<_>>>()?;

            Ok(())
        })
    }
}
