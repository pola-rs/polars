use std::sync::Arc;

use polars_core::prelude::Schema;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::prelude::{ExecutionState, PhysicalExpr};
use polars_expr::reduce::Reduction;

use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::nodes::ComputeNode;

pub struct ReduceNode {
    inputs: Vec<Arc<dyn PhysicalExpr>>,
    reductions: Vec<Box<dyn Reduction>>,
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
        todo!()
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        todo!()
    }
}
