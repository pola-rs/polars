use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::prelude::{ExecutionState, PhysicalExpr};
use polars_expr::reduce::Reduction;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq};
use crate::nodes::ComputeNode;

// All reductions in a single operation.
// `select(sum, min) -> vec![sum, min]
type Reductions = Vec<Box<dyn Reduction>>;

enum ReduceState {
    Sink {
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        reductions: Reductions,
    },
    Source(Mutex<Option<DataFrame>>),
    Done,
}

pub struct ReduceNode {
    // Reductions that are ready to finalize
    full: Mutex<Vec<Reductions>>,
    state: ReduceState,
    output_schema: SchemaRef,
}

impl ReduceNode {
    pub fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        reductions: Vec<Box<dyn Reduction>>,
        output_schema: SchemaRef,
    ) -> Self {
        Self {
            state: ReduceState::Sink { inputs, reductions },
            output_schema,
            full: Default::default(),
        }
    }

    fn spawn_sink<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        _send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        let ReduceState::Sink {
            inputs, reductions, ..
        } = &self.state
        else {
            unreachable!()
        };

        let mut recv = recv[0].take().unwrap();

        scope.spawn_task(TaskPriority::High, async move {
            let mut reductions = (*reductions).clone();

            while let Ok(morsel) = recv.recv().await {
                let df = morsel.into_df();

                for (i, input) in inputs.iter().map(|s| s.evaluate(&df, state)).enumerate() {
                    let input = input?;
                    reductions[i].update(&input)?;
                }
            }

            self.full.lock().push(reductions);

            Ok(())
        })
    }

    fn spawn_source<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        _recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        let ReduceState::Source(df) = &self.state else {
            unreachable!()
        };

        scope.spawn_task(TaskPriority::Low, async move {
            if let Some(df) = df.lock().take() {
                let morsel = Morsel::new(df, MorselSeq::new(0));
                let mut send = send[1].take().unwrap();
                send.send(morsel).await?;
            }
            Ok(())
        })
    }
}

impl ComputeNode for ReduceNode {
    fn name(&self) -> &str {
        "reduce"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(recv.len() == 1 && send.len() == 1);

        // State transitions
        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(&self.state, ReduceState::Done) {
            *self.state = ReduceState::Done;
        }

        match self.state {
            ReduceState::Sink if matches!(recv[0], PortState::Done) => {
                let reductions = std::mem::take(&mut *self.full.lock());

                let reductions = reductions
                    .into_iter()
                    .map(PolarsResult::Ok)
                    .reduce(|a, b| {
                        let mut a = a?;
                        let mut b = b?;
                        for (a, b) in a.iter_mut().zip(b.iter_mut()) {
                            a.combine(b)?
                        }
                        Ok(a)
                    })
                    .expect("at least on thread");

                // TODO! make `update_state` fallible.
                let reductions = reductions.unwrap();

                let columns = reductions
                    .into_iter()
                    .zip(self.output_schema.iter_names())
                    .map(|(mut r, name)| r.finalize().map(|scalar| scalar.into_series(name)))
                    .collect::<PolarsResult<Vec<_>>>()
                    .unwrap();
                let out = unsafe { DataFrame::new_no_checks(columns) };

                *self.state = ReduceState::Source(Mutex::new(Some(out)));
            },
            ReduceState::Source(..) => {
                *self.state = ReduceState::Done;
            },
            ReduceState::Done => {},
        }

        // Communicate state
        match &self.state {
            ReduceState::Sink{..} => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            ReduceState::Source(..) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            ReduceState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        match self.state {
            ReduceState::Sink { .. } => self.spawn_sink(scope, pipeline, recv, send, state),
            ReduceState::Source(..) => self.spawn_source(scope, pipeline, recv, send, state),
            ReduceState::Done => unreachable!(),
        }
    }
}
