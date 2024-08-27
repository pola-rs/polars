use std::sync::Arc;

use polars_core::schema::Schema;
use polars_expr::reduce::Reduction;
use polars_utils::itertools::Itertools;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::morsel::SourceToken;

enum ReduceState {
    Sink {
        selectors: Vec<StreamExpr>,
        reductions: Vec<Box<dyn Reduction>>,
    },
    Source(Option<DataFrame>),
    Done,
}

pub struct ReduceNode {
    state: ReduceState,
    output_schema: Arc<Schema>,
}

impl ReduceNode {
    pub fn new(
        selectors: Vec<StreamExpr>,
        reductions: Vec<Box<dyn Reduction>>,
        output_schema: Arc<Schema>,
    ) -> Self {
        Self {
            state: ReduceState::Sink {
                selectors,
                reductions,
            },
            output_schema,
        }
    }

    fn spawn_sink<'env, 's>(
        selectors: &'env [StreamExpr],
        reductions: &'env mut [Box<dyn Reduction>],
        scope: &'s TaskScope<'s, 'env>,
        recv: RecvPort<'_>,
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let parallel_tasks: Vec<_> = recv
            .parallel()
            .into_iter()
            .map(|mut recv| {
                let mut local_reductions: Vec<_> =
                    reductions.iter().map(|d| d.init_dyn()).collect();

                scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = recv.recv().await {
                        for (reduction, selector) in local_reductions.iter_mut().zip(selectors) {
                            // TODO: don't convert to physical representation here.
                            let input = selector.evaluate(morsel.df(), state).await?;
                            reduction.update(&input.to_physical_repr())?;
                        }
                    }

                    PolarsResult::Ok(local_reductions)
                })
            })
            .collect();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            for task in parallel_tasks {
                let local_reductions = task.await?;
                for (r1, r2) in reductions.iter_mut().zip(local_reductions) {
                    r1.combine(&*r2)?;
                }
            }

            Ok(())
        }));
    }

    fn spawn_source<'env, 's>(
        df: &'env mut Option<DataFrame>,
        scope: &'s TaskScope<'s, 'env>,
        send: SendPort<'_>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let mut send = send.serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let morsel = Morsel::new(df.take().unwrap(), MorselSeq::new(0), SourceToken::new());
            let _ = send.send(morsel).await;
            Ok(())
        }));
    }
}

impl ComputeNode for ReduceNode {
    fn name(&self) -> &str {
        "reduce"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = ReduceState::Done;
            },
            // Input is done, transition to being a source.
            ReduceState::Sink { reductions, .. } if matches!(recv[0], PortState::Done) => {
                let columns = reductions
                    .iter_mut()
                    .zip(self.output_schema.iter_fields())
                    .map(|(r, field)| {
                        r.finalize().map(|scalar| {
                            scalar.into_series(&field.name).cast(&field.dtype).unwrap()
                        })
                    })
                    .try_collect_vec()?;
                let out = DataFrame::new(columns).unwrap();

                self.state = ReduceState::Source(Some(out));
            },
            // We have sent the reduced dataframe, we are done.
            ReduceState::Source(df) if df.is_none() => {
                self.state = ReduceState::Done;
            },
            // Nothing to change.
            ReduceState::Done | ReduceState::Sink { .. } | ReduceState::Source(_) => {},
        }

        // Communicate our state.
        match &self.state {
            ReduceState::Sink { .. } => {
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
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send.len() == 1 && recv.len() == 1);
        match &mut self.state {
            ReduceState::Sink {
                selectors,
                reductions,
            } => {
                assert!(send[0].is_none());
                let recv_port = recv[0].take().unwrap();
                Self::spawn_sink(selectors, reductions, scope, recv_port, state, join_handles)
            },
            ReduceState::Source(df) => {
                assert!(recv[0].is_none());
                let send_port = send[0].take().unwrap();
                Self::spawn_source(df, scope, send_port, join_handles)
            },
            ReduceState::Done => unreachable!(),
        }
    }
}
