use std::sync::Arc;

use polars_core::schema::Schema;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;

pub struct SelectNode {
    selectors: Vec<StreamExpr>,
    schema: Arc<Schema>,
    extend_original: bool,
}

impl SelectNode {
    pub fn new(selectors: Vec<StreamExpr>, schema: Arc<Schema>, extend_original: bool) -> Self {
        Self {
            selectors,
            schema,
            extend_original,
        }
    }
}

impl ComputeNode for SelectNode {
    fn name(&self) -> &str {
        "select"
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
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.len() == 1 && send.len() == 1);
        let receivers = recv[0].take().unwrap().parallel();
        let senders = send[0].take().unwrap().parallel();

        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {
                    let (df, seq, source_token, consume_token) = morsel.into_inner();
                    let mut selected = Vec::new();
                    for selector in slf.selectors.iter() {
                        let s = selector.evaluate(&df, state).await?;
                        selected.push(s);
                    }

                    let ret = if slf.extend_original {
                        let mut out = df;
                        out._add_columns(selected, &slf.schema)?;
                        out
                    } else {
                        // Broadcast scalars.
                        let max_non_unit_length = selected
                            .iter()
                            .map(|s| s.len())
                            .filter(|l| *l != 1)
                            .max()
                            .unwrap_or(1);
                        for s in &mut selected {
                            if s.len() != max_non_unit_length {
                                assert!(s.len() == 1, "got series of incompatible lengths");
                                *s = s.new_from_index(0, max_non_unit_length);
                            }
                        }
                        unsafe { DataFrame::new_no_checks(selected) }
                    };

                    let mut morsel = Morsel::new(ret, seq, source_token);
                    if let Some(token) = consume_token {
                        morsel.set_consume_token(token);
                    }

                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
