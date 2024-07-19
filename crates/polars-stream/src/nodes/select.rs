use std::sync::Arc;

use polars_core::prelude::Series;
use polars_core::schema::Schema;
use polars_expr::prelude::PhysicalExpr;

use super::compute_node_prelude::*;

pub struct SelectNode {
    selectors: Vec<Arc<dyn PhysicalExpr>>,
    schema: Arc<Schema>,
    extend_original: bool,
    maybe_re_entrant: bool,
}

impl SelectNode {
    pub fn new(
        selectors: Vec<Arc<dyn PhysicalExpr>>,
        schema: Arc<Schema>,
        extend_original: bool,
        maybe_re_entrant: bool,
    ) -> Self {
        Self {
            selectors,
            schema,
            extend_original,
            maybe_re_entrant,
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
                // Helper to finish the operation.
                let extend_or_create = |df: DataFrame, mut selected: Vec<Series>| {
                    // Extend or create new dataframe.
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
                    PolarsResult::Ok(ret)
                };

                while let Ok(morsel) = recv.recv().await {
                    // We need spawn_blocking because evaluate could contain Python UDFs which
                    // recursively call the executor again.
                    let morsel = if slf.maybe_re_entrant {
                        let (df, seq, consume_token) = morsel.into_inner();
                        let mut selected = Vec::new();
                        for selector in &slf.selectors {
                            let df = df.clone();
                            let selector = selector.clone();
                            let state = state.clone();
                            selected.push(
                                polars_io::pl_async::get_runtime()
                                    .spawn_blocking(move || selector.evaluate(&df, &state))
                                    .await
                                    .unwrap()?,
                            );
                        }

                        let ret = extend_or_create(df, selected)?;

                        let mut morsel = Morsel::new(ret, seq);
                        if let Some(token) = consume_token {
                            morsel.set_consume_token(token);
                        }
                        morsel
                    }
                    // This is the happy path where there are guaranteed to be no re-entrant expressions.
                    else {
                        morsel.try_map(|df| {
                            // Select columns.
                            let selected: Vec<_> = slf
                                .selectors
                                .iter()
                                .map(|s| s.evaluate(&df, state))
                                .collect::<PolarsResult<_>>()?;

                            extend_or_create(df, selected)
                        })?
                    };

                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
