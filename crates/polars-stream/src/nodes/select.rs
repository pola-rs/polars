use std::sync::Arc;

use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_expr::prelude::PhysicalExpr;

use super::compute_node_prelude::*;

pub struct SelectNode {
    selectors: Vec<Arc<dyn PhysicalExpr>>,
    schema: Arc<Schema>,
    extend_original: bool,
}

impl SelectNode {
    pub fn new(
        selectors: Vec<Arc<dyn PhysicalExpr>>,
        schema: Arc<Schema>,
        extend_original: bool,
    ) -> Self {
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
            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel.try_map(|df| {
                    // Select columns.
                    let mut selected: Vec<Series> = self
                        .selectors
                        .iter()
                        .map(|s| s.evaluate(&df, state))
                        .collect::<PolarsResult<_>>()?;

                    // Extend or create new dataframe.
                    let ret = if self.extend_original {
                        let mut out = df.clone();
                        out._add_columns(selected, &self.schema)?;
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
                })?;

                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        })
    }
}
