use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, format_pl_smallstr};
use tokio::sync::oneshot;

use super::{
    JoinHandle, Morsel, MorselSeq, SourceNode, SourceOutput, StreamingExecutionState, TaskPriority,
};
use crate::async_executor::spawn;
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;

type GetBatchFn =
    Box<dyn Fn(&StreamingExecutionState) -> PolarsResult<Option<DataFrame>> + Send + Sync>;

pub struct BatchSourceNode {
    pub name: PlSmallStr,
    pub output_schema: SchemaRef,
    pub get_batch_fn: Option<GetBatchFn>,
}

impl BatchSourceNode {
    pub fn new(name: &str, output_schema: SchemaRef, get_batch_fn: Option<GetBatchFn>) -> Self {
        let name = format_pl_smallstr!("batch_source[{name}]");
        Self {
            name,
            output_schema,
            get_batch_fn,
        }
    }
}

impl SourceNode for BatchSourceNode {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn is_source_output_parallel(&self, _is_receiver_serial: bool) -> bool {
        false
    }

    fn spawn_source(
        &mut self,
        mut output_recv: Receiver<SourceOutput>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<oneshot::Sender<polars_utils::IdxSize>>,
    ) {
        // We only spawn this once, so this is all fine.
        let output_schema = self.output_schema.clone();
        let get_batch_fn = self.get_batch_fn.take().unwrap();
        let state = state.clone();
        join_handles.push(spawn(TaskPriority::Low, async move {
            let mut seq = MorselSeq::default();
            let mut n_rows_seen = 0;

            'phase_loop: while let Ok(phase_output) = output_recv.recv().await {
                let mut sender = phase_output.port.serial();
                let source_token = SourceToken::new();
                let wait_group = WaitGroup::default();

                loop {
                    let df = (get_batch_fn)(&state)?;
                    let Some(df) = df else {
                        if let Some(unrestricted_row_count) = unrestricted_row_count {
                            if unrestricted_row_count.send(n_rows_seen).is_err() {
                                return Ok(());
                            }
                        }

                        if n_rows_seen == 0 {
                            let morsel = Morsel::new(
                                DataFrame::empty_with_schema(output_schema.as_ref()),
                                seq,
                                source_token.clone(),
                            );
                            if sender.send(morsel).await.is_err() {
                                return Ok(());
                            }
                        }

                        break 'phase_loop;
                    };

                    let num_rows = IdxSize::try_from(df.height()).map_err(|_| {
                        polars_err!(bigidx, ctx = "batch source", size = df.height())
                    })?;
                    n_rows_seen = n_rows_seen.checked_add(num_rows).ok_or_else(|| {
                        polars_err!(
                            bigidx,
                            ctx = "batch source",
                            size = n_rows_seen as usize + num_rows as usize
                        )
                    })?;

                    let mut morsel = Morsel::new(df, seq, source_token.clone());
                    morsel.set_consume_token(wait_group.token());
                    seq = seq.successor();

                    if sender.send(morsel).await.is_err() {
                        return Ok(());
                    }

                    wait_group.wait().await;
                    if source_token.stop_requested() {
                        phase_output.outcome.stop();
                        continue 'phase_loop;
                    }
                }
            }

            Ok(())
        }));
    }
}
