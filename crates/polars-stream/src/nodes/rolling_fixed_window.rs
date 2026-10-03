use std::fmt;
use std::sync::Arc;

use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsError, PolarsResult};
use polars_plan::dsl::ColumnsUdf;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::relaxed_cell::RelaxedCell;

use super::ComputeNode;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::{RecvPort, SendPort};
use crate::utils::morsel_distributor::morsel_distributor;

/// The rows `[i + offset, i + offset + length)` form the window of row `i`, clipped to the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RollingFixedWindow {
    pub offset: i64,
    pub length: usize,
}

impl RollingFixedWindow {
    /// The number of rows after row `i` that the window of row `i` needs.
    fn ahead(&self) -> u64 {
        (self.offset + self.length as i64 - 1).max(0) as u64
    }

    /// The number of rows before row `i` that the window of row `i` needs.
    fn behind(&self) -> u64 {
        (-self.offset).max(0) as u64
    }
}

impl fmt::Display for RollingFixedWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let end = self.offset + self.length as i64;
        write!(f, "{}..{end}", self.offset)
    }
}

/// Applies a function to consecutive rows, where the output of each row only depends on the rows
/// in its window.
///
/// The function is applied to contiguous batches of rows and must return one value per row, with
/// the windows truncated at the edges of the batch.
pub struct RollingFixedWindowNode {
    func: Arc<dyn ColumnsUdf>,
    window: RollingFixedWindow,
    output_name: PlSmallStr,

    /// The input rows `[buf_start, buf_start + buf.height())`.
    buf: DataFrame,
    buf_start: u64,
    /// The first row that has not been sent yet.
    next_row: u64,
    is_finished: bool,

    seq: MorselSeq,
    seq_offset: Arc<RelaxedCell<u64>>,
}

impl RollingFixedWindowNode {
    pub fn new(
        func: Arc<dyn ColumnsUdf>,
        window: RollingFixedWindow,
        output_name: PlSmallStr,
        schema: SchemaRef,
    ) -> Self {
        Self {
            func,
            window,
            output_name,
            buf: DataFrame::empty_with_arc_schema(schema),
            buf_start: 0,
            next_row: 0,
            is_finished: false,
            seq: MorselSeq::default(),
            seq_offset: Arc::default(),
        }
    }

    /// Returns the rows to apply the function to, together with the range of its output to send,
    /// if enough rows are ready to be sent.
    fn next_batch(&mut self) -> Option<(DataFrame, usize, usize)> {
        let buf_end = self.buf_start + self.buf.height() as u64;
        let ahead = self.window.ahead();
        let behind = self.window.behind();

        let ready_end = buf_end.saturating_sub(ahead);
        let num_ready = ready_end.saturating_sub(self.next_row);

        // Every batch also computes up to `behind + ahead + 1` rows around the ready rows that it
        // doesn't send, so wait until four times as many rows are ready to keep this overhead at
        // most 25%.
        let min_ready = 4 * (behind + ahead + 1);
        if num_ready < min_ready {
            return None;
        }

        let batch = self.buf.clone();
        let out_offset = (self.next_row - self.buf_start) as usize;
        self.next_row = ready_end;

        // Some functions limit the window size to the batch size, so keep at least a full window
        // for the next batch. The first batch holds one as `min_ready >= length`.
        let keep_start = ready_end
            .saturating_sub(behind)
            .min(buf_end.saturating_sub(self.window.length as u64))
            .max(self.buf_start);
        self.buf = self
            .buf
            .slice((keep_start - self.buf_start) as i64, usize::MAX);
        self.buf_start = keep_start;

        Some((batch, out_offset, num_ready as usize))
    }

    fn evaluate(
        func: &dyn ColumnsUdf,
        output_name: &PlSmallStr,
        batch: DataFrame,
        out_offset: usize,
        out_len: usize,
    ) -> PolarsResult<DataFrame> {
        let mut columns = batch.into_columns();
        let out = func.call_udf(&mut columns)?;
        Ok(out
            .slice(out_offset as i64, out_len)
            .with_name(output_name.clone())
            .into_frame())
    }
}

impl ComputeNode for RollingFixedWindowNode {
    fn name(&self) -> &str {
        "rolling-fixed-window-function"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            self.is_finished = true;
            self.buf = self.buf.clear();
        } else if recv[0] == PortState::Done {
            send[0] = if self.is_finished {
                PortState::Done
            } else {
                PortState::Ready
            };
        } else {
            recv.swap_with_slice(send);
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);

        let Some(recv) = recv_ports[0].take() else {
            // The input is finished, send all remaining rows.
            let mut send = send_ports[0].take().unwrap().serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                self.is_finished = true;
                let buf_end = self.buf_start + self.buf.height() as u64;
                if self.next_row < buf_end {
                    let out_offset = (self.next_row - self.buf_start) as usize;
                    let batch = std::mem::take(&mut self.buf);
                    let df = Self::evaluate(
                        &*self.func,
                        &self.output_name,
                        batch,
                        out_offset,
                        usize::MAX,
                    )?;
                    let seq = self.seq.successor().offset_by_u64(self.seq_offset.load());
                    _ = send
                        .send(Morsel::new_unregistered(df, seq, SourceToken::new()))
                        .await;
                }

                Ok(())
            }));
            return;
        };

        let mut recv = recv.serial();
        let send = send_ports[0].take().unwrap().parallel();

        let (mut distributor, rxs) = morsel_distributor(
            send.len(),
            *DEFAULT_DISTRIBUTOR_BUFFER_SIZE,
            self.seq_offset.clone(),
        );

        // Worker tasks.
        //
        // These apply the function to the batches.
        join_handles.extend(rxs.into_iter().zip(send).map(|(mut rx, mut tx)| {
            let wg = WaitGroup::default();
            let func = self.func.clone();
            let output_name = self.output_name.clone();
            scope.spawn_task(TaskPriority::High, async move {
                while let Ok((mut morsel, (out_offset, out_len))) = rx.recv().await {
                    morsel = morsel
                        .try_map::<PolarsError, _>(|df| {
                            Self::evaluate(&*func, &output_name, df, out_offset, out_len)
                        })
                        .await?;
                    morsel.set_consume_token(wg.token());

                    if tx.send(morsel).await.is_err() {
                        break;
                    }
                    wg.wait().await;
                }

                Ok(())
            })
        }));

        // Distributor task.
        //
        // This buffers the input and cuts it into batches.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                let (sf, seq, source_token, wait_token) = morsel.into_inner();
                let df = sf.into_df().await;
                self.seq = seq;
                drop(wait_token);

                if df.height() == 0 {
                    continue;
                }

                self.buf.vstack_mut_owned(df)?;

                if let Some((batch, out_offset, out_len)) = self.next_batch() {
                    let morsel = Morsel::new_unregistered(batch, seq, source_token);
                    if distributor
                        .send((morsel, (out_offset, out_len)))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }

            Ok(())
        }));
    }
}
