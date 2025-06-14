use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::schema::Schema;
use polars_core::utils::Container;
use polars_error::polars_ensure;

use super::compute_node_prelude::*;
use crate::async_primitives::connector::{Receiver, Sender};
use crate::expression::StreamExpr;
use crate::morsel::SourceToken;

pub struct ShiftNode {
    state: State,
    output_schema: Arc<Schema>,
}

#[derive(Default)]
enum State {
    #[default]
    Uninit,
    Positive {
        buffer: Buffer,
        head: usize,
        offset: usize,
        seq: u64,
    },
    Negative {
        // Rows to skip at the start of the DataFrame
        // This state will be decremented until it reaches 0
        // and we don't have to skip any rows anymore
        skip: usize,
        // Nulls to append at the end of the DataFrame
        tail: usize,
        seq: u64,
    },
}

// Keeps track of the total number rows in the buffers
struct Buffer {
    inner: VecDeque<Morsel>,
    rows: usize,
}

impl Buffer {
    fn new() -> Self {
        Self {
            inner: Default::default(),
            rows: 0,
        }
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn len_after_pop_front(&self) -> usize {
        self.len()
            - self
                .inner
                .front()
                .map(|morsel| morsel.df().height())
                .unwrap_or(0)
    }

    fn add(&mut self, morsel: &mut Morsel) {
        self.rows += morsel.df().height();
        let _ = morsel.take_consume_token();
    }

    fn push_back(&mut self, mut morsel: Morsel) {
        self.add(&mut morsel);
        self.inner.push_back(morsel);
    }
    fn push_front(&mut self, mut morsel: Morsel) {
        self.add(&mut morsel);
        self.inner.push_front(morsel);
    }
    fn pop_front(&mut self) -> Morsel {
        let morsel = self.inner.pop_front().unwrap();
        self.rows -= morsel.df().height();
        morsel
    }
}

/// Send a morsel and update the MorselSeq
async fn send_morsel(
    mut morsel: Morsel,
    sender: &mut Sender<Morsel>,
    seq: &mut u64,
) -> Result<(), Morsel> {
    *seq += 1;

    morsel.set_seq(MorselSeq::new(*seq));
    sender.send(morsel).await
}

impl ShiftNode {
    pub fn new(output_schema: Arc<Schema>) -> Self {
        Self {
            state: State::Uninit,
            output_schema,
        }
    }

    async fn recv(
        &self,
        receiver: &mut Receiver<Morsel>,
        state: &StreamingExecutionState,
    ) -> Result<PolarsResult<Morsel>, ()> {
        let Ok(morsel) = receiver.recv().await else {
            return Err(());
        };
        Ok(Ok(morsel))
    }

    async fn spawn_uninit(
        &mut self,
        shift_state: &mut State,
        receiver: &mut Receiver<Morsel>,
        sender: &mut Sender<Morsel>,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        todo!()
    }

    async fn eval(
        &mut self,
        shift_state: &mut State,
        receiver: &mut Receiver<Morsel>,
        sender: &mut Sender<Morsel>,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        match shift_state {
            State::Uninit => {
                let mut buffer = Buffer::new();

                let Ok(first) = self.recv(receiver, state).await else {
                    return Ok(());
                };
                let first = first?;

                let offset_column = self
                    .offset
                    .evaluate(first.df(), &state.in_memory_exec_state)
                    .await?;
                polars_ensure!(offset_column.len() == 1, InvalidOperation: "expected a scalar for 'offset' in the 'shift'");
                let offset_column = offset_column.get(0).unwrap();
                let offset = offset_column
                    .extract::<i64>()
                    .expect("type checked at dsl resolving");

                // Deal with the trivial case
                if offset == 0 {
                    let _ = sender.send(first).await;

                    while let Ok(morsel) = receiver.recv().await {
                        if sender.send(morsel).await.is_err() {
                            break;
                        }
                    }
                } else {
                    if offset > 0 {
                        buffer.push_back(first);
                        *shift_state = State::Positive {
                            buffer,
                            head: offset as _,
                            offset: offset as _,
                            seq: 0,
                        };
                    } else {
                        let mut skip = -offset as usize;
                        let tail = -offset as usize;
                        let len = first.df().height();
                        let mut seq = 0;

                        if skip >= len {
                            skip -= len
                        } else {
                            let morsel = first.map(|df| df.slice(skip as _, usize::MAX));

                            if send_morsel(morsel, sender, &mut seq).await.is_err() {
                                return Ok(());
                            };
                            skip = 0;
                        }

                        *shift_state = State::Negative { skip, tail, seq };
                    }
                    Box::pin(self.eval(shift_state, receiver, sender, state)).await?
                }
            },
            State::Positive {
                buffer,
                head,
                offset,
                seq,
            } => {
                let offset = *offset;

                // 1. We need to offset the array by `offset` and insert nulls
                // 2. Then we ensure we insert the nulls (also in the middle of a morsel if needed)
                let tail_shift = offset;
                while *head > 0 {
                    if let Ok(next) = self.recv(receiver, state).await {
                        let next = next?;
                        buffer.push_back(next);
                    } else {
                        break;
                    }
                    let mut morsel = buffer.pop_front();
                    let len = morsel.df().height();

                    // Last iteration
                    if len > *head {
                        let shifted_morsel = morsel.clone().map(|df| df.shift_seq(*head as _));

                        let tail = morsel.map(|df| df.tail(Some(*head)));
                        buffer.push_front(tail);
                        // Don't immediately return.
                        // It can be that this is the last morsel and it has to be sliced
                        // In that case
                        buffer.push_front(shifted_morsel);
                        *head = 0;
                    }
                    // Return full morsel of nulls
                    else {
                        *head -= len;
                        let nulls = morsel
                            .clone()
                            .map(|df| DataFrame::full_null(&self.output_schema, df.height()));
                        buffer.push_front(morsel);

                        morsel = nulls;
                        if send_morsel(morsel, sender, seq).await.is_err() {
                            break;
                        }
                    }
                }

                // 3. Now we have dealt with the starting 'offset'
                // we can pump the remaining morsel through, until we have reached the tail offset
                loop {
                    if buffer.len_after_pop_front() > tail_shift {
                        let morsel = buffer.pop_front();

                        if send_morsel(morsel, sender, seq).await.is_err() {
                            break;
                        }
                    } else {
                        let Ok(morsel) = self.recv(receiver, state).await else {
                            break;
                        };
                        buffer.push_back(morsel?);
                    }
                }

                // 4. Deal with the tail by slicing the last morsel
                // Slices last morsel length.
                if buffer.len() > tail_shift {
                    let len = buffer.len() - tail_shift;
                    let last_morsel = buffer.pop_front();
                    let last_morsel = last_morsel.map(|df| df.slice(0, len));
                    let _ = send_morsel(last_morsel, sender, seq).await;
                }
            },
            State::Negative { skip, tail, seq } => {
                while let Ok(morsel) = self.recv(receiver, state).await {
                    let morsel = morsel?;

                    // 1. Deals with the boundary effects.
                    // We 'skip' the initial row until we have reached the 'offset'.
                    // After that we pump all morsel.
                    if *skip != 0 {
                        let len = morsel.df().len();

                        if *skip >= len {
                            *skip -= len
                        } else {
                            let morsel = morsel.map(|df| df.slice(*skip as _, usize::MAX));

                            if morsel.df().is_empty() {
                                continue;
                            }

                            if send_morsel(morsel, sender, seq).await.is_err() {
                                return Ok(());
                            };
                            *skip = 0;
                        }
                    }
                    // 2. The non-boundary case, where we can pump morsels.
                    else if send_morsel(morsel, sender, seq).await.is_err() {
                        return Ok(());
                    }
                }

                // 3. When all morsels are processed, pass a full-null morsel of 'offset' length.
                let last_morsel = Morsel::new(
                    DataFrame::full_null(&self.output_schema, *tail),
                    Default::default(),
                    SourceToken::new(),
                );
                let _ = send_morsel(last_morsel, sender, seq).await;
            },
        }
        Ok(())
    }
}

impl ComputeNode for ShiftNode {
    fn name(&self) -> &str {
        "shift"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 2);

        let t = match self.state {
            State::Uninit => {
                let mut receiver = recv_ports[1].take().unwrap().serial();
                let mut sender = send_ports[1].take().unwrap().serial();

                let slf = &mut *self;
                scope.spawn_task(TaskPriority::High, async move {
                    let mut shift_state = std::mem::take(&mut slf.state);

                    slf.eval(&mut shift_state, &mut receiver, &mut sender, state)
                        .await?;
                    slf.state = shift_state;

                    Ok(())
                })
            },
            _ => {
                todo!()
            },
        };

        //let mut receiver = recv_ports[0].take().unwrap().serial();
        //let mut sender = send_ports[0].take().unwrap().serial();

        //let slf = &mut *self;
        //let t = scope.spawn_task(TaskPriority::High, async move {
        //    let mut shift_state = std::mem::take(&mut slf.state);
        //
        //    slf.eval(&mut shift_state, &mut receiver, &mut sender, state)
        //        .await?;
        //    slf.state = shift_state;
        //
        //    Ok(())
        //});
        join_handles.push(t);
    }
}
