use std::collections::VecDeque;
use std::sync::Arc;

use arrow::offset;
use polars_core::schema::Schema;
use polars_core::utils::Container;
use polars_error::polars_ensure;

use super::compute_node_prelude::*;
use crate::async_primitives::connector::{Receiver, Sender};
use crate::morsel::SourceToken;

pub struct ShiftNode {
    state: State,
    output_schema: Arc<Schema>,
}

#[derive(Default)]
enum State {
    #[default]
    Uninit,
    PositiveHead {
        buffer: Buffer,
        head: usize,
        offset: usize,
        seq: u64,
    },
    PostivePassthrough {
        buffer: Buffer,
        offset: usize,
        seq: u64,
    },
    PositiveTail {
        buffer: Buffer,
        tail_shift: usize,
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
    Done,
}

// Keeps track of the total number rows in the buffers
#[derive(Default)]
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
    dbg!(&morsel.df());
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
            _ => todo!(),
            State::Uninit => {
                todo!()
                //let mut buffer = Buffer::new();
                //
                //let Ok(first) = self.recv(receiver, state).await else {
                //    return Ok(());
                //};
                //let first = first?;
                //
                //let offset_column = self
                //    .offset
                //    .evaluate(first.df(), &state.in_memory_exec_state)
                //    .await?;
                //polars_ensure!(offset_column.len() == 1, InvalidOperation: "expected a scalar for 'offset' in the 'shift'");
                //let offset_column = offset_column.get(0).unwrap();
                //let offset = offset_column
                //    .extract::<i64>()
                //    .expect("type checked at dsl resolving");
                //
                //// Deal with the trivial case
                //if offset == 0 {
                //    let _ = sender.send(first).await;
                //
                //    while let Ok(morsel) = receiver.recv().await {
                //        if sender.send(morsel).await.is_err() {
                //            break;
                //        }
                //    }
                //} else {
                //    if offset > 0 {
                //        buffer.push_back(first);
                //        *shift_state = State::Positive {
                //            buffer,
                //            head: offset as _,
                //            offset: offset as _,
                //            seq: 0,
                //        };
                //    } else {
                //        let mut skip = -offset as usize;
                //        let tail = -offset as usize;
                //        let len = first.df().height();
                //        let mut seq = 0;
                //
                //        if skip >= len {
                //            skip -= len
                //        } else {
                //            let morsel = first.map(|df| df.slice(skip as _, usize::MAX));
                //
                //            if send_morsel(morsel, sender, &mut seq).await.is_err() {
                //                return Ok(());
                //            };
                //            skip = 0;
                //        }
                //
                //        *shift_state = State::Negative { skip, tail, seq };
                //    }
                //    Box::pin(self.eval(shift_state, receiver, sender, state)).await?
                //}
            },
            State::PositiveHead {
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
            State::Done => todo!(),
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
        dbg!("update state");
        dbg!(&recv, &send);
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = State::Done;
        }

        // Handle offset
        if matches!(&self.state, State::Uninit) {
            recv[1] = PortState::Ready
        } else {
            recv[1] = PortState::Done
        }

        if recv[0] == PortState::Done {
            dbg!("left");
            match &mut self.state {
                State::Uninit => self.state = State::Done,
                State::Done => {
                    send[0] = PortState::Done;
                    recv[0] = PortState::Done;
                },
                State::PositiveHead {
                    buffer,
                    seq,
                    offset,
                    ..
                } => {
                    let buffer = std::mem::take(buffer);

                    //let buffer = std::mem::take(buffer);
                    self.state = State::PostivePassthrough {
                        buffer,
                        seq: *seq,
                        offset: *offset,
                    };

                    //self.state = State::PositiveTail {
                    //    buffer,
                    //    tail_shift: *offset,
                    //    seq: *seq,
                    //};
                },
                State::PositiveTail {
                    buffer, tail_shift, ..
                } => {
                    if buffer.len() <= *tail_shift {
                        self.state = State::Done;
                        recv[0] = PortState::Done;
                        send[0] = PortState::Done;
                    }
                },
                State::PostivePassthrough {
                    buffer,
                    seq,
                    offset,
                } => {
                    let buffer = std::mem::take(buffer);
                    dbg!(&buffer.inner);
                    self.state = State::PositiveTail {
                        buffer,
                        tail_shift: *offset,
                        seq: *seq,
                    }
                },
                State::Negative { .. } => todo!(),
            }
        } else {
            dbg!("right");
            match &mut self.state {
                State::Uninit => {
                    recv[0] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                },
                State::Done => {
                    recv[0] = PortState::Done;
                    send[0] = PortState::Done;
                },
                State::Negative { .. } => {
                    recv[0] = PortState::Ready;
                    send[0] = PortState::Ready;
                },
                State::PositiveHead {
                    head,
                    buffer,
                    offset,
                    seq,
                } => {
                    dbg!("head", *head, &buffer.inner);
                    if *head == 0 {
                        let buffer = std::mem::take(buffer);
                        self.state = State::PostivePassthrough {
                            buffer,
                            seq: *seq,
                            offset: *offset,
                        };
                    }
                },
                State::PostivePassthrough { .. } => {
                    dbg!("passthorugh");
                },
                State::PositiveTail { .. } => {
                    dbg!("tail");
                    recv[0] = PortState::Done;
                    send[0] = PortState::Ready;
                },
            }
        }

        //recv.swap_with_slice(send);
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
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        let t = match &mut self.state {
            shift_state @ State::Uninit => {
                dbg!("init");
                let receiver = recv_ports[1].take().unwrap().serial();

                scope.spawn_task(TaskPriority::Low, async move {
                    *shift_state = init(receiver).await?;
                    Ok(())
                })
            },
            State::PositiveHead {
                buffer,
                head,
                offset,
                seq,
            } => {
                dbg!("pos head");
                let receiver = recv_ports[0].take().unwrap().serial();
                let sender = send_ports[0].take().unwrap().serial();
                let schema = self.output_schema.clone();
                scope.spawn_task(TaskPriority::High, async move {
                    positive_head(receiver, sender, buffer, head, *offset, seq, schema).await?;
                    Ok(())
                })
            },
            State::PositiveTail {
                buffer,
                tail_shift,
                seq,
            } => {
                dbg!("pos tail");
                let sender = send_ports[0].take().unwrap().serial();
                scope.spawn_task(TaskPriority::High, async move {
                    positive_tail(sender, buffer, *tail_shift, seq).await;
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

async fn init(mut receiver: Receiver<Morsel>) -> PolarsResult<State> {
    let Ok(offset) = receiver.recv().await else {
        unreachable!("expected morsel")
    };

    let offset = offset.df();
    polars_ensure!(offset.shape() == (1, 1), InvalidOperation: "expected a scalar as 'offset' in 'shift'");
    let item = offset.get_columns()[0].get(0).unwrap();
    let offset = item
        .extract::<i64>()
        .expect("type checked at dsl resolving");

    if offset > 0 {
        Ok(State::PositiveHead {
            buffer: Buffer::new(),
            head: offset as _,
            offset: offset as _,
            seq: 0,
        })
    } else {
        Ok(State::Negative {
            skip: -offset as _,
            tail: -offset as _,
            seq: 0,
        })
    }
}

async fn positive_head(
    mut receiver: Receiver<Morsel>,
    mut sender: Sender<Morsel>,
    buffer: &mut Buffer,
    head: &mut usize,
    offset: usize,
    seq: &mut u64,
    output_schema: Arc<Schema>,
) -> PolarsResult<()> {
    // 1. We need to offset the array by `offset` and insert nulls
    // 2. Then we ensure we insert the nulls (also in the middle of a morsel if needed)
    let tail_shift = offset;
    while *head > 0 {
        if let Ok(next) = receiver.recv().await {
            buffer.push_back(next);
        } else {
            dbg!("EMPTIED BUFFER");
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
                .map(|df| DataFrame::full_null(&output_schema, df.height()));
            buffer.push_front(morsel);

            morsel = nulls;
            if send_morsel(morsel, &mut sender, seq).await.is_err() {
                break;
            }
        }
    }
    dbg!("FINISHED", *head);

    Ok(())
}

async fn positive_tail(
    mut sender: Sender<Morsel>,
    buffer: &mut Buffer,
    tail_shift: usize,
    seq: &mut u64,
) {
    while buffer.len_after_pop_front() > tail_shift {
        dbg!("LOOP");
        let morsel = buffer.pop_front();

        if send_morsel(morsel, &mut sender, seq).await.is_err() {
            break;
        }
    }
    // 4. Deal with the tail by slicing the last morsel
    // Slices last morsel length.
    if buffer.len() > tail_shift {
        let len = buffer.len() - tail_shift;
        let last_morsel = buffer.pop_front();
        let last_morsel = last_morsel.map(|df| df.slice(0, len));
        let _ = send_morsel(last_morsel, &mut sender, seq).await;
    }
}
