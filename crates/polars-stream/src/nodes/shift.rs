use std::collections::VecDeque;

use polars_error::polars_ensure;

use super::compute_node_prelude::*;
use crate::async_primitives::connector::{Receiver, Sender};
use crate::expression::StreamExpr;

pub struct ShiftNode {
    column: StreamExpr,
    offset: StreamExpr,
    state: State,
}

#[derive(Default)]
enum State {
    #[default]
    Uninit,
    Init {
        buffer: Buffer,
        offset: i64,
        seq: u64,
    },
}

struct Buffer {
    inner: VecDeque<Morsel>,
    len: usize,
}

impl Buffer {
    fn new() -> Self {
        Self {
            inner: Default::default(),
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
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
        self.len += morsel.df().height();
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
        self.len -= morsel.df().height();
        morsel
    }
}

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
    pub fn new(column: StreamExpr, offset: StreamExpr) -> Self {
        Self {
            column,
            offset,
            state: State::Uninit,
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
        Ok(morsel
            .async_try_map(|df| async move {
                let column = self.column.evaluate(&df, &state.in_memory_exec_state).await;

                column.map(|column| column.into_frame())
            })
            .await)
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
                    buffer.push_back(first);
                    *shift_state = State::Init {
                        buffer,
                        offset,
                        seq: 0,
                    };
                    Box::pin(self.eval(shift_state, receiver, sender, state)).await?
                }
            },
            State::Init {
                buffer,
                offset,
                seq,
            } if *offset >= 0 => {
                let offset = *offset as usize;

                // Buffer until the offset is reached.
                while buffer.len() < offset {
                    if let Ok(next) = self.recv(receiver, state).await {
                        let next = next?;

                        buffer.push_back(next);
                    } else {
                        break;
                    }
                }

                let mut shifted = offset;
                while buffer.len() >= shifted && shifted > 0 {
                    let mut morsel = buffer.pop_front();
                    let len = morsel.df().height();

                    // Last iteration
                    if len > shifted {
                        let shifted_morsel = morsel.clone().map(|df| df.shift_seq(shifted as _));

                        let tail = morsel.map(|df| df.tail(Some(len - shifted)));
                        buffer.push_front(tail);

                        morsel = shifted_morsel;
                        shifted = 0;
                    }
                    // Return full morsel of nulls
                    else {
                        shifted -= len;
                        let nulls = morsel
                            .clone()
                            .map(|df| DataFrame::full_null(df.schema(), df.height()));
                        buffer.push_front(morsel);

                        morsel = nulls;
                    }

                    if send_morsel(morsel, sender, seq).await.is_err() {
                        break;
                    }
                }

                let tail_shift = offset;
                loop {
                    if buffer.len_after_pop_front() >= tail_shift {
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

                if buffer.len() != tail_shift {
                    assert!(buffer.len_after_pop_front() < tail_shift);
                    let last_morsel = buffer.pop_front();
                    let len = tail_shift - buffer.len();
                    let last_morsel = last_morsel.map(|df| df.slice(0, len));
                    let _ = send_morsel(last_morsel, sender, seq).await;
                }
            },
            State::Init {
                buffer: _,
                offset: _,
                seq: _,
            } => {
                todo!()
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let mut receiver = recv_ports[0].take().unwrap().serial();
        let mut sender = send_ports[0].take().unwrap().serial();

        let slf = &mut *self;
        let t = scope.spawn_task(TaskPriority::High, async move {
            let mut shift_state = std::mem::take(&mut slf.state);

            slf.eval(&mut shift_state, &mut receiver, &mut sender, state)
                .await?;
            slf.state = shift_state;

            Ok(())
        });
        join_handles.push(t);
    }
}
