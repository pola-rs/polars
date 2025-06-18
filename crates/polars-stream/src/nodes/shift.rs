use std::collections::VecDeque;
use std::sync::Arc;

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
    PositiveTail {
        buffer: Buffer,
        tail_shift: usize,
        seq: u64,
    },
    PositivePass {
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
    NegativeFlush {
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
    sender.send(morsel).await
}

impl ShiftNode {
    pub fn new(output_schema: Arc<Schema>) -> Self {
        Self {
            state: State::Uninit,
            output_schema,
        }
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
        assert!(recv.len() == 2 && send.len() == 1);
        dbg!(&recv, &send);

        if send[0] == PortState::Done {
            self.state = State::Done;
            return Ok(());
        }

        // Handle offset
        if matches!(&self.state, State::Uninit) {
            recv[1] = PortState::Ready
        } else {
            recv[1] = PortState::Done
        }

        match &mut self.state {
            State::Uninit => {
                recv[0] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            State::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
            State::PositivePass {
                buffer,
                tail_shift,
                seq,
            } => {
                if matches!(recv[0], PortState::Done) {
                    self.state = State::PositiveTail {
                        buffer: std::mem::take(buffer),
                        tail_shift: *tail_shift,
                        seq: *seq,
                    }
                }
            },
            State::Negative { tail, seq, .. } => {
                if matches!(recv[0], PortState::Done) {
                    self.state = State::NegativeFlush {
                        tail: *tail,
                        seq: *seq,
                    }
                }
            },
            State::PositiveHead { .. } => {
                recv[0] = PortState::Ready;
                send[0] = PortState::Ready;
            },
            State::PositiveTail { .. } | State::NegativeFlush { .. } => {},
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
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        let t = match std::mem::take(&mut self.state) {
            State::Uninit => {
                let slf = self;
                let receiver = recv_ports[1].take().unwrap().serial();
                //let _ = recv_ports[0].take().unwrap();
                //let _ = send_ports[0].take().unwrap();

                scope.spawn_task(TaskPriority::Low, async move {
                    slf.state = init(receiver).await?;
                    Ok(())
                })
            },
            State::PositiveHead {
                buffer,
                mut head,
                offset,
                mut seq,
            } => {
                let receiver = recv_ports[0].take().unwrap().serial();
                let sender = send_ports[0].take().unwrap().serial();
                let slf = self;
                let schema = slf.output_schema.clone();
                scope.spawn_task(TaskPriority::High, async move {
                    slf.state = positive_head(
                        receiver, sender, buffer, &mut head, offset, &mut seq, schema,
                    )
                    .await?;

                    Ok(())
                })
            },
            State::PositivePass {
                buffer,
                tail_shift,
                mut seq,
            } => {
                let receiver = recv_ports[0].take().unwrap().serial();
                let sender = send_ports[0].take().unwrap().serial();
                let slf = self;
                scope.spawn_task(TaskPriority::High, async move {
                    slf.state =
                        positive_pass(receiver, sender, buffer, tail_shift, &mut seq).await?;

                    Ok(())
                })
            },
            State::PositiveTail {
                buffer,
                tail_shift,
                mut seq,
            } => {
                let sender = send_ports[0].take().unwrap().serial();
                let slf = self;
                scope.spawn_task(TaskPriority::High, async move {
                    positive_tail(sender, buffer, tail_shift, &mut seq).await;
                    slf.state = State::Done;
                    Ok(())
                })
            },
            State::Negative {
                mut skip,
                tail,
                mut seq,
            } => {
                let receiver = recv_ports[0].take().unwrap().serial();
                let sender = send_ports[0].take().unwrap().serial();
                let slf = self;
                scope.spawn_task(TaskPriority::High, async move {
                    negative(receiver, sender, &mut skip, &mut seq).await?;

                    slf.state = State::Negative { skip, tail, seq };
                    Ok(())
                })
            },
            State::NegativeFlush { tail, mut seq } => {
                let sender = send_ports[0].take().unwrap().serial();
                let slf = self;
                scope.spawn_task(TaskPriority::High, async move {
                    negative_flush(sender, tail, &mut seq, &slf.output_schema).await?;

                    slf.state = State::Done;
                    Ok(())
                })
            },
            State::Done => unreachable!(),
        };

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
    mut buffer: Buffer,
    head: &mut usize,
    offset: usize,
    seq: &mut u64,
    output_schema: Arc<Schema>,
) -> PolarsResult<State> {
    // 1. We need to offset the array by `offset` and insert nulls
    // 2. Then we ensure we insert the nulls (also in the middle of a morsel if needed)
    while *head > 0 {
        if let Ok(next) = receiver.recv().await {
            buffer.push_back(next);
        } else {
            return Ok(State::PositiveHead {
                buffer,
                head: *head,
                offset,
                seq: *seq,
            });
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
                return Ok(State::Done);
            }
        }
    }

    positive_pass(receiver, sender, buffer, offset, seq).await
}

async fn positive_pass(
    mut recv: Receiver<Morsel>,
    mut sender: Sender<Morsel>,
    mut buffer: Buffer,
    tail_shift: usize,
    seq: &mut u64,
) -> PolarsResult<State> {
    // 3. Now we have dealt with the starting 'offset'
    // we can pump the remaining morsel through, until we have reached the tail offset

    loop {
        if buffer.len_after_pop_front() > tail_shift {
            let morsel = buffer.pop_front();

            if send_morsel(morsel, &mut sender, seq).await.is_err() {
                return Ok(State::Done);
            }
        } else {
            let Ok(morsel) = recv.recv().await else {
                // There might come a new phase
                return Ok(State::PositivePass {
                    buffer,
                    tail_shift,
                    seq: *seq,
                });
            };
            buffer.push_back(morsel);
        }
    }
}

async fn positive_tail(
    mut sender: Sender<Morsel>,
    mut buffer: Buffer,
    tail_shift: usize,
    seq: &mut u64,
) {
    while buffer.len_after_pop_front() > tail_shift {
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

async fn negative(
    mut receiver: Receiver<Morsel>,
    mut sender: Sender<Morsel>,
    skip: &mut usize,
    seq: &mut u64,
) -> PolarsResult<()> {
    while let Ok(morsel) = receiver.recv().await {
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

                if send_morsel(morsel, &mut sender, seq).await.is_err() {
                    return Ok(());
                };
                *skip = 0;
            }
        }
        // 2. The non-boundary case, where we can pump morsels.
        else if send_morsel(morsel, &mut sender, seq).await.is_err() {
            return Ok(());
        }
    }
    Ok(())
}

async fn negative_flush(
    mut sender: Sender<Morsel>,
    tail: usize,
    seq: &mut u64,
    output_schema: &Schema,
) -> PolarsResult<()> {
    let last_morsel = Morsel::new(
        DataFrame::full_null(output_schema, tail),
        Default::default(),
        SourceToken::new(),
    );
    let _ = send_morsel(last_morsel, &mut sender, seq).await;
    Ok(())
}
