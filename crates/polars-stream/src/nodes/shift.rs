use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::prelude::*;
use polars_core::schema::Schema;

use super::compute_node_prelude::*;
use crate::async_primitives::connector::{Receiver, Sender};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{SourceToken, get_ideal_morsel_size};
use crate::nodes::in_memory_sink::InMemorySinkNode;

#[allow(private_interfaces)]
pub enum ShiftNode {
    GatheringParams {
        offset: InMemorySinkNode,
        fill: Option<InMemorySinkNode>,
        output_schema: Arc<Schema>,
    },
    Shifting(ShiftState),
    Done,
}

struct ShiftState {
    offset: i64,
    rows_received: usize,
    rows_sent: usize,
    buffer: VecDeque<DataFrame>,
    fill: DataFrame,
    seq: MorselSeq,
}

impl ShiftState {
    async fn shift_positive(
        &mut self,
        mut recv: Option<Receiver<Morsel>>,
        mut send: Sender<Morsel>,
    ) -> PolarsResult<()> {
        let mut source_token = SourceToken::new();
        let wait_group = WaitGroup::default();

        while recv.is_some() || self.rows_received != self.rows_sent {
            // Try to get more data if necessary.
            if self.rows_received == self.rows_sent {
                if let Some(r) = &mut recv {
                    let Ok(morsel) = r.recv().await else { break };
                    source_token = morsel.source_token().clone();
                    if morsel.df().is_empty() {
                        continue;
                    }
                    self.rows_received += morsel.df().height();
                    self.buffer.push_back(morsel.into_df());
                }
            }

            // Send along a morsel.
            let df;
            if self.rows_sent < self.offset as usize {
                let len = self.rows_received.min(self.offset as usize) - self.rows_sent;
                df = self.fill.new_from_index(0, len);
            } else {
                let src = self.buffer.front_mut().unwrap();
                let len = self.rows_received - self.rows_sent;
                (df, *src) = src.split_at(len as i64);
                if src.is_empty() {
                    self.buffer.pop_front();
                }
            };
            self.rows_sent += df.height();

            let mut morsel = Morsel::new(df, self.seq, source_token.clone());
            self.seq = self.seq.successor();
            morsel.set_consume_token(wait_group.token());
            if send.send(morsel).await.is_err() {
                break;
            }
            wait_group.wait().await;
            if source_token.stop_requested() {
                break;
            }
        }

        Ok(())
    }

    async fn shift_negative(
        &mut self,
        mut recv: Receiver<Morsel>,
        mut send: Sender<Morsel>,
    ) -> PolarsResult<()> {
        let shift = self.offset.unsigned_abs() as usize;

        while let Ok(mut morsel) = recv.recv().await {
            let shift_needed = shift.saturating_sub(self.rows_received);
            self.rows_received += morsel.df().height();
            if shift_needed > 0 {
                morsel =
                    morsel.map(|df| df.slice(shift_needed.min(df.height()) as i64, df.height()));
            }
            if morsel.df().is_empty() {
                continue;
            }

            morsel.set_seq(self.seq);
            self.seq = self.seq.successor();
            self.rows_sent += morsel.df().height();
            if send.send(morsel).await.is_err() {
                break;
            }
        }

        Ok(())
    }

    async fn flush_negative(
        &mut self,
        mut send: Sender<Morsel>,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let source_token = SourceToken::new();
        let wait_group = WaitGroup::default();

        let total_len = self.rows_received - self.rows_sent;
        let ideal_morsel_count = (total_len / get_ideal_morsel_size()).max(1);
        let morsel_count = ideal_morsel_count.next_multiple_of(state.num_pipelines);
        let morsel_size = total_len.div_ceil(morsel_count).max(1);

        while self.rows_sent != self.rows_received {
            let len = morsel_size.min(self.rows_received - self.rows_sent);
            let df = self.fill.new_from_index(0, len);
            self.rows_sent += len;

            let mut morsel = Morsel::new(df, self.seq, source_token.clone());
            self.seq = self.seq.successor();
            morsel.set_consume_token(wait_group.token());
            if send.send(morsel).await.is_err() {
                break;
            }
            wait_group.wait().await;
            if source_token.stop_requested() {
                break;
            }
        }

        Ok(())
    }
}

impl ShiftNode {
    pub fn new(output_schema: Arc<Schema>, offset_schema: Arc<Schema>, with_fill: bool) -> Self {
        assert!(offset_schema.len() == 1);
        Self::GatheringParams {
            offset: InMemorySinkNode::new(offset_schema),
            fill: with_fill.then(|| InMemorySinkNode::new(output_schema.clone())),
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
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() <= 3 && send.len() == 1);

        // Are we done?
        if recv[0] == PortState::Done {
            if let Self::Shifting(shift_state) = self {
                if shift_state.rows_sent == shift_state.rows_received {
                    *self = Self::Done;
                }
            }
        }

        // Do we have the parameters to start shifting?
        if recv[1..].iter().all(|p| *p == PortState::Done) {
            if let Self::GatheringParams {
                offset,
                fill,
                output_schema,
            } = self
            {
                let offset_frame = offset.get_output()?.unwrap();
                polars_ensure!(offset_frame.height() == 1, ComputeError: "got more than one value for 'n' in shift");
                let offset_item = offset_frame.get_columns()[0].get(0)?;
                let offset = if offset_item.is_null() {
                    polars_warn!(
                        Deprecation, // @2.0
                        "shift value 'n' is null, which currently returns a column of null values. This will become an error in the future.",
                    );
                    // @2.0: Currently we still require the entire output to become null
                    // if the shift is null, simulate this with an infinite negative shift.
                    *fill = None;
                    i64::MIN
                } else {
                    offset_item.extract::<i64>().ok_or_else(
                        || polars_err!(ComputeError: "invalid value of 'n' in shift: {:?}", offset_item),
                    )?
                };

                let fill_frame = if let Some(fill) = fill {
                    fill.get_output()?.unwrap()
                } else {
                    DataFrame::empty_with_schema(output_schema)
                };

                *self = Self::Shifting(ShiftState {
                    offset,
                    rows_received: 0,
                    rows_sent: 0,
                    buffer: VecDeque::new(),
                    fill: fill_frame,
                    seq: MorselSeq::default(),
                })
            }
        }

        match self {
            Self::GatheringParams { offset, fill, .. } => {
                offset.update_state(&mut recv[1..2], &mut [], state)?;
                if let Some(fill) = fill {
                    fill.update_state(&mut recv[2..3], &mut [], state)?;
                }
                recv[0] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            Self::Shifting(shift_state) => {
                if recv[0] == PortState::Done && shift_state.rows_sent < shift_state.rows_received {
                    send[0] = PortState::Ready;
                } else {
                    recv[..1].swap_with_slice(send);
                }
                recv[1..].fill(PortState::Done);
            },
            Self::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }
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
        assert!(recv_ports.len() <= 3 && send_ports.len() == 1);
        match self {
            Self::GatheringParams {
                offset,
                fill,
                output_schema: _,
            } => {
                assert!(recv_ports[0].is_none());
                assert!(send_ports[0].is_none());
                if recv_ports[1].is_some() {
                    offset.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
                if matches!(recv_ports.get(2), Some(Some(_))) {
                    fill.as_mut().unwrap().spawn(
                        scope,
                        &mut recv_ports[2..3],
                        &mut [],
                        state,
                        join_handles,
                    );
                }
            },
            Self::Shifting(shift_state) => {
                assert!(recv_ports[1..].iter().all(|p| p.is_none()));
                let recv = recv_ports[0].take().map(|p| p.serial());
                let send = send_ports[0].take().unwrap().serial();
                join_handles.push(if shift_state.offset >= 0 {
                    scope.spawn_task(TaskPriority::High, shift_state.shift_positive(recv, send))
                } else if let Some(r) = recv {
                    scope.spawn_task(TaskPriority::High, shift_state.shift_negative(r, send))
                } else {
                    scope.spawn_task(TaskPriority::High, shift_state.flush_negative(send, state))
                });
            },
            Self::Done => unreachable!(),
        }
    }
}
