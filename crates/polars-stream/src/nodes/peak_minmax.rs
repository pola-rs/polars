use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, Column, IntoColumn};
use polars_error::PolarsResult;
use polars_ops::prelude::peaks;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::wait_group::WaitGroup;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::{RecvPort, SendPort};

enum State {
    /// No morsels seen yet.
    Start,
    /// We have seen one morsel. Wait until 1 more to start streaming out data.
    One(MorselSeq, Column),
    /// We have seen two morsels. We have saved the last value of 2 morsels ago and the last
    /// morsel.
    Two(AnyValue<'static>, MorselSeq, Column),
    /// No more morsels will be received.
    Done,
}

pub struct PeakMinMaxNode {
    state: State,

    /// Is the node the `peak_max`?
    is_peak_max: bool,
}

impl PeakMinMaxNode {
    pub fn new(is_peak_max: bool) -> Self {
        Self {
            state: State::Start,
            is_peak_max,
        }
    }
}

impl ComputeNode for PeakMinMaxNode {
    fn name(&self) -> &str {
        if self.is_peak_max {
            "peaks_max"
        } else {
            "peaks_min"
        }
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if matches!(self.state, State::Done) {
            send[0] = PortState::Done;
            recv[0] = PortState::Done;
        } else if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            self.state = State::Done;
        } else if recv[0] == PortState::Done {
            if matches!(self.state, State::Start) {
                send[0] = PortState::Done;
            } else {
                send[0] = PortState::Ready;
            }
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
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        let recv = recv_ports[0].take();
        let mut send = send_ports[0].take().unwrap().serial();

        match recv {
            // No more morsels to receive. Flush out the remaining data.
            None => {
                if matches!(self.state, State::Start) {
                    return;
                }

                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    let (start, seq, prev_column) = match &self.state {
                        State::Start => unreachable!(),
                        State::One(seq, df) => (&AnyValue::Int8(0), *seq, df),
                        State::Two(av, seq, df) => (av, *seq, df),
                        State::Done => unreachable!(),
                    };

                    let column = peaks::peak_min_max(
                        prev_column,
                        start,
                        &AnyValue::Int8(0),
                        self.is_peak_max,
                    )?
                    .into_column();
                    let df = DataFrame::new(vec![column]).unwrap();
                    _ = send.send(Morsel::new(df, seq, SourceToken::new())).await;

                    self.state = State::Done;
                    Ok(())
                }));
            },

            Some(recv) => {
                let mut recv = recv.serial();
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    let source_token = SourceToken::new();

                    while let Ok(m) = recv.recv().await {
                        let (df, seq, in_source_token, in_wait_token) = m.into_inner();
                        drop(in_wait_token);
                        if df.height() == 0 {
                            continue;
                        }

                        assert_eq!(df.width(), 1);
                        let column = &df[0];

                        let (start, prev_seq, prev_column) = match &self.state {
                            State::Start => {
                                self.state = State::One(seq, column.clone());
                                continue;
                            },
                            State::One(prev_seq, prev_column) => {
                                (&AnyValue::Int8(0), *prev_seq, prev_column)
                            },
                            State::Two(prev_start, prev_seq, prev_column) => {
                                (prev_start, *prev_seq, prev_column)
                            },
                            State::Done => unreachable!(),
                        };
                        let end = &column.get(0).unwrap();
                        let out = peaks::peak_min_max(prev_column, start, end, self.is_peak_max)?
                            .into_column();

                        let wg = WaitGroup::default();
                        let mut m = Morsel::new(
                            DataFrame::new(vec![out]).unwrap(),
                            prev_seq,
                            source_token.clone(),
                        );
                        m.set_consume_token(wg.token());

                        if send.send(m).await.is_err() {
                            self.state = State::Done;
                            break;
                        }

                        wg.wait().await;
                        if source_token.stop_requested() {
                            in_source_token.stop();
                        }

                        let prev_end = prev_column
                            .get(prev_column.len() - 1)
                            .unwrap()
                            .to_physical()
                            .into_static();
                        self.state = State::Two(prev_end, seq, column.clone());
                    }
                    Ok(())
                }));
            },
        }
    }
}
