use polars_core::datatypes::AnyValue;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, PlSmallStr, SortOptions};
use polars_ops::prelude::SeriesMethods;

use super::compute_node_prelude::*;
use crate::morsel::SourceToken;
use crate::nodes::ComputeNode;

enum IsSortedState {
    Sink {
        is_sorted: bool,
        last_value: Option<AnyValue<'static>>,
        output_name: Option<PlSmallStr>,
    },
    Source(Option<DataFrame>),
    Done,
}

pub struct IsSortedNode {
    state: IsSortedState,
    descending: Option<bool>,
    nulls_last: Option<bool>,
}

impl IsSortedNode {
    pub fn new(descending: Option<bool>, nulls_last: Option<bool>) -> Self {
        Self {
            state: IsSortedState::Sink {
                is_sorted: true,
                last_value: None,
                output_name: None,
            },
            descending,
            nulls_last,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_sink<'env, 's>(
        is_sorted: &'env mut bool,
        last_value: &'env mut Option<AnyValue<'static>>,
        output_name: &'env mut Option<PlSmallStr>,
        descending: Option<bool>,
        nulls_last: Option<bool>,
        scope: &'s TaskScope<'s, 'env>,
        recv: RecvPort<'_>,
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let mut recv = recv.serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                if !*is_sorted {
                    continue;
                }

                let (df, _seq, _in_source_token, in_wait_token) = morsel.into_inner();
                drop(in_wait_token);
                if df.height() == 0 {
                    continue;
                }
                assert_eq!(df.width(), 1);
                let column = &df[0];

                if output_name.is_none() {
                    *output_name = Some(column.name().clone());
                }

                let series = column.as_materialized_series();

                let desc_values = match descending {
                    Some(d) => vec![d],
                    None => vec![false, true],
                };
                let nulls_last_values = match nulls_last {
                    Some(n) => vec![n],
                    None => vec![false, true],
                };

                let mut any_sorted = false;
                for d in desc_values {
                    for &n in &nulls_last_values {
                        let options = SortOptions {
                            descending: d,
                            nulls_last: n,
                            ..Default::default()
                        };

                        let internally_sorted = series.is_sorted(options)?;

                        let boundary_ok = if let Some(ref prev) = *last_value {
                            let first = series.get(0).unwrap();
                            match (prev.is_null(), first.is_null()) {
                                (true, true) => true,
                                (true, false) => !n,
                                (false, true) => n,
                                (false, false) => {
                                    if d {
                                        prev >= &first
                                    } else {
                                        prev <= &first
                                    }
                                },
                            }
                        } else {
                            true
                        };

                        if internally_sorted && boundary_ok {
                            any_sorted = true;
                            break;
                        }
                    }
                    if any_sorted {
                        break;
                    }
                }

                if !any_sorted {
                    *is_sorted = false;
                }

                let last = series.get(series.len() - 1).unwrap().into_static();
                *last_value = Some(last);
            }
            Ok(())
        }))
    }

    fn spawn_source<'env, 's>(
        df: &'env mut Option<DataFrame>,
        scope: &'s TaskScope<'s, 'env>,
        send: SendPort<'_>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let mut send = send.serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let morsel = Morsel::new(df.take().unwrap(), MorselSeq::new(0), SourceToken::new());
            let _ = send.send(morsel).await;
            Ok(())
        }));
    }
}

impl ComputeNode for IsSortedNode {
    fn name(&self) -> &str {
        "is_sorted"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        match &mut self.state {
            _ if send[0] == PortState::Done => {
                self.state = IsSortedState::Done;
            },
            IsSortedState::Sink {
                is_sorted,
                output_name,
                ..
            } if matches!(recv[0], PortState::Done) => {
                let name = output_name.clone().unwrap_or("is_sorted".into());
                let column = Column::new(name, &[*is_sorted]);
                let out = unsafe { DataFrame::new_unchecked(1, vec![column]) };
                self.state = IsSortedState::Source(Some(out));
            },
            IsSortedState::Source(df) if df.is_none() => {
                self.state = IsSortedState::Done;
            },
            IsSortedState::Done | IsSortedState::Sink { .. } | IsSortedState::Source(_) => {},
        }

        match &self.state {
            IsSortedState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            IsSortedState::Source(..) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            IsSortedState::Done => {
                recv[0] = PortState::Done;
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
        assert!(send_ports.len() == 1 && recv_ports.len() == 1);

        match &mut self.state {
            IsSortedState::Sink {
                is_sorted,
                last_value,
                output_name,
            } => {
                assert!(send_ports[0].is_none());
                let recv = recv_ports[0].take().unwrap();
                Self::spawn_sink(
                    is_sorted,
                    last_value,
                    output_name,
                    self.descending,
                    self.nulls_last,
                    scope,
                    recv,
                    state,
                    join_handles,
                );
            },
            IsSortedState::Source(df) => {
                assert!(recv_ports[0].is_none());
                let send = send_ports[0].take().unwrap();
                Self::spawn_source(df, scope, send, join_handles);
            },
            IsSortedState::Done => unreachable!(),
        }
    }
}
