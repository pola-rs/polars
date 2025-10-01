use polars_core::prelude::{Column, DataType, StringChunked};
use polars_error::PolarsResult;
use polars_time::prelude::string::Pattern;
use polars_time::prelude::string::infer::{
    infer_pattern_date_single, infer_pattern_datetime_single, infer_pattern_time_single,
};

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

enum State {
    Nulls,
    Infer(Vec<Morsel>),
    Inferred(Pattern),
}

pub struct TemporalInferParse {
    dtype: DataType,
    state: State,
}

impl TemporalInferParse {
    pub fn new(dtype: DataType) -> Self {
        assert!(matches!(
            dtype,
            DataType::Time | DataType::Datetime(_, _) | DataType::Date
        ));
        Self {
            dtype,
            state: State::Nulls,
        }
    }
}

fn find_pattern(ca: &StringChunked, dtype: &DataType) -> Option<Pattern> {
    match dtype {
        DataType::Date => ca
            .iter()
            .filter_map(|v| v)
            .find_map(infer_pattern_date_single),
        DataType::Time => ca
            .iter()
            .filter_map(|v| v)
            .find_map(infer_pattern_time_single),
        DataType::Datetime(_, _) => ca
            .iter()
            .filter_map(|v| v)
            .find_map(infer_pattern_datetime_single),
        _ => unreachable!(),
    }
}

fn propagate_ca(ca: &StringChunked, dtype: &DataType, pattern: Pattern) -> PolarsResult<Column> {
    todo!()
}

impl ComputeNode for TemporalInferParse {
    fn name(&self) -> &str {
        "temporal_infer_parse"
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        match self.state {
            State::Nulls | State::Infer(_) => {
                let mut recv = recv_ports[0].take().unwrap().serial();
                let mut send = send_ports[0].take().unwrap().serial();

                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    loop {
                        match &mut self.state {
                            State::Nulls => {
                                while let Ok(mut m) = recv.recv().await {
                                    let df = m.df_mut();
                                    if df.height() == 0 {
                                        continue;
                                    }

                                    assert_eq!(df.width(), 1);
                                    let column = &df[0];
                                    let ca = column.str()?;

                                    if let Some(offset) = ca.first_non_null() {
                                        m.take_consume_token();
                                        self.state = State::Infer(vec![m]);
                                        break;
                                    }

                                    let out = Column::full_null(
                                        column.name().clone(),
                                        ca.len(),
                                        &self.dtype,
                                    );
                                    unsafe { df.get_columns_mut()[0] = out };
                                    df.clear_schema();
                                    if send.send(m).await.is_err() {
                                        break;
                                    }
                                }
                            },
                            State::Infer(morsels) => {
                                while let Ok(mut m) = recv.recv().await {
                                    let df = m.df_mut();
                                    if df.height() == 0 {
                                        continue;
                                    }

                                    assert_eq!(df.width(), 1);
                                    let column = &df[0];
                                    let ca = column.str()?;

                                    let Some(pattern) = find_pattern(ca, &self.dtype) else {
                                        m.take_consume_token();
                                        morsels.push(m);
                                        continue;
                                    };

                                    m.source_token().stop();
                                    for m in std::mem::take(morsels) {
                                        let df = m.df_mut();
                                        let column = &df[0];
                                        let ca = column.str()?;
                                        let out = propagate_ca(ca, &self.dtype, pattern);
                                        unsafe { df.get_columns_mut()[0] = out };
                                        df.clear_schema();
                                        if send.send(m).await.is_err() {
                                            break;
                                        }
                                    }
                                    self.state = State::Inferred(pattern);
                                }
                            },
                            State::Inferred(pattern) => {
                                while let Ok(mut m) = recv.recv().await {
                                    m.source_token().stop();

                                    let df = m.df_mut();
                                    if df.height() == 0 {
                                        continue;
                                    }

                                    assert_eq!(df.width(), 1);
                                    let column = &df[0];
                                    let ca = column.str()?;
                                    let out = propagate_ca(ca, &self.dtype, *pattern)?;
                                    unsafe { df.get_columns_mut()[0] = out };
                                    df.clear_schema();
                                    if send.send(m).await.is_err() {
                                        break;
                                    }
                                }

                                break;
                            },
                        }
                    }

                    Ok(())
                }));
            },
            State::Inferred(pattern) => {
                let mut recv = recv_ports[0].take().unwrap().parallel();
                let mut send = send_ports[0].take().unwrap().parallel();

                join_handles.extend(recv.into_iter().zip(send).map(|(mut rx, mut tx)| {
                    scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(mut m) = rx.recv().await {
                            let df = m.df_mut();
                            if df.height() == 0 {
                                continue;
                            }

                            assert_eq!(df.width(), 1);
                            let column = &df[0];
                            let ca = column.str()?;
                            let out = propagate_ca(ca, &self.dtype, pattern)?;
                            unsafe { df.get_columns_mut()[0] = out };
                            df.clear_schema();
                            if tx.send(m).await.is_err() {
                                break;
                            }
                        }
                        Ok(())
                    })
                }));
            },
        }
    }
}
