use std::sync::Arc;

use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_plan::dsl::v2::{PluginV2, PluginV2State, PluginV2Flags};
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;

#[derive(Debug, Clone, Copy)]
enum Action {
    Insert,
    Finalize,
    Done,
}

pub struct StatefulUdfNode {
    name: PlSmallStr,
    action: Action,
    buffer: Vec<(Series, PluginV2State, MorselSeq)>,
    input_schema: Arc<Schema>,
    udf: Arc<PluginV2>,
    output_name: PlSmallStr,
}

impl StatefulUdfNode {
    pub fn new(
        udf: Arc<PluginV2>,
        input_schema: Arc<Schema>,
        output_name: PlSmallStr,
    ) -> PolarsResult<Self> {
        let name = udf.name().into();
        Ok(Self {
            name,
            action: Action::Insert,
            buffer: Vec::new(),
            input_schema,
            udf,
            output_name,
        })
    }

    pub async fn task(
        output_name: PlSmallStr,
        (buffer, state, state_seq): &mut (Series, PluginV2State, MorselSeq),
        rx: Option<PortReceiver>,
        mut tx: Option<PortSender>,
    ) -> PolarsResult<()> {
        let wait_group = WaitGroup::default();
        match rx {
            None if buffer.is_empty() => {},
            None => {
                if let Some(mut tx) = tx {
                    let new_buffer = Series::new_empty(buffer.name().clone(), buffer.dtype());
                    let df = std::mem::replace(buffer, new_buffer).into_frame();
                    let mut morsel = Morsel::new(df, state_seq.successor(), SourceToken::new());
                    morsel.set_consume_token(wait_group.token());
                    if tx.send(morsel).await.is_err() {
                        return Ok(());
                    }
                    wait_group.wait().await;
                }
            },
            Some(mut rx) => {
                while let Ok(morsel) = rx.recv().await {
                    let (df, seq, source_token, _) = morsel.into_inner();
                    *state_seq = seq;
                    let inputs = &df
                        .take_columns()
                        .into_iter()
                        .map(|c| c.take_materialized_series())
                        .collect::<Vec<Series>>();
                    match (state.insert(&inputs)?, tx.as_mut()) {
                        (None, _) => continue,
                        (Some(out), None) => {
                            _ = buffer.append_owned(out.with_name(output_name.clone()))?
                        },
                        (Some(out), Some(tx)) => {
                            let df = out.with_name(output_name.clone()).into_frame();
                            let mut morsel = Morsel::new(df, seq, source_token);
                            morsel.set_consume_token(wait_group.token());
                            if tx.send(morsel).await.is_err() {
                                break;
                            }
                            wait_group.wait().await;
                        },
                    }
                }
            },
        }

        Ok(())
    }
}

impl ComputeNode for StatefulUdfNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if matches!(self.action, Action::Done) {
            recv[0] = P::Done;
            send[0] = P::Done;
            return Ok(());
        }

        let flags = self.udf.flags();

        if matches!(self.action, Action::Insert) && self.buffer.is_empty() {
            let udf_state = self.udf.clone().initialize(&self.input_schema)?;
            let field = self.udf.to_field(&self.input_schema)?;
            self.buffer = Vec::with_capacity(state.num_pipelines);
            if flags.allows_concurrent_evaluation() {
                for _ in 1..state.num_pipelines {
                    self.buffer.push((
                        Series::new_empty(self.output_name.clone(), &field.dtype),
                        udf_state.new_empty()?,
                        MorselSeq::default(),
                    ));
                }
            }
            self.buffer.push((
                Series::new_empty(self.output_name.clone(), &field.dtype),
                udf_state,
                MorselSeq::default(),
            ));
        }

        use PortState as P;
        self.action = match (recv[0], send[0]) {
            (_, P::Done) => Action::Done,
            (P::Done, _) if self.buffer.is_empty() => Action::Done,
            (P::Done, _) if self.buffer.iter().all(|b| b.0.is_empty()) => {
                if flags.needs_finalize() {
                    Action::Finalize
                } else {
                    Action::Done
                }
            },

            (rx, _) => {
                if rx == P::Ready || self.buffer.iter().any(|b| !b.0.is_empty()) {
                    send[0] = P::Ready;
                }

                Action::Insert
            },
        };

        match self.action {
            Action::Insert => {
                if !flags.contains(PluginV2Flags::INSERT_HAS_OUTPUT) {
                    send[0] = P::Blocked
                }
            },
            Action::Finalize => send[0] = P::Ready,
            Action::Done => {
                recv[0] = P::Done;
                send[0] = P::Done;
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);

        let flags = self.udf.flags();

        match &mut self.action {
            Action::Done => unreachable!(),
            Action::Insert => {
                if flags.allows_concurrent_evaluation() {
                    let receiver: Vec<Option<PortReceiver>> = recv_ports[0].take().map_or_else(
                        || (0..state.num_pipelines).map(|_| None).collect::<Vec<_>>(),
                        |rxs| rxs.parallel().into_iter().map(Some).collect::<Vec<_>>(),
                    );
                    let senders: Vec<Option<PortSender>> = send_ports[0].take().map_or_else(
                        || (0..state.num_pipelines).map(|_| None).collect::<Vec<_>>(),
                        |txs| txs.parallel().into_iter().map(Some).collect::<Vec<_>>(),
                    );

                    for ((send, recv), udf_state) in senders
                        .into_iter()
                        .zip(receiver)
                        .zip(self.buffer.iter_mut())
                    {
                        join_handles.push(scope.spawn_task(
                            TaskPriority::High,
                            Self::task(self.output_name.clone(), udf_state, recv, send),
                        ));
                    }
                } else {
                    let recv = recv_ports[0].take().map(|tx| tx.serial());
                    let send = send_ports[0].take().map(|tx| tx.serial());

                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        Self::task(self.output_name.clone(), &mut self.buffer[0], recv, send),
                    ));
                }
            },
            Action::Finalize => {
                assert!(!self.buffer.is_empty());
                assert!(flags.contains(PluginV2Flags::NEEDS_FINALIZE));
                assert!(self.buffer.len() == 1 || flags.contains(PluginV2Flags::STATES_COMBINABLE));

                let mut send = send_ports[0].take().unwrap().serial();

                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    let (buffer, mut udf_state, mut seq) = self.buffer.pop().unwrap();
                    assert!(buffer.is_empty());
                    for (_, other, other_seq) in std::mem::take(&mut self.buffer).iter() {
                        udf_state.combine(other)?;
                        seq = seq.max(*other_seq);
                    }
                    match udf_state.finalize()? {
                        None => {},
                        Some(series) => {
                            let df = series.with_name(self.output_name.clone()).into_frame();
                            _ = send
                                .send(Morsel::new(df, seq.successor(), SourceToken::new()))
                                .await;
                        },
                    }
                    Ok(())
                }));
            },
        }
    }
}
