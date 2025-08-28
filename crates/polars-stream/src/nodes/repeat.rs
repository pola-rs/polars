use std::sync::Arc;

use polars_core::schema::Schema;

use super::compute_node_prelude::*;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{SourceToken, get_ideal_morsel_size};
use crate::nodes::in_memory_sink::InMemorySinkNode;
pub enum RepeatNode {
    GatheringParams {
        value: InMemorySinkNode,
        repeats: InMemorySinkNode,
    },
    Repeating {
        value: DataFrame,
        seq: MorselSeq,
        repeats_left: usize,
    },
}

impl RepeatNode {
    pub fn new(value_schema: Arc<Schema>, repeats_schema: Arc<Schema>) -> Self {
        assert!(value_schema.len() == 1);
        assert!(repeats_schema.len() == 1);
        Self::GatheringParams {
            value: InMemorySinkNode::new(value_schema),
            repeats: InMemorySinkNode::new(repeats_schema),
        }
    }
}

impl ComputeNode for RepeatNode {
    fn name(&self) -> &str {
        "repeat"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        if recv[0] == PortState::Done && recv[1] == PortState::Done {
            if let Self::GatheringParams { value, repeats } = self {
                let repeats = repeats.get_output()?.unwrap();
                let repeats_item = repeats.get_columns()[0].get(0)?;
                let repeats_left = repeats_item.extract::<usize>().unwrap();

                let value = value.get_output()?.unwrap();
                let seq = MorselSeq::default();
                *self = Self::Repeating {
                    value,
                    seq,
                    repeats_left,
                };
            }
        }

        match self {
            Self::GatheringParams { value, repeats } => {
                value.update_state(&mut recv[0..1], &mut [], state)?;
                repeats.update_state(&mut recv[1..2], &mut [], state)?;
                send[0] = PortState::Blocked;
            },
            Self::Repeating { repeats_left, .. } => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = if *repeats_left > 0 {
                    PortState::Ready
                } else {
                    PortState::Done
                };
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
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);
        match self {
            Self::GatheringParams { value, repeats } => {
                assert!(send_ports[0].is_none());
                if recv_ports[0].is_some() {
                    value.spawn(scope, &mut recv_ports[0..1], &mut [], state, join_handles);
                }
                if recv_ports[1].is_some() {
                    repeats.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
            },
            Self::Repeating {
                value,
                seq,
                repeats_left,
            } => {
                assert!(recv_ports[0].is_none());
                assert!(recv_ports[1].is_none());

                let mut send = send_ports[0].take().unwrap().serial();

                let ideal_morsel_count = (*repeats_left / get_ideal_morsel_size()).max(1);
                let morsel_count = ideal_morsel_count.next_multiple_of(state.num_pipelines);
                let morsel_size = repeats_left.div_ceil(morsel_count).max(1);

                join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                    let source_token = SourceToken::new();

                    let wait_group = WaitGroup::default();
                    while *repeats_left > 0 && !source_token.stop_requested() {
                        let height = morsel_size.min(*repeats_left);
                        let df = value.new_from_index(0, height);
                        let mut morsel = Morsel::new(df, *seq, source_token.clone());
                        morsel.set_consume_token(wait_group.token());

                        *seq = seq.successor();
                        *repeats_left -= height;

                        if send.send(morsel).await.is_err() {
                            break;
                        }
                        wait_group.wait().await;
                    }

                    Ok(())
                }));
            },
        }
    }
}
