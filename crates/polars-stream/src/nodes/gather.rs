use std::sync::Arc;

use polars_core::prelude::*;
use polars_ops::series::convert_and_bound_index;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;

pub struct GatherNode {
    state: GatherState,
    null_on_oob: bool,
}

enum GatherState {
    Sink(InMemorySinkNode),
    Gather(DataFrame),
    Done,
}

impl GatherNode {
    pub fn new(input_schema: Arc<Schema>, null_on_oob: bool) -> Self {
        Self {
            state: GatherState::Sink(InMemorySinkNode::new(input_schema)),
            null_on_oob,
        }
    }
}

impl ComputeNode for GatherNode {
    fn name(&self) -> &str {
        "gather"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, or there are no more indices,
        // transition to being done.
        if send[0] == PortState::Done || recv[1] == PortState::Done {
            self.state = GatherState::Done;
        }

        // If the payload input is done, transition to gathering.
        if recv[0] == PortState::Done {
            if let GatherState::Sink(sink_node) = &mut self.state {
                let mut df = sink_node.get_output()?.unwrap();
                df.rechunk_mut_par();
                self.state = GatherState::Gather(df);
            }
        }

        match &mut self.state {
            GatherState::Sink(sink_node) => {
                sink_node.update_state(&mut recv[0..1], &mut [], state)?;
                recv[1] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            GatherState::Gather(_) => {
                recv[0] = PortState::Done;
                recv[1..2].swap_with_slice(send);
            },
            GatherState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, GatherState::Sink(_))
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

        match &mut self.state {
            GatherState::Sink(sink_node) => {
                assert!(recv_ports[1].is_none());
                sink_node.spawn(scope, &mut recv_ports[0..1], &mut [], state, join_handles)
            },
            GatherState::Gather(target) => {
                assert!(recv_ports[0].is_none());
                let receivers = recv_ports[1].take().unwrap().parallel();
                let senders = send_ports[0].take().unwrap().parallel();

                for (mut recv, mut send) in receivers.into_iter().zip(senders) {
                    let null_on_oob = self.null_on_oob;
                    let target = &*target;
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            let morsel = morsel.try_map(|idx_df| {
                                assert!(idx_df.width() == 1);
                                if idx_df.height() == 0 {
                                    return Ok(target.clear());
                                }

                                match &idx_df.columns()[0] {
                                    Column::Series(idx_s) => {
                                        let idx_ca = convert_and_bound_index(
                                            idx_s,
                                            target.height(),
                                            null_on_oob,
                                        )?;
                                        target.take(&idx_ca)
                                    },
                                    Column::Scalar(idx_c) => {
                                        let idx_s = idx_c.as_single_value_series();
                                        let idx_ca = convert_and_bound_index(
                                            &idx_s,
                                            target.height(),
                                            null_on_oob,
                                        )?;
                                        match idx_ca.get(0) {
                                            Some(idx) => {
                                                Ok(target.new_from_index(idx as usize, idx_c.len()))
                                            },
                                            None => Ok(DataFrame::full_null(
                                                target.schema(),
                                                idx_c.len(),
                                            )),
                                        }
                                    },
                                }
                            })?;

                            if send.send(morsel).await.is_err() {
                                break;
                            }
                        }

                        Ok(())
                    }));
                }
            },
            GatherState::Done => unreachable!(),
        }
    }
}
