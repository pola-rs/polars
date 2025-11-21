use std::sync::Arc;

use arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::schema::Schema;
use polars_ops::frame::{JoinArgs, MaintainOrderJoin};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use crate::morsel::get_ideal_morsel_size;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_sink::InMemorySinkNode;

pub struct CrossJoinNode {
    left_is_build: bool,
    left_input_schema: Arc<Schema>,
    right_input_schema: Arc<Schema>,
    right_rename: Vec<Option<PlSmallStr>>,
    state: CrossJoinState,
}

impl CrossJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        args: &JoinArgs,
    ) -> Self {
        let left_is_build = match args.maintain_order {
            MaintainOrderJoin::None => true, // TODO: size estimation.
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => false,
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => true,
        };
        let build_input_schema = if left_is_build {
            &left_input_schema
        } else {
            &right_input_schema
        };
        let sink_node = InMemorySinkNode::new(build_input_schema.clone());
        let right_rename = right_input_schema
            .iter_names()
            .map(|rname| {
                if left_input_schema.contains(rname) {
                    Some(format_pl_smallstr!("{}{}", rname, args.suffix()))
                } else {
                    None
                }
            })
            .collect();

        Self {
            left_is_build,
            left_input_schema,
            right_input_schema,
            right_rename,
            state: CrossJoinState::Build(sink_node),
        }
    }
}

enum CrossJoinState {
    Build(InMemorySinkNode),
    Probe(DataFrame),
    Done,
}

impl ComputeNode for CrossJoinNode {
    fn name(&self) -> &str {
        "cross-join"
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        true
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        let build_idx = if self.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;

        // Are we done?
        if send[0] == PortState::Done || recv[probe_idx] == PortState::Done {
            self.state = CrossJoinState::Done;
        }

        // Transition to build?
        if recv[build_idx] == PortState::Done {
            if let CrossJoinState::Build(sink_node) = &mut self.state {
                let df = sink_node.get_output()?.unwrap();
                if df.height() > 0 {
                    self.state = CrossJoinState::Probe(df);
                } else {
                    self.state = CrossJoinState::Done;
                }
            }
        }

        match &self.state {
            CrossJoinState::Build(_) => {
                recv[build_idx] = PortState::Ready;
                recv[probe_idx] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            CrossJoinState::Probe(_) => {
                recv[build_idx] = PortState::Done;
                core::mem::swap(&mut recv[probe_idx], &mut send[0]);
            },
            CrossJoinState::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
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
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);
        let build_idx = if self.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;
        match &mut self.state {
            CrossJoinState::Build(sink_node) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                sink_node.spawn(
                    scope,
                    &mut recv_ports[build_idx..build_idx + 1],
                    &mut [],
                    state,
                    join_handles,
                );
            },
            CrossJoinState::Probe(build_df) => {
                assert!(recv_ports[build_idx].is_none());
                let receivers = recv_ports[probe_idx].take().unwrap().parallel();
                let senders = send_ports[0].take().unwrap().parallel();
                let ideal_morsel_size = get_ideal_morsel_size();

                for (mut recv, mut send) in receivers.into_iter().zip(senders) {
                    let left_is_build = self.left_is_build;
                    let left_input_schema = self.left_input_schema.clone();
                    let right_input_schema = self.right_input_schema.clone();
                    let right_rename = &self.right_rename;
                    let build_df = &*build_df;
                    join_handles.push(
                        scope.spawn_task(TaskPriority::High, async move {
                            let mut build_repeater = DataFrameBuilder::new(left_input_schema);
                            let mut probe_repeater = DataFrameBuilder::new(right_input_schema);
                            if !left_is_build {
                                core::mem::swap(&mut build_repeater, &mut probe_repeater);
                            }
                            let mut cached_build_df_repeated = DataFrame::empty();

                            while let Ok(morsel) = recv.recv().await {
                                let combine =
                                    |build_join_df: DataFrame, probe_join_df: DataFrame| unsafe {
                                        let (mut left_join_df, mut right_join_df);
                                        left_join_df = build_join_df;
                                        right_join_df = probe_join_df;
                                        if !left_is_build {
                                            core::mem::swap(&mut left_join_df, &mut right_join_df);
                                        }

                                        for (col, opt_rename) in right_join_df
                                            .get_columns_mut()
                                            .iter_mut()
                                            .zip(right_rename)
                                        {
                                            if let Some(rename) = opt_rename {
                                                col.rename(rename.clone());
                                            }
                                        }

                                        left_join_df
                                            .hstack_mut_unchecked(right_join_df.get_columns());
                                        Morsel::new(
                                            left_join_df,
                                            morsel.seq(),
                                            morsel.source_token().clone(),
                                        )
                                    };

                                let probe_df = morsel.df();
                                if build_df.height() >= ideal_morsel_size {
                                    for probe_offset in 0..probe_df.height() {
                                        let mut build_offset = 0;
                                        while build_offset < build_df.height() {
                                            let height = (build_df.height() - build_offset)
                                                .min(ideal_morsel_size);
                                            let build_join_df =
                                                build_df.slice(build_offset as i64, height);
                                            let probe_join_df =
                                                probe_df.new_from_index(probe_offset, height);
                                            let combined = combine(build_join_df, probe_join_df);
                                            if send.send(combined).await.is_err() {
                                                return Ok(());
                                            }
                                            build_offset += height;
                                        }
                                    }
                                } else {
                                    let max_build_repeats = ideal_morsel_size / build_df.height();
                                    let mut probe_offset = 0;
                                    while probe_offset < probe_df.height() {
                                        let build_repeats = (probe_df.height() - probe_offset)
                                            .min(max_build_repeats);
                                        let build_height = build_repeats * build_df.height();
                                        if build_height > cached_build_df_repeated.height() {
                                            build_repeater.subslice_extend_repeated(
                                                build_df,
                                                0,
                                                build_df.height(),
                                                build_repeats,
                                                ShareStrategy::Never,
                                            );
                                            cached_build_df_repeated =
                                                build_repeater.freeze_reset();
                                        }
                                        let build_join_df =
                                            cached_build_df_repeated.slice(0, build_height);

                                        probe_repeater.subslice_extend_each_repeated(
                                            probe_df,
                                            probe_offset,
                                            build_repeats,
                                            build_df.height(),
                                            ShareStrategy::Always,
                                        );
                                        let probe_join_df = probe_repeater.freeze_reset();

                                        let combined = combine(build_join_df, probe_join_df);
                                        if send.send(combined).await.is_err() {
                                            return Ok(());
                                        }

                                        probe_offset += build_repeats;
                                    }
                                }
                            }
                            Ok(())
                        }),
                    );
                }
            },
            CrossJoinState::Done => unreachable!(),
        }
    }
}
