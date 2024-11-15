use std::sync::Arc;

use polars_core::prelude::PlHashSet;
use polars_core::schema::Schema;
use polars_expr::hash_keys::HashKeys;
use polars_ops::frame::{JoinArgs, JoinType};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;

/// A payload selector contains for each column whether that column should be
/// included in the payload, and if yes with what name.
fn compute_payload_selector(
    this: &Schema,
    other: &Schema,
    is_left: bool,
    args: &JoinArgs,
) -> Vec<Option<PlSmallStr>> {
    let should_coalesce = args.should_coalesce();
    let other_col_names: PlHashSet<PlSmallStr> = other.iter_names_cloned().collect();

    this.iter_names()
        .map(|c| {
            if !other_col_names.contains(c) {
                return Some(c.clone());
            }

            if is_left {
                if should_coalesce && args.how == JoinType::Right {
                    None
                } else {
                    Some(c.clone())
                }
            } else {
                if should_coalesce {
                    if args.how == JoinType::Right {
                        Some(c.clone())
                    } else {
                        None
                    }
                } else {
                    Some(format_pl_smallstr!("{}{}", c, args.suffix()))
                }
            }
        })
        .collect()
}

fn select_payload(df: DataFrame, selector: &[Option<PlSmallStr>]) -> DataFrame {
    // Maintain height of zero-width dataframes.
    if df.width() == 0 {
        return df;
    }

    df.take_columns()
        .into_iter()
        .zip(selector)
        .filter_map(|(c, name)| Some(c.with_name(name.clone()?)))
        .collect()
}

struct BuildPartition {
    hash_keys: Vec<HashKeys>,
    frames: Vec<DataFrame>,
}

struct BuildState {
    partitions: Vec<BuildPartition>,
}

struct ProbeState {}

enum EquiJoinState {
    Build(BuildState),
    Probe(ProbeState),
    Done,
}

pub struct EquiJoinNode {
    state: EquiJoinState,
    num_pipelines: usize,
    left_is_build: bool,
    emit_unmatched_build: bool,
    emit_unmatched_probe: bool,
    left_payload_select: Vec<Option<PlSmallStr>>,
    right_payload_select: Vec<Option<PlSmallStr>>,
    args: JoinArgs,
}

impl EquiJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        args: JoinArgs,
    ) -> Self {
        // TODO: use cardinality estimation to determine this.
        let left_is_build = args.how != JoinType::Left;

        let emit_unmatched_left = args.how == JoinType::Left || args.how == JoinType::Full;
        let emit_unmatched_right = args.how == JoinType::Right || args.how == JoinType::Full;
        let emit_unmatched_build = if left_is_build { emit_unmatched_left } else { emit_unmatched_right };
        let emit_unmatched_probe = if left_is_build { emit_unmatched_right } else { emit_unmatched_left };
        let left_payload_select = compute_payload_selector(&left_input_schema, &right_input_schema, true, &args);
        let right_payload_select = compute_payload_selector(&right_input_schema, &left_input_schema, false, &args);
        Self {
            state: EquiJoinState::Build(BuildState {
                partitions: Vec::new()
            }),
            num_pipelines: 0,
            left_is_build,
            emit_unmatched_build,
            emit_unmatched_probe,
            left_payload_select,
            right_payload_select,
            args
        }
    }
}

/*
impl ComputeNode for EquiJoinNode {
    fn name(&self) -> &str {
        "in_memory_join"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self.state, EquiJoinState::Done) {
            self.state = EquiJoinState::Done;
        }

        // If the input is done, transition to being a source.
        if let EquiJoinState::Sink { left, right } = &mut self.state {
            if recv[0] == PortState::Done && recv[1] == PortState::Done {
                let left_df = left.get_output()?.unwrap();
                let right_df = right.get_output()?.unwrap();
                let mut source_node =
                    InMemorySourceNode::new(Arc::new((self.joiner)(left_df, right_df)?));
                source_node.initialize(self.num_pipelines);
                self.state = EquiJoinState::Source(source_node);
            }
        }

        match &mut self.state {
            EquiJoinState::Sink { left, right, .. } => {
                left.update_state(&mut recv[0..1], &mut [])?;
                right.update_state(&mut recv[1..2], &mut [])?;
                send[0] = PortState::Blocked;
            },
            EquiJoinState::Source(source_node) => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                source_node.update_state(&mut [], send)?;
            },
            EquiJoinState::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, EquiJoinState::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);
        match &mut self.state {
            EquiJoinState::Sink { left, right, .. } => {
                if recv_ports[0].is_some() {
                    left.spawn(scope, &mut recv_ports[0..1], &mut [], state, join_handles);
                }
                if recv_ports[1].is_some() {
                    right.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
            },
            EquiJoinState::Source(source) => {
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            EquiJoinState::Done => unreachable!(),
        }
    }
}
*/
