use std::sync::Arc;

use polars_core::prelude::{PlHashSet, PlRandomState};
use polars_core::schema::Schema;
use polars_expr::hash_keys::HashKeys;
use polars_ops::frame::{JoinArgs, JoinType};
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::format_pl_smallstr;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;

use crate::async_primitives::connector::Receiver;
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

#[derive(Default)]
struct BuildPartition {
    hash_keys: Vec<HashKeys>,
    frames: Vec<DataFrame>,
    sketch: Option<CardinalitySketch>,
}

struct BuildState {
    partitions_per_worker: Vec<Vec<BuildPartition>>,
}

impl BuildState {
    async fn partition_and_sink(
        mut recv: Receiver<Morsel>,
        partitions: &mut Vec<BuildPartition>,
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
    ) -> PolarsResult<()> {
        let mut partition_idxs = vec![Vec::new(); partitioner.num_partitions()];
        partitions.resize_with(partitioner.num_partitions(), BuildPartition::default);
        
        let mut sketches = vec![CardinalitySketch::default(); partitioner.num_partitions()];

        while let Ok(morsel) = recv.recv().await {
            let df = morsel.into_df();
            let hash_keys = HashKeys::from_df(&df, params.random_state.clone(), params.args.join_nulls, true);
            let selector = if params.left_is_build {
                &params.left_payload_select
            } else {
                &params.right_payload_select
            };

            // We must rechunk the payload for later chunked gathers.
            let mut payload = select_payload(df, selector);
            payload.rechunk_mut();
            
            unsafe {
                hash_keys.gen_partition_idxs(&partitioner, &mut partition_idxs, &mut sketches);
                for (p, idxs_in_p) in partitions.iter_mut().zip(&partition_idxs) {
                    p.hash_keys.push(hash_keys.gather(idxs_in_p));
                    p.frames.push(payload.take_slice_unchecked_impl(idxs_in_p, false));
                }
            }
        }
        
        for (p, sketch) in sketches.into_iter().enumerate() {
            partitions[p].sketch = Some(sketch);
        }
        
        Ok(())
    }
}

struct ProbeState {}

enum EquiJoinState {
    Build(BuildState),
    Probe(ProbeState),
    Done,
}

struct EquiJoinParams {
    left_is_build: bool,
    left_payload_select: Vec<Option<PlSmallStr>>,
    right_payload_select: Vec<Option<PlSmallStr>>,
    args: JoinArgs,
    random_state: PlRandomState,
}

impl EquiJoinParams {
    /// Should we emit unmatched rows from the build side?
    fn emit_unmatched_build(&self) -> bool {
        if self.left_is_build {
            self.args.how == JoinType::Left || self.args.how == JoinType::Full
        } else {
            self.args.how == JoinType::Right || self.args.how == JoinType::Full
        }
    }

    /// Should we emit unmatched rows from the probe side?
    fn emit_unmatched_probe(&self) -> bool {
        if self.left_is_build {
            self.args.how == JoinType::Right || self.args.how == JoinType::Full
        } else {
            self.args.how == JoinType::Left || self.args.how == JoinType::Full
        }
    }
}

pub struct EquiJoinNode {
    state: EquiJoinState,
    params: EquiJoinParams,
    num_pipelines: usize,
}

impl EquiJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        args: JoinArgs,
    ) -> Self {
        // TODO: use cardinality estimation to determine this.
        let left_is_build = args.how != JoinType::Left;

        let left_payload_select =
            compute_payload_selector(&left_input_schema, &right_input_schema, true, &args);
        let right_payload_select =
            compute_payload_selector(&right_input_schema, &left_input_schema, false, &args);
        Self {
            state: EquiJoinState::Build(BuildState {
                partitions_per_worker: Vec::new(),
            }),
            num_pipelines: 0,
            params: EquiJoinParams {
                left_is_build,
                left_payload_select,
                right_payload_select,
                args,
                random_state: PlRandomState::new(),
            }
        }
    }
}

impl ComputeNode for EquiJoinNode {
    fn name(&self) -> &str {
        "equi_join"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        let build_idx = if self.params.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;

        // If the output doesn't want any more data, or the probe side is done,
        // transition to being done.
        if send[0] == PortState::Done || recv[probe_idx] == PortState::Done {
            self.state = EquiJoinState::Done;
        }

        // If we are building and the build input is done, transition to probing.
        if let EquiJoinState::Build(build_state) = &mut self.state {
            if recv[build_idx] == PortState::Done {
                todo!()
            }
        }

        match &mut self.state {
            EquiJoinState::Build(_) => {
                recv[build_idx] = PortState::Ready;
                recv[probe_idx] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            EquiJoinState::Probe(_) => {
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Ready;
                send[0] = PortState::Ready;
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
        matches!(self.state, EquiJoinState::Build { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);

        let build_idx = if self.params.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;

        match &mut self.state {
            EquiJoinState::Build(build_state) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let receivers = recv_ports[build_idx].take().unwrap().parallel();

                build_state
                    .partitions_per_worker
                    .resize_with(self.num_pipelines, || Vec::new());
                let partitioner = HashPartitioner::new(self.num_pipelines, 0);
                for (worker_ps, recv) in build_state.partitions_per_worker.iter_mut().zip(receivers)
                {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::partition_and_sink(recv, worker_ps, partitioner.clone(), &self.params),
                    ));
                }
            },
            EquiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
                todo!()
            },
            EquiJoinState::Done => unreachable!(),
        }
    }
}
