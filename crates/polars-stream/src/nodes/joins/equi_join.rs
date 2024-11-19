use std::sync::Arc;

use polars_core::prelude::{PlHashSet, PlRandomState};
use polars_core::schema::{Schema, SchemaExt};
use polars_core::series::IsSorted;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_expr::chunked_idx_table::{new_chunked_idx_table, ChunkedIdxTable};
use polars_expr::hash_keys::HashKeys;
use polars_ops::frame::{JoinArgs, JoinType};
use polars_ops::prelude::TakeChunked;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, IdxSize};
use rayon::prelude::*;

use crate::async_primitives::connector::{Receiver, Sender};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
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

fn select_schema(schema: &Schema, selector: &[Option<PlSmallStr>]) -> Schema {
    schema.iter_fields()
        .zip(selector)
        .filter_map(|(f, name)| Some(f.with_name(name.clone()?)))
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
            let hash_keys = HashKeys::from_df(
                &df,
                params.random_state.clone(),
                params.args.join_nulls,
                true,
            );
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
                    p.frames
                        .push(payload.take_slice_unchecked_impl(idxs_in_p, false));
                }
            }
        }

        for (p, sketch) in sketches.into_iter().enumerate() {
            partitions[p].sketch = Some(sketch);
        }

        Ok(())
    }

    fn finalize(&mut self, table: &dyn ChunkedIdxTable) -> ProbeState {
        let num_partitions = self.partitions_per_worker.len();
        let table_per_partition: Vec<_> = (0..num_partitions)
            .into_par_iter()
            .with_max_len(1)
            .map(|p| {
                // Estimate sizes and cardinality.
                let mut sketch = CardinalitySketch::new();
                let mut num_frames = 0;
                for worker in &self.partitions_per_worker {
                    sketch.combine(worker[p].sketch.as_ref().unwrap());
                    num_frames += worker[p].frames.len();
                }

                // Build table for this partition.
                let mut combined_frames = Vec::with_capacity(num_frames);
                let mut table = table.new_empty();
                table.reserve(sketch.estimate() * 5 / 4);
                for worker in &self.partitions_per_worker {
                    for (hash_keys, frame) in worker[p].hash_keys.iter().zip(&worker[p].frames) {
                        table.insert_key_chunk(hash_keys.clone());
                        combined_frames.push(frame.clone());
                    }
                }

                let df = accumulate_dataframes_vertical_unchecked(combined_frames);
                ProbeTable { table, df }
            })
            .collect();

        ProbeState {
            table_per_partition,
        }
    }
}

struct ProbeTable {
    // Important that df is not rechunked, the chunks it was inserted with
    // into the table must be preserved for chunked gathers.
    table: Box<dyn ChunkedIdxTable>,
    df: DataFrame,
}

struct ProbeState {
    table_per_partition: Vec<ProbeTable>,
}

impl ProbeState {
    // TODO: shuffle after partitioning and keep probe tables thread-local.
    async fn partition_and_probe(
        mut recv: Receiver<Morsel>,
        mut send: Sender<Morsel>,
        partitions: &[ProbeTable],
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
    ) -> PolarsResult<()> {
        let mut partition_idxs = Vec::new();
        let mut table_match = Vec::new();
        let mut probe_match = Vec::new();

        let probe_limit = get_ideal_morsel_size() as IdxSize;
        let mark_matches = params.emit_unmatched_build();
        let emit_unmatched = params.emit_unmatched_probe();

        while let Ok(morsel) = recv.recv().await {
            let (df, seq, src_token, wait_token) = morsel.into_inner();
            let hash_keys = HashKeys::from_df(
                &df,
                params.random_state.clone(),
                params.args.join_nulls,
                true,
            );
            let selector = if params.left_is_build {
                &params.right_payload_select
            } else {
                &params.left_payload_select
            };
            let payload = select_payload(df, selector);

            unsafe {
                hash_keys.gen_partition_idxs(&partitioner, &mut partition_idxs, &mut []);
                for (p, idxs_in_p) in partitions.iter().zip(&partition_idxs) {
                    let mut offset = 0;
                    while let Some(idxs_in_p_slice) = idxs_in_p.get(offset as usize..) {
                        offset += p.table.probe_subset(
                            &hash_keys,
                            idxs_in_p_slice,
                            &mut table_match,
                            &mut probe_match,
                            mark_matches,
                            emit_unmatched,
                            probe_limit,
                        );
                        let mut build_df = if emit_unmatched {
                            p.df.take_opt_chunked_unchecked(&table_match)
                        } else {
                            p.df.take_chunked_unchecked(&table_match, IsSorted::Not)
                        };
                        let mut probe_df = payload.take_slice_unchecked(&probe_match);

                        let out_df = if params.left_is_build {
                            build_df.hstack_mut_unchecked(probe_df.get_columns());
                            build_df
                        } else {
                            probe_df.hstack_mut_unchecked(build_df.get_columns());
                            probe_df
                        };

                        let out_morsel = Morsel::new(out_df, seq, src_token.clone());
                        if send.send(out_morsel).await.is_err() {
                            break;
                        }
                    }
                }
            }

            drop(wait_token);
        }

        Ok(())
    }
}

struct EmitUnmatchedState {
    partitions: Vec<ProbeTable>,
    active_partition_idx: usize,
    offset_in_active_p: usize,
}

impl EmitUnmatchedState {
    async fn emit_unmatched(
        &mut self,
        mut send: Sender<Morsel>,
        params: &EquiJoinParams,
        num_pipelines: usize,
    ) -> PolarsResult<()> {
        let total_len: usize = self.partitions.iter().map(|p| p.table.num_keys() as usize).sum();
        let ideal_morsel_count = (total_len / get_ideal_morsel_size()).max(1);
        let morsel_count = ideal_morsel_count.next_multiple_of(num_pipelines);
        let morsel_size = total_len.div_ceil(morsel_count).max(1);

        let mut morsel_seq = MorselSeq::default();
        let wait_group = WaitGroup::default();
        let source_token = SourceToken::new();
        let mut unmarked_idxs = Vec::new();
        while let Some(p) = self.partitions.get(self.active_partition_idx) {
            loop {
                p.table.unmarked_keys(&mut unmarked_idxs, self.offset_in_active_p as IdxSize, morsel_size as IdxSize);
                self.offset_in_active_p += unmarked_idxs.len();
                if unmarked_idxs.is_empty() {
                    break;
                }
                
                let out_df = unsafe {
                    let mut build_df = p.df.take_chunked_unchecked(&unmarked_idxs, IsSorted::Not);
                    let len = build_df.height();
                    if params.left_is_build {
                        let probe_df = DataFrame::full_null(&params.right_payload_schema, len);
                        build_df.hstack_mut_unchecked(probe_df.get_columns());
                        build_df
                    } else {
                        let mut probe_df = DataFrame::full_null(&params.left_payload_schema, len);
                        probe_df.hstack_mut_unchecked(build_df.get_columns());
                        probe_df
                    }
                };

                let mut morsel = Morsel::new(out_df, morsel_seq, source_token.clone());
                morsel_seq = morsel_seq.successor();
                morsel.set_consume_token(wait_group.token());
                if send.send(morsel).await.is_err() {
                    return Ok(());
                }

                wait_group.wait().await;
                if source_token.stop_requested() {
                    return Ok(());
                }
            }
                
            self.active_partition_idx += 1;
            self.offset_in_active_p = 0;
        }

        Ok(())
    }
}

enum EquiJoinState {
    Build(BuildState),
    Probe(ProbeState),
    EmitUnmatchedBuild(EmitUnmatchedState),
    Done,
}

struct EquiJoinParams {
    left_is_build: bool,
    left_payload_select: Vec<Option<PlSmallStr>>,
    right_payload_select: Vec<Option<PlSmallStr>>,
    left_payload_schema: Schema,
    right_payload_schema: Schema,
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
    table: Box<dyn ChunkedIdxTable>,
}

impl EquiJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        args: JoinArgs,
    ) -> Self {
        // TODO: use cardinality estimation to determine this.
        let left_is_build = args.how != JoinType::Left;
        let table = if left_is_build {
            new_chunked_idx_table(left_input_schema.clone())
        } else {
            new_chunked_idx_table(right_input_schema.clone())
        };

        let left_payload_select =
            compute_payload_selector(&left_input_schema, &right_input_schema, true, &args);
        let right_payload_select =
            compute_payload_selector(&right_input_schema, &left_input_schema, false, &args);
        
        let left_payload_schema = select_schema(&left_input_schema, &left_payload_select);
        let right_payload_schema = select_schema(&right_input_schema, &right_payload_select);
        Self {
            state: EquiJoinState::Build(BuildState {
                partitions_per_worker: Vec::new(),
            }),
            num_pipelines: 0,
            params: EquiJoinParams {
                left_is_build,
                left_payload_select,
                right_payload_select,
                left_payload_schema,
                right_payload_schema,
                args,
                random_state: PlRandomState::new(),
            },
            table,
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

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done {
            self.state = EquiJoinState::Done;
        }

        // If we are building and the build input is done, transition to probing.
        if let EquiJoinState::Build(build_state) = &mut self.state {
            if recv[build_idx] == PortState::Done {
                self.state = EquiJoinState::Probe(build_state.finalize(&*self.table));
            }
        }
        
        // If we are probing and the probe input is done, emit unmatched if
        // necessary, otherwise we're done.
        if let EquiJoinState::Probe(probe_state) = &mut self.state {
            if recv[probe_idx] == PortState::Done {
                if self.params.emit_unmatched_build() {
                    self.state = EquiJoinState::EmitUnmatchedBuild(EmitUnmatchedState {
                        partitions: core::mem::take(&mut probe_state.table_per_partition),
                        active_partition_idx: 0,
                        offset_in_active_p: 0,
                    });
                } else {
                    self.state = EquiJoinState::Done;
                }
            }
        }
        
        // Finally, check if we are done emitting unmatched keys.
        if let EquiJoinState::EmitUnmatchedBuild(emit_state) = &mut self.state {
            if emit_state.active_partition_idx >= emit_state.partitions.len() {
                self.state = EquiJoinState::Done;
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
            EquiJoinState::EmitUnmatchedBuild(_) => {
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Done;
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
                        BuildState::partition_and_sink(
                            recv,
                            worker_ps,
                            partitioner.clone(),
                            &self.params,
                        ),
                    ));
                }
            },
            EquiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
                let receivers = recv_ports[probe_idx].take().unwrap().parallel();
                let senders = send_ports[0].take().unwrap().parallel();

                let partitioner = HashPartitioner::new(self.num_pipelines, 0);
                for (recv, send) in receivers.into_iter().zip(senders.into_iter()) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        ProbeState::partition_and_probe(
                            recv,
                            send,
                            &probe_state.table_per_partition,
                            partitioner.clone(),
                            &self.params,
                        ),
                    ));
                }
            },
            EquiJoinState::EmitUnmatchedBuild(emit_state) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let send = send_ports[0].take().unwrap().serial();
                join_handles.push(scope.spawn_task(
                    TaskPriority::Low,
                    emit_state.emit_unmatched(send, &self.params, self.num_pipelines)
                ));
            },
            EquiJoinState::Done => unreachable!(),
        }
    }
}
