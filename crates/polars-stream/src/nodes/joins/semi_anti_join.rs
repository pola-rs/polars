use std::sync::Arc;

use arrow::array::BooleanArray;
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_expr::groups::{Grouper, new_hash_grouper};
use polars_expr::hash_keys::HashKeys;
use polars_ops::frame::{JoinArgs, JoinType};
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::sparse_init_vec::SparseInitVec;

use crate::async_executor;
use crate::async_primitives::connector::{Receiver, Sender};
use crate::expression::StreamExpr;
use crate::nodes::compute_node_prelude::*;

async fn select_keys(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &SemiAntiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<HashKeys> {
    let mut key_columns = Vec::new();
    for selector in key_selectors {
        key_columns.push(selector.evaluate(df, state).await?.into_column());
    }
    let keys = DataFrame::new_with_broadcast_len(key_columns, df.height())?;
    Ok(HashKeys::from_df(
        &keys,
        params.random_state,
        params.nulls_equal,
        false,
    ))
}

struct SemiAntiJoinParams {
    left_is_build: bool,
    left_key_selectors: Vec<StreamExpr>,
    right_key_selectors: Vec<StreamExpr>,
    nulls_equal: bool,
    is_anti: bool,
    return_bool: bool,
    random_state: PlRandomState,
}

pub struct SemiAntiJoinNode {
    state: SemiAntiJoinState,
    params: SemiAntiJoinParams,
    grouper: Box<dyn Grouper>,
}

impl SemiAntiJoinNode {
    pub fn new(
        unique_key_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        args: JoinArgs,
        return_bool: bool,
        num_pipelines: usize,
    ) -> PolarsResult<Self> {
        let left_is_build = false;
        let is_anti = args.how == JoinType::Anti;

        let state = SemiAntiJoinState::Build(BuildState::new(num_pipelines, num_pipelines));

        Ok(Self {
            state,
            params: SemiAntiJoinParams {
                left_is_build,
                left_key_selectors,
                right_key_selectors,
                random_state: PlRandomState::default(),
                nulls_equal: args.nulls_equal,
                return_bool,
                is_anti,
            },
            grouper: new_hash_grouper(unique_key_schema),
        })
    }
}

enum SemiAntiJoinState {
    Build(BuildState),
    Probe(ProbeState),
    Done,
}

#[derive(Default)]
struct LocalBuilder {
    // The complete list of keys as seen by this builder.
    keys: Vec<HashKeys>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // key_idxs_values_per_p[p][start..stop] contains the offsets into morsels[i]
    // for partition p, where start, stop are:
    // let start = key_idxs_offsets[i * num_partitions + p];
    // let stop = key_idxs_offsets[(i + 1) * num_partitions + p];
    key_idxs_values_per_p: Vec<Vec<IdxSize>>,
    key_idxs_offsets_per_p: Vec<usize>,
}

struct BuildState {
    local_builders: Vec<LocalBuilder>,
}

impl BuildState {
    fn new(num_pipelines: usize, num_partitions: usize) -> Self {
        let local_builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                keys: Vec::new(),
                sketch_per_p: vec![CardinalitySketch::default(); num_partitions],
                key_idxs_values_per_p: vec![Vec::new(); num_partitions],
                key_idxs_offsets_per_p: vec![0; num_partitions],
            })
            .collect();
        Self { local_builders }
    }

    async fn partition_and_sink(
        mut recv: Receiver<Morsel>,
        local: &mut LocalBuilder,
        partitioner: HashPartitioner,
        params: &SemiAntiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let key_selectors = if params.left_is_build {
            &params.left_key_selectors
        } else {
            &params.right_key_selectors
        };

        while let Ok(morsel) = recv.recv().await {
            let hash_keys = select_keys(
                morsel.df(),
                key_selectors,
                params,
                &state.in_memory_exec_state,
            )
            .await?;

            hash_keys.gen_idxs_per_partition(
                &partitioner,
                &mut local.key_idxs_values_per_p,
                &mut local.sketch_per_p,
                false,
            );

            local
                .key_idxs_offsets_per_p
                .extend(local.key_idxs_values_per_p.iter().map(|vp| vp.len()));
            local.keys.push(hash_keys);
        }
        Ok(())
    }

    fn finalize(&mut self, grouper: &dyn Grouper) -> ProbeState {
        // To reduce maximum memory usage we want to drop the original keys
        // as soon as they're processed, so we move into Arcs. The drops might
        // also be expensive, so instead of directly dropping we put that on
        // a work queue.
        let keys_per_local_builder = self
            .local_builders
            .iter_mut()
            .map(|b| Arc::new(core::mem::take(&mut b.keys)))
            .collect_vec();
        let (key_drop_q_send, key_drop_q_recv) =
            async_channel::bounded(keys_per_local_builder.len());
        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let groupers: SparseInitVec<Box<dyn Grouper>> =
            SparseInitVec::with_capacity(num_partitions);

        async_executor::task_scope(|s| {
            // Wrap in outer Arc to move to each thread, performing the
            // expensive clone on that thread.
            let arc_keys_per_local_builder = Arc::new(keys_per_local_builder);
            let mut join_handles = Vec::new();
            for p in 0..num_partitions {
                let arc_keys_per_local_builder = Arc::clone(&arc_keys_per_local_builder);
                let key_drop_q_send = key_drop_q_send.clone();
                let key_drop_q_recv = key_drop_q_recv.clone();
                let groupers = &groupers;
                join_handles.push(s.spawn_task(TaskPriority::High, async move {
                    // Extract from outer arc and drop outer arc.
                    let keys_per_local_builder = Arc::unwrap_or_clone(arc_keys_per_local_builder);

                    // Compute cardinality estimate.
                    let mut sketch = CardinalitySketch::new();
                    for l in local_builders {
                        sketch.combine(&l.sketch_per_p[p]);
                    }

                    // Allocate hash table.
                    let mut p_grouper = grouper.new_empty();
                    p_grouper.reserve(sketch.estimate() * 5 / 4);

                    // Build.
                    let mut skip_drop_attempt = false;
                    for (l, l_keys) in local_builders.iter().zip(keys_per_local_builder) {
                        // Try to help with dropping the processed keys.
                        if !skip_drop_attempt {
                            drop(key_drop_q_recv.try_recv());
                        }

                        for (i, keys) in l_keys.iter().enumerate() {
                            unsafe {
                                let p_key_idxs_start =
                                    l.key_idxs_offsets_per_p[i * num_partitions + p];
                                let p_key_idxs_stop =
                                    l.key_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_key_idxs =
                                    &l.key_idxs_values_per_p[p][p_key_idxs_start..p_key_idxs_stop];
                                p_grouper.insert_keys_subset(keys, p_key_idxs, None);
                            }
                        }

                        if let Some(l) = Arc::into_inner(l_keys) {
                            // If we're the last thread to process this set of keys we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            drop(key_drop_q_send.try_send(l));
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }

                    // We're done, help others out by doing drops.
                    drop(key_drop_q_send); // So we don't deadlock trying to receive from ourselves.
                    while let Ok(l_keys) = key_drop_q_recv.recv().await {
                        drop(l_keys);
                    }

                    groupers.try_set(p, p_grouper).ok().unwrap();
                }));
            }

            // Drop outer arc after spawning each thread so the inner arcs
            // can get dropped as soon as they're processed. We also have to
            // drop the drop queue sender so we don't deadlock waiting for it
            // to end.
            drop(arc_keys_per_local_builder);
            drop(key_drop_q_send);

            polars_io::pl_async::get_runtime().block_on(async move {
                for handle in join_handles {
                    handle.await;
                }
            });
        });

        ProbeState {
            grouper_per_partition: groupers.try_assume_init().ok().unwrap(),
        }
    }
}

struct ProbeState {
    grouper_per_partition: Vec<Box<dyn Grouper>>,
}

impl ProbeState {
    /// Returns the max morsel sequence sent.
    async fn partition_and_probe(
        mut recv: Receiver<Morsel>,
        mut send: Sender<Morsel>,
        partitions: &[Box<dyn Grouper>],
        partitioner: HashPartitioner,
        params: &SemiAntiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let mut probe_match = Vec::new();
        let key_selectors = if params.left_is_build {
            &params.right_key_selectors
        } else {
            &params.left_key_selectors
        };

        while let Ok(morsel) = recv.recv().await {
            let (df, in_seq, src_token, wait_token) = morsel.into_inner();
            if df.height() == 0 {
                continue;
            }

            let hash_keys =
                select_keys(&df, key_selectors, params, &state.in_memory_exec_state).await?;

            unsafe {
                let out_df = if params.return_bool {
                    let mut builder = BitmapBuilder::with_capacity(df.height());
                    partitions[0].contains_key_partitioned_groupers(
                        partitions,
                        &hash_keys,
                        &partitioner,
                        params.is_anti,
                        &mut builder,
                    );
                    let mut arr = BooleanArray::from(builder.freeze());
                    if !params.nulls_equal {
                        arr.set_validity(hash_keys.validity().cloned());
                    }
                    let s = BooleanChunked::with_chunk(df[0].name().clone(), arr).into_series();
                    DataFrame::new(vec![Column::from(s)])?
                } else {
                    probe_match.clear();
                    partitions[0].probe_partitioned_groupers(
                        partitions,
                        &hash_keys,
                        &partitioner,
                        params.is_anti,
                        &mut probe_match,
                    );
                    if probe_match.is_empty() {
                        continue;
                    }
                    df.take_slice_unchecked(&probe_match)
                };

                let mut morsel = Morsel::new(out_df, in_seq, src_token.clone());
                if let Some(token) = wait_token {
                    morsel.set_consume_token(token);
                }
                if send.send(morsel).await.is_err() {
                    return Ok(());
                }
            }
        }

        Ok(())
    }
}

impl ComputeNode for SemiAntiJoinNode {
    fn name(&self) -> &str {
        match (self.params.return_bool, self.params.is_anti) {
            (false, false) => "semi-join",
            (false, true) => "anti-join",
            (true, false) => "is-in",
            (true, true) => "is-not-in",
        }
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done {
            self.state = SemiAntiJoinState::Done;
        }

        let build_idx = if self.params.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;

        // If we are building and the build input is done, transition to probing.
        if let SemiAntiJoinState::Build(build_state) = &mut self.state {
            if recv[build_idx] == PortState::Done {
                let probe_state = build_state.finalize(&*self.grouper);
                self.state = SemiAntiJoinState::Probe(probe_state);
            }
        }

        // If we are probing and the probe input is done, we're done.
        if let SemiAntiJoinState::Probe(_) = &mut self.state {
            if recv[probe_idx] == PortState::Done {
                self.state = SemiAntiJoinState::Done;
            }
        }

        match &mut self.state {
            SemiAntiJoinState::Build(_) => {
                send[0] = PortState::Blocked;
                if recv[build_idx] != PortState::Done {
                    recv[build_idx] = PortState::Ready;
                }
                if recv[probe_idx] != PortState::Done {
                    recv[probe_idx] = PortState::Blocked;
                }
            },
            SemiAntiJoinState::Probe(_) => {
                if recv[probe_idx] != PortState::Done {
                    core::mem::swap(&mut send[0], &mut recv[probe_idx]);
                } else {
                    send[0] = PortState::Done;
                }
                recv[build_idx] = PortState::Done;
            },
            SemiAntiJoinState::Done => {
                send[0] = PortState::Done;
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, SemiAntiJoinState::Build { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);

        let build_idx = if self.params.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;

        match &mut self.state {
            SemiAntiJoinState::Build(build_state) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let receivers = recv_ports[build_idx].take().unwrap().parallel();

                let partitioner = HashPartitioner::new(state.num_pipelines, 0);
                for (local_builder, recv) in build_state.local_builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::partition_and_sink(
                            recv,
                            local_builder,
                            partitioner.clone(),
                            &self.params,
                            state,
                        ),
                    ));
                }
            },
            SemiAntiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                let receivers = recv_ports[probe_idx].take().unwrap().parallel();

                let partitioner = HashPartitioner::new(state.num_pipelines, 0);
                for (recv, send) in receivers.into_iter().zip(senders) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        ProbeState::partition_and_probe(
                            recv,
                            send,
                            &probe_state.grouper_per_partition,
                            partitioner.clone(),
                            &self.params,
                            state,
                        ),
                    ));
                }
            },
            SemiAntiJoinState::Done => unreachable!(),
        }
    }
}
