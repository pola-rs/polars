use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use arrow::array::builder::ShareStrategy;
use crossbeam_queue::ArrayQueue;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::{config, POOL};
use polars_expr::hash_keys::HashKeys;
use polars_expr::idx_table::{new_idx_table, IdxTable};
use polars_io::pl_async::get_runtime;
use polars_ops::frame::{JoinArgs, JoinType, MaintainOrderJoin};
use polars_ops::series::coalesce_columns;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::sparse_init_vec::SparseInitVec;
use polars_utils::{format_pl_smallstr, IdxSize};
use rayon::prelude::*;

use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::wait_group::WaitGroup;
use crate::expression::StreamExpr;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_source::InMemorySourceNode;

static SAMPLE_LIMIT: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_JOIN_SAMPLE_LIMIT")
        .map(|limit| limit.parse().unwrap())
        .unwrap_or(10_000_000)
});

// If one side is this much bigger than the other side we'll always use the
// smaller side as the build side without checking cardinalities.
const LOPSIDED_SAMPLE_FACTOR: usize = 10;

/// A payload selector contains for each column whether that column should be
/// included in the payload, and if yes with what name.
fn compute_payload_selector(
    this: &Schema,
    other: &Schema,
    this_key_schema: &Schema,
    is_left: bool,
    args: &JoinArgs,
) -> PolarsResult<Vec<Option<PlSmallStr>>> {
    let should_coalesce = args.should_coalesce();

    this.iter_names()
        .enumerate()
        .map(|(i, c)| {
            let selector = if should_coalesce && this_key_schema.contains(c) {
                if is_left != (args.how == JoinType::Right) {
                    Some(c.clone())
                } else if args.how == JoinType::Full {
                    // We must keep the right-hand side keycols around for
                    // coalescing.
                    Some(format_pl_smallstr!("__POLARS_COALESCE_KEYCOL{i}"))
                } else {
                    None
                }
            } else if !other.contains(c) || is_left {
                Some(c.clone())
            } else {
                let suffixed = format_pl_smallstr!("{}{}", c, args.suffix());
                if other.contains(&suffixed) {
                    polars_bail!(Duplicate: "column with name '{suffixed}' already exists\n\n\
                    You may want to try:\n\
                    - renaming the column prior to joining\n\
                    - using the `suffix` parameter to specify a suffix different to the default one ('_right')")
                }
                Some(suffixed)
            };
            Ok(selector)
        })
        .collect()
}

/// Fixes names and does coalescing of columns post-join.
fn postprocess_join(df: DataFrame, params: &EquiJoinParams) -> DataFrame {
    if params.args.how == JoinType::Full && params.args.should_coalesce() {
        // TODO: don't do string-based column lookups for each dataframe, pre-compute coalesce indices.
        let mut key_idx = 0;
        df.get_columns()
            .iter()
            .filter_map(|c| {
                if let Some((key_name, _)) = params.left_key_schema.get_at_index(key_idx) {
                    if c.name() == key_name {
                        let other = df
                            .column(&format_pl_smallstr!("__POLARS_COALESCE_KEYCOL{key_idx}"))
                            .unwrap();
                        key_idx += 1;
                        return Some(coalesce_columns(&[c.clone(), other.clone()]).unwrap());
                    }
                }

                if c.name().starts_with("__POLARS_COALESCE_KEYCOL") {
                    return None;
                }

                Some(c.clone())
            })
            .collect()
    } else {
        df
    }
}

fn select_schema(schema: &Schema, selector: &[Option<PlSmallStr>]) -> Schema {
    schema
        .iter_fields()
        .zip(selector)
        .filter_map(|(f, name)| Some(f.with_name(name.clone()?)))
        .collect()
}

async fn select_keys(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &EquiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<HashKeys> {
    let mut key_columns = Vec::new();
    for (i, selector) in key_selectors.iter().enumerate() {
        // We use key columns entirely by position, and allow duplicate names,
        // so just assign arbitrary unique names.
        let unique_name = format_pl_smallstr!("__POLARS_KEYCOL_{i}");
        let s = selector.evaluate(df, state).await?;
        key_columns.push(s.into_column().with_name(unique_name));
    }
    let keys = DataFrame::new_with_broadcast_len(key_columns, df.height())?;
    Ok(HashKeys::from_df(
        &keys,
        params.random_state.clone(),
        params.args.nulls_equal,
        true,
    ))
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

fn estimate_cardinality(
    morsels: &[Morsel],
    key_selectors: &[StreamExpr],
    params: &EquiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<usize> {
    // TODO: parallelize.
    let mut sketch = CardinalitySketch::new();
    for morsel in morsels {
        let hash_keys =
            get_runtime().block_on(select_keys(morsel.df(), key_selectors, params, state))?;
        hash_keys.sketch_cardinality(&mut sketch);
    }
    Ok(sketch.estimate())
}

struct BufferedStream {
    morsels: ArrayQueue<Morsel>,
    post_buffer_offset: MorselSeq,
}

impl BufferedStream {
    pub fn new(morsels: Vec<Morsel>, start_offset: MorselSeq) -> Self {
        // Relabel so we can insert into parallel streams later.
        let mut seq = start_offset;
        let queue = ArrayQueue::new(morsels.len().max(1));
        for mut morsel in morsels {
            morsel.set_seq(seq);
            queue.push(morsel).unwrap();
            seq = seq.successor();
        }

        Self {
            morsels: queue,
            post_buffer_offset: seq,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.morsels.is_empty()
    }

    #[expect(clippy::needless_lifetimes)]
    pub fn reinsert<'s, 'env>(
        &'s self,
        num_pipelines: usize,
        recv_port: Option<RecvPort<'_>>,
        scope: &'s TaskScope<'s, 'env>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) -> Option<Vec<Receiver<Morsel>>> {
        let receivers = if let Some(p) = recv_port {
            p.parallel().into_iter().map(Some).collect_vec()
        } else {
            (0..num_pipelines).map(|_| None).collect_vec()
        };

        let source_token = SourceToken::new();
        let mut out = Vec::new();
        for orig_recv in receivers {
            let (mut new_send, new_recv) = connector();
            out.push(new_recv);
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                // Act like an InMemorySource node until cached morsels are consumed.
                let wait_group = WaitGroup::default();
                loop {
                    let Some(mut morsel) = self.morsels.pop() else {
                        break;
                    };
                    morsel.replace_source_token(source_token.clone());
                    morsel.set_consume_token(wait_group.token());
                    if new_send.send(morsel).await.is_err() {
                        return Ok(());
                    }
                    wait_group.wait().await;
                    // TODO: Unfortunately we can't actually stop here without
                    // re-buffering morsels from the stream that comes after.
                    // if source_token.stop_requested() {
                    //     break;
                    // }
                }

                if let Some(mut recv) = orig_recv {
                    while let Ok(mut morsel) = recv.recv().await {
                        if source_token.stop_requested() {
                            morsel.source_token().stop();
                        }
                        morsel.set_seq(morsel.seq().offset_by(self.post_buffer_offset));
                        if new_send.send(morsel).await.is_err() {
                            break;
                        }
                    }
                }
                Ok(())
            }));
        }
        Some(out)
    }
}

impl Default for BufferedStream {
    fn default() -> Self {
        Self {
            morsels: ArrayQueue::new(1),
            post_buffer_offset: MorselSeq::default(),
        }
    }
}

impl Drop for BufferedStream {
    fn drop(&mut self) {
        POOL.install(|| {
            // Parallel drop as the state might be quite big.
            (0..self.morsels.len())
                .into_par_iter()
                .for_each(|_| drop(self.morsels.pop()));
        })
    }
}

#[derive(Default)]
struct SampleState {
    left: Vec<Morsel>,
    left_len: usize,
    right: Vec<Morsel>,
    right_len: usize,
}

impl SampleState {
    async fn sink(
        mut recv: Receiver<Morsel>,
        morsels: &mut Vec<Morsel>,
        len: &mut usize,
        this_final_len: Arc<AtomicUsize>,
        other_final_len: Arc<AtomicUsize>,
    ) -> PolarsResult<()> {
        while let Ok(mut morsel) = recv.recv().await {
            *len += morsel.df().height();
            if *len >= *SAMPLE_LIMIT
                || *len
                    >= other_final_len
                        .load(Ordering::Relaxed)
                        .saturating_mul(LOPSIDED_SAMPLE_FACTOR)
            {
                morsel.source_token().stop();
            }

            drop(morsel.take_consume_token());
            morsels.push(morsel);
        }
        this_final_len.store(*len, Ordering::Relaxed);
        Ok(())
    }

    fn try_transition_to_build(
        &mut self,
        recv: &[PortState],
        num_pipelines: usize,
        params: &mut EquiJoinParams,
        table: &mut Option<Box<dyn IdxTable>>,
    ) -> PolarsResult<Option<BuildState>> {
        let left_saturated = self.left_len >= *SAMPLE_LIMIT;
        let right_saturated = self.right_len >= *SAMPLE_LIMIT;
        let left_done = recv[0] == PortState::Done || left_saturated;
        let right_done = recv[1] == PortState::Done || right_saturated;
        #[expect(clippy::nonminimal_bool)]
        let stop_sampling = (left_done && right_done)
            || (left_done && self.right_len >= LOPSIDED_SAMPLE_FACTOR * self.left_len)
            || (right_done && self.left_len >= LOPSIDED_SAMPLE_FACTOR * self.right_len);
        if !stop_sampling {
            return Ok(None);
        }

        if config::verbose() {
            eprintln!(
                "choosing equi-join build side, sample lengths are: {} vs. {}",
                self.left_len, self.right_len
            );
        }

        let estimate_cardinalities = || {
            let execution_state = ExecutionState::new();
            let left_cardinality = estimate_cardinality(
                &self.left,
                &params.left_key_selectors,
                params,
                &execution_state,
            )?;
            let right_cardinality = estimate_cardinality(
                &self.right,
                &params.right_key_selectors,
                params,
                &execution_state,
            )?;
            let norm_left_factor = self.left_len.min(*SAMPLE_LIMIT) as f64 / self.left_len as f64;
            let norm_right_factor =
                self.right_len.min(*SAMPLE_LIMIT) as f64 / self.right_len as f64;
            let norm_left_cardinality = (left_cardinality as f64 * norm_left_factor) as usize;
            let norm_right_cardinality = (right_cardinality as f64 * norm_right_factor) as usize;
            if config::verbose() {
                eprintln!("estimated cardinalities are: {norm_left_cardinality} vs. {norm_right_cardinality}");
            }
            PolarsResult::Ok((norm_left_cardinality, norm_right_cardinality))
        };

        let left_is_build = match (left_saturated, right_saturated) {
            (false, false) => {
                if self.left_len * LOPSIDED_SAMPLE_FACTOR < self.right_len
                    || self.left_len > self.right_len * LOPSIDED_SAMPLE_FACTOR
                {
                    // Don't bother estimating cardinality, just choose smaller as it's highly
                    // imbalanced.
                    self.left_len < self.right_len
                } else {
                    let (lc, rc) = estimate_cardinalities()?;
                    // Let's assume for now that per element building a
                    // table is 3x more expensive than a probe, with
                    // unique keys getting an additional 3x factor for
                    // having to update the hash table in addition to the probe.
                    let left_build_cost = self.left_len * 3 + 3 * lc;
                    let left_probe_cost = self.left_len;
                    let right_build_cost = self.right_len * 3 + 3 * rc;
                    let right_probe_cost = self.right_len;
                    left_build_cost + right_probe_cost < left_probe_cost + right_build_cost
                }
            },

            // Choose the unsaturated side, the saturated side could be
            // arbitrarily big.
            (false, true) => true,
            (true, false) => false,

            // Estimate cardinality and choose smaller.
            (true, true) => {
                let (lc, rc) = estimate_cardinalities()?;
                lc < rc
            },
        };

        if config::verbose() {
            eprintln!(
                "build side chosen: {}",
                if left_is_build { "left" } else { "right" }
            );
        }

        // Transition to building state.
        params.left_is_build = Some(left_is_build);
        *table = Some(if left_is_build {
            new_idx_table(params.left_key_schema.clone())
        } else {
            new_idx_table(params.right_key_schema.clone())
        });

        let mut sampled_build_morsels =
            BufferedStream::new(core::mem::take(&mut self.left), MorselSeq::default());
        let mut sampled_probe_morsels =
            BufferedStream::new(core::mem::take(&mut self.right), MorselSeq::default());
        if !left_is_build {
            core::mem::swap(&mut sampled_build_morsels, &mut sampled_probe_morsels);
        }

        let partitioner = HashPartitioner::new(num_pipelines, 0);
        let mut build_state = BuildState {
            local_builders: (0..num_pipelines)
                .map(|_| LocalBuilder::default())
                .collect(),
            sampled_probe_morsels,
        };

        // Simulate the sample build morsels flowing into the build side.
        if !sampled_build_morsels.is_empty() {
            let state = ExecutionState::new();
            crate::async_executor::task_scope(|scope| {
                let mut join_handles = Vec::new();
                let receivers = sampled_build_morsels
                    .reinsert(num_pipelines, None, scope, &mut join_handles)
                    .unwrap();

                for (local_builder, recv) in build_state.local_builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::partition_and_sink(
                            recv,
                            local_builder,
                            partitioner.clone(),
                            params,
                            &state,
                        ),
                    ));
                }

                polars_io::pl_async::get_runtime().block_on(async move {
                    for handle in join_handles {
                        handle.await?;
                    }
                    PolarsResult::Ok(())
                })
            })?;
        }

        Ok(Some(build_state))
    }
}

#[derive(Default)]
struct LocalBuilder {
    // The complete list of morsels and their computed hashes seen by this builder.
    morsels: Vec<(MorselSeq, DataFrame, HashKeys)>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // morsel_idxs_values_per_p[p][start..stop] contains the offsets into morsels[i]
    // for partition p, where start, stop are:
    // let start = morsel_idxs_offsets[i * num_partitions + p];
    // let stop = morsel_idxs_offsets[(i + 1) * num_partitions + p];
    morsel_idxs_values_per_p: Vec<Vec<IdxSize>>,
    morsel_idxs_offsets_per_p: Vec<usize>,
}

#[derive(Default)]
struct BuildState {
    local_builders: Vec<LocalBuilder>,
    sampled_probe_morsels: BufferedStream,
}

impl BuildState {
    async fn partition_and_sink(
        mut recv: Receiver<Morsel>,
        local: &mut LocalBuilder,
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
        state: &ExecutionState,
    ) -> PolarsResult<()> {
        let track_unmatchable = params.emit_unmatched_build();
        local
            .sketch_per_p
            .resize_with(partitioner.num_partitions(), Default::default);
        local
            .morsel_idxs_values_per_p
            .resize_with(partitioner.num_partitions(), Default::default);

        if local.morsel_idxs_offsets_per_p.is_empty() {
            local
                .morsel_idxs_offsets_per_p
                .resize(partitioner.num_partitions(), 0);
        }

        let (key_selectors, payload_selector);
        if params.left_is_build.unwrap() {
            payload_selector = &params.left_payload_select;
            key_selectors = &params.left_key_selectors;
        } else {
            payload_selector = &params.right_payload_select;
            key_selectors = &params.right_key_selectors;
        };

        while let Ok(morsel) = recv.recv().await {
            // Compute hashed keys and payload. We must rechunk the payload for
            // later gathers.
            let hash_keys = select_keys(morsel.df(), key_selectors, params, state).await?;
            let mut payload = select_payload(morsel.df().clone(), payload_selector);
            payload.rechunk_mut();

            hash_keys.gen_idxs_per_partition(
                &partitioner,
                &mut local.morsel_idxs_values_per_p,
                &mut local.sketch_per_p,
                track_unmatchable,
            );

            local
                .morsel_idxs_offsets_per_p
                .extend(local.morsel_idxs_values_per_p.iter().map(|vp| vp.len()));
            local.morsels.push((morsel.seq(), payload, hash_keys));
        }
        Ok(())
    }

    fn finalize(&mut self, params: &EquiJoinParams, table: &dyn IdxTable) -> ProbeState {
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        // To reduce maximum memory usage we want to drop the morsels
        // as soon as they're processed, so we move into Arcs. The drops might
        // also be expensive, so instead of directly dropping we put that on
        // a work queue.
        let morsels_per_local_builder = self
            .local_builders
            .iter_mut()
            .map(|b| Arc::new(core::mem::take(&mut b.morsels)))
            .collect_vec();
        let (morsel_drop_q_send, morsel_drop_q_recv) = crossbeam_channel::bounded(morsels_per_local_builder.len());
        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let probe_tables: SparseInitVec<ProbeTable> = SparseInitVec::with_capacity(num_partitions);

        POOL.scope(|s| {
            // Wrap in outer Arc to move to each thread, performing the
            // expensive clone on that thread.
            let arc_morsels_per_local_builder = Arc::new(morsels_per_local_builder);
            for p in 0..num_partitions {
                let arc_morsels_per_local_builder = Arc::clone(&arc_morsels_per_local_builder);
                let morsel_drop_q_send = morsel_drop_q_send.clone();
                let morsel_drop_q_recv = morsel_drop_q_recv.clone();
                let probe_tables = &probe_tables;
                s.spawn(move |_| {
                    // Extract from outer arc and drop outer arc.
                    let morsels_per_local_builder =
                        Arc::unwrap_or_clone(arc_morsels_per_local_builder);

                    // Compute cardinality estimate and total amount of
                    // payload for this partition.
                    let mut sketch = CardinalitySketch::new();
                    let mut payload_rows = 0;
                    for l in local_builders {
                        sketch.combine(&l.sketch_per_p[p]);
                        let offsets_len = l.morsel_idxs_offsets_per_p.len();
                        payload_rows +=
                            l.morsel_idxs_offsets_per_p[offsets_len - num_partitions + p];
                    }

                    // Allocate hash table and payload builder.
                    let mut p_table = table.new_empty();
                    p_table.reserve(sketch.estimate() * 5 / 4);
                    let mut p_payload = DataFrameBuilder::new(payload_schema.clone());
                    p_payload.reserve(payload_rows);

                    // Build.
                    let mut skip_drop_attempt = false;
                    for (l, l_morsels) in local_builders.iter().zip(morsels_per_local_builder) {
                        // Try to help with dropping the processed morsels.
                        if !skip_drop_attempt {
                            drop(morsel_drop_q_recv.try_recv());
                        }

                        for (i, morsel) in l_morsels.iter().enumerate() {
                            let (_mseq, payload, keys) = morsel;
                            unsafe {
                                let p_morsel_idxs_start =
                                    l.morsel_idxs_offsets_per_p[i * num_partitions + p];
                                let p_morsel_idxs_stop =
                                    l.morsel_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_morsel_idxs = &l.morsel_idxs_values_per_p[p]
                                    [p_morsel_idxs_start..p_morsel_idxs_stop];
                                p_table.insert_keys_subset(keys, p_morsel_idxs, track_unmatchable);
                                p_payload.gather_extend(
                                    payload,
                                    p_morsel_idxs,
                                    ShareStrategy::Never,
                                );
                            }
                        }
                        
                        if let Some(l) = Arc::into_inner(l_morsels) {
                            // If we're the last thread to process this set of morsels we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            morsel_drop_q_send.send(l).unwrap();
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }
                    
                    // We're done, help others out by doing drops.
                    drop(morsel_drop_q_send); // So we don't deadlock.
                    while let Ok(l_morsels) = morsel_drop_q_recv.recv() {
                        drop(l_morsels);
                    }

                    probe_tables.try_set(p, ProbeTable {
                        hash_table: p_table,
                        payload: p_payload.freeze(),
                    }).ok().unwrap();
                });
            }

            // Drop outer arc after spawning each thread so the inner arcs
            // can get dropped as soon as they're processed. We also have to
            // drop the drop queue sender so we don't deadlock waiting for it
            // to end.
            drop(arc_morsels_per_local_builder);
            drop(morsel_drop_q_send);
        });

        ProbeState {
            table_per_partition: probe_tables.try_assume_init().ok().unwrap(),
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
        }
    }
}

struct ProbeTable {
    hash_table: Box<dyn IdxTable>,
    payload: DataFrame,
}

struct ProbeState {
    table_per_partition: Vec<ProbeTable>,
    max_seq_sent: MorselSeq,
    sampled_probe_morsels: BufferedStream,
}

impl ProbeState {
    /// Returns the max morsel sequence sent.
    async fn partition_and_probe(
        mut recv: Receiver<Morsel>,
        mut send: Sender<Morsel>,
        partitions: &[ProbeTable],
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
        state: &ExecutionState,
    ) -> PolarsResult<MorselSeq> {
        // TODO: shuffle after partitioning and keep probe tables thread-local.
        let mut partition_idxs = vec![Vec::new(); partitioner.num_partitions()];
        let mut probe_partitions = Vec::new();
        let mut materialized_idxsize_range = Vec::new();
        let mut table_match = Vec::new();
        let mut probe_match = Vec::new();
        let mut max_seq = MorselSeq::default();

        let probe_limit = get_ideal_morsel_size() as IdxSize;
        let mark_matches = params.emit_unmatched_build();
        let emit_unmatched = params.emit_unmatched_probe();

        let (key_selectors, payload_selector, build_payload_schema, probe_payload_schema);
        if params.left_is_build.unwrap() {
            key_selectors = &params.right_key_selectors;
            payload_selector = &params.right_payload_select;
            build_payload_schema = &params.left_payload_schema;
            probe_payload_schema = &params.right_payload_schema;
        } else {
            key_selectors = &params.left_key_selectors;
            payload_selector = &params.left_payload_select;
            build_payload_schema = &params.right_payload_schema;
            probe_payload_schema = &params.left_payload_schema;
        };

        let mut build_out = DataFrameBuilder::new(build_payload_schema.clone());
        let mut probe_out = DataFrameBuilder::new(probe_payload_schema.clone());

        // A simple estimate used to size reserves.
        let mut selectivity_estimate = 1.0;

        while let Ok(morsel) = recv.recv().await {
            // Compute hashed keys and payload.
            let (df, seq, src_token, wait_token) = morsel.into_inner();
            max_seq = seq;

            let df_height = df.height();
            if df_height == 0 {
                continue;
            }

            let hash_keys = select_keys(&df, key_selectors, params, state).await?;
            let mut payload = select_payload(df, payload_selector);
            let mut payload_rechunked = false; // We don't eagerly rechunk because there might be no matches.
            let mut total_matches = 0;

            // Use selectivity estimate to reserve for morsel builders.
            let max_match_per_key_est = selectivity_estimate as usize + 16;
            let out_est_size = ((selectivity_estimate * 1.2 * df_height as f64) as usize)
                .min(probe_limit as usize);
            build_out.reserve(out_est_size + max_match_per_key_est);

            unsafe {
                let new_morsel = |build: &mut DataFrameBuilder, probe: &mut DataFrameBuilder| {
                    let mut build_df = build.freeze_reset();
                    let mut probe_df = probe.freeze_reset();
                    let out_df = if params.left_is_build.unwrap() {
                        build_df.hstack_mut_unchecked(probe_df.get_columns());
                        build_df
                    } else {
                        probe_df.hstack_mut_unchecked(build_df.get_columns());
                        probe_df
                    };
                    let out_df = postprocess_join(out_df, params);
                    Morsel::new(out_df, seq, src_token.clone())
                };

                if params.preserve_order_probe {
                    // To preserve the order we can't do bulk probes per partition and must follow
                    // the order of the probe morsel. We can still group probes that are
                    // consecutively on the same partition.
                    hash_keys.gen_partitions(&partitioner, &mut probe_partitions, emit_unmatched);
                    let mut probe_group_start = 0;
                    while probe_group_start < probe_partitions.len() {
                        let p_idx = probe_partitions[probe_group_start];
                        let mut probe_group_end = probe_group_start + 1;
                        while probe_partitions.get(probe_group_end) == Some(&p_idx) {
                            probe_group_end += 1;
                        }
                        let Some(p) = partitions.get(p_idx as usize) else {
                            probe_group_start = probe_group_end;
                            continue;
                        };

                        materialized_idxsize_range.extend(
                            materialized_idxsize_range.len() as IdxSize..probe_group_end as IdxSize,
                        );

                        while probe_group_start < probe_group_end {
                            let matches_before_limit = probe_limit - probe_match.len() as IdxSize;
                            table_match.clear();
                            probe_group_start += p.hash_table.probe_subset(
                                &hash_keys,
                                &materialized_idxsize_range[probe_group_start..probe_group_end],
                                &mut table_match,
                                &mut probe_match,
                                mark_matches,
                                emit_unmatched,
                                matches_before_limit,
                            ) as usize;

                            if emit_unmatched {
                                build_out.opt_gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            } else {
                                build_out.gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            };

                            if probe_match.len() >= probe_limit as usize
                                || probe_group_start == probe_partitions.len()
                            {
                                if !payload_rechunked {
                                    payload.rechunk_mut();
                                    payload_rechunked = true;
                                }
                                probe_out.gather_extend(
                                    &payload,
                                    &probe_match,
                                    ShareStrategy::Always,
                                );
                                probe_match.clear();
                                let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                                if send.send(out_morsel).await.is_err() {
                                    return Ok(max_seq);
                                }
                                if probe_group_end != probe_partitions.len() {
                                    // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                    // matches and just do a large reserve.
                                    build_out.reserve(probe_limit as usize + max_match_per_key_est);
                                }
                            }
                        }
                    }
                } else {
                    // Partition and probe the tables.
                    for p in partition_idxs.iter_mut() {
                        p.clear();
                    }
                    hash_keys.gen_idxs_per_partition(
                        &partitioner,
                        &mut partition_idxs,
                        &mut [],
                        emit_unmatched,
                    );

                    for (p, idxs_in_p) in partitions.iter().zip(&partition_idxs) {
                        let mut offset = 0;
                        while offset < idxs_in_p.len() {
                            let matches_before_limit = probe_limit - probe_match.len() as IdxSize;
                            table_match.clear();
                            offset += p.hash_table.probe_subset(
                                &hash_keys,
                                &idxs_in_p[offset..],
                                &mut table_match,
                                &mut probe_match,
                                mark_matches,
                                emit_unmatched,
                                matches_before_limit,
                            ) as usize;

                            if table_match.is_empty() {
                                continue;
                            }
                            total_matches += table_match.len();

                            if emit_unmatched {
                                build_out.opt_gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            } else {
                                build_out.gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            };

                            if probe_match.len() >= probe_limit as usize {
                                if !payload_rechunked {
                                    payload.rechunk_mut();
                                    payload_rechunked = true;
                                }
                                probe_out.gather_extend(
                                    &payload,
                                    &probe_match,
                                    ShareStrategy::Always,
                                );
                                probe_match.clear();
                                let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                                if send.send(out_morsel).await.is_err() {
                                    return Ok(max_seq);
                                }
                                // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                // matches and just do a large reserve.
                                build_out.reserve(probe_limit as usize + max_match_per_key_est);
                            }
                        }
                    }

                    if !probe_match.is_empty() {
                        if !payload_rechunked {
                            payload.rechunk_mut();
                        }
                        probe_out.gather_extend(&payload, &probe_match, ShareStrategy::Always);
                        probe_match.clear();
                        let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                        if send.send(out_morsel).await.is_err() {
                            return Ok(max_seq);
                        }
                    }
                }
            }

            drop(wait_token);

            // Move selectivity estimate a bit towards latest value.
            selectivity_estimate =
                0.8 * selectivity_estimate + 0.2 * (total_matches as f64 / df_height as f64);
        }

        Ok(max_seq)
    }

    fn ordered_unmatched(
        &mut self,
        _partitioner: &HashPartitioner,
        _params: &EquiJoinParams,
    ) -> DataFrame {
        todo!()
    }
}

impl Drop for ProbeState {
    fn drop(&mut self) {
        POOL.install(|| {
            // Parallel drop as the state might be quite big.
            self.table_per_partition.par_drain(..).for_each(drop);
        })
    }
}

struct EmitUnmatchedState {
    partitions: Vec<ProbeTable>,
    active_partition_idx: usize,
    offset_in_active_p: usize,
    morsel_seq: MorselSeq,
}

impl EmitUnmatchedState {
    async fn emit_unmatched(
        &mut self,
        mut send: Sender<Morsel>,
        params: &EquiJoinParams,
        num_pipelines: usize,
    ) -> PolarsResult<()> {
        let total_len: usize = self
            .partitions
            .iter()
            .map(|p| p.hash_table.num_keys() as usize)
            .sum();
        let ideal_morsel_count = (total_len / get_ideal_morsel_size()).max(1);
        let morsel_count = ideal_morsel_count.next_multiple_of(num_pipelines);
        let morsel_size = total_len.div_ceil(morsel_count).max(1);

        let wait_group = WaitGroup::default();
        let source_token = SourceToken::new();
        let mut unmarked_idxs = Vec::new();
        while let Some(p) = self.partitions.get(self.active_partition_idx) {
            loop {
                // Generate a chunk of unmarked key indices.
                self.offset_in_active_p += p.hash_table.unmarked_keys(
                    &mut unmarked_idxs,
                    self.offset_in_active_p as IdxSize,
                    morsel_size as IdxSize,
                ) as usize;
                if unmarked_idxs.is_empty() {
                    break;
                }

                // Gather and create full-null counterpart.
                let out_df = unsafe {
                    let mut build_df = p.payload.take_slice_unchecked_impl(&unmarked_idxs, false);
                    let len = build_df.height();
                    if params.left_is_build.unwrap() {
                        let probe_df = DataFrame::full_null(&params.right_payload_schema, len);
                        build_df.hstack_mut_unchecked(probe_df.get_columns());
                        build_df
                    } else {
                        let mut probe_df = DataFrame::full_null(&params.left_payload_schema, len);
                        probe_df.hstack_mut_unchecked(build_df.get_columns());
                        probe_df
                    }
                };
                let out_df = postprocess_join(out_df, params);

                // Send and wait until consume token is consumed.
                let mut morsel = Morsel::new(out_df, self.morsel_seq, source_token.clone());
                self.morsel_seq = self.morsel_seq.successor();
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
    Sample(SampleState),
    Build(BuildState),
    Probe(ProbeState),
    EmitUnmatchedBuild(EmitUnmatchedState),
    EmitUnmatchedBuildInOrder(InMemorySourceNode),
    Done,
}

struct EquiJoinParams {
    left_is_build: Option<bool>,
    preserve_order_build: bool,
    preserve_order_probe: bool,
    left_key_schema: Arc<Schema>,
    left_key_selectors: Vec<StreamExpr>,
    right_key_schema: Arc<Schema>,
    right_key_selectors: Vec<StreamExpr>,
    left_payload_select: Vec<Option<PlSmallStr>>,
    right_payload_select: Vec<Option<PlSmallStr>>,
    left_payload_schema: Arc<Schema>,
    right_payload_schema: Arc<Schema>,
    args: JoinArgs,
    random_state: PlRandomState,
}

impl EquiJoinParams {
    /// Should we emit unmatched rows from the build side?
    fn emit_unmatched_build(&self) -> bool {
        if self.left_is_build.unwrap() {
            self.args.how == JoinType::Left || self.args.how == JoinType::Full
        } else {
            self.args.how == JoinType::Right || self.args.how == JoinType::Full
        }
    }

    /// Should we emit unmatched rows from the probe side?
    fn emit_unmatched_probe(&self) -> bool {
        if self.left_is_build.unwrap() {
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
    table: Option<Box<dyn IdxTable>>,
}

impl EquiJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        left_key_schema: Arc<Schema>,
        right_key_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        let left_is_build = match args.maintain_order {
            MaintainOrderJoin::None => {
                if *SAMPLE_LIMIT == 0 {
                    Some(true)
                } else {
                    None
                }
            },
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => Some(false),
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => Some(true),
        };

        let table = left_is_build.map(|lib| {
            if lib {
                new_idx_table(left_key_schema.clone())
            } else {
                new_idx_table(right_key_schema.clone())
            }
        });

        let preserve_order_probe = args.maintain_order != MaintainOrderJoin::None;
        let preserve_order_build = matches!(
            args.maintain_order,
            MaintainOrderJoin::LeftRight | MaintainOrderJoin::RightLeft
        );

        let left_payload_select = compute_payload_selector(
            &left_input_schema,
            &right_input_schema,
            &left_key_schema,
            true,
            &args,
        )?;
        let right_payload_select = compute_payload_selector(
            &right_input_schema,
            &left_input_schema,
            &right_key_schema,
            false,
            &args,
        )?;

        let state = if left_is_build.is_some() {
            EquiJoinState::Build(BuildState::default())
        } else {
            EquiJoinState::Sample(SampleState::default())
        };

        let left_payload_schema = Arc::new(select_schema(&left_input_schema, &left_payload_select));
        let right_payload_schema =
            Arc::new(select_schema(&right_input_schema, &right_payload_select));
        Ok(Self {
            state,
            num_pipelines: 0,
            params: EquiJoinParams {
                left_is_build,
                preserve_order_build,
                preserve_order_probe,
                left_key_schema,
                left_key_selectors,
                right_key_schema,
                right_key_selectors,
                left_payload_select,
                right_payload_select,
                left_payload_schema,
                right_payload_schema,
                args,
                random_state: PlRandomState::new(),
            },
            table,
        })
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

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done {
            self.state = EquiJoinState::Done;
        }

        // If we are sampling and both sides are done/filled, transition to building.
        if let EquiJoinState::Sample(sample_state) = &mut self.state {
            if let Some(build_state) = sample_state.try_transition_to_build(
                recv,
                self.num_pipelines,
                &mut self.params,
                &mut self.table,
            )? {
                self.state = EquiJoinState::Build(build_state);
            }
        }

        let build_idx = if self.params.left_is_build == Some(true) {
            0
        } else {
            1
        };
        let probe_idx = 1 - build_idx;

        // If we are building and the build input is done, transition to probing.
        if let EquiJoinState::Build(build_state) = &mut self.state {
            if recv[build_idx] == PortState::Done {
                self.state = EquiJoinState::Probe(
                    build_state.finalize(&self.params, self.table.as_deref().unwrap()),
                );
            }
        }

        // If we are probing and the probe input is done, emit unmatched if
        // necessary, otherwise we're done.
        if let EquiJoinState::Probe(probe_state) = &mut self.state {
            let samples_consumed = probe_state.sampled_probe_morsels.is_empty();
            if samples_consumed && recv[probe_idx] == PortState::Done {
                if self.params.emit_unmatched_build() {
                    if self.params.preserve_order_build {
                        let partitioner = HashPartitioner::new(self.num_pipelines, 0);
                        let unmatched = probe_state.ordered_unmatched(&partitioner, &self.params);
                        let mut src = InMemorySourceNode::new(
                            Arc::new(unmatched),
                            probe_state.max_seq_sent.successor(),
                        );
                        src.initialize(self.num_pipelines);
                        self.state = EquiJoinState::EmitUnmatchedBuildInOrder(src);
                    } else {
                        self.state = EquiJoinState::EmitUnmatchedBuild(EmitUnmatchedState {
                            partitions: core::mem::take(&mut probe_state.table_per_partition),
                            active_partition_idx: 0,
                            offset_in_active_p: 0,
                            morsel_seq: probe_state.max_seq_sent.successor(),
                        });
                    }
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
            EquiJoinState::Sample(sample_state) => {
                send[0] = PortState::Blocked;
                if recv[0] != PortState::Done {
                    recv[0] = if sample_state.left_len < *SAMPLE_LIMIT {
                        PortState::Ready
                    } else {
                        PortState::Blocked
                    };
                }
                if recv[1] != PortState::Done {
                    recv[1] = if sample_state.right_len < *SAMPLE_LIMIT {
                        PortState::Ready
                    } else {
                        PortState::Blocked
                    };
                }
            },
            EquiJoinState::Build(_) => {
                send[0] = PortState::Blocked;
                if recv[build_idx] != PortState::Done {
                    recv[build_idx] = PortState::Ready;
                }
                if recv[probe_idx] != PortState::Done {
                    recv[probe_idx] = PortState::Blocked;
                }
            },
            EquiJoinState::Probe(probe_state) => {
                if recv[probe_idx] != PortState::Done {
                    core::mem::swap(&mut send[0], &mut recv[probe_idx]);
                } else {
                    let samples_consumed = probe_state.sampled_probe_morsels.is_empty();
                    send[0] = if samples_consumed {
                        PortState::Done
                    } else {
                        PortState::Ready
                    };
                }
                recv[build_idx] = PortState::Done;
            },
            EquiJoinState::EmitUnmatchedBuild(_) => {
                send[0] = PortState::Ready;
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Done;
            },
            EquiJoinState::EmitUnmatchedBuildInOrder(src_node) => {
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Done;
                src_node.update_state(&mut [], &mut send[0..1])?;
                if send[0] == PortState::Done {
                    self.state = EquiJoinState::Done;
                }
            },
            EquiJoinState::Done => {
                send[0] = PortState::Done;
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(
            self.state,
            EquiJoinState::Sample { .. } | EquiJoinState::Build { .. }
        )
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

        let build_idx = if self.params.left_is_build == Some(true) {
            0
        } else {
            1
        };
        let probe_idx = 1 - build_idx;

        match &mut self.state {
            EquiJoinState::Sample(sample_state) => {
                assert!(send_ports[0].is_none());
                let left_final_len = Arc::new(AtomicUsize::new(if recv_ports[0].is_none() {
                    sample_state.left_len
                } else {
                    usize::MAX
                }));
                let right_final_len = Arc::new(AtomicUsize::new(if recv_ports[1].is_none() {
                    sample_state.right_len
                } else {
                    usize::MAX
                }));

                if let Some(left_recv) = recv_ports[0].take() {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        SampleState::sink(
                            left_recv.serial(),
                            &mut sample_state.left,
                            &mut sample_state.left_len,
                            left_final_len.clone(),
                            right_final_len.clone(),
                        ),
                    ));
                }
                if let Some(right_recv) = recv_ports[1].take() {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        SampleState::sink(
                            right_recv.serial(),
                            &mut sample_state.right,
                            &mut sample_state.right_len,
                            right_final_len,
                            left_final_len,
                        ),
                    ));
                }
            },
            EquiJoinState::Build(build_state) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let receivers = recv_ports[build_idx].take().unwrap().parallel();

                build_state
                    .local_builders
                    .resize_with(self.num_pipelines, Default::default);
                let partitioner = HashPartitioner::new(self.num_pipelines, 0);
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
            EquiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                let receivers = probe_state
                    .sampled_probe_morsels
                    .reinsert(
                        self.num_pipelines,
                        recv_ports[probe_idx].take(),
                        scope,
                        join_handles,
                    )
                    .unwrap();

                let partitioner = HashPartitioner::new(self.num_pipelines, 0);
                let probe_tasks = receivers
                    .into_iter()
                    .zip(senders)
                    .map(|(recv, send)| {
                        scope.spawn_task(
                            TaskPriority::High,
                            ProbeState::partition_and_probe(
                                recv,
                                send,
                                &probe_state.table_per_partition,
                                partitioner.clone(),
                                &self.params,
                                state,
                            ),
                        )
                    })
                    .collect_vec();

                let max_seq_sent = &mut probe_state.max_seq_sent;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    for probe_task in probe_tasks {
                        *max_seq_sent = (*max_seq_sent).max(probe_task.await?);
                    }
                    Ok(())
                }));
            },
            EquiJoinState::EmitUnmatchedBuild(emit_state) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let send = send_ports[0].take().unwrap().serial();
                join_handles.push(scope.spawn_task(
                    TaskPriority::Low,
                    emit_state.emit_unmatched(send, &self.params, self.num_pipelines),
                ));
            },
            EquiJoinState::EmitUnmatchedBuildInOrder(src_node) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                src_node.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            EquiJoinState::Done => unreachable!(),
        }
    }
}
