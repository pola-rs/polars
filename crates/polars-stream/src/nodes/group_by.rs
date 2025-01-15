use std::sync::Arc;

use polars_core::prelude::{IntoColumn, PlRandomState};
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::POOL;
use polars_expr::groups::Grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::reduce::GroupedReduction;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use crate::async_primitives::connector::Receiver;
use crate::expression::StreamExpr;
use crate::nodes::in_memory_source::InMemorySourceNode;
use crate::GROUP_BY_MIN_ROWS_PER_PARTITION;

struct LocalGroupBySinkState {
    grouper: Box<dyn Grouper>,
    grouped_reductions: Vec<Box<dyn GroupedReduction>>,
}

impl LocalGroupBySinkState {
    fn into_df(self, output_schema: &Schema) -> PolarsResult<DataFrame> {
        let mut out = self.grouper.get_keys_in_group_order();
        let out_names = output_schema.iter_names().skip(out.width());
        for (mut r, name) in self.grouped_reductions.into_iter().zip(out_names) {
            unsafe {
                out.with_column_unchecked(r.finalize()?.with_name(name.clone()).into_column());
            }
        }
        Ok(out)
    }
}

struct GroupBySinkState {
    key_selectors: Vec<StreamExpr>,
    grouped_reduction_selectors: Vec<StreamExpr>,
    grouper: Box<dyn Grouper>,
    grouped_reductions: Vec<Box<dyn GroupedReduction>>,
    local: Vec<LocalGroupBySinkState>,
    random_state: PlRandomState,
}

impl GroupBySinkState {
    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        receivers: Vec<Receiver<Morsel>>,
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(receivers.len() >= self.local.len());
        self.local
            .resize_with(receivers.len(), || LocalGroupBySinkState {
                grouper: self.grouper.new_empty(),
                grouped_reductions: self
                    .grouped_reductions
                    .iter()
                    .map(|r| r.new_empty())
                    .collect(),
            });
        for (mut recv, local) in receivers.into_iter().zip(&mut self.local) {
            let key_selectors = &self.key_selectors;
            let grouped_reduction_selectors = &self.grouped_reduction_selectors;
            let random_state = &self.random_state;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut group_idxs = Vec::new();
                while let Ok(morsel) = recv.recv().await {
                    // Compute group indices from key.
                    let seq = morsel.seq().to_u64();
                    let df = morsel.into_df();
                    let mut key_columns = Vec::new();
                    for selector in key_selectors {
                        let s = selector.evaluate(&df, state).await?;
                        key_columns.push(s.into_column());
                    }
                    let keys = DataFrame::new_with_broadcast_len(key_columns, df.height())?;
                    let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, true);
                    local.grouper.insert_keys(hash_keys, &mut group_idxs);

                    // Update reductions.
                    for (selector, reduction) in grouped_reduction_selectors
                        .iter()
                        .zip(&mut local.grouped_reductions)
                    {
                        unsafe {
                            // SAFETY: we resize the reduction to the number of groups beforehand.
                            reduction.resize(local.grouper.num_groups());
                            reduction.update_groups(
                                selector
                                    .evaluate(&df, state)
                                    .await?
                                    .as_materialized_series(),
                                &group_idxs,
                                seq,
                            )?;
                        }
                    }
                }
                Ok(())
            }));
        }
    }

    fn combine_locals(
        output_schema: &Schema,
        mut locals: Vec<LocalGroupBySinkState>,
    ) -> PolarsResult<DataFrame> {
        let mut group_idxs = Vec::new();
        let mut combined = locals.pop().unwrap();
        for local in locals {
            combined.grouper.combine(&*local.grouper, &mut group_idxs);
            for (l, r) in combined
                .grouped_reductions
                .iter_mut()
                .zip(&local.grouped_reductions)
            {
                unsafe {
                    l.resize(combined.grouper.num_groups());
                    l.combine(&**r, &group_idxs)?;
                }
            }
        }
        combined.into_df(output_schema)
    }

    fn combine_locals_parallel(
        num_partitions: usize,
        output_schema: &Schema,
        locals: Vec<LocalGroupBySinkState>,
    ) -> PolarsResult<DataFrame> {
        let partitioner = HashPartitioner::new(num_partitions, 0);
        POOL.install(|| {
            let l_partitions: Vec<_> = locals
                .as_slice()
                .into_par_iter()
                .with_max_len(1)
                .map(|local| {
                    let mut partition_idxs = vec![Vec::new(); num_partitions];
                    let mut sketches = vec![CardinalitySketch::new(); num_partitions];
                    local.grouper.gen_partition_idxs(
                        &partitioner,
                        &mut partition_idxs,
                        &mut sketches,
                    );
                    (partition_idxs, sketches)
                })
                .collect();

            let frames = unsafe {
                (0..num_partitions)
                    .into_par_iter()
                    .with_max_len(1)
                    .map(|p| {
                        // Estimate combined cardinality.
                        let mut combined_sketch = CardinalitySketch::new();
                        for l_partition in &l_partitions {
                            combined_sketch.combine(&l_partition.1[p]);
                        }
                        let combined_cardinality = combined_sketch.estimate() * 5 / 4;

                        // Allocate with the estimated cardinality.
                        let mut combined = LocalGroupBySinkState {
                            grouper: locals[0].grouper.new_empty(),
                            grouped_reductions: locals[0]
                                .grouped_reductions
                                .iter()
                                .map(|r| r.new_empty())
                                .collect(),
                        };
                        combined.grouper.reserve(combined_cardinality);
                        for r in combined.grouped_reductions.iter_mut() {
                            r.reserve(combined_cardinality);
                        }

                        // Combine everything.
                        let mut group_idxs = Vec::new();
                        for l in 0..num_partitions {
                            combined.grouper.gather_combine(
                                &*locals[l].grouper,
                                &l_partitions[l].0[p],
                                &mut group_idxs,
                            );
                            for (a, b) in combined
                                .grouped_reductions
                                .iter_mut()
                                .zip(&locals[l].grouped_reductions)
                            {
                                a.resize(combined.grouper.num_groups());
                                a.gather_combine(&**b, &l_partitions[l].0[p], &group_idxs)?;
                            }
                        }
                        combined.into_df(output_schema)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            };

            Ok(accumulate_dataframes_vertical_unchecked(frames))
        })
    }

    fn into_source(self, output_schema: &Schema) -> PolarsResult<InMemorySourceNode> {
        let num_pipelines = self.local.len();
        let num_rows: usize = self
            .local
            .iter()
            .map(|l| l.grouper.num_groups() as usize)
            .sum();
        let ideal_num_partitions = num_rows.div_ceil(GROUP_BY_MIN_ROWS_PER_PARTITION);
        let num_partitions = if ideal_num_partitions >= 4 {
            ideal_num_partitions.min(self.local.len())
        } else {
            // If the ideal number of partitions is this low, don't even bother.
            1
        };

        let df = if num_partitions == 1 {
            Self::combine_locals(output_schema, self.local)
        } else {
            Self::combine_locals_parallel(num_partitions, output_schema, self.local)
        };

        let mut source_node = InMemorySourceNode::new(Arc::new(df?), MorselSeq::default());
        source_node.initialize(num_pipelines);
        Ok(source_node)
    }
}

enum GroupByState {
    Sink(GroupBySinkState),
    Source(InMemorySourceNode),
    Done,
}

pub struct GroupByNode {
    state: GroupByState,
    output_schema: Arc<Schema>,
}

impl GroupByNode {
    pub fn new(
        key_selectors: Vec<StreamExpr>,
        grouped_reduction_selectors: Vec<StreamExpr>,
        grouped_reductions: Vec<Box<dyn GroupedReduction>>,
        grouper: Box<dyn Grouper>,
        output_schema: Arc<Schema>,
        random_state: PlRandomState,
    ) -> Self {
        Self {
            state: GroupByState::Sink(GroupBySinkState {
                key_selectors,
                grouped_reduction_selectors,
                grouped_reductions,
                grouper,
                local: Vec::new(),
                random_state,
            }),
            output_schema,
        }
    }
}

impl ComputeNode for GroupByNode {
    fn name(&self) -> &str {
        "group_by"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = GroupByState::Done;
            },
            // Input is done, transition to being a source.
            GroupByState::Sink(_) if matches!(recv[0], PortState::Done) => {
                let GroupByState::Sink(sink) =
                    core::mem::replace(&mut self.state, GroupByState::Done)
                else {
                    unreachable!()
                };
                self.state = GroupByState::Source(sink.into_source(&self.output_schema)?);
            },
            // Defer to source node implementation.
            GroupByState::Source(src) => {
                src.update_state(&mut [], send)?;
                if send[0] == PortState::Done {
                    self.state = GroupByState::Done;
                }
            },
            // Nothing to change.
            GroupByState::Done | GroupByState::Sink(_) => {},
        }

        // Communicate our state.
        match &self.state {
            GroupByState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            GroupByState::Source(..) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            GroupByState::Done => {
                recv[0] = PortState::Done;
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
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send_ports.len() == 1 && recv_ports.len() == 1);
        match &mut self.state {
            GroupByState::Sink(sink) => {
                assert!(send_ports[0].is_none());
                sink.spawn(
                    scope,
                    recv_ports[0].take().unwrap().parallel(),
                    state,
                    join_handles,
                )
            },
            GroupByState::Source(source) => {
                assert!(recv_ports[0].is_none());
                source.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            GroupByState::Done => unreachable!(),
        }
    }
}
