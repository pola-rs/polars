use std::mem::ManuallyDrop;
use std::sync::Arc;

use polars_core::prelude::IntoColumn;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::POOL;
use polars_expr::groups::Grouper;
use polars_expr::reduce::GroupedReduction;
use polars_utils::itertools::Itertools;
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use crate::async_primitives::connector::Receiver;
use crate::expression::StreamExpr;
use crate::nodes::in_memory_source::InMemorySourceNode;

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
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut group_idxs = Vec::new();
                while let Ok(morsel) = recv.recv().await {
                    // Compute group indices from key.
                    let df = morsel.into_df();
                    let mut key_columns = Vec::new();
                    for selector in key_selectors {
                        let s = selector.evaluate(&df, state).await?;
                        key_columns.push(s.into_column());
                    }
                    let keys = DataFrame::new_with_broadcast_len(key_columns, df.height())?;
                    local.grouper.insert_keys(&keys, &mut group_idxs);

                    // Update reductions.
                    for (selector, reduction) in grouped_reduction_selectors
                        .iter()
                        .zip(&mut local.grouped_reductions)
                    {
                        unsafe {
                            // SAFETY: we resize the reduction to the number of groups beforehand.
                            reduction.resize(local.grouper.num_groups());
                            reduction.update_groups(
                                &selector.evaluate(&df, state).await?,
                                &group_idxs,
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

    fn into_source_parallel(self, output_schema: &Schema) -> PolarsResult<InMemorySourceNode> {
        let num_partitions = self.local.len();
        let seed = 0xdeadbeef;
        POOL.install(|| {
            let l_partition_idxs: Vec<_> = self
                .local
                .as_slice()
                .into_par_iter()
                .with_max_len(1)
                .map(|local| {
                    local
                        .grouper
                        .gen_partition_idxs(seed, num_partitions)
                })
                .collect();

            let frames = unsafe {
                (0..num_partitions)
                    .into_par_iter()
                    .with_max_len(1)
                    .map(|p| {
                        let mut group_idxs = Vec::new();
                        let mut combined = LocalGroupBySinkState {
                            grouper: self.grouper.new_empty(),
                            grouped_reductions: self.grouped_reductions.iter().map(|r| r.new_empty()).collect(),
                        };
                        for l in 0..num_partitions { 
                            combined.grouper.gather_combine(&*self.local[l].grouper, &l_partition_idxs[l][p], &mut group_idxs);
                            for (a, b) in combined
                                .grouped_reductions
                                .iter_mut()
                                .zip(&self.local[l].grouped_reductions)
                            {
                                a.resize(combined.grouper.num_groups());
                                a.gather_combine(&**b, &l_partition_idxs[l][p], &group_idxs)?;
                            }
                        }
                        combined.into_df(output_schema)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            };

            let df = accumulate_dataframes_vertical_unchecked(frames);
            let mut source_node = InMemorySourceNode::new(Arc::new(df));
            source_node.initialize(num_partitions);
            Ok(source_node)
        })
    }

    fn into_source(self, output_schema: &Schema) -> PolarsResult<InMemorySourceNode> {
        if std::env::var("POLARS_PARALLEL_GROUPBY_FINALIZE").as_deref() == Ok("1") {
            self.into_source_parallel(output_schema)
        } else {
            let num_pipelines = self.local.len();
            let df = Self::combine_locals(output_schema, self.local);
            let mut source_node = InMemorySourceNode::new(Arc::new(df?));
            source_node.initialize(num_pipelines);
            Ok(source_node)
        }
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
    ) -> Self {
        Self {
            state: GroupByState::Sink(GroupBySinkState {
                key_selectors,
                grouped_reduction_selectors,
                grouped_reductions,
                grouper,
                local: Vec::new(),
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
