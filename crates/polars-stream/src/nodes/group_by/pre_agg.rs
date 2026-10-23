use polars_async::parallel::for_each_parallel;
use polars_error::PolarsResult;
use polars_expr::hash_keys::HashKeys;
use polars_expr::reduce::GroupedReduction;
use polars_ooc::memory_manager;
use polars_utils::hashing::HashPartitioner;
use polars_utils::{IdxSize, UnitVec};

use crate::execute::StreamingExecutionState;
use crate::hash_shuffle::split::{SplitRows, split};

/// Partial states of groups that left a hot table (evictions), or a whole hot table.
pub struct PreAgg {
    pub keys: HashKeys,
    /// The indices of the grouped reductions the states belong to.
    pub reduction_idxs: UnitVec<usize>,
    pub reductions: Vec<Box<dyn GroupedReduction>>,
    /// The groups of each partition; `None` with one partition.
    pub split: Option<SplitRows>,
}

/// All pre-aggregates, pipeline by pipeline in creation order (the order first/last need), each
/// split into its groups per partition. Splitting only computes group indices; the keys and
/// states stay where they are.
#[derive(Default)]
pub struct SplitPreAggs(Vec<PreAgg>);

impl SplitPreAggs {
    /// Splits every pre-aggregate in parallel when there is more than one partition.
    pub fn new(
        mut pre_aggs: Vec<PreAgg>,
        partitioner: &HashPartitioner,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Self> {
        if partitioner.num_partitions() > 1 {
            for_each_parallel(
                pre_aggs.iter_mut(),
                state.task_metrics(),
                &|pre_agg| async move {
                    pre_agg.split = split(&pre_agg.keys, None, partitioner);
                    memory_manager().spill().await;
                    Ok(())
                },
            )?;
        }
        Ok(Self(pre_aggs))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// The pre-aggregates with groups in partition `p`, in order, each with those groups (`None`:
    /// all of them).
    pub fn in_partition(&self, p: usize) -> impl Iterator<Item = (&PreAgg, Option<&[IdxSize]>)> {
        self.0
            .iter()
            .filter_map(move |pre_agg| match &pre_agg.split {
                None => Some((pre_agg, None)),
                Some(split_rows) => {
                    let groups = split_rows.partition(p);
                    (!groups.is_empty()).then_some((pre_agg, Some(groups)))
                },
            })
    }

    /// Drops the pre-aggregates in parallel.
    pub fn free(self, state: &StreamingExecutionState) -> PolarsResult<()> {
        for_each_parallel(self.0, state.task_metrics(), &|pre_agg| async move {
            drop(pre_agg);
            Ok(())
        })
    }
}
