use polars_error::PolarsResult;
use polars_expr::hash_keys::HashKeys;
use polars_expr::reduce::GroupedReduction;
use polars_utils::hashing::HashPartitioner;
use polars_utils::{IdxSize, UnitVec};

use crate::execute::StreamingExecutionState;
use crate::hash_shuffle::parallel::for_each_parallel;
use crate::hash_shuffle::split::{SplitRows, split};

/// Partial states of groups that left a hot table (evictions), or a whole hot table.
pub(super) struct PreAgg {
    pub(super) keys: HashKeys,
    /// The indices of the grouped reductions the states belong to.
    pub(super) reduction_idxs: UnitVec<usize>,
    pub(super) reductions: Vec<Box<dyn GroupedReduction>>,
    /// The groups of each partition; `None` with one partition.
    pub(super) split: Option<SplitRows>,
}

/// All pre-aggregates, pipeline by pipeline in creation order (the order first/last need), each
/// split into its groups per partition. Splitting only computes group indices; the keys and
/// states stay where they are.
#[derive(Default)]
pub(super) struct SplitPreAggs(Vec<PreAgg>);

impl SplitPreAggs {
    /// Splits every pre-aggregate in parallel when there is more than one partition.
    pub(super) fn new(
        mut pre_aggs: Vec<PreAgg>,
        partitioner: &HashPartitioner,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Self> {
        if partitioner.num_partitions() > 1 {
            for_each_parallel(
                pre_aggs.iter_mut(),
                &|pre_agg: &mut PreAgg| {
                    pre_agg.split = split(&pre_agg.keys, None, partitioner);
                    Ok(())
                },
                state,
            )?;
        }
        Ok(Self(pre_aggs))
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    /// The pre-aggregates with groups in partition `p`, in order, each with those groups (`None`:
    /// all of them).
    pub(super) fn in_partition(
        &self,
        p: usize,
    ) -> impl Iterator<Item = (&PreAgg, Option<&[IdxSize]>)> {
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
    pub(super) fn free(self, state: &StreamingExecutionState) -> PolarsResult<()> {
        for_each_parallel(
            self.0.into_iter(),
            &|pre_agg: PreAgg| {
                drop(pre_agg);
                Ok(())
            },
            state,
        )
    }
}
