use std::sync::Arc;

use polars_core::POOL;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{GroupPositions, GroupsType};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::prelude::{AggState, AggregationContext, PhysicalExpr};
use crate::state::ExecutionState;

pub fn reverse<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    // Length preserving operation on scalars keeps scalar.
    if let AggState::AggregatedScalar(_) | AggState::LiteralScalar(_) = &ac.agg_state() {
        return Ok(ac);
    }

    POOL.install(|| {
        let positions = GroupsType::Idx(match &**ac.groups().as_ref() {
            GroupsType::Idx(idx) => idx
                .into_par_iter()
                .map(|(first, idx)| {
                    (
                        idx.last().copied().unwrap_or(first),
                        idx.iter().copied().rev().collect(),
                    )
                })
                .collect(),
            GroupsType::Slice {
                groups,
                overlapping: _,
            } => groups
                .into_par_iter()
                .map(|[start, len]| {
                    (
                        start + len.saturating_sub(1),
                        (*start..*start + *len).rev().collect(),
                    )
                })
                .collect(),
        })
        .into_sliceable();
        ac.with_groups(positions);
    });

    Ok(ac)
}
