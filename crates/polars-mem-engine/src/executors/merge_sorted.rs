use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_ops::prelude::*;
use recursive::recursive;

use super::*;

pub(crate) struct MergeSorted {
    pub(crate) input_left: Box<dyn Executor>,
    pub(crate) input_right: Box<dyn Executor>,
    pub(crate) key: Arc<[PlSmallStr]>,
}

/// Build the series that the merge order is decided on.
///
/// With a single key we use that column directly so that the per dtype merge
/// logic (categoricals, nulls, ...) keeps working as before. With multiple keys
/// we row encode them into a single ordered binary series so the lexicographic
/// order over all keys is respected.
fn merge_key_series(df: &DataFrame, key: &[PlSmallStr]) -> PolarsResult<Series> {
    if key.len() == 1 {
        return Ok(df.column(key[0].as_str())?.as_materialized_series().clone());
    }

    let columns = key
        .iter()
        .map(|name| df.column(name.as_str()).cloned())
        .collect::<PolarsResult<Vec<_>>>()?;
    let descending = vec![false; columns.len()];
    let nulls_last = vec![false; columns.len()];
    let encoded =
        _get_rows_encoded_ca(PlSmallStr::EMPTY, &columns, &descending, &nulls_last, false)?;
    Ok(encoded.into_series())
}

impl Executor for MergeSorted {
    #[recursive]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run MergeSorted")
            }
        }
        let (left, right) = {
            let mut state2 = state.split();
            state2.branch_idx += 1;
            let (left, right) = RAYON.join(
                || self.input_left.execute(state),
                || self.input_right.execute(&mut state2),
            );
            (left?, right?)
        };

        let profile_name = Cow::Borrowed("Merge Sorted");
        state.record(
            || {
                let lhs = merge_key_series(&left, &self.key)?;
                let rhs = merge_key_series(&right, &self.key)?;

                _merge_sorted_dfs(&left, &right, &lhs, &rhs, true)
            },
            profile_name,
        )
    }
}
