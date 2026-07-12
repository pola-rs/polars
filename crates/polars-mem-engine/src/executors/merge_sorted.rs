use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_ops::prelude::*;
use recursive::recursive;

use super::*;

pub(crate) struct MergeSorted {
    pub(crate) input_left: Box<dyn Executor>,
    pub(crate) input_right: Box<dyn Executor>,
    pub(crate) key: Arc<[PlSmallStr]>,
    pub(crate) descending: bool,
    pub(crate) nulls_last: bool,
}

/// Build the series that the merge order is decided on.
///
/// With a single key we use that column directly so that the per dtype merge
/// logic (categoricals, nulls, ...) keeps working as before. With multiple keys
/// we row encode them into a single ordered binary series so the lexicographic
/// order over all keys is respected. The `descending` / `nulls_last` options are
/// baked into the encoding, so the caller must not re-apply them for the
/// row-encoded path.
fn merge_key_series(
    df: &DataFrame,
    key: &[PlSmallStr],
    descending: bool,
    nulls_last: bool,
) -> PolarsResult<Series> {
    if key.len() == 1 {
        return Ok(df.column(key[0].as_str())?.as_materialized_series().clone());
    }

    let columns = key
        .iter()
        .map(|name| df.column(name.as_str()).cloned())
        .collect::<PolarsResult<Vec<_>>>()?;
    let descending = vec![descending; columns.len()];
    let nulls_last = vec![nulls_last; columns.len()];
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
                let lhs = merge_key_series(&left, &self.key, self.descending, self.nulls_last)?;
                let rhs = merge_key_series(&right, &self.key, self.descending, self.nulls_last)?;

                // For the row-encoded (multi-key) path the ordering is already
                // baked into the encoding, so the merge itself compares the
                // encoded column ascending with nulls first. For a single key we
                // compare the raw column and apply the options here.
                let (descending, nulls_last) = if self.key.len() == 1 {
                    (self.descending, self.nulls_last)
                } else {
                    (false, false)
                };

                _merge_sorted_dfs(&left, &right, &lhs, &rhs, true, descending, nulls_last)
            },
            profile_name,
        )
    }
}
