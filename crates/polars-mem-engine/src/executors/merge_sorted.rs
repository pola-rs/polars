use polars_ops::prelude::*;

use super::*;

pub(crate) struct MergeSorted {
    pub(crate) input_left: Box<dyn Executor>,
    pub(crate) input_right: Box<dyn Executor>,
    pub(crate) key: PlSmallStr,
}

impl Executor for MergeSorted {
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
            let (left, right) = POOL.join(
                || self.input_left.execute(state),
                || self.input_right.execute(&mut state2),
            );
            (left?, right?)
        };

        let profile_name = Cow::Borrowed("Merge Sorted");
        state.record(
            || {
                let lhs = left.column(self.key.as_str())?;
                let rhs = right.column(self.key.as_str())?;

                _merge_sorted_dfs(
                    &left,
                    &right,
                    lhs.as_materialized_series(),
                    rhs.as_materialized_series(),
                    true,
                )
            },
            profile_name,
        )
    }
}
