//! Implementations of PhysicalAggregation. These aggregations are called by the groupby context,
//! and nowhere else. Note, that this differes from evaluate on groups, which is also called in that
//! context, but typically before aggregation

use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::array::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupTuples};
use polars_core::prelude::*;
use polars_core::utils::NoNull;

impl PhysicalAggregation for SliceExpr {
    // As a final aggregation a Slice returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let s = self.input.evaluate(df, state)?;
        let agg_s = s.agg_list(groups);
        let out = agg_s.map(|s| {
            s.list()
                .unwrap()
                .into_iter()
                .map(|opt_s| opt_s.map(|s| s.slice(self.offset, self.len)))
                .collect::<ListChunked>()
                .into_series()
        });
        Ok(out)
    }
}
