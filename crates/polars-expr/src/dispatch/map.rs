use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRMapFunction;

pub fn function_expr_to_udf(func: IRMapFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRMapFunction::*;
    match func {
        Entries => map!(map_entries),
    }
}

fn map_entries(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| Ok(s.map()?.storage().clone()))
}
