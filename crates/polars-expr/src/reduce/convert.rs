// use polars_core::error::feature_gated;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::*;
use crate::reduce::first_last::{new_first_reduction, new_last_reduction};
use crate::reduce::len::LenReduce;
use crate::reduce::mean::new_mean_reduction;
use crate::reduce::min_max::{new_max_reduction, new_min_reduction};
use crate::reduce::sum::new_sum_reduction;
use crate::reduce::var_std::new_var_std_reduction;

/// Converts a node into a reduction + its associated selector expression.
pub fn into_reduction(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<(Box<dyn GroupedReduction>, Node)> {
    let get_dt = |node| {
        expr_arena
            .get(node)
            .to_dtype(schema, Context::Default, expr_arena)?
            .materialize_unknown(false)
    };
    let out = match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(input) => (new_sum_reduction(get_dt(*input)?), *input),
            IRAggExpr::Mean(input) => (new_mean_reduction(get_dt(*input)?), *input),
            IRAggExpr::Min {
                propagate_nans,
                input,
            } => (new_min_reduction(get_dt(*input)?, *propagate_nans), *input),
            IRAggExpr::Max {
                propagate_nans,
                input,
            } => (new_max_reduction(get_dt(*input)?, *propagate_nans), *input),
            IRAggExpr::Var(input, ddof) => {
                (new_var_std_reduction(get_dt(*input)?, false, *ddof), *input)
            },
            IRAggExpr::Std(input, ddof) => {
                (new_var_std_reduction(get_dt(*input)?, true, *ddof), *input)
            },
            IRAggExpr::First(input) => (new_first_reduction(get_dt(*input)?), *input),
            IRAggExpr::Last(input) => (new_last_reduction(get_dt(*input)?), *input),
            _ => todo!(),
        },
        AExpr::Len => {
            // Compute length on the first column, or if none exist we'll use
            // a zero-length dummy series.
            let out: Box<dyn GroupedReduction> = Box::new(LenReduce::default());
            let expr = if let Some(first_column) = schema.iter_names().next() {
                expr_arena.add(AExpr::Column(first_column.as_str().into()))
            } else {
                let dummy = Series::new_null(PlSmallStr::from_static("dummy"), 0);
                expr_arena.add(AExpr::Literal(LiteralValue::Series(SpecialEq::new(dummy))))
            };
            (out, expr)
        },
        _ => unreachable!(),
    };
    Ok(out)
}
