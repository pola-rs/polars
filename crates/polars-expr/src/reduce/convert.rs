// use polars_core::error::feature_gated;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::*;
use crate::reduce::count::CountReduce;
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
            IRAggExpr::Count(input, include_nulls) => {
                let count = Box::new(CountReduce::new(*include_nulls)) as Box<_>;
                (count, *input)
            },
            IRAggExpr::Quantile { .. } => todo!(),
            IRAggExpr::Median(_) => todo!(),
            IRAggExpr::NUnique(_) => todo!(),
            IRAggExpr::Implode(_) => todo!(),
            IRAggExpr::AggGroups(_) => todo!(),
        },
        AExpr::Len => {
            if let Some(first_column) = schema.iter_names().next() {
                let out: Box<dyn GroupedReduction> = Box::new(LenReduce::default());
                let expr = expr_arena.add(AExpr::Column(first_column.as_str().into()));

                (out, expr)
            } else {
                // Support len aggregation on 0-width morsels.
                // Notes:
                // * We do this instead of projecting a scalar, because scalar literals don't
                //   project to the height of the DataFrame (in the PhysicalExpr impl).
                // * This approach is not sound for `update_groups()`, but currently that case is
                //   not hit (it would need group-by -> len on empty morsels).
                let out: Box<dyn GroupedReduction> = new_sum_reduction(DataType::IDX_DTYPE);
                let expr = expr_arena.add(AExpr::Len);

                (out, expr)
            }
        },
        _ => unreachable!(),
    };
    Ok(out)
}
