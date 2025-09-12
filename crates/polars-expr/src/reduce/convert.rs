// use polars_core::error::feature_gated;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::*;
use crate::reduce::any_all::{new_all_reduction, new_any_reduction};
#[cfg(feature = "bitwise")]
use crate::reduce::bitwise::{
    new_bitwise_and_reduction, new_bitwise_or_reduction, new_bitwise_xor_reduction,
};
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
            .to_dtype(schema, expr_arena)?
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
            IRAggExpr::Count {
                input,
                include_nulls,
            } => {
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
        #[cfg(feature = "bitwise")]
        AExpr::Function {
            input: inner_exprs,
            function: IRFunctionExpr::Bitwise(inner_fn),
            options: _,
        } => {
            assert!(inner_exprs.len() == 1);
            let input = inner_exprs[0].node();
            match inner_fn {
                IRBitwiseFunction::And => (new_bitwise_and_reduction(get_dt(input)?), input),
                IRBitwiseFunction::Or => (new_bitwise_or_reduction(get_dt(input)?), input),
                IRBitwiseFunction::Xor => (new_bitwise_xor_reduction(get_dt(input)?), input),
                _ => unreachable!(),
            }
        },

        AExpr::Function {
            input: inner_exprs,
            function: IRFunctionExpr::Boolean(inner_fn),
            options: _,
        } => {
            assert!(inner_exprs.len() == 1);
            let input = inner_exprs[0].node();
            match inner_fn {
                IRBooleanFunction::Any { ignore_nulls } => {
                    (new_any_reduction(*ignore_nulls), input)
                },
                IRBooleanFunction::All { ignore_nulls } => {
                    (new_all_reduction(*ignore_nulls), input)
                },
                _ => unreachable!(),
            }
        },
        _ => unreachable!(),
    };
    Ok(out)
}
