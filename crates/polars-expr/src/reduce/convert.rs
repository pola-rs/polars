// use polars_core::error::feature_gated;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::*;
use crate::reduce::any_all::{new_all_reduction, new_any_reduction};
#[cfg(feature = "approx_unique")]
use crate::reduce::approx_n_unique::new_approx_n_unique_reduction;
#[cfg(feature = "bitwise")]
use crate::reduce::bitwise::{
    new_bitwise_and_reduction, new_bitwise_or_reduction, new_bitwise_xor_reduction,
};
use crate::reduce::count::{CountReduce, NullCountReduce};
use crate::reduce::first_last::{new_first_reduction, new_item_reduction, new_last_reduction};
use crate::reduce::first_last_nonnull::{new_first_nonnull_reduction, new_last_nonnull_reduction};
use crate::reduce::len::LenReduce;
use crate::reduce::mean::new_mean_reduction;
use crate::reduce::min_max::{new_max_reduction, new_min_reduction};
use crate::reduce::min_max_by::{new_max_by_reduction, new_min_by_reduction};
use crate::reduce::sum::new_sum_reduction;
use crate::reduce::var_std::new_var_std_reduction;

/// Converts a node into a reduction + its associated selector expression.
pub fn into_reduction(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
    is_aggregation_context: bool,
) -> PolarsResult<(Box<dyn GroupedReduction>, Vec<Node>)> {
    let get_dt = |node| {
        expr_arena
            .get(node)
            .to_dtype(&ToFieldContext::new(expr_arena, schema))?
            .materialize_unknown(false)
    };
    let (gr, in_node) = match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(input) => (new_sum_reduction(get_dt(*input)?)?, *input),
            IRAggExpr::Mean(input) => (new_mean_reduction(get_dt(*input)?)?, *input),
            IRAggExpr::Min {
                propagate_nans,
                input,
            } => (new_min_reduction(get_dt(*input)?, *propagate_nans)?, *input),
            IRAggExpr::Max {
                propagate_nans,
                input,
            } => (new_max_reduction(get_dt(*input)?, *propagate_nans)?, *input),
            IRAggExpr::Var(input, ddof) => (
                new_var_std_reduction(get_dt(*input)?, false, *ddof)?,
                *input,
            ),
            IRAggExpr::Std(input, ddof) => {
                (new_var_std_reduction(get_dt(*input)?, true, *ddof)?, *input)
            },
            IRAggExpr::First(input) => (new_first_reduction(get_dt(*input)?), *input),
            IRAggExpr::FirstNonNull(input) => {
                (new_first_nonnull_reduction(get_dt(*input)?), *input)
            },
            IRAggExpr::Last(input) => (new_last_reduction(get_dt(*input)?), *input),
            IRAggExpr::LastNonNull(input) => (new_last_nonnull_reduction(get_dt(*input)?), *input),
            IRAggExpr::Item { input, allow_empty } => {
                (new_item_reduction(get_dt(*input)?, *allow_empty), *input)
            },
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
                polars_ensure!(
                    !is_aggregation_context,
                    ComputeError:
                    "not implemented: len() of groups with no columns"
                );

                let out: Box<dyn GroupedReduction> = new_sum_reduction(DataType::IDX_DTYPE)?;
                let expr = expr_arena.add(AExpr::Len);

                (out, expr)
            }
        },

        AExpr::Function {
            input: inner_exprs,
            function: IRFunctionExpr::NullCount,
            options: _,
        } => {
            assert!(inner_exprs.len() == 1);
            let input = inner_exprs[0].node();
            let count = Box::new(NullCountReduce::new()) as Box<_>;
            (count, input)
        },

        #[cfg(feature = "approx_unique")]
        AExpr::Function {
            input: inner_exprs,
            function: IRFunctionExpr::ApproxNUnique,
            options: _,
        } => {
            assert!(inner_exprs.len() == 1);
            let input = inner_exprs[0].node();
            let out = new_approx_n_unique_reduction(get_dt(input)?)?;
            (out, input)
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

        AExpr::Function {
            input: inner_exprs,
            function: IRFunctionExpr::MinBy,
            options: _,
        } => {
            assert!(inner_exprs.len() == 2);
            let input = inner_exprs[0].node();
            let by = inner_exprs[1].node();
            let gr = new_min_by_reduction(get_dt(input)?, get_dt(by)?)?;
            return Ok((gr, vec![input, by]));
        },

        AExpr::Function {
            input: inner_exprs,
            function: IRFunctionExpr::MaxBy,
            options: _,
        } => {
            assert!(inner_exprs.len() == 2);
            let input = inner_exprs[0].node();
            let by = inner_exprs[1].node();
            let gr = new_max_by_reduction(get_dt(input)?, get_dt(by)?)?;
            return Ok((gr, vec![input, by]));
        },

        AExpr::AnonymousAgg {
            input: inner_exprs,
            fmt_str: _,
            function,
        } => {
            let ann_agg = function.materialize()?;
            assert!(inner_exprs.len() == 1);
            let input = inner_exprs[0].node();
            let reduction = ann_agg.as_any();
            let reduction = reduction
                .downcast_ref::<Box<dyn GroupedReduction>>()
                .unwrap();
            (reduction.new_empty(), input)
        },
        _ => unreachable!(),
    };
    Ok((gr, vec![in_node]))
}
