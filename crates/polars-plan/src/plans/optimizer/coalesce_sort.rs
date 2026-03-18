use IR::*;
use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::{AExpr, DynamicPred, ExprIR, is_sorted};
use crate::prelude::IR;

pub struct CoalesceSort {}

impl OptimizationRule for CoalesceSort {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let lp = lp_arena.get(node);

        match lp {
            Sort {
                input,
                by_column,
                slice,
                sort_options,
            } => {
                let input_ir = lp_arena.get(*input);
                let input_sortedness = is_sorted(*input, lp_arena, expr_arena);

                if let IR::Sort {
                    input: in_input,
                    by_column: in_by_column,
                    slice: in_slice,
                    sort_options: in_sort_options,
                } = input_ir
                {
                    if let Some(result) = try_coalesce_sorts(
                        *in_input,
                        by_column,
                        in_by_column,
                        slice,
                        in_slice,
                        sort_options,
                        in_sort_options,
                        expr_arena,
                    ) {
                        return Ok(Some(result));
                    }
                }

                if let Some(_s) = input_sortedness {
                    // TODO: [amber] If the sortedness information starts with the
                    // sortedness requirement of this sort node, then remove the sort node
                }

                Ok(None)
            },
            _ => Ok(None),
        }
    }
}

/// If two consecutive sort nodes share a prefix of sort columns, replace them with
/// the sort node that covers the most columns.
fn try_coalesce_sorts(
    in_input: Node,
    by_column: &Vec<ExprIR>,
    in_by_column: &Vec<ExprIR>,
    slice: &Option<(i64, usize, Option<DynamicPred>)>,
    in_slice: &Option<(i64, usize, Option<DynamicPred>)>,
    sort_options: &SortMultipleOptions,
    in_sort_options: &SortMultipleOptions,
    expr_arena: &Arena<AExpr>,
) -> Option<IR> {
    if slice != in_slice {
        return None;
    }

    let expr_eq = |e1: &&ExprIR, e2: &&ExprIR| {
        AExpr::is_expr_equal_to(
            expr_arena.get(e1.node()),
            expr_arena.get(e2.node()),
            expr_arena,
        )
    };

    let main_has_most_cols = prefix_dominance(by_column.iter(), in_by_column.iter(), expr_eq)?;
    let (merged_options, main_has_most_cols2) =
        coalesce_sort_multiple_options(sort_options, in_sort_options)?;
    debug_assert_eq!(main_has_most_cols, main_has_most_cols2);

    Some(if main_has_most_cols {
        Sort {
            input: in_input,
            by_column: by_column.clone(),
            slice: slice.clone(),
            sort_options: merged_options,
        }
    } else {
        Sort {
            input: in_input,
            by_column: in_by_column.clone(),
            slice: slice.clone(),
            sort_options: merged_options,
        }
    })
}

/// Checks whether one iterator is a prefix of the other (or they are equal).
///
/// Returns `Some(true)` if the left iterator has at least as many elements as the right,
/// `Some(false)` if the right iterator is strictly longer, and `None` if the iterators
/// diverge before either is exhausted.
fn prefix_dominance<T, I, EQ>(iter1: I, iter2: I, eq: EQ) -> Option<bool>
where
    I: IntoIterator<Item = T>,
    EQ: Fn(&T, &T) -> bool,
{
    let mut iter1 = iter1.into_iter();
    let mut iter2 = iter2.into_iter();
    loop {
        match (iter1.next(), iter2.next()) {
            (Some(a), Some(b)) if eq(&a, &b) => {},
            (Some(_), Some(_)) => return None,
            (_, None) => return Some(true),
            (None, Some(_)) => return Some(false),
        }
    }
}

fn coalesce_sort_multiple_options(
    main: &SortMultipleOptions,
    input: &SortMultipleOptions,
) -> Option<(SortMultipleOptions, bool)> {
    let bool_eq = |b1: &&_, b2: &&_| b1 == b2;
    let main_sorts_most_cols =
        prefix_dominance(main.descending.iter(), input.descending.iter(), bool_eq)?;
    let main_sorts_most_cols2 =
        prefix_dominance(main.nulls_last.iter(), input.nulls_last.iter(), bool_eq)?;
    debug_assert_eq!(main_sorts_most_cols, main_sorts_most_cols2);
    // We don't care about the multithreaded criterion

    let maintain_order = main.maintain_order && input.maintain_order;
    let limit = match (main.limit, input.limit) {
        (Some(l1), Some(l2)) => Some(IdxSize::min(l1, l2)),
        (Some(l), None) | (None, Some(l)) => Some(l),
        (None, None) => None,
    };
    let result = SortMultipleOptions {
        maintain_order,
        limit,
        ..if main_sorts_most_cols {
            main.clone()
        } else {
            input.clone()
        }
    };
    Some((result, main_sorts_most_cols))
}
