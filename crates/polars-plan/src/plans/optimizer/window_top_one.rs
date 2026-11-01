//! Rewrites a filter that keeps the first row number of each partition into a group-by that picks
//! that row.
//!
//! ```text
//! FILTER col(rn) == 1                         FILTER col(rn) == 1
//!   SELECT [a, col(tmp) as rn]                  SELECT [a, col(tmp) as rn]
//!     WINDOW PARTITION BY [g] ORDER BY k:   →     SELECT [g, k, a, 1 as tmp]
//!       [(int_range(len()) + 1).over(..)            GROUP BY [g]: [k.sort_by(k).first(), a.sort_by(k).first()]
//!        as tmp]                                      input
//!       input
//! ```
//!
//! Only done if the window may reorder its output and may pick any of the rows with equal order
//! keys, as the group-by does.

use polars_core::chunked_array::cast::CastOptions;
use polars_core::scalar::Scalar;
use polars_utils::arena::{Arena, Node};

use super::simplify_expr::lower_sort_reads;
use crate::plans::*;
use crate::unique_column_name;

/// With `maintain_errors`, only projections of plain columns are crossed, so that no
/// expression that could fail runs on fewer rows.
pub(super) fn rewrite_top_one_windows(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    maintain_errors: bool,
) {
    let mut visited = PlIndexSet::new();
    let mut stack = vec![root];
    let mut found = Vec::new();
    while let Some(node) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }
        let ir = ir_arena.get(node);
        if let IR::Filter { input, predicate } = ir
            && let Some((name, keep)) = row_number_filter(predicate.node(), expr_arena)
            && let Some(found_window) =
                find_window(*input, name, keep, ir_arena, expr_arena, maintain_errors)
        {
            found.push(found_window);
        }
        ir.copy_inputs(&mut stack);
    }

    for window in found {
        rewrite_window(window, ir_arena, expr_arena);
    }
}

/// Which row numbers a filter keeps.
#[derive(Clone, Copy)]
enum Keep {
    Equal(i64),
    LessThan(i64),
}

impl Keep {
    fn is_only(self, first: i64) -> bool {
        match self {
            Self::Equal(v) => v == first,
            Self::LessThan(v) => v == first + 1,
        }
    }
}

/// The column and the row numbers kept by a predicate `col(name) {==, <=, <} lit`.
fn row_number_filter(predicate: Node, expr_arena: &Arena<AExpr>) -> Option<(PlSmallStr, Keep)> {
    let AExpr::BinaryExpr { left, op, right } = expr_arena.get(predicate) else {
        return None;
    };
    let (name, op, value) = match (expr_arena.get(*left), expr_arena.get(*right)) {
        (AExpr::Column(name), AExpr::Literal(lv)) => (name, *op, lv),
        (AExpr::Literal(lv), AExpr::Column(name)) => (name, op.swap_operands()?, lv),
        _ => return None,
    };
    let value = int_literal(value)?;
    let keep = match op {
        Operator::Eq | Operator::EqValidity | Operator::LtEq => Keep::Equal(value),
        Operator::Lt => Keep::LessThan(value),
        _ => return None,
    };
    Some((name.clone(), keep))
}

fn int_literal(lv: &LiteralValue) -> Option<i64> {
    lv.get_datatype()
        .is_integer()
        .then(|| lv.extract_i64().ok())
        .flatten()
}

/// A window with a single row number expression, and what the rewrite needs to know about it.
struct FoundWindow {
    node: Node,
    name: PlSmallStr,
    keep: Keep,
    /// The window output columns that are read above it, if known.
    live: Option<PlIndexSet<PlSmallStr>>,
}

/// Follows `name` down to the window that computes it, through projections that compute every
/// row on its own.
fn find_window(
    mut node: Node,
    mut name: PlSmallStr,
    keep: Keep,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    maintain_errors: bool,
) -> Option<FoundWindow> {
    let mut live: Option<PlIndexSet<PlSmallStr>> = None;
    loop {
        match ir_arena.get(node) {
            IR::Window { exprs, .. } => {
                return (exprs.len() == 1 && exprs[0].output_name() == &name).then_some(
                    FoundWindow {
                        node,
                        name,
                        keep,
                        live,
                    },
                );
            },
            IR::SimpleProjection { input, columns } => {
                live.get_or_insert_with(|| columns.iter_names().cloned().collect());
                node = *input;
            },
            IR::Select {
                input, expr: exprs, ..
            }
            | IR::HStack { input, exprs, .. } => {
                let is_select = matches!(ir_arena.get(node), IR::Select { .. });
                let crossable = |e: &ExprIR| {
                    if maintain_errors {
                        matches!(expr_arena.get(e.node()), AExpr::Column(_))
                    } else {
                        is_elementwise_rec(e.node(), expr_arena)
                    }
                };
                if !exprs.iter().all(crossable) {
                    return None;
                }
                match exprs.iter().find(|e| e.output_name() == &name) {
                    Some(e) => {
                        let AExpr::Column(input_name) = expr_arena.get(e.node()) else {
                            return None;
                        };
                        name = input_name.clone();
                    },
                    None if is_select => return None,
                    None => {},
                }

                let is_live =
                    |e: &&ExprIR| live.as_ref().is_none_or(|l| l.contains(e.output_name()));
                let read = exprs
                    .iter()
                    .filter(is_live)
                    .flat_map(|e| aexpr_to_leaf_names_iter(e.node(), expr_arena))
                    .cloned()
                    .collect::<Vec<_>>();
                live = if is_select {
                    Some(read.into_iter().collect())
                } else {
                    live.map(|mut l| {
                        for e in exprs {
                            l.swap_remove(e.output_name());
                        }
                        l.extend(read);
                        l
                    })
                };
                node = *input;
            },
            _ => return None,
        }
    }
}

/// Whether a dtype has a value per row that `int_range(0, len())` cannot overflow.
#[cfg(feature = "range")]
fn is_wide_index(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Int64 | DataType::UInt64) || dtype == &IDX_DTYPE
}

/// The first row number of `int_range(0, len())` or `int_range(0, len()) + 1`.
fn first_row_number(node: Node, expr_arena: &Arena<AExpr>) -> Option<i64> {
    let is_one = |node: Node| matches!(expr_arena.get(node), AExpr::Literal(lv) if int_literal(lv) == Some(1));
    match expr_arena.get(node) {
        AExpr::BinaryExpr {
            left,
            op: Operator::Plus,
            right,
        } => {
            let range = match (is_one(*left), is_one(*right)) {
                (false, true) => *left,
                (true, false) => *right,
                _ => return None,
            };
            is_row_index(range, expr_arena).then_some(1)
        },
        _ => is_row_index(node, expr_arena).then_some(0),
    }
}

/// Whether `node` is `int_range(0, len())`.
#[cfg(feature = "range")]
fn is_row_index(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let AExpr::Function {
        input,
        function: IRFunctionExpr::Range(IRRangeFunction::IntRange { step: 1, dtype }),
        ..
    } = expr_arena.get(node)
    else {
        return false;
    };
    let [start, end] = input.as_slice() else {
        return false;
    };
    if !is_wide_index(dtype)
        || !matches!(expr_arena.get(start.node()), AExpr::Literal(lv) if int_literal(lv) == Some(0))
    {
        return false;
    }
    match expr_arena.get(end.node()) {
        AExpr::Len => true,
        AExpr::Cast { expr, dtype, .. } => {
            is_wide_index(dtype) && matches!(expr_arena.get(*expr), AExpr::Len)
        },
        _ => false,
    }
}

#[cfg(not(feature = "range"))]
fn is_row_index(_node: Node, _expr_arena: &Arena<AExpr>) -> bool {
    false
}

/// Whether rows of a group have the same value in a key of this dtype, so that the key of the
/// group is the key of every row.
fn equal_keys_are_identical(dtype: &DataType) -> bool {
    dtype.is_integer()
        || dtype.is_bool()
        || dtype.is_string()
        || dtype.is_binary()
        || dtype.is_temporal()
        || dtype.is_decimal()
        || dtype.is_categorical()
        || dtype.is_enum()
}

fn rewrite_window(
    window: FoundWindow,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<()> {
    let FoundWindow {
        node,
        name,
        keep,
        live,
    } = window;
    let IR::Window {
        input,
        partition_by,
        order_by: Some((order_by, sort_options)),
        exprs,
        schema,
        maintain_order: false,
        ordered_eval: false,
    } = ir_arena.get(node)
    else {
        return None;
    };
    if partition_by.is_empty() {
        return None;
    }
    let AExpr::Over {
        function,
        mapping: WindowMapping::GroupsToRows,
        ..
    } = expr_arena.get(exprs[0].node())
    else {
        return None;
    };
    let first = first_row_number(*function, expr_arena)?;
    if !keep.is_only(first) {
        return None;
    }

    let input = *input;
    let partition_by = partition_by.clone();
    let order_by = order_by.clone();
    let sort_options = SortMultipleOptions {
        maintain_order: false,
        ..sort_options.clone()
    };
    let output_schema = schema.clone();
    let input_schema = ir_arena.get(input).schema(ir_arena).into_owned();

    let by = order_by
        .iter()
        .map(|c| expr_arena.add(AExpr::Column(c.clone())))
        .collect::<Vec<_>>();
    let mut keys = Vec::new();
    let mut aggs = Vec::new();
    for (column, dtype) in input_schema.iter() {
        let is_key = partition_by.contains(column);
        if is_key {
            let key = expr_arena.add(AExpr::Column(column.clone()));
            if equal_keys_are_identical(dtype) {
                keys.push(ExprIR::new(key, OutputName::ColumnLhs(column.clone())));
                continue;
            }
            keys.push(ExprIR::new(key, OutputName::Alias(unique_column_name())));
        } else if live.as_ref().is_some_and(|l| !l.contains(column)) {
            continue;
        }
        let expr = expr_arena.add(AExpr::Column(column.clone()));
        let sort_by = expr_arena.add(AExpr::SortBy {
            expr,
            by: by.clone(),
            sort_options: sort_options.clone(),
        });
        let first = expr_arena.add(AExpr::Agg(IRAggExpr::First(sort_by)));
        let lowered = lower_sort_reads(first, expr_arena, &input_schema)?;
        let lowered = expr_arena.add(lowered);
        aggs.push(ExprIR::new(lowered, OutputName::Alias(column.clone())));
    }

    let projection = output_schema
        .iter()
        .map(|(column, dtype)| {
            let value = if column == &name {
                let first = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                    DataType::Int64,
                    AnyValue::Int64(first),
                ))));
                AExpr::Cast {
                    expr: first,
                    dtype: dtype.clone(),
                    options: CastOptions::Strict,
                }
            } else if partition_by.contains(column)
                || aggs.iter().any(|a| a.output_name() == column)
            {
                AExpr::Column(column.clone())
            } else {
                AExpr::Literal(LiteralValue::Scalar(Scalar::null(dtype.clone())))
            };
            ExprIR::new(expr_arena.add(value), OutputName::Alias(column.clone()))
        })
        .collect();

    let new = IRBuilder::new(input, expr_arena, ir_arena)
        .group_by(keys, aggs, None, false, Default::default())
        .ok()?
        .project(projection, ProjectionOptions::default())
        .build();
    ir_arena.replace(node, new);
    Some(())
}
