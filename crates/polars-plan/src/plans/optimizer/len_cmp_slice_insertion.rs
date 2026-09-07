//! Inserts a `head` slice below `select(len() <cmp> n)` (and similar single-output
//! selections that compare the dataframe length against an integer literal), so
//! that the general slice-pushdown machinery can push the row limit further down
//! the plan (e.g. into scans, filters, or union branches) instead of needing to
//! materialize more rows than necessary just to answer the comparison.
//!
//! For example, `df.filter(pred).select(pl.len() > 100)` only needs to know
//! whether more than 100 rows match; inserting `head(101)` below the `select`
//! lets slice-pushdown limit how much of `pred` actually needs to be evaluated.

use polars_core::prelude::IdxSize;
use polars_utils::arena::{Arena, Node};
use polars_utils::index::idxsize_try_from;

use crate::prelude::{AExpr, IR, Operator};

pub(super) fn insert_slice_before_len_cmp(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) {
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        ir_arena.get(node).copy_inputs(&mut stack);

        let IR::Select { input, expr, .. } = ir_arena.get(node) else {
            continue;
        };

        let [e] = expr.as_slice() else {
            continue;
        };

        let Some(head_len) = len_cmp_head_len(e.node(), expr_arena) else {
            continue;
        };

        let input = *input;
        let new_input = ir_arena.add(IR::Slice {
            input,
            offset: 0,
            len: head_len,
        });

        let IR::Select { input, .. } = ir_arena.get_mut(node) else {
            unreachable!()
        };
        *input = new_input;
    }
}

/// If `node` is a comparison between `len()` and a non-negative integer literal,
/// returns the number of leading rows that need to be materialized to answer it.
fn len_cmp_head_len(node: Node, expr_arena: &Arena<AExpr>) -> Option<IdxSize> {
    let AExpr::BinaryExpr { left, op, right } = expr_arena.get(node) else {
        return None;
    };

    let (op, literal_node) = if matches!(expr_arena.get(*left), AExpr::Len) {
        (*op, *right)
    } else if matches!(expr_arena.get(*right), AExpr::Len) {
        (flip_comparison(*op), *left)
    } else {
        return None;
    };

    let AExpr::Literal(lv) = expr_arena.get(literal_node) else {
        return None;
    };
    let n: u64 = lv.extract_i64().ok()?.try_into().ok()?;

    let head_len = match op {
        // `len() < n` doesn't need the extra row: `head(n).len() < n` is
        // already equivalent to `len() < n`.
        Operator::Lt => n,
        Operator::Eq | Operator::NotEq | Operator::LtEq | Operator::Gt | Operator::GtEq => {
            n.saturating_add(1)
        },
        _ => return None,
    };

    idxsize_try_from(head_len).ok()
}

/// Flips a comparison operator so that `n <op> len()` becomes `len() <flip(op)> n`.
fn flip_comparison(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use polars_core::prelude::{DataFrame, Scalar, Schema};
    use polars_utils::arena::Arena;

    use super::*;
    use crate::plans::options::ProjectionOptions;
    use crate::plans::{ExprIR, LiteralValue};

    fn leaf(ir_arena: &mut Arena<IR>) -> Node {
        let schema = Schema::default();
        ir_arena.add(IR::DataFrameScan {
            df: Arc::new(DataFrame::empty_with_schema(&schema)),
            schema: Arc::new(schema),
            output_schema: None,
        })
    }

    fn len_cmp_select(
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        op: Operator,
        n: i64,
        literal_on_left: bool,
    ) -> Node {
        let len_node = expr_arena.add(AExpr::Len);
        let lit_node = expr_arena.add(AExpr::Literal(LiteralValue::from(Scalar::from(n))));
        // `len() <op> n` is the same statement as `n <flip(op)> len()`.
        let (left, op, right) = if literal_on_left {
            (lit_node, flip_comparison(op), len_node)
        } else {
            (len_node, op, lit_node)
        };
        let cmp_node = expr_arena.add(AExpr::BinaryExpr { left, op, right });

        let input = leaf(ir_arena);
        ir_arena.add(IR::Select {
            input,
            expr: vec![ExprIR::from_node(cmp_node, expr_arena)],
            schema: Arc::new(Schema::default()),
            options: ProjectionOptions::default(),
        })
    }

    fn slice_len_below(ir_arena: &Arena<IR>, select_node: Node) -> Option<IdxSize> {
        let IR::Select { input, .. } = ir_arena.get(select_node) else {
            panic!("expected Select node")
        };
        match ir_arena.get(*input) {
            IR::Slice { offset, len, .. } => {
                assert_eq!(*offset, 0);
                Some(*len)
            },
            _ => None,
        }
    }

    #[test]
    fn test_insert_slice_for_len_cmp() {
        let cases = [
            (Operator::Eq, 5, 6),
            (Operator::NotEq, 5, 6),
            (Operator::Lt, 5, 5),
            (Operator::LtEq, 5, 6),
            (Operator::Gt, 5, 6),
            (Operator::GtEq, 5, 6),
        ];

        for (op, n, expected_head_len) in cases {
            for literal_on_left in [false, true] {
                let mut expr_arena = Arena::new();
                let mut ir_arena = Arena::new();
                let select_node =
                    len_cmp_select(&mut ir_arena, &mut expr_arena, op, n, literal_on_left);

                insert_slice_before_len_cmp(select_node, &mut ir_arena, &expr_arena);

                assert_eq!(
                    slice_len_below(&ir_arena, select_node),
                    Some(expected_head_len as IdxSize),
                    "op={op:?}, n={n}, literal_on_left={literal_on_left}"
                );
            }
        }
    }

    #[test]
    fn test_no_slice_for_non_len_cmp() {
        let mut expr_arena = Arena::new();
        let mut ir_arena = Arena::new();

        // `len() + 1` is not a comparison, so nothing should be inserted.
        let len_node = expr_arena.add(AExpr::Len);
        let lit_node = expr_arena.add(AExpr::Literal(LiteralValue::from(Scalar::from(1i64))));
        let plus_node = expr_arena.add(AExpr::BinaryExpr {
            left: len_node,
            op: Operator::Plus,
            right: lit_node,
        });
        let input = leaf(&mut ir_arena);
        let select_node = ir_arena.add(IR::Select {
            input,
            expr: vec![ExprIR::from_node(plus_node, &expr_arena)],
            schema: Arc::new(Schema::default()),
            options: ProjectionOptions::default(),
        });

        insert_slice_before_len_cmp(select_node, &mut ir_arena, &expr_arena);

        assert_eq!(slice_len_below(&ir_arena, select_node), None);
    }
}
