use polars_core::chunked_array::cast::CastOptions;

use super::*;

struct Bound {
    index: usize,
    column: Node,
    name: PlSmallStr,
    dtype: DataType,
    op: Operator,
    date: Scalar,
}

impl Bound {
    fn new(index: usize, node: Node, schema: &Schema, arena: &Arena<AExpr>) -> Option<Self> {
        let AExpr::BinaryExpr { left, op, right } = arena.get(node) else {
            return None;
        };
        let (column, op, value) = match (arena.get(*left), arena.get(*right)) {
            (AExpr::Cast { .. }, AExpr::Literal(LiteralValue::Scalar(value))) => {
                (*left, *op, value)
            },
            (AExpr::Literal(LiteralValue::Scalar(value)), AExpr::Cast { .. }) => {
                (*right, op.swap_operands()?, value)
            },
            _ => return None,
        };
        if !matches!(
            op,
            Operator::Gt | Operator::GtEq | Operator::Lt | Operator::LtEq
        ) {
            return None;
        }
        let AExpr::Cast {
            expr: column,
            dtype,
            options: CastOptions::NonStrict,
        } = arena.get(column)
        else {
            return None;
        };
        let AExpr::Column(name) = arena.get(*column) else {
            return None;
        };
        if schema.get(name) != Some(&DataType::Date)
            || !matches!(dtype, DataType::Datetime(_, None))
            || value.dtype() != dtype
            || value.is_null()
        {
            return None;
        }
        let date = value
            .clone()
            .cast_with_options(&DataType::Date, CastOptions::Strict)
            .ok()?;
        let roundtrip = date
            .clone()
            .cast_with_options(dtype, CastOptions::Strict)
            .ok()?;
        if roundtrip != *value {
            return None;
        }
        Some(Self {
            index,
            column: *column,
            name: name.clone(),
            dtype: dtype.clone(),
            op,
            date,
        })
    }

    fn is_lower(&self) -> bool {
        matches!(self.op, Operator::Gt | Operator::GtEq)
    }
}

pub(super) fn narrow_date_filter(
    mut predicate: ExprIR,
    schema: &Schema,
    expr_arena: &mut Arena<AExpr>,
) -> ExprIR {
    if !matches!(
        ExprPushdownGroup::Pushable.update_with_expr_rec(
            expr_arena.get(predicate.node()),
            expr_arena,
            None,
        ),
        ExprPushdownGroup::Pushable
    ) {
        return predicate;
    }

    let mut conjuncts: Vec<_> = MintermIter::new(predicate.node(), expr_arena).collect();
    let bounds: Vec<_> = conjuncts
        .iter()
        .enumerate()
        .filter_map(|(i, &node)| Bound::new(i, node, schema, expr_arena))
        .collect();
    let mut directions = PlHashMap::default();
    for bound in &bounds {
        let flags = directions
            .entry((bound.name.clone(), bound.dtype.clone()))
            .or_insert(0u8);
        *flags |= if bound.is_lower() { 1 } else { 2 };
    }
    let mut changed = false;
    for bound in bounds {
        // Both finite bounds reject dates whose timestamp cast would produce NULL.
        if directions[&(bound.name, bound.dtype)] == 3 {
            let right = expr_arena.add(AExpr::Literal(bound.date.into()));
            conjuncts[bound.index] = expr_arena.add(AExpr::BinaryExpr {
                left: bound.column,
                op: bound.op,
                right,
            });
            changed = true;
        }
    }
    if changed {
        let node = conjuncts
            .into_iter()
            .reduce(|left, right| combine_by_and(left, right, expr_arena))
            .unwrap();
        predicate.set_node(node);
    }
    predicate
}
