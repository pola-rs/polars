use polars_core::chunked_array::cast::CastOptions;

use super::*;
use crate::plans::aexpr::{ExprPushdownGroup, is_inherently_nondeterministic};
use crate::plans::optimizer::EvaluateFunctionFn;

pub(crate) struct FoldTemporalConstants {
    pub evaluate_function: EvaluateFunctionFn,
}

impl OptimizationRule for FoldTemporalConstants {
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _schema: &Schema,
        _ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        let AExpr::Function {
            input, function, ..
        } = expr_arena.get(expr_node)
        else {
            return Ok(None);
        };
        let foldable = match function {
            #[cfg(feature = "offset_by")]
            IRFunctionExpr::TemporalExpr(IRTemporalFunction::OffsetBy) => true,
            #[cfg(feature = "strings")]
            IRFunctionExpr::StringExpr(IRStringFunction::Strptime(_, _)) => true,
            _ => false,
        };
        if !foldable {
            return Ok(None);
        }
        let mut columns = Vec::with_capacity(input.len());
        for expr in input {
            let AExpr::Literal(LiteralValue::Scalar(value)) = expr_arena.get(expr.node()) else {
                return Ok(None);
            };
            columns.push(value.clone().into_column(PlSmallStr::EMPTY));
        }
        // Keep errors at execution time, including invalid calendar offsets.
        let Ok(result) = (self.evaluate_function)(function.clone(), &mut columns) else {
            return Ok(None);
        };
        if result.len() != 1 {
            return Ok(None);
        }
        let scalar = Scalar::new(result.dtype().clone(), result.get(0)?.into_static());
        Ok(Some(AExpr::Literal(scalar.into())))
    }
}

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

/// Visit each filter once; shared inputs are left unchanged when rebuilding a chain.
pub(crate) fn narrow_date_filters(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let mut stack = vec![root];
    let mut visited = PlHashSet::default();
    while let Some(node) = stack.pop() {
        if visited.contains(&node) {
            continue;
        }
        let mut input = node;
        let mut conjuncts = Vec::new();
        while let IR::Filter {
            input: next,
            predicate,
        } = lp_arena.get(input)
        {
            if visited.contains(&input)
                || !matches!(
                    ExprPushdownGroup::Pushable.update_with_expr_rec(
                        expr_arena.get(predicate.node()),
                        expr_arena,
                        None,
                    ),
                    ExprPushdownGroup::Pushable
                )
                || is_inherently_nondeterministic(predicate.node(), expr_arena)
            {
                break;
            }
            visited.insert(input);
            conjuncts.extend(MintermIter::new(predicate.node(), expr_arena));
            input = *next;
        }
        if input == node {
            visited.insert(node);
            lp_arena.get(node).copy_inputs(&mut stack);
            continue;
        }
        stack.push(input);
        let schema = lp_arena.get(input).schema(lp_arena);
        let bounds: Vec<_> = conjuncts
            .iter()
            .enumerate()
            .filter_map(|(i, &node)| Bound::new(i, node, &schema, expr_arena))
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
            // Preserve separate filters for predicate pushdown without mutating shared inputs.
            let mut conjuncts = conjuncts.into_iter();
            let last = conjuncts.next_back().unwrap();
            for predicate in conjuncts {
                input = lp_arena.add(IR::Filter {
                    input,
                    predicate: ExprIR::from_node(predicate, expr_arena),
                });
            }
            lp_arena.replace(
                node,
                IR::Filter {
                    input,
                    predicate: ExprIR::from_node(last, expr_arena),
                },
            );
        }
    }
}
