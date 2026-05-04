use super::stack_opt::OptimizeExprContext;
use super::*;

pub struct FusedArithmetic {}

fn get_expr(input: &[Node], op: FusedOperator, expr_arena: &Arena<AExpr>) -> AExpr {
    let input = input
        .iter()
        .copied()
        .map(|n| ExprIR::from_node(n, expr_arena))
        .collect();
    let mut options =
        FunctionOptions::elementwise().with_casting_rules(CastingRules::cast_to_supertypes());
    // order of operations change because of FMA
    // so we must toggle this check off
    // it is still safe as it is a trusted operation
    unsafe { options.no_check_lengths() }
    AExpr::Function {
        input,
        function: IRFunctionExpr::Fused(op),
        options,
    }
}

fn check_eligible(
    nodes: &[Node],
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<bool> {
    // Exclude scalars for now as these will not benefit from fused operations downstream #9857
    // This optimization would also interfere with the `col -> lit` type-coercion rules
    // And it might also interfere with constant folding which is a more suitable optimizations here
    for node in nodes {
        let field = expr_arena
            .get(*node)
            .to_field(&ToFieldContext::new(expr_arena, schema))?;
        if !field.dtype.is_primitive_numeric() || is_scalar_ae(*node, expr_arena) {
            return Ok(false);
        }
    }

    Ok(true)
}

impl OptimizationRule for FusedArithmetic {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        schema: &Schema,
        ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        // We don't want to fuse arithmetic that we send to pyarrow.
        if ctx.in_pyarrow_scan || ctx.in_io_plugin {
            return Ok(None);
        }

        let expr = expr_arena.get(expr_node);

        use AExpr::*;
        match expr {
            BinaryExpr {
                left,
                op: outer_op @ (Operator::Plus | Operator::Minus),
                right,
            } => {
                let (a, b, other, mul_is_left) = if let BinaryExpr {
                    left: a,
                    op: Operator::Multiply,
                    right: b,
                } = expr_arena.get(*left)
                {
                    (*a, *b, *right, true)
                } else if let BinaryExpr {
                    left: a,
                    op: Operator::Multiply,
                    right: b,
                } = expr_arena.get(*right)
                {
                    (*a, *b, *left, false)
                } else {
                    return Ok(None);
                };

                if !check_eligible(&[a, b], expr_arena, schema)? {
                    return Ok(None);
                }

                let (input, fused_op) = match (outer_op, mul_is_left) {
                    (Operator::Plus, _) => ([a, b, other], FusedOperator::MultiplyAdd),
                    (Operator::Minus, true) => ([a, b, other], FusedOperator::MultiplySub),
                    (Operator::Minus, false) => ([other, a, b], FusedOperator::SubMultiply),
                    _ => unreachable!(),
                };

                Ok(Some(get_expr(&input, fused_op, expr_arena)))
            },
            _ => Ok(None),
        }
    }
}
