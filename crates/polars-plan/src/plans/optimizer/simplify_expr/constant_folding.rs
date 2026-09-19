use super::*;
use crate::plans::optimizer::EvaluateFunctionFn;

pub(crate) struct ConstantFoldingRule {
    evaluate_function: EvaluateFunctionFn,
    coerce: Option<TypeCoercionRule>,
}

impl ConstantFoldingRule {
    pub(crate) fn new(evaluate_function: EvaluateFunctionFn, type_coercion: bool) -> Self {
        Self {
            evaluate_function,
            coerce: type_coercion.then_some(TypeCoercionRule {}),
        }
    }

    fn can_fold(function: &IRFunctionExpr) -> bool {
        match function {
            #[cfg(feature = "offset_by")]
            IRFunctionExpr::TemporalExpr(IRTemporalFunction::OffsetBy) => true,
            #[cfg(feature = "strings")]
            IRFunctionExpr::StringExpr(IRStringFunction::Strptime(_, _)) => true,
            _ => false,
        }
    }
}

impl OptimizationRule for ConstantFoldingRule {
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        schema: &Schema,
        ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        if matches!(expr_arena.get(expr_node), AExpr::Cast { expr, .. } if matches!(expr_arena.get(*expr), AExpr::Literal(_)))
        {
            let Some(rule) = &mut self.coerce else {
                return Ok(None);
            };
            // Folding an argument can expose a literal cast. Keep failures at execution time.
            return Ok(rule
                .optimize_expr(expr_arena, expr_node, schema, ctx)
                .unwrap_or(None));
        }
        let AExpr::Function {
            input, function, ..
        } = expr_arena.get(expr_node)
        else {
            return Ok(None);
        };
        if !Self::can_fold(function) {
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
