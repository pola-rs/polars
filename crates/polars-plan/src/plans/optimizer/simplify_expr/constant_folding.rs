use super::*;
use crate::plans::optimizer::EvaluateFunctionFn;

pub(super) fn fold_function(
    input: &[ExprIR],
    function: &IRFunctionExpr,
    expr_arena: &Arena<AExpr>,
    evaluate_function: EvaluateFunctionFn,
) -> PolarsResult<Option<AExpr>> {
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
    let Ok(result) = evaluate_function(function.clone(), &mut columns) else {
        return Ok(None);
    };
    if result.len() != 1 {
        return Ok(None);
    }
    let scalar = Scalar::new(result.dtype().clone(), result.get(0)?.into_static());
    Ok(Some(AExpr::Literal(scalar.into())))
}
