use recursive::recursive;
use super::*;


#[recursive]
pub fn is_scalar_ae(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    match expr_arena.get(node) {
        AExpr::Literal(lv) => lv.is_scalar(),
        AExpr::Function {options, input, .. } | AExpr::AnonymousFunction{options , input, ..} => {
            if options.is_elementwise() {
                input.iter().all(|e| e.is_scalar(expr_arena))
            } else {
                options.flags.contains(FunctionFlags::RETURNS_SCALAR)
            }
        },
        AExpr::BinaryExpr {left, right, ..} => {
            is_scalar_ae(*left, expr_arena) && is_scalar_ae(*right, expr_arena)
        },
        AExpr::Ternary {truthy, falsy, ..} => {
            is_scalar_ae(*truthy, expr_arena) && is_scalar_ae(*falsy, expr_arena)
        },
        AExpr::Agg(_) | AExpr::Len => true,
        AExpr::Cast{expr , ..}  | AExpr::Alias(expr, _)  => is_scalar_ae(*expr, expr_arena),
        _ => false
    }
}
