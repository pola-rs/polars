use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::predicates::aexpr_to_skip_batch_expr;
use super::{AExpr, FileScan, IR};
use crate::plans::{ExprIR, OutputName};

pub fn optimize(root: Node, lp_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let mut ir_stack = Vec::with_capacity(16);
    ir_stack.push(root);

    while let Some(current) = ir_stack.pop() {
        let current_ir = lp_arena.get(current);
        current_ir.copy_inputs(&mut ir_stack);
        let IR::Scan {
            scan_type: FileScan::Parquet { .. },
            predicate: Some(expr),
            file_info,
            ..
        } = current_ir
        else {
            continue;
        };

        let skip_batches_expr =
            aexpr_to_skip_batch_expr(expr.node(), expr_arena, &file_info.schema);
        if let Some(skip_batches_expr) = skip_batches_expr {
            eprintln!(
                "{}",
                ExprIR::new(skip_batches_expr, OutputName::Alias(PlSmallStr::EMPTY))
                    .display(expr_arena)
            );
        }
    }
}
