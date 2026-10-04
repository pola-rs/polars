use polars_utils::pl_str::PlSmallStr;

use super::*;

pub(super) fn process_rename(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    existing: &[PlSmallStr],
    new: &[PlSmallStr],
) {
    let rename_map: PlIndexMap<PlSmallStr, PlSmallStr> =
        new.iter().cloned().zip(existing.iter().cloned()).collect();

    if !rename_map.is_empty() {
        for (_, expr_ir) in acc_predicates.iter_mut() {
            map_column_references(expr_ir, expr_arena, &rename_map);
        }
    }
}
