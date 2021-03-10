use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::aexpr_to_root_nodes;
use polars_core::prelude::*;

pub(crate) fn aexpr_to_root_names(node: Node, arena: &Arena<AExpr>) -> Vec<Arc<String>> {
    aexpr_to_root_nodes(node, arena)
        .into_iter()
        .map(|node| aexpr_to_root_column_name(node, arena).unwrap())
        .collect()
}

/// unpack alias(col) to name of the root column name
pub(crate) fn aexpr_to_root_column_name(root: Node, arena: &Arena<AExpr>) -> Result<Arc<String>> {
    let mut roots = aexpr_to_root_nodes(root, arena);
    match roots.len() {
        0 => Err(PolarsError::Other("no root column name found".into())),
        1 => match arena.get(roots.pop().unwrap()) {
            AExpr::Wildcard => Err(PolarsError::Other(
                "wildcard has not root column name".into(),
            )),
            AExpr::Column(name) => Ok(name.clone()),
            _ => {
                unreachable!();
            }
        },
        _ => Err(PolarsError::Other(
            "found more than one root column name".into(),
        )),
    }
}

/// check if a selection/projection can be done on the downwards schema
pub(crate) fn check_down_node(node: Node, down_schema: &Schema, expr_arena: &Arena<AExpr>) -> bool {
    let roots = aexpr_to_root_nodes(node, expr_arena);

    match roots.is_empty() {
        true => false,
        false => roots
            .iter()
            .map(|e| {
                expr_arena
                    .get(*e)
                    .to_field(down_schema, Context::Other, expr_arena)
                    .is_ok()
            })
            .all(|b| b),
    }
}

pub(crate) fn aexprs_to_schema(
    expr: &[Node],
    schema: &Schema,
    ctxt: Context,
    arena: &Arena<AExpr>,
) -> Schema {
    let fields = expr
        .iter()
        .map(|expr| arena.get(*expr).to_field(schema, ctxt, arena))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Schema::new(fields)
}
