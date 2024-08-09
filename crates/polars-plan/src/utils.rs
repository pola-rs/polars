use std::fmt::Formatter;
use std::iter::FlatMap;

use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use smartstring::alias::String as SmartString;

use crate::constants::{get_len_name, LEN};
use crate::prelude::*;

/// Utility to write comma delimited strings
pub fn comma_delimited(mut s: String, items: &[SmartString]) -> String {
    s.push('(');
    for c in items {
        s.push_str(c);
        s.push_str(", ");
    }
    s.pop();
    s.pop();
    s.push(')');
    s
}

/// Utility to write comma delimited
pub(crate) fn fmt_column_delimited<S: AsRef<str>>(
    f: &mut Formatter<'_>,
    items: &[S],
    container_start: &str,
    container_end: &str,
) -> std::fmt::Result {
    write!(f, "{container_start}")?;
    for (i, c) in items.iter().enumerate() {
        write!(f, "{}", c.as_ref())?;
        if i != (items.len() - 1) {
            write!(f, ", ")?;
        }
    }
    write!(f, "{container_end}")
}

pub trait PushNode {
    fn push_node(&mut self, value: Node);

    fn extend_from_slice(&mut self, values: &[Node]);
}

impl PushNode for Vec<Node> {
    fn push_node(&mut self, value: Node) {
        self.push(value)
    }

    fn extend_from_slice(&mut self, values: &[Node]) {
        Vec::extend_from_slice(self, values)
    }
}

impl PushNode for UnitVec<Node> {
    fn push_node(&mut self, value: Node) {
        self.push(value)
    }

    fn extend_from_slice(&mut self, values: &[Node]) {
        UnitVec::extend(self, values.iter().copied())
    }
}

pub(crate) fn is_scan(plan: &IR) -> bool {
    matches!(plan, IR::Scan { .. } | IR::DataFrameScan { .. })
}

/// A projection that only takes a column or a column + alias.
#[cfg(feature = "meta")]
pub(crate) fn aexpr_is_simple_projection(current_node: Node, arena: &Arena<AExpr>) -> bool {
    arena
        .iter(current_node)
        .all(|(_node, e)| matches!(e, AExpr::Column(_) | AExpr::Alias(_, _)))
}

pub fn has_aexpr<F>(current_node: Node, arena: &Arena<AExpr>, matches: F) -> bool
where
    F: Fn(&AExpr) -> bool,
{
    arena.iter(current_node).any(|(_node, e)| matches(e))
}

pub fn has_aexpr_window(current_node: Node, arena: &Arena<AExpr>) -> bool {
    has_aexpr(current_node, arena, |e| matches!(e, AExpr::Window { .. }))
}

pub fn has_aexpr_literal(current_node: Node, arena: &Arena<AExpr>) -> bool {
    has_aexpr(current_node, arena, |e| matches!(e, AExpr::Literal(_)))
}

/// Can check if an expression tree has a matching_expr. This
/// requires a dummy expression to be created that will be used to pattern match against.
pub fn has_expr<F>(current_expr: &Expr, matches: F) -> bool
where
    F: Fn(&Expr) -> bool,
{
    current_expr.into_iter().any(matches)
}

/// Check if leaf expression is a literal
#[cfg(feature = "is_in")]
pub(crate) fn has_leaf_literal(e: &Expr) -> bool {
    match e {
        Expr::Literal(_) => true,
        _ => expr_to_leaf_column_exprs_iter(e).any(|e| matches!(e, Expr::Literal(_))),
    }
}
/// Check if leaf expression returns a scalar
#[cfg(feature = "is_in")]
pub(crate) fn all_return_scalar(e: &Expr) -> bool {
    match e {
        Expr::Literal(lv) => lv.projects_as_scalar(),
        Expr::Function { options: opt, .. } => opt.flags.contains(FunctionFlags::RETURNS_SCALAR),
        Expr::Agg(_) => true,
        Expr::Column(_) | Expr::Wildcard => false,
        _ => {
            let mut empty = true;
            for leaf in expr_to_leaf_column_exprs_iter(e) {
                if !all_return_scalar(leaf) {
                    return false;
                }
                empty = false;
            }
            !empty
        },
    }
}

pub fn has_null(current_expr: &Expr) -> bool {
    has_expr(current_expr, |e| {
        matches!(e, Expr::Literal(LiteralValue::Null))
    })
}

pub fn aexpr_output_name(node: Node, arena: &Arena<AExpr>) -> PolarsResult<Arc<str>> {
    for (_, ae) in arena.iter(node) {
        match ae {
            // don't follow the partition by branch
            AExpr::Window { function, .. } => return aexpr_output_name(*function, arena),
            AExpr::Column(name) => return Ok(name.clone()),
            AExpr::Alias(_, name) => return Ok(name.clone()),
            AExpr::Len => return Ok(get_len_name()),
            AExpr::Literal(val) => return Ok(val.output_column_name()),
            _ => {},
        }
    }
    let expr = node_to_expr(node, arena);
    polars_bail!(
        ComputeError:
        "unable to find root column name for expr '{expr:?}' when calling 'output_name'",
    );
}

/// output name of expr
pub fn expr_output_name(expr: &Expr) -> PolarsResult<Arc<str>> {
    for e in expr {
        match e {
            // don't follow the partition by branch
            Expr::Window { function, .. } => return expr_output_name(function),
            Expr::Column(name) => return Ok(name.clone()),
            Expr::Alias(_, name) => return Ok(name.clone()),
            Expr::KeepName(_) | Expr::Wildcard | Expr::RenameAlias { .. } => polars_bail!(
                ComputeError:
                "cannot determine output column without a context for this expression"
            ),
            Expr::Columns(_) | Expr::DtypeColumn(_) | Expr::IndexColumn(_) => polars_bail!(
                ComputeError:
                "this expression may produce multiple output names"
            ),
            Expr::Len => return Ok(get_len_name()),
            Expr::Literal(val) => return Ok(val.output_column_name()),
            _ => {},
        }
    }
    polars_bail!(
        ComputeError:
        "unable to find root column name for expr '{expr:?}' when calling 'output_name'",
    );
}

/// This function should be used to find the name of the start of an expression
/// Normal iteration would just return the first root column it found
pub(crate) fn get_single_leaf(expr: &Expr) -> PolarsResult<Arc<str>> {
    for e in expr {
        match e {
            Expr::Filter { input, .. } => return get_single_leaf(input),
            Expr::Gather { expr, .. } => return get_single_leaf(expr),
            Expr::SortBy { expr, .. } => return get_single_leaf(expr),
            Expr::Window { function, .. } => return get_single_leaf(function),
            Expr::Column(name) => return Ok(name.clone()),
            Expr::Len => return Ok(ColumnName::from(LEN)),
            _ => {},
        }
    }
    polars_bail!(
        ComputeError: "unable to find a single leaf column in expr {:?}", expr
    );
}

#[allow(clippy::type_complexity)]
pub fn expr_to_leaf_column_names_iter(expr: &Expr) -> impl Iterator<Item = Arc<str>> + '_ {
    expr_to_leaf_column_exprs_iter(expr).flat_map(|e| expr_to_leaf_column_name(e).ok())
}

/// This should gradually replace expr_to_root_column as this will get all names in the tree.
pub fn expr_to_leaf_column_names(expr: &Expr) -> Vec<Arc<str>> {
    expr_to_leaf_column_names_iter(expr).collect()
}

/// unpack alias(col) to name of the root column name
pub fn expr_to_leaf_column_name(expr: &Expr) -> PolarsResult<Arc<str>> {
    let mut leaves = expr_to_leaf_column_exprs_iter(expr).collect::<Vec<_>>();
    polars_ensure!(leaves.len() <= 1, ComputeError: "found more than one root column name");
    match leaves.pop() {
        Some(Expr::Column(name)) => Ok(name.clone()),
        Some(Expr::Wildcard) => polars_bail!(
            ComputeError: "wildcard has no root column name",
        ),
        Some(_) => unreachable!(),
        None => polars_bail!(
            ComputeError: "no root column name found",
        ),
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn aexpr_to_column_nodes_iter<'a>(
    root: Node,
    arena: &'a Arena<AExpr>,
) -> FlatMap<AExprIter<'a>, Option<ColumnNode>, fn((Node, &'a AExpr)) -> Option<ColumnNode>> {
    arena.iter(root).flat_map(|(node, ae)| {
        if matches!(ae, AExpr::Column(_)) {
            Some(ColumnNode(node))
        } else {
            None
        }
    })
}

pub fn column_node_to_name(node: ColumnNode, arena: &Arena<AExpr>) -> &Arc<str> {
    if let AExpr::Column(name) = arena.get(node.0) {
        name
    } else {
        unreachable!()
    }
}

/// If the leaf names match `current`, the node will be replaced
/// with a renamed expression.
pub(crate) fn rename_matching_aexpr_leaf_names(
    node: Node,
    arena: &mut Arena<AExpr>,
    current: &str,
    new_name: &str,
) -> Node {
    let mut leaves = aexpr_to_column_nodes_iter(node, arena);

    if leaves.any(|node| matches!(arena.get(node.0), AExpr::Column(name) if &**name == current)) {
        // we convert to expression as we cannot easily copy the aexpr.
        let mut new_expr = node_to_expr(node, arena);
        new_expr = new_expr.map_expr(|e| match e {
            Expr::Column(name) if &*name == current => Expr::Column(ColumnName::from(new_name)),
            e => e,
        });
        to_aexpr(new_expr, arena).expect("infallible")
    } else {
        node
    }
}

/// Get all leaf column expressions in the expression tree.
pub(crate) fn expr_to_leaf_column_exprs_iter(expr: &Expr) -> impl Iterator<Item = &Expr> {
    expr.into_iter().flat_map(|e| match e {
        Expr::Column(_) | Expr::Wildcard => Some(e),
        _ => None,
    })
}

/// Take a list of expressions and a schema and determine the output schema.
pub fn expressions_to_schema(
    expr: &[Expr],
    schema: &Schema,
    ctxt: Context,
) -> PolarsResult<Schema> {
    let mut expr_arena = Arena::with_capacity(4 * expr.len());
    expr.iter()
        .map(|expr| expr.to_field_amortized(schema, ctxt, &mut expr_arena))
        .collect()
}

pub fn aexpr_to_leaf_names_iter(
    node: Node,
    arena: &Arena<AExpr>,
) -> impl Iterator<Item = Arc<str>> + '_ {
    aexpr_to_column_nodes_iter(node, arena).map(|node| match arena.get(node.0) {
        AExpr::Column(name) => name.clone(),
        _ => unreachable!(),
    })
}

pub fn aexpr_to_leaf_names(node: Node, arena: &Arena<AExpr>) -> Vec<Arc<str>> {
    aexpr_to_leaf_names_iter(node, arena).collect()
}

pub fn aexpr_to_leaf_name(node: Node, arena: &Arena<AExpr>) -> Arc<str> {
    aexpr_to_leaf_names_iter(node, arena).next().unwrap()
}

/// check if a selection/projection can be done on the downwards schema
pub(crate) fn check_input_node(
    node: Node,
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
) -> bool {
    aexpr_to_leaf_names_iter(node, expr_arena).all(|name| input_schema.contains(name.as_ref()))
}

pub(crate) fn check_input_column_node(
    node: ColumnNode,
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
) -> bool {
    match expr_arena.get(node.0) {
        AExpr::Column(name) => input_schema.contains(name.as_ref()),
        // Invariant of `ColumnNode`
        _ => unreachable!(),
    }
}

pub(crate) fn aexprs_to_schema<I: IntoIterator<Item = K>, K: Into<Node>>(
    expr: I,
    schema: &Schema,
    ctxt: Context,
    arena: &Arena<AExpr>,
) -> Schema {
    expr.into_iter()
        .map(|node| {
            arena
                .get(node.into())
                .to_field(schema, ctxt, arena)
                .unwrap()
        })
        .collect()
}

pub(crate) fn expr_irs_to_schema<I: IntoIterator<Item = K>, K: AsRef<ExprIR>>(
    expr: I,
    schema: &Schema,
    ctxt: Context,
    arena: &Arena<AExpr>,
) -> Schema {
    expr.into_iter()
        .map(|e| {
            let e = e.as_ref();
            let mut field = arena.get(e.node()).to_field(schema, ctxt, arena).unwrap();

            if let Some(name) = e.get_alias() {
                field.name = name.as_ref().into()
            }
            field
        })
        .collect()
}

/// Concatenate multiple schemas into one, disallowing duplicate field names
pub fn merge_schemas(schemas: &[SchemaRef]) -> PolarsResult<Schema> {
    let schema_size = schemas.iter().map(|schema| schema.len()).sum();
    let mut merged_schema = Schema::with_capacity(schema_size);

    for schema in schemas {
        schema.iter().try_for_each(|(name, dtype)| {
            if merged_schema.with_column(name.clone(), dtype.clone()).is_none() {
                Ok(())
            } else {
                Err(polars_err!(Duplicate: "Column with name '{}' has more than one occurrence", name))
            }
        })?;
    }

    Ok(merged_schema)
}
