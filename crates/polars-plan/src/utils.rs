use std::fmt::Formatter;
use std::iter::FlatMap;
use std::sync::Arc;

use polars_core::prelude::*;
use smartstring::alias::String as SmartString;

use crate::logical_plan::iterator::ArenaExprIter;
use crate::logical_plan::Context;
use crate::prelude::consts::{LEN, LITERAL_NAME};
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
}

impl PushNode for Vec<Node> {
    fn push_node(&mut self, value: Node) {
        self.push(value)
    }
}

impl PushNode for [Option<Node>; 2] {
    fn push_node(&mut self, value: Node) {
        match self {
            [None, None] => self[0] = Some(value),
            [Some(_), None] => self[1] = Some(value),
            _ => panic!("cannot push more than 2 nodes"),
        }
    }
}

impl PushNode for [Option<Node>; 1] {
    fn push_node(&mut self, value: Node) {
        match self {
            [None] => self[0] = Some(value),
            _ => panic!("cannot push more than 1 node"),
        }
    }
}

pub(crate) fn is_scan(plan: &ALogicalPlan) -> bool {
    matches!(
        plan,
        ALogicalPlan::Scan { .. } | ALogicalPlan::DataFrameScan { .. }
    )
}

impl PushNode for &mut [Option<Node>] {
    fn push_node(&mut self, value: Node) {
        if self[0].is_some() {
            self[1] = Some(value)
        } else {
            self[0] = Some(value)
        }
    }
}

/// A projection that only takes a column or a column + alias.
#[cfg(feature = "meta")]
pub(crate) fn aexpr_is_simple_projection(current_node: Node, arena: &Arena<AExpr>) -> bool {
    arena
        .iter(current_node)
        .all(|(_node, e)| matches!(e, AExpr::Column(_) | AExpr::Alias(_, _)))
}

pub(crate) fn aexpr_is_elementwise(current_node: Node, arena: &Arena<AExpr>) -> bool {
    arena.iter(current_node).all(|(_node, e)| {
        use AExpr::*;
        match e {
            AnonymousFunction { options, .. } | Function { options, .. } => {
                !matches!(options.collect_groups, ApplyOptions::GroupWise)
            },
            Column(_)
            | Alias(_, _)
            | Literal(_)
            | BinaryExpr { .. }
            | Ternary { .. }
            | Cast { .. } => true,
            _ => false,
        }
    })
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
/// requires a dummy expression to be created that will be used to patter match against.
pub(crate) fn has_expr<F>(current_expr: &Expr, matches: F) -> bool
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
        Expr::Function { options: opt, .. } => opt.returns_scalar,
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
            Expr::Columns(_) | Expr::DtypeColumn(_) => polars_bail!(
                ComputeError:
                "this expression may produce multiple output names"
            ),
            Expr::Len => return Ok(Arc::from(LEN)),
            Expr::Literal(val) => {
                return match val {
                    LiteralValue::Series(s) => Ok(Arc::from(s.name())),
                    _ => Ok(Arc::from(LITERAL_NAME)),
                }
            },
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
            Expr::Len => return Ok(Arc::from(LEN)),
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

fn is_column_aexpr(ae: &AExpr) -> bool {
    matches!(ae, AExpr::Column(_) | AExpr::Wildcard)
}

#[allow(clippy::type_complexity)]
pub(crate) fn aexpr_to_column_nodes_iter<'a>(
    root: Node,
    arena: &'a Arena<AExpr>,
) -> FlatMap<AExprIter<'a>, Option<Node>, fn((Node, &'a AExpr)) -> Option<Node>> {
    arena.iter(root).flat_map(|(node, ae)| {
        if is_column_aexpr(ae) {
            Some(node)
        } else {
            None
        }
    })
}

pub(crate) fn aexpr_to_column_nodes(root: Node, arena: &Arena<AExpr>) -> Vec<Node> {
    aexpr_to_column_nodes_iter(root, arena).collect()
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

    if leaves.any(|node| matches!(arena.get(node), AExpr::Column(name) if &**name == current)) {
        // we convert to expression as we cannot easily copy the aexpr.
        let mut new_expr = node_to_expr(node, arena);
        new_expr.mutate().apply(|e| match e {
            Expr::Column(name) if &**name == current => {
                *name = Arc::from(new_name);
                true
            },
            _ => true,
        });
        to_aexpr(new_expr, arena)
    } else {
        node
    }
}

/// Rename the root of the expression from `current` to `new` and assign to new node in arena.
/// Returns `Node` on first successful rename.
pub(crate) fn aexpr_assign_renamed_leaf(
    node: Node,
    arena: &mut Arena<AExpr>,
    current: &str,
    new_name: &str,
) -> Node {
    let leafs = aexpr_to_column_nodes_iter(node, arena);

    for node in leafs {
        match arena.get(node) {
            AExpr::Column(name) if &**name == current => {
                return arena.add(AExpr::Column(Arc::from(new_name)))
            },
            _ => {},
        }
    }
    panic!("should be a root column that is renamed");
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
    aexpr_to_column_nodes_iter(node, arena).map(|node| match arena.get(node) {
        // expecting only columns here, wildcards and dtypes should already be replaced
        AExpr::Column(name) => name.clone(),
        e => {
            panic!("{e:?} not expected")
        },
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
    aexpr_to_leaf_names_iter(node, expr_arena)
        .all(|name| input_schema.index_of(name.as_ref()).is_some())
}

pub(crate) fn aexprs_to_schema(
    expr: &[Node],
    schema: &Schema,
    ctxt: Context,
    arena: &Arena<AExpr>,
) -> Schema {
    expr.iter()
        .map(|expr| arena.get(*expr).to_field(schema, ctxt, arena).unwrap())
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

pub fn combine_predicates_expr<I>(iter: I) -> Expr
where
    I: Iterator<Item = Expr>,
{
    let mut single_pred = None;
    for expr in iter {
        single_pred = match single_pred {
            None => Some(expr),
            Some(e) => Some(e.and(expr)),
        };
    }
    single_pred.expect("an empty iterator was passed")
}

pub fn expr_is_projected_upstream(
    e: &Node,
    input: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
    projected_names: &PlHashSet<Arc<str>>,
) -> bool {
    let input_schema = lp_arena.get(input).schema(lp_arena);
    // don't do projection that is not used in upstream selection
    let output_field = expr_arena
        .get(*e)
        .to_field(input_schema.as_ref(), Context::Default, expr_arena)
        .unwrap();
    let output_name = output_field.name();
    projected_names.contains(output_name.as_str())
}
