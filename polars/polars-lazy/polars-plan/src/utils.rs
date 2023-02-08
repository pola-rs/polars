use std::fmt::Formatter;
use std::iter::FlatMap;
use std::sync::Arc;

use polars_core::prelude::*;

use crate::logical_plan::iterator::ArenaExprIter;
use crate::logical_plan::Context;
use crate::prelude::names::COUNT;
use crate::prelude::*;

/// Utility to write comma delimited
pub fn column_delimited(mut s: String, items: &[String]) -> String {
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
    match plan {
        #[cfg(feature = "csv-file")]
        ALogicalPlan::CsvScan { .. } => true,
        ALogicalPlan::DataFrameScan { .. } => true,
        #[cfg(feature = "parquet")]
        ALogicalPlan::ParquetScan { .. } => true,
        _ => false,
    }
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

/// Can check if an expression tree has a matching_expr. This
/// requires a dummy expression to be created that will be used to patter match against.
pub(crate) fn has_expr<F>(current_expr: &Expr, matches: F) -> bool
where
    F: Fn(&Expr) -> bool,
{
    current_expr.into_iter().any(matches)
}

/// Check if root expression is a literal
#[cfg(feature = "is_in")]
pub(crate) fn has_root_literal_expr(e: &Expr) -> bool {
    match e {
        Expr::Literal(_) => true,
        _ => {
            let roots = expr_to_root_column_exprs(e);
            roots.iter().any(|e| matches!(e, Expr::Literal(_)))
        }
    }
}

pub fn has_null(current_expr: &Expr) -> bool {
    has_expr(current_expr, |e| {
        matches!(e, Expr::Literal(LiteralValue::Null))
    })
}

/// output name of expr
pub(crate) fn expr_output_name(expr: &Expr) -> PolarsResult<Arc<str>> {
    for e in expr {
        match e {
            // don't follow the partition by branch
            Expr::Window { function, .. } => return expr_output_name(function),
            Expr::Column(name) => return Ok(name.clone()),
            Expr::Alias(_, name) => return Ok(name.clone()),
            Expr::KeepName(_) | Expr::Wildcard | Expr::RenameAlias { .. } => {
                return Err(PolarsError::ComputeError(
                    "Cannot determine an output column without a context for this expression"
                        .into(),
                ))
            }
            Expr::Columns(_) | Expr::DtypeColumn(_) => {
                return Err(PolarsError::ComputeError(
                    "This expression might produce multiple output names".into(),
                ))
            }
            Expr::Count => return Ok(Arc::from(COUNT)),
            _ => {}
        }
    }
    Err(PolarsError::ComputeError(
        format!(
            "No root column name could be found for expr '{expr:?}' when calling 'output_name'",
        )
        .into(),
    ))
}

/// This function should be used to find the name of the start of an expression
/// Normal iteration would just return the first root column it found
pub(crate) fn get_single_leaf(expr: &Expr) -> PolarsResult<Arc<str>> {
    for e in expr {
        match e {
            Expr::Filter { input, .. } => return get_single_leaf(input),
            Expr::Take { expr, .. } => return get_single_leaf(expr),
            Expr::SortBy { expr, .. } => return get_single_leaf(expr),
            Expr::Window { function, .. } => return get_single_leaf(function),
            Expr::Column(name) => return Ok(name.clone()),
            _ => {}
        }
    }
    Err(PolarsError::ComputeError(
        format!("no single leaf column found in {expr:?}").into(),
    ))
}

/// This should gradually replace expr_to_root_column as this will get all names in the tree.
pub fn expr_to_leaf_column_names(expr: &Expr) -> Vec<Arc<str>> {
    expr_to_root_column_exprs(expr)
        .into_iter()
        .map(|e| expr_to_leaf_column_name(&e).unwrap())
        .collect()
}

/// unpack alias(col) to name of the root column name
pub fn expr_to_leaf_column_name(expr: &Expr) -> PolarsResult<Arc<str>> {
    let mut roots = expr_to_root_column_exprs(expr);
    match roots.len() {
        0 => Err(PolarsError::ComputeError(
            "no root column name found".into(),
        )),
        1 => match roots.pop().unwrap() {
            Expr::Wildcard => Err(PolarsError::ComputeError(
                "wildcard has not root column name".into(),
            )),
            Expr::Column(name) => Ok(name),
            _ => {
                unreachable!();
            }
        },
        _ => Err(PolarsError::ComputeError(
            "found more than one root column name".into(),
        )),
    }
}

fn is_leaf_aexpr(ae: &AExpr) -> bool {
    matches!(ae, AExpr::Column(_) | AExpr::Wildcard)
}

#[allow(clippy::type_complexity)]
pub(crate) fn aexpr_to_leaf_nodes_iter<'a>(
    root: Node,
    arena: &'a Arena<AExpr>,
) -> FlatMap<AExprIter<'a>, Option<Node>, fn((Node, &'a AExpr)) -> Option<Node>> {
    arena.iter(root).flat_map(
        |(node, ae)| {
            if is_leaf_aexpr(ae) {
                Some(node)
            } else {
                None
            }
        },
    )
}

pub(crate) fn aexpr_to_leaf_nodes(root: Node, arena: &Arena<AExpr>) -> Vec<Node> {
    aexpr_to_leaf_nodes_iter(root, arena).collect()
}

/// Rename the roots of the expression to a single name.
/// Most of the times used with columns that have a single root.
/// In some cases we can have multiple roots.
/// For instance in predicate pushdown the predicates are combined by their root column
/// When combined they may be a binary expression with the same root columns
pub(crate) fn rename_aexpr_leaf_names(
    node: Node,
    arena: &mut Arena<AExpr>,
    new_name: Arc<str>,
) -> Node {
    // we convert to expression as we cannot easily copy the aexpr.
    let mut new_expr = node_to_expr(node, arena);
    new_expr.mutate().apply(|e| {
        if let Expr::Column(name) = e {
            *name = new_name.clone()
        }
        true
    });
    to_aexpr(new_expr, arena)
}

/// If the leaf names match `current`, the node will be replaced
/// with a renamed expression.
pub(crate) fn rename_matching_aexpr_leaf_names(
    node: Node,
    arena: &mut Arena<AExpr>,
    current: &str,
    new_name: &str,
) -> Node {
    let mut leaves = aexpr_to_leaf_nodes_iter(node, arena);

    if leaves.any(|node| matches!(arena.get(node), AExpr::Column(name) if &**name == current)) {
        // we convert to expression as we cannot easily copy the aexpr.
        let mut new_expr = node_to_expr(node, arena);
        new_expr.mutate().apply(|e| match e {
            Expr::Column(name) if &**name == current => {
                *name = Arc::from(new_name);
                true
            }
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
    let leafs = aexpr_to_leaf_nodes_iter(node, arena);

    for node in leafs {
        match arena.get(node) {
            AExpr::Column(name) if &**name == current => {
                return arena.add(AExpr::Column(Arc::from(new_name)))
            }
            _ => {}
        }
    }
    panic!("should be a root column that is renamed");
}

/// Get all root column expressions in the expression tree.
pub(crate) fn expr_to_root_column_exprs(expr: &Expr) -> Vec<Expr> {
    let mut out = vec![];
    expr.into_iter().for_each(|e| match e {
        Expr::Column(_) | Expr::Wildcard => {
            out.push(e.clone());
        }
        _ => {}
    });
    out
}

/// Take a list of expressions and a schema and determine the output schema.
pub fn expressions_to_schema(
    expr: &[Expr],
    schema: &Schema,
    ctxt: Context,
) -> PolarsResult<Schema> {
    let mut expr_arena = Arena::with_capacity(4 * expr.len());
    let fields = expr
        .iter()
        .map(|expr| expr.to_field_amortized(schema, ctxt, &mut expr_arena));
    Schema::try_from_fallible(fields)
}

pub fn aexpr_to_leaf_names_iter(
    node: Node,
    arena: &Arena<AExpr>,
) -> impl Iterator<Item = Arc<str>> + '_ {
    aexpr_to_leaf_nodes_iter(node, arena).map(|node| match arena.get(node) {
        // expecting only columns here, wildcards and dtypes should already be replaced
        AExpr::Column(name) => name.clone(),
        e => {
            panic!("{e:?} not expected")
        }
    })
}

pub fn aexpr_to_leaf_names(node: Node, arena: &Arena<AExpr>) -> Vec<Arc<str>> {
    aexpr_to_leaf_names_iter(node, arena).collect()
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
    let fields = expr
        .iter()
        .map(|expr| arena.get(*expr).to_field(schema, ctxt, arena).unwrap());
    Schema::from(fields)
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
    lp_arena: &mut Arena<ALogicalPlan>,
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
