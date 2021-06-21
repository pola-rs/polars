use crate::logical_plan::iterator::ArenaExprIter;
use crate::logical_plan::Context;
use crate::prelude::*;
use ahash::RandomState;
use polars_core::prelude::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "private")]
pub(crate) fn equal_aexprs(left: &[Node], right: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| AExpr::eq(*l, *r, expr_arena))
}

#[cfg(feature = "private")]
pub(crate) fn remove_duplicate_aexprs(exprs: &[Node], expr_arena: &Arena<AExpr>) -> Vec<Node> {
    let mut unique = HashSet::with_capacity_and_hasher(exprs.len(), RandomState::new());
    let mut new = Vec::with_capacity(exprs.len());
    for node in exprs {
        let mut can_insert = false;
        for name in aexpr_to_root_names(*node, expr_arena) {
            if unique.insert(name) {
                can_insert = true
            }
        }
        if can_insert {
            new.push(*node)
        }
    }
    new
}

pub(crate) trait PushNode {
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
            [Some(_)] => self[0] = Some(value),
            _ => panic!("cannot push more than 2 nodes"),
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

pub(crate) fn has_aexpr<F>(current_node: Node, arena: &Arena<AExpr>, matches: F) -> bool
where
    F: Fn(&AExpr) -> bool,
{
    arena.iter(current_node).any(|(_node, e)| matches(e))
}

/// Can check if an expression tree has a matching_expr. This
/// requires a dummy expression to be created that will be used to patter match against.
pub(crate) fn has_expr<F>(current_expr: &Expr, matches: F) -> bool
where
    F: Fn(&Expr) -> bool,
{
    current_expr.into_iter().any(|e| matches(e))
}

/// output name of expr
pub(crate) fn output_name(expr: &Expr) -> Result<Arc<String>> {
    for e in expr {
        match e {
            Expr::Column(name) => return Ok(name.clone()),
            Expr::Alias(_, name) => return Ok(name.clone()),
            _ => {}
        }
    }
    Err(PolarsError::Other(
        format!(
            "No root column name could be found for expr {:?} in output name utillity",
            expr
        )
        .into(),
    ))
}

pub(crate) fn rename_field(field: &Field, name: &str) -> Field {
    Field::new(name, field.data_type().clone())
}

/// This should gradually replace expr_to_root_column as this will get all names in the tree.
pub(crate) fn expr_to_root_column_names(expr: &Expr) -> Vec<Arc<String>> {
    expr_to_root_column_exprs(expr)
        .into_iter()
        .map(|e| expr_to_root_column_name(&e).unwrap())
        .collect()
}

/// unpack alias(col) to name of the root column name
pub(crate) fn expr_to_root_column_name(expr: &Expr) -> Result<Arc<String>> {
    let mut roots = expr_to_root_column_exprs(expr);
    match roots.len() {
        0 => Err(PolarsError::Other("no root column name found".into())),
        1 => match roots.pop().unwrap() {
            Expr::Wildcard => Err(PolarsError::Other(
                "wildcard has not root column name".into(),
            )),
            Expr::Column(name) => Ok(name),
            _ => {
                unreachable!();
            }
        },
        _ => Err(PolarsError::Other(
            "found more than one root column name".into(),
        )),
    }
}

pub(crate) fn aexpr_to_root_nodes(root: Node, arena: &Arena<AExpr>) -> Vec<Node> {
    let mut out = vec![];
    arena.iter(root).for_each(|(node, e)| match e {
        AExpr::Column(_) | AExpr::Wildcard => {
            out.push(node);
        }
        _ => {}
    });
    out
}

pub(crate) fn rename_aexpr_root_name(
    node: Node,
    arena: &mut Arena<AExpr>,
    new_name: Arc<String>,
) -> Result<()> {
    let roots = aexpr_to_root_nodes(node, arena);
    match roots.len() {
        1 => {
            let node = roots[0];
            arena.replace_with(node, |ae| match ae {
                AExpr::Column(_) => AExpr::Column(new_name),
                _ => panic!("should be only a column"),
            });
            Ok(())
        }
        _ => {
            panic!("had more than one root columns");
        }
    }
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

pub(crate) fn rename_expr_root_name(expr: &Expr, new_name: Arc<String>) -> Result<Expr> {
    let mut arena = Arena::with_capacity(32);
    let root = to_aexpr(expr.clone(), &mut arena);
    rename_aexpr_root_name(root, &mut arena, new_name)?;
    Ok(node_to_exp(root, &arena))
}

pub(crate) fn expressions_to_schema(expr: &[Expr], schema: &Schema, ctxt: Context) -> Schema {
    let fields = expr
        .iter()
        .map(|expr| expr.to_field(schema, ctxt))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Schema::new(fields)
}

/// Get a set of the data source paths in this LogicalPlan
pub(crate) fn agg_source_paths(
    root_lp: Node,
    paths: &mut HashSet<PathBuf, RandomState>,
    lp_arena: &Arena<ALogicalPlan>,
) {
    use ALogicalPlan::*;
    let logical_plan = lp_arena.get(root_lp);
    match logical_plan {
        Slice { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Selection { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Cache { input } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        #[cfg(feature = "csv-file")]
        CsvScan { path, .. } => {
            paths.insert(path.clone());
        }
        #[cfg(feature = "parquet")]
        ParquetScan { path, .. } => {
            paths.insert(path.clone());
        }
        DataFrameScan { .. } => (),
        Projection { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        LocalProjection { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Sort { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Explode { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Distinct { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Aggregate { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Join {
            input_left,
            input_right,
            ..
        } => {
            agg_source_paths(*input_left, paths, lp_arena);
            agg_source_paths(*input_right, paths, lp_arena);
        }
        HStack { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Melt { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
        Udf { input, .. } => {
            agg_source_paths(*input, paths, lp_arena);
        }
    }
}

pub(crate) fn try_path_to_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        PolarsError::Other(format!("Non-UTF8 file path: {}", path.to_string_lossy()).into())
    })
}

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
                    .to_field(down_schema, Context::Default, expr_arena)
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

pub(crate) fn combine_predicates_expr<I>(iter: I) -> Expr
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
