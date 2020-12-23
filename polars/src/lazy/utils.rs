use crate::lazy::logical_plan::Context;
use crate::{lazy::prelude::*, prelude::*};
use ahash::RandomState;
use std::collections::HashSet;
use std::sync::Arc;

/// A pushed down projection can create an alias, create a new expr only containing the new names.
pub(crate) fn projected_name(expr: &Expr) -> Result<Expr> {
    match expr {
        Expr::Column(name) => Ok(Expr::Column(name.clone())),
        Expr::Alias(_, name) => Ok(Expr::Column(name.clone())),
        Expr::Sort { expr, .. } => projected_name(expr),
        Expr::Agg(agg) => {
            use AggExpr::*;
            match agg {
                First(expr) => projected_name(expr),
                Last(expr) => projected_name(expr),
                Quantile { expr, .. } => projected_name(expr),
                Sum(expr) => projected_name(expr),
                Min(expr) => projected_name(expr),
                List(expr) => projected_name(expr),
                Max(expr) => projected_name(expr),
                Median(expr) => projected_name(expr),
                Mean(expr) => projected_name(expr),
                Count(expr) => projected_name(expr),
                AggGroups(expr) => projected_name(expr),
                NUnique(expr) => projected_name(expr),
            }
        }
        Expr::Cast { expr, .. } => projected_name(expr),
        Expr::Apply { input, .. } => projected_name(input),
        Expr::Shift { input, .. } => projected_name(input),
        Expr::Window { function, .. } => projected_name(function),
        Expr::Ternary { predicate, .. } => projected_name(predicate),
        a => Err(PolarsError::Other(
            format!(
                "No root column name could be found for expr {:?} in projected_name utillity",
                a
            )
            .into(),
        )),
    }
}

pub(crate) fn has_aexpr(current_node: Node, arena: &Arena<AExpr>, matching_expr: &AExpr) -> bool {
    let current_expr = arena.get(current_node);

    match current_expr {
        AExpr::Window {
            function,
            partition_by,
            order_by,
        } => {
            if matches!(matching_expr, AExpr::Window {..}) {
                true
            } else {
                has_aexpr(*function, arena, matching_expr)
                    || has_aexpr(*partition_by, arena, matching_expr)
                    || order_by
                        .map(|ob| has_aexpr(ob, arena, matching_expr))
                        .unwrap_or(false)
            }
        }
        AExpr::Duplicated(node) => {
            if matches!(matching_expr, AExpr::Duplicated(_)) {
                true
            } else {
                has_aexpr(*node, arena, matching_expr)
            }
        }
        AExpr::Unique(node) => {
            if matches!(matching_expr, AExpr::Unique(_)) {
                true
            } else {
                has_aexpr(*node, arena, matching_expr)
            }
        }
        AExpr::Reverse(node) => {
            if matches!(matching_expr, AExpr::Reverse(_)) {
                true
            } else {
                has_aexpr(*node, arena, matching_expr)
            }
        }
        AExpr::Alias(node, _) => {
            if matches!(matching_expr, AExpr::Alias(_, _)) {
                true
            } else {
                has_aexpr(*node, arena, matching_expr)
            }
        }
        AExpr::Column(_) => {
            matches!(matching_expr, AExpr::Column(_))
        }
        AExpr::Literal(_) => {
            matches!(matching_expr, AExpr::Literal(_))
        }
        AExpr::BinaryExpr { left, right, .. } => {
            if matches!(matching_expr, AExpr::BinaryExpr{..}) {
                true
            } else {
                has_aexpr(*left, arena, matching_expr) | has_aexpr(*right, arena, matching_expr)
            }
        }
        AExpr::Not(e) => {
            if matches!(matching_expr, AExpr::Not(_)) {
                true
            } else {
                has_aexpr(*e, arena, matching_expr)
            }
        }
        AExpr::IsNotNull(e) => {
            if matches!(matching_expr, AExpr::IsNotNull(_)) {
                true
            } else {
                has_aexpr(*e, arena, matching_expr)
            }
        }
        AExpr::IsNull(e) => {
            if matches!(matching_expr, AExpr::IsNull(_)) {
                true
            } else {
                has_aexpr(*e, arena, matching_expr)
            }
        }
        AExpr::Cast { expr, .. } => {
            if matches!(matching_expr, AExpr::Cast{..}) {
                true
            } else {
                has_aexpr(*expr, arena, matching_expr)
            }
        }
        AExpr::Sort { expr, .. } => {
            if matches!(matching_expr, AExpr::Sort{..}) {
                true
            } else {
                has_aexpr(*expr, arena, matching_expr)
            }
        }
        AExpr::Agg(agg) => {
            if let AExpr::Agg(tmp_matching_expr) = matching_expr {
                match agg {
                    AAggExpr::Min(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Min(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::Max(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Max(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::Median(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Median(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::NUnique(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::NUnique(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::First(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::First(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::Last(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Last(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::Mean(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Mean(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::List(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::List(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::Count(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Count(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::Quantile { expr, .. } => {
                        if matches!(tmp_matching_expr, AAggExpr::Quantile{..}) {
                            true
                        } else {
                            has_aexpr(*expr, arena, matching_expr)
                        }
                    }
                    AAggExpr::Sum(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::Sum(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                    AAggExpr::AggGroups(e) => {
                        if matches!(tmp_matching_expr, AAggExpr::AggGroups(_)) {
                            true
                        } else {
                            has_aexpr(*e, arena, matching_expr)
                        }
                    }
                }
            } else {
                false
            }
        }
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            if matches!(matching_expr, AExpr::Ternary{..}) {
                true
            } else {
                has_aexpr(*predicate, arena, matching_expr)
                    | has_aexpr(*truthy, arena, matching_expr)
                    | has_aexpr(*falsy, arena, matching_expr)
            }
        }
        AExpr::Apply { input, .. } => {
            if matches!(matching_expr, AExpr::Apply{..}) {
                true
            } else {
                has_aexpr(*input, arena, matching_expr)
            }
        }
        AExpr::Shift { input, .. } => {
            if matches!(matching_expr, AExpr::Shift{..}) {
                true
            } else {
                has_aexpr(*input, arena, matching_expr)
            }
        }
        AExpr::Wildcard => {
            matches!(matching_expr, AExpr::Wildcard)
        }
    }
}

/// Can check if an expression tree has a matching_expr. This
/// requires a dummy expression to be created that will be used to patter match against.
///
/// Another option was to create a recursive macro but would increase code bloat.
pub(crate) fn has_expr(current_expr: &Expr, matching_expr: &Expr) -> bool {
    match current_expr {
        Expr::Window {
            function,
            partition_by,
            order_by,
        } => {
            if matches!(matching_expr, Expr::Window {..}) {
                true
            } else {
                has_expr(function, matching_expr)
                    || has_expr(partition_by, matching_expr)
                    || order_by
                        .as_ref()
                        .map(|ob| has_expr(&*ob, matching_expr))
                        .unwrap_or(false)
            }
        }
        Expr::Duplicated(e) => {
            if matches!(matching_expr, Expr::Duplicated(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Unique(e) => {
            if matches!(matching_expr, Expr::Unique(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Reverse(e) => {
            if matches!(matching_expr, Expr::Reverse(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Alias(e, _) => {
            if matches!(matching_expr, Expr::Alias(_, _)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Column(_) => {
            matches!(matching_expr, Expr::Column(_))
        }
        Expr::Literal(_) => {
            matches!(matching_expr, Expr::Literal(_))
        }
        Expr::BinaryExpr { left, right, .. } => {
            if matches!(matching_expr, Expr::BinaryExpr{..}) {
                true
            } else {
                has_expr(left, matching_expr) | has_expr(right, matching_expr)
            }
        }
        Expr::Not(e) => {
            if matches!(matching_expr, Expr::Not(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::IsNotNull(e) => {
            if matches!(matching_expr, Expr::IsNotNull(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::IsNull(e) => {
            if matches!(matching_expr, Expr::IsNull(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Cast { expr, .. } => {
            if matches!(matching_expr, Expr::Cast{..}) {
                true
            } else {
                has_expr(expr, matching_expr)
            }
        }
        Expr::Sort { expr, .. } => {
            if matches!(matching_expr, Expr::Sort{..}) {
                true
            } else {
                has_expr(expr, matching_expr)
            }
        }
        Expr::Agg(agg) => {
            if let Expr::Agg(tmp_matching_expr) = matching_expr {
                match agg {
                    AggExpr::Min(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Min(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::Max(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Max(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::Median(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Median(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::NUnique(e) => {
                        if matches!(tmp_matching_expr, AggExpr::NUnique(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::First(e) => {
                        if matches!(tmp_matching_expr, AggExpr::First(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::Last(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Last(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::Mean(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Mean(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::List(e) => {
                        if matches!(tmp_matching_expr, AggExpr::List(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::Count(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Count(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::Quantile { expr, .. } => {
                        if matches!(tmp_matching_expr, AggExpr::Quantile{..}) {
                            true
                        } else {
                            has_expr(expr, matching_expr)
                        }
                    }
                    AggExpr::Sum(e) => {
                        if matches!(tmp_matching_expr, AggExpr::Sum(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                    AggExpr::AggGroups(e) => {
                        if matches!(tmp_matching_expr, AggExpr::AggGroups(_)) {
                            true
                        } else {
                            has_expr(e, matching_expr)
                        }
                    }
                }
            } else {
                false
            }
        }
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            if matches!(matching_expr, Expr::Ternary{..}) {
                true
            } else {
                has_expr(predicate, matching_expr)
                    | has_expr(truthy, matching_expr)
                    | has_expr(falsy, matching_expr)
            }
        }
        Expr::Apply { input, .. } => {
            if matches!(matching_expr, Expr::Apply{..}) {
                true
            } else {
                has_expr(input, matching_expr)
            }
        }
        Expr::Shift { input, .. } => {
            if matches!(matching_expr, Expr::Shift{..}) {
                true
            } else {
                has_expr(input, matching_expr)
            }
        }
        Expr::Wildcard => {
            matches!(matching_expr, Expr::Wildcard)
        }
    }
}

/// output name of expr
pub(crate) fn output_name(expr: &Expr) -> Result<Arc<String>> {
    match expr {
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias(_, name) => Ok(name.clone()),
        Expr::Sort { expr, .. } => output_name(expr),
        Expr::Cast { expr, .. } => output_name(expr),
        Expr::BinaryExpr { left, right, .. } => {
            let left = output_name(left);
            let right = output_name(right);

            match (left, right) {
                (Ok(_), Ok(_)) => Err(PolarsError::Other(
                    "could not determine output name between two root columns".into(),
                )),
                (Ok(left), _) => Ok(left),
                (_, Ok(right)) => Ok(right),
                _ => panic!("no output name found for any expression?"),
            }
        }
        Expr::Ternary { truthy, .. } => output_name(truthy),
        Expr::Window { function, .. } => output_name(function),
        a => Err(PolarsError::Other(
            format!(
                "No root column name could be found for expr {:?} in output name utillity",
                a
            )
            .into(),
        )),
    }
}

pub(crate) fn projected_names(exprs: &[Expr]) -> Result<Vec<Expr>> {
    exprs.iter().map(|expr| projected_name(expr)).collect()
}

pub(crate) fn rename_field(field: &Field, name: &str) -> Field {
    Field::new(name, field.data_type().clone(), field.is_nullable())
}

/// This should gradually replace expr_to_root_column as this will get all names in the tree.
pub(crate) fn expr_to_root_column_names(expr: &Expr) -> Vec<Arc<String>> {
    expr_to_root_column_exprs(expr)
        .into_iter()
        .map(|e| expr_to_root_column_name(&e).unwrap())
        .collect()
}

/// unpack alias(col) to name of the root column name
// TODO! reuse expr_to_root_column_expr
pub(crate) fn expr_to_root_column_name(expr: &Expr) -> Result<Arc<String>> {
    match expr {
        Expr::Duplicated(expr) => expr_to_root_column_name(expr),
        Expr::Unique(expr) => expr_to_root_column_name(expr),
        Expr::Reverse(expr) => expr_to_root_column_name(expr),
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias(expr, _) => expr_to_root_column_name(expr),
        Expr::Not(expr) => expr_to_root_column_name(expr),
        Expr::IsNull(expr) => expr_to_root_column_name(expr),
        Expr::IsNotNull(expr) => expr_to_root_column_name(expr),
        Expr::BinaryExpr { left, right, .. } => {
            let left_result = expr_to_root_column_name(left);
            let right_result = expr_to_root_column_name(right);

            let err = || {
                Err(PolarsError::Other(
                    format!(
                        "cannot find root column name for binary expression {:?}, {:?}",
                        left_result, right_result
                    )
                    .into(),
                ))
            };

            match (&left_result, &right_result) {
                (Ok(left), Err(_)) => match &**right {
                    Expr::BinaryExpr { .. } => err(),
                    _ => Ok(left.clone()),
                },
                (Err(_), Ok(right)) => match &**left {
                    Expr::BinaryExpr { .. } => err(),
                    _ => Ok(right.clone()),
                },
                _ => err(),
            }
        }
        Expr::Sort { expr, .. } => expr_to_root_column_name(expr),
        Expr::Agg(agg) => match agg {
            AggExpr::First(expr) => expr_to_root_column_name(expr),
            AggExpr::Last(expr) => expr_to_root_column_name(expr),
            AggExpr::AggGroups(expr) => expr_to_root_column_name(expr),
            AggExpr::NUnique(expr) => expr_to_root_column_name(expr),
            AggExpr::Quantile { expr, .. } => expr_to_root_column_name(expr),
            AggExpr::Sum(expr) => expr_to_root_column_name(expr),
            AggExpr::Min(expr) => expr_to_root_column_name(expr),
            AggExpr::Max(expr) => expr_to_root_column_name(expr),
            AggExpr::Median(expr) => expr_to_root_column_name(expr),
            AggExpr::List(expr) => expr_to_root_column_name(expr),
            AggExpr::Mean(expr) => expr_to_root_column_name(expr),
            AggExpr::Count(expr) => expr_to_root_column_name(expr),
        },
        Expr::Cast { expr, .. } => expr_to_root_column_name(expr),
        Expr::Apply { input, .. } => expr_to_root_column_name(input),
        Expr::Shift { input, .. } => expr_to_root_column_name(input),
        Expr::Window { function, .. } => expr_to_root_column_name(function),
        Expr::Ternary { predicate, .. } => expr_to_root_column_name(predicate),
        a => Err(PolarsError::Other(
            format!("No root column name could be found for {:?}", a).into(),
        )),
    }
}

// Find the first binary expressions somewhere in the tree.
pub(crate) fn unpack_binary_exprs(expr: &Expr) -> Result<(&Expr, &Expr)> {
    match expr {
        Expr::Unique(expr) => unpack_binary_exprs(expr),
        Expr::Duplicated(expr) => unpack_binary_exprs(expr),
        Expr::Reverse(expr) => unpack_binary_exprs(expr),
        Expr::Alias(expr, _) => unpack_binary_exprs(expr),
        Expr::Not(expr) => unpack_binary_exprs(expr),
        Expr::IsNull(expr) => unpack_binary_exprs(expr),
        Expr::IsNotNull(expr) => unpack_binary_exprs(expr),
        Expr::Agg(agg) => match agg {
            AggExpr::First(expr) => unpack_binary_exprs(expr),
            AggExpr::Last(expr) => unpack_binary_exprs(expr),
            AggExpr::AggGroups(expr) => unpack_binary_exprs(expr),
            AggExpr::NUnique(expr) => unpack_binary_exprs(expr),
            AggExpr::Quantile { expr, .. } => unpack_binary_exprs(expr),
            AggExpr::Sum(expr) => unpack_binary_exprs(expr),
            AggExpr::Min(expr) => unpack_binary_exprs(expr),
            AggExpr::Max(expr) => unpack_binary_exprs(expr),
            AggExpr::Median(expr) => unpack_binary_exprs(expr),
            AggExpr::List(expr) => unpack_binary_exprs(expr),
            AggExpr::Mean(expr) => unpack_binary_exprs(expr),
            AggExpr::Count(expr) => unpack_binary_exprs(expr),
        },
        Expr::BinaryExpr { left, right, .. } => Ok((&**left, &**right)),
        Expr::Sort { expr, .. } => unpack_binary_exprs(expr),
        Expr::Shift { input, .. } => unpack_binary_exprs(input),
        Expr::Apply { input, .. } => unpack_binary_exprs(input),
        Expr::Cast { expr, .. } => unpack_binary_exprs(expr),
        a => Err(PolarsError::Other(
            format!("No binary expression could be found for {:?}", a).into(),
        )),
    }
}

/// Get all root column expressions in the expression tree.
pub(crate) fn expr_to_root_column_exprs(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Column(_) => vec![expr.clone()],
        Expr::Duplicated(expr) => expr_to_root_column_exprs(expr),
        Expr::Unique(expr) => expr_to_root_column_exprs(expr),
        Expr::Reverse(expr) => expr_to_root_column_exprs(expr),
        Expr::Alias(expr, _) => expr_to_root_column_exprs(expr),
        Expr::Not(expr) => expr_to_root_column_exprs(expr),
        Expr::IsNull(expr) => expr_to_root_column_exprs(expr),
        Expr::IsNotNull(expr) => expr_to_root_column_exprs(expr),
        Expr::Agg(agg) => match agg {
            AggExpr::First(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Last(expr) => expr_to_root_column_exprs(expr),
            AggExpr::AggGroups(expr) => expr_to_root_column_exprs(expr),
            AggExpr::NUnique(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Quantile { expr, .. } => expr_to_root_column_exprs(expr),
            AggExpr::Sum(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Min(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Max(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Median(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Mean(expr) => expr_to_root_column_exprs(expr),
            AggExpr::Count(expr) => expr_to_root_column_exprs(expr),
            AggExpr::List(expr) => expr_to_root_column_exprs(expr),
        },
        Expr::BinaryExpr { left, right, .. } => {
            let mut results = Vec::with_capacity(16);
            results.extend(expr_to_root_column_exprs(left).into_iter());
            results.extend(expr_to_root_column_exprs(right).into_iter());
            results
        }
        Expr::Sort { expr, .. } => expr_to_root_column_exprs(expr),
        Expr::Shift { input, .. } => expr_to_root_column_exprs(input),
        Expr::Apply { input, .. } => expr_to_root_column_exprs(input),
        Expr::Cast { expr, .. } => expr_to_root_column_exprs(expr),
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let mut results = Vec::with_capacity(16);
            results.extend(expr_to_root_column_exprs(predicate).into_iter());
            results.extend(expr_to_root_column_exprs(truthy).into_iter());
            results.extend(expr_to_root_column_exprs(falsy).into_iter());
            results
        }
        Expr::Window {
            function,
            partition_by,
            order_by,
        } => {
            let mut results = Vec::with_capacity(16);
            let order_by_res = order_by.as_ref().map(|ob| expr_to_root_column_exprs(&*ob));

            results.extend(expr_to_root_column_exprs(function).into_iter());
            results.extend(expr_to_root_column_exprs(partition_by).into_iter());
            if let Some(exprs) = order_by_res {
                results.extend(exprs.into_iter())
            }
            results
        }
        Expr::Wildcard => vec![Expr::Wildcard],
        Expr::Literal(_) => vec![],
    }
}

pub(crate) fn rename_expr_root_name(expr: &Expr, new_name: Arc<String>) -> Result<Expr> {
    match expr {
        Expr::Window {
            function,
            partition_by,
            order_by,
        } => {
            let function = Box::new(rename_expr_root_name(function, new_name)?);
            Ok(Expr::Window {
                function,
                partition_by: partition_by.clone(),
                order_by: order_by.clone(),
            })
        }
        Expr::Column(_) => Ok(Expr::Column(new_name)),
        Expr::Reverse(expr) => rename_expr_root_name(expr, new_name),
        Expr::Unique(expr) => rename_expr_root_name(expr, new_name),
        Expr::Duplicated(expr) => rename_expr_root_name(expr, new_name),
        Expr::Alias(expr, alias) => rename_expr_root_name(expr, new_name)
            .map(|expr| Expr::Alias(Box::new(expr), alias.clone())),
        Expr::Not(expr) => {
            rename_expr_root_name(expr, new_name).map(|expr| Expr::Not(Box::new(expr)))
        }
        Expr::IsNull(expr) => {
            rename_expr_root_name(expr, new_name).map(|expr| Expr::IsNull(Box::new(expr)))
        }
        Expr::IsNotNull(expr) => {
            rename_expr_root_name(expr, new_name).map(|expr| Expr::IsNotNull(Box::new(expr)))
        }
        Expr::BinaryExpr { left, right, op } => {
            match rename_expr_root_name(left, new_name.clone()) {
                Err(_) => rename_expr_root_name(right, new_name).map(|right| Expr::BinaryExpr {
                    left: Box::new(*left.clone()),
                    op: *op,
                    right: Box::new(right),
                }),
                Ok(expr_left) => match rename_expr_root_name(right, new_name) {
                    Ok(_) => Err(PolarsError::Other(
                        format!(
                            "cannot find root column for binary expression {:?}, {:?}",
                            left, right
                        )
                        .into(),
                    )),
                    Err(_) => Ok(Expr::BinaryExpr {
                        left: Box::new(expr_left),
                        op: *op,
                        right: Box::new(*right.clone()),
                    }),
                },
            }
        }
        Expr::Sort { expr, reverse } => {
            rename_expr_root_name(expr, new_name).map(|expr| Expr::Sort {
                expr: Box::new(expr),
                reverse: *reverse,
            })
        }
        Expr::Cast { expr, .. } => rename_expr_root_name(expr, new_name),
        Expr::Apply {
            input,
            function,
            output_type,
        } => Ok(Expr::Apply {
            input: Box::new(rename_expr_root_name(input, new_name)?),
            function: function.clone(),
            output_type: output_type.clone(),
        }),
        Expr::Shift { input, .. } => rename_expr_root_name(input, new_name),
        Expr::Ternary { predicate, .. } => rename_expr_root_name(predicate, new_name),
        a => Err(PolarsError::Other(
            format!(
                "No root column name could be found for {:?} when trying to rename",
                a
            )
            .into(),
        )),
    }
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
    logical_plan: &LogicalPlan,
    paths: &mut HashSet<String, RandomState>,
) {
    use LogicalPlan::*;
    match logical_plan {
        Slice { input, .. } => {
            agg_source_paths(input, paths);
        }
        Selection { input, .. } => {
            agg_source_paths(input, paths);
        }
        Cache { input } => {
            agg_source_paths(input, paths);
        }
        CsvScan { path, .. } => {
            paths.insert(path.clone());
        }
        #[cfg(feature = "parquet")]
        ParquetScan { path, .. } => {
            paths.insert(path.clone());
        }
        DataFrameScan { .. } => (),
        Projection { input, .. } => {
            agg_source_paths(input, paths);
        }
        LocalProjection { input, .. } => {
            agg_source_paths(input, paths);
        }
        Sort { input, .. } => {
            agg_source_paths(input, paths);
        }
        Explode { input, .. } => {
            agg_source_paths(input, paths);
        }
        Distinct { input, .. } => {
            agg_source_paths(input, paths);
        }
        Aggregate { input, .. } => {
            agg_source_paths(input, paths);
        }
        Join {
            input_left,
            input_right,
            ..
        } => {
            agg_source_paths(input_left, paths);
            agg_source_paths(input_right, paths);
        }
        HStack { input, .. } => {
            agg_source_paths(input, paths);
        }
    }
}
