use crate::lazy::logical_plan::Context;
use crate::{lazy::prelude::*, prelude::*};
use std::sync::Arc;

/// A pushed down projection can create an alias, create a new expr only containing the new names.
pub(crate) fn projected_name(expr: &Expr) -> Result<Expr> {
    match expr {
        Expr::Column(name) => Ok(Expr::Column(name.clone())),
        Expr::Alias(_, name) => Ok(Expr::Column(name.clone())),
        Expr::Sort { expr, .. } => projected_name(expr),
        Expr::Cast { expr, .. } => projected_name(expr),
        a => Err(PolarsError::Other(
            format!(
                "No root column name could be found for expr {:?} in projected_name utillity",
                a
            )
            .into(),
        )),
    }
}

/// Can check if an expression tree has a matching_expr. This
/// requires a dummy expression to be created that will be used to patter match against.
///
/// Another option was to create a recursive macro but would increase code bloat.
pub(crate) fn has_expr(current_expr: &Expr, matching_expr: &Expr) -> bool {
    match current_expr {
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
        Expr::Min(e) => {
            if matches!(matching_expr, Expr::Min(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Max(e) => {
            if matches!(matching_expr, Expr::Max(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Median(e) => {
            if matches!(matching_expr, Expr::Median(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::NUnique(e) => {
            if matches!(matching_expr, Expr::NUnique(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::First(e) => {
            if matches!(matching_expr, Expr::First(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Last(e) => {
            if matches!(matching_expr, Expr::Last(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Mean(e) => {
            if matches!(matching_expr, Expr::Mean(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::List(e) => {
            if matches!(matching_expr, Expr::List(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Count(e) => {
            if matches!(matching_expr, Expr::Count(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::Quantile { expr, .. } => {
            if matches!(**expr, Expr::Quantile{..}) {
                true
            } else {
                has_expr(expr, matching_expr)
            }
        }
        Expr::Sum(e) => {
            if matches!(matching_expr, Expr::Sum(_)) {
                true
            } else {
                has_expr(e, matching_expr)
            }
        }
        Expr::AggGroups(e) => {
            if matches!(matching_expr, Expr::AggGroups(_)) {
                true
            } else {
                has_expr(e, matching_expr)
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

// unpack alias(col) to name of the root column name
// TODO! reuse expr_to_root_column_expr
pub(crate) fn expr_to_root_column(expr: &Expr) -> Result<Arc<String>> {
    match expr {
        Expr::Duplicated(expr) => expr_to_root_column(expr),
        Expr::Unique(expr) => expr_to_root_column(expr),
        Expr::Reverse(expr) => expr_to_root_column(expr),
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias(expr, _) => expr_to_root_column(expr),
        Expr::Not(expr) => expr_to_root_column(expr),
        Expr::IsNull(expr) => expr_to_root_column(expr),
        Expr::IsNotNull(expr) => expr_to_root_column(expr),
        Expr::BinaryExpr { left, right, .. } => {
            let mut left = expr_to_root_column(left);
            let mut right = expr_to_root_column(right);

            match (&mut left, &mut right) {
                (Ok(left), Err(_)) => Ok(std::mem::take(left)),
                (Err(_), Ok(right)) => Ok(std::mem::take(right)),
                _ => Err(PolarsError::Other(
                    format!(
                        "cannot find root column for binary expression {:?}, {:?}",
                        left, right
                    )
                    .into(),
                )),
            }
        }
        Expr::Sort { expr, .. } => expr_to_root_column(expr),
        Expr::First(expr) => expr_to_root_column(expr),
        Expr::Last(expr) => expr_to_root_column(expr),
        Expr::AggGroups(expr) => expr_to_root_column(expr),
        Expr::NUnique(expr) => expr_to_root_column(expr),
        Expr::Quantile { expr, .. } => expr_to_root_column(expr),
        Expr::Sum(expr) => expr_to_root_column(expr),
        Expr::Min(expr) => expr_to_root_column(expr),
        Expr::Max(expr) => expr_to_root_column(expr),
        Expr::Median(expr) => expr_to_root_column(expr),
        Expr::Mean(expr) => expr_to_root_column(expr),
        Expr::Count(expr) => expr_to_root_column(expr),
        Expr::Cast { expr, .. } => expr_to_root_column(expr),
        Expr::Apply { input, .. } => expr_to_root_column(input),
        Expr::Shift { input, .. } => expr_to_root_column(input),
        Expr::Ternary { predicate, .. } => expr_to_root_column(predicate),
        a => Err(PolarsError::Other(
            format!("No root column name could be found for {:?}", a).into(),
        )),
    }
}

pub(crate) fn expressions_to_root_columns(exprs: &[Expr]) -> Result<Vec<Arc<String>>> {
    exprs.iter().map(expr_to_root_column).collect()
}
pub(crate) fn expressions_to_root_column_exprs(exprs: &[Expr]) -> Result<Vec<Expr>> {
    exprs
        .iter()
        .map(|e| match expr_to_root_column_expr(e) {
            Ok(e) => Ok(e.clone()),
            Err(e) => Err(e),
        })
        .collect()
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
        Expr::First(expr) => unpack_binary_exprs(expr),
        Expr::Last(expr) => unpack_binary_exprs(expr),
        Expr::AggGroups(expr) => unpack_binary_exprs(expr),
        Expr::NUnique(expr) => unpack_binary_exprs(expr),
        Expr::Quantile { expr, .. } => unpack_binary_exprs(expr),
        Expr::Sum(expr) => unpack_binary_exprs(expr),
        Expr::Min(expr) => unpack_binary_exprs(expr),
        Expr::Max(expr) => unpack_binary_exprs(expr),
        Expr::Median(expr) => unpack_binary_exprs(expr),
        Expr::Mean(expr) => unpack_binary_exprs(expr),
        Expr::Count(expr) => unpack_binary_exprs(expr),
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

// unpack alias(col) to name of the root column name
pub(crate) fn expr_to_root_column_expr(expr: &Expr) -> Result<&Expr> {
    match expr {
        Expr::Column(_) => Ok(expr),
        Expr::Duplicated(expr) => expr_to_root_column_expr(expr),
        Expr::Unique(expr) => expr_to_root_column_expr(expr),
        Expr::Reverse(expr) => expr_to_root_column_expr(expr),
        Expr::Alias(expr, _) => expr_to_root_column_expr(expr),
        Expr::Not(expr) => expr_to_root_column_expr(expr),
        Expr::IsNull(expr) => expr_to_root_column_expr(expr),
        Expr::IsNotNull(expr) => expr_to_root_column_expr(expr),
        Expr::First(expr) => expr_to_root_column_expr(expr),
        Expr::Last(expr) => expr_to_root_column_expr(expr),
        Expr::AggGroups(expr) => expr_to_root_column_expr(expr),
        Expr::NUnique(expr) => expr_to_root_column_expr(expr),
        Expr::Quantile { expr, .. } => expr_to_root_column_expr(expr),
        Expr::Sum(expr) => expr_to_root_column_expr(expr),
        Expr::Min(expr) => expr_to_root_column_expr(expr),
        Expr::Max(expr) => expr_to_root_column_expr(expr),
        Expr::Median(expr) => expr_to_root_column_expr(expr),
        Expr::Mean(expr) => expr_to_root_column_expr(expr),
        Expr::Count(expr) => expr_to_root_column_expr(expr),
        Expr::BinaryExpr { left, right, .. } => match expr_to_root_column_expr(left) {
            Err(_) => expr_to_root_column_expr(right),
            Ok(expr) => match expr_to_root_column_expr(right) {
                Ok(_) => Err(PolarsError::Other(
                    format!(
                        "cannot find root column expr for binary expression {:?}, {:?}",
                        left, right
                    )
                    .into(),
                )),
                Err(_) => Ok(expr),
            },
        },
        Expr::Sort { expr, .. } => expr_to_root_column_expr(expr),
        Expr::Shift { input, .. } => expr_to_root_column_expr(input),
        Expr::Apply { input, .. } => expr_to_root_column_expr(input),
        Expr::Cast { expr, .. } => expr_to_root_column_expr(expr),
        Expr::Ternary { predicate, .. } => expr_to_root_column_expr(predicate),
        Expr::Wildcard => Ok(expr),
        a => Err(PolarsError::Other(
            format!("No root column expr could be found for {:?}", a).into(),
        )),
    }
}

pub(crate) fn rename_expr_root_name(expr: &Expr, new_name: Arc<String>) -> Result<Expr> {
    match expr {
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
