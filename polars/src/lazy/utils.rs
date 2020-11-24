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

// count the number of projections down in the tree
pub(crate) fn count_downtree_projections(lp: &LogicalPlan, n: usize) -> usize {
    use LogicalPlan::*;
    match lp {
        Selection { input, .. } => count_downtree_projections(input, n),
        DataFrameOp { input, .. } => count_downtree_projections(input, n),
        Distinct { input, .. } => count_downtree_projections(input, n),
        CsvScan { .. } => n,
        DataFrameScan { .. } => n,
        Aggregate { input, .. } => count_downtree_projections(input, n),
        Join {
            input_left,
            input_right,
            ..
        } => count_downtree_projections(input_left, n) + count_downtree_projections(input_right, n),
        HStack { input, .. } => count_downtree_projections(input, n),
        Projection { input, .. } => count_downtree_projections(input, n + 1),
        LocalProjection { input, .. } => count_downtree_projections(input, n + 1),
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
        Expr::AggFirst(expr) => expr_to_root_column(expr),
        Expr::AggLast(expr) => expr_to_root_column(expr),
        Expr::AggGroups(expr) => expr_to_root_column(expr),
        Expr::AggNUnique(expr) => expr_to_root_column(expr),
        Expr::AggQuantile { expr, .. } => expr_to_root_column(expr),
        Expr::AggSum(expr) => expr_to_root_column(expr),
        Expr::AggMin(expr) => expr_to_root_column(expr),
        Expr::AggMax(expr) => expr_to_root_column(expr),
        Expr::AggMedian(expr) => expr_to_root_column(expr),
        Expr::AggMean(expr) => expr_to_root_column(expr),
        Expr::AggCount(expr) => expr_to_root_column(expr),
        Expr::Cast { expr, .. } => expr_to_root_column(expr),
        Expr::Apply { input, .. } => expr_to_root_column(input),
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
        Expr::Reverse(expr) => unpack_binary_exprs(expr),
        Expr::Alias(expr, _) => unpack_binary_exprs(expr),
        Expr::Not(expr) => unpack_binary_exprs(expr),
        Expr::IsNull(expr) => unpack_binary_exprs(expr),
        Expr::IsNotNull(expr) => unpack_binary_exprs(expr),
        Expr::AggFirst(expr) => unpack_binary_exprs(expr),
        Expr::AggLast(expr) => unpack_binary_exprs(expr),
        Expr::AggGroups(expr) => unpack_binary_exprs(expr),
        Expr::AggNUnique(expr) => unpack_binary_exprs(expr),
        Expr::AggQuantile { expr, .. } => unpack_binary_exprs(expr),
        Expr::AggSum(expr) => unpack_binary_exprs(expr),
        Expr::AggMin(expr) => unpack_binary_exprs(expr),
        Expr::AggMax(expr) => unpack_binary_exprs(expr),
        Expr::AggMedian(expr) => unpack_binary_exprs(expr),
        Expr::AggMean(expr) => unpack_binary_exprs(expr),
        Expr::AggCount(expr) => unpack_binary_exprs(expr),
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
        Expr::Reverse(expr) => expr_to_root_column_expr(expr),
        Expr::Alias(expr, _) => expr_to_root_column_expr(expr),
        Expr::Not(expr) => expr_to_root_column_expr(expr),
        Expr::IsNull(expr) => expr_to_root_column_expr(expr),
        Expr::IsNotNull(expr) => expr_to_root_column_expr(expr),
        Expr::AggFirst(expr) => expr_to_root_column_expr(expr),
        Expr::AggLast(expr) => expr_to_root_column_expr(expr),
        Expr::AggGroups(expr) => expr_to_root_column_expr(expr),
        Expr::AggNUnique(expr) => expr_to_root_column_expr(expr),
        Expr::AggQuantile { expr, .. } => expr_to_root_column_expr(expr),
        Expr::AggSum(expr) => expr_to_root_column_expr(expr),
        Expr::AggMin(expr) => expr_to_root_column_expr(expr),
        Expr::AggMax(expr) => expr_to_root_column_expr(expr),
        Expr::AggMedian(expr) => expr_to_root_column_expr(expr),
        Expr::AggMean(expr) => expr_to_root_column_expr(expr),
        Expr::AggCount(expr) => expr_to_root_column_expr(expr),
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

pub(crate) fn expressions_to_schema(expr: &[Expr], schema: &Schema) -> Schema {
    let fields = expr
        .iter()
        .map(|expr| expr.to_field(schema))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    Schema::new(fields)
}
