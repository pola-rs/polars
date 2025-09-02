use super::*;

/// Accumulate over multiple columns horizontally / row wise.
pub fn fold_exprs<E>(
    acc: Expr,
    f: PlanCallback<(Series, Series), Series>,
    exprs: E,
    returns_scalar: bool,
    return_dtype: Option<DataTypeExpr>,
) -> Expr
where
    E: AsRef<[Expr]>,
{
    let mut exprs_v = Vec::with_capacity(exprs.as_ref().len() + 1);
    exprs_v.push(acc);
    exprs_v.extend(exprs.as_ref().iter().cloned());

    Expr::Function {
        input: exprs_v,
        function: FunctionExpr::FoldHorizontal {
            callback: f,
            returns_scalar,
            return_dtype,
        },
    }
}

/// Analogous to [`Iterator::reduce`](std::iter::Iterator::reduce).
///
/// An accumulator is initialized to the series given by the first expression in `exprs`, and then each subsequent value
/// of the accumulator is computed from `f(acc, next_expr_series)`. If `exprs` is empty, an error is returned when
/// `collect` is called.
pub fn reduce_exprs<E>(
    f: PlanCallback<(Series, Series), Series>,
    exprs: E,
    returns_scalar: bool,
    return_dtype: Option<DataTypeExpr>,
) -> Expr
where
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref().to_vec();

    Expr::Function {
        input: exprs,
        function: FunctionExpr::ReduceHorizontal {
            callback: f,
            returns_scalar,
            return_dtype,
        },
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cum_reduce_exprs<E>(
    f: PlanCallback<(Series, Series), Series>,
    exprs: E,

    returns_scalar: bool,
    return_dtype: Option<DataTypeExpr>,
) -> Expr
where
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref().to_vec();

    Expr::Function {
        input: exprs,
        function: FunctionExpr::CumReduceHorizontal {
            callback: f,
            returns_scalar,
            return_dtype,
        },
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cum_fold_exprs<E>(
    acc: Expr,
    f: PlanCallback<(Series, Series), Series>,
    exprs: E,
    returns_scalar: bool,
    return_dtype: Option<DataTypeExpr>,
    include_init: bool,
) -> Expr
where
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref();
    let mut exprs_v = Vec::with_capacity(exprs.len());
    exprs_v.push(acc);
    exprs_v.extend(exprs.iter().cloned());

    Expr::Function {
        input: exprs_v,
        function: FunctionExpr::CumFoldHorizontal {
            callback: f,
            returns_scalar,
            return_dtype,
            include_init,
        },
    }
}

/// Create a new column with the bitwise-and of the elements in each row.
///
/// The name of the resulting column will be "all"; use [`alias`](Expr::alias) to choose a different name.
pub fn all_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    // This will be reduced to `expr & expr` during conversion to IR.
    Ok(Expr::n_ary(
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal),
        exprs,
    ))
}

/// Create a new column with the bitwise-or of the elements in each row.
///
/// The name of the resulting column will be "any"; use [`alias`](Expr::alias) to choose a different name.
pub fn any_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    // This will be reduced to `expr | expr` during conversion to IR.
    Ok(Expr::n_ary(
        FunctionExpr::Boolean(BooleanFunction::AnyHorizontal),
        exprs,
    ))
}

/// Create a new column with the maximum value per row.
///
/// The name of the resulting column will be `"max"`; use [`alias`](Expr::alias) to choose a different name.
pub fn max_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    Ok(Expr::n_ary(FunctionExpr::MaxHorizontal, exprs))
}

/// Create a new column with the minimum value per row.
///
/// The name of the resulting column will be `"min"`; use [`alias`](Expr::alias) to choose a different name.
pub fn min_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    Ok(Expr::n_ary(FunctionExpr::MinHorizontal, exprs))
}

/// Sum all values horizontally across columns.
pub fn sum_horizontal<E: AsRef<[Expr]>>(exprs: E, ignore_nulls: bool) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    Ok(Expr::n_ary(
        FunctionExpr::SumHorizontal { ignore_nulls },
        exprs,
    ))
}

/// Compute the mean of all values horizontally across columns.
pub fn mean_horizontal<E: AsRef<[Expr]>>(exprs: E, ignore_nulls: bool) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    Ok(Expr::n_ary(
        FunctionExpr::MeanHorizontal { ignore_nulls },
        exprs,
    ))
}

/// Folds the expressions from left to right keeping the first non-null values.
///
/// It is an error to provide an empty `exprs`.
pub fn coalesce(exprs: &[Expr]) -> Expr {
    Expr::n_ary(FunctionExpr::Coalesce, exprs.to_vec())
}
