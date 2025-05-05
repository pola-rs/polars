use super::*;

#[cfg(feature = "dtype-struct")]
fn cum_fold_dtype() -> GetOutput {
    GetOutput::map_fields(|fields| {
        let mut st = fields[0].dtype.clone();
        for fld in &fields[1..] {
            st = get_supertype(&st, &fld.dtype).unwrap();
        }
        Ok(Field::new(
            fields[0].name.clone(),
            DataType::Struct(
                fields
                    .iter()
                    .map(|fld| Field::new(fld.name().clone(), st.clone()))
                    .collect(),
            ),
        ))
    })
}

/// Accumulate over multiple columns horizontally / row wise.
pub fn fold_exprs<F, E>(
    acc: Expr,
    f: F,
    exprs: E,
    returns_scalar: bool,
    return_dtype: Option<DataType>,
) -> Expr
where
    F: 'static + Fn(Column, Column) -> PolarsResult<Option<Column>> + Send + Sync,
    E: AsRef<[Expr]>,
{
    let mut exprs_v = Vec::with_capacity(exprs.as_ref().len() + 1);
    exprs_v.push(acc);
    exprs_v.extend(exprs.as_ref().iter().cloned());
    let exprs = exprs_v;

    let function = new_column_udf(move |columns: &mut [Column]| {
        let mut acc = columns.first().unwrap().clone();
        for c in &columns[1..] {
            if let Some(a) = f(acc.clone(), c.clone())? {
                acc = a
            }
        }
        Ok(Some(acc))
    });

    let output_type = return_dtype
        .map(GetOutput::from_type)
        .unwrap_or_else(|| GetOutput::first());

    Expr::AnonymousFunction {
        input: exprs,
        function,
        // Take the type of the accumulator.
        output_type,
        options: FunctionOptions::groupwise()
            .with_fmt_str("fold")
            .with_flags(|mut f| {
                f |= FunctionFlags::INPUT_WILDCARD_EXPANSION;
                f.set(FunctionFlags::RETURNS_SCALAR, returns_scalar);
                f
            }),
    }
}

/// Analogous to [`Iterator::reduce`](std::iter::Iterator::reduce).
///
/// An accumulator is initialized to the series given by the first expression in `exprs`, and then each subsequent value
/// of the accumulator is computed from `f(acc, next_expr_series)`. If `exprs` is empty, an error is returned when
/// `collect` is called.
pub fn reduce_exprs<F, E>(f: F, exprs: E) -> Expr
where
    F: 'static + Fn(Column, Column) -> PolarsResult<Option<Column>> + Send + Sync,
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref().to_vec();

    let function = new_column_udf(move |columns: &mut [Column]| {
        let mut c_iter = columns.iter();

        match c_iter.next() {
            Some(acc) => {
                let mut acc = acc.clone();

                for c in c_iter {
                    if let Some(a) = f(acc.clone(), c.clone())? {
                        acc = a
                    }
                }
                Ok(Some(acc))
            },
            None => Err(polars_err!(ComputeError: "`reduce` did not have any expressions to fold")),
        }
    });

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: GetOutput::super_type(),
        options: FunctionOptions::aggregation()
            .with_fmt_str("reduce")
            .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cum_reduce_exprs<F, E>(f: F, exprs: E) -> Expr
where
    F: 'static + Fn(Column, Column) -> PolarsResult<Option<Column>> + Send + Sync,
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref().to_vec();

    let function = new_column_udf(move |columns: &mut [Column]| {
        let mut c_iter = columns.iter();

        match c_iter.next() {
            Some(acc) => {
                let mut acc = acc.clone();
                let mut result = vec![acc.clone()];

                for c in c_iter {
                    let name = c.name().clone();
                    if let Some(a) = f(acc.clone(), c.clone())? {
                        acc = a;
                    }
                    acc.rename(name);
                    result.push(acc.clone());
                }

                StructChunked::from_columns(acc.name().clone(), result[0].len(), &result)
                    .map(|ca| Some(ca.into_column()))
            },
            None => Err(polars_err!(ComputeError: "`reduce` did not have any expressions to fold")),
        }
    });

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: cum_fold_dtype(),
        options: FunctionOptions::aggregation()
            .with_fmt_str("cum_reduce")
            .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cum_fold_exprs<F, E>(acc: Expr, f: F, exprs: E, include_init: bool) -> Expr
where
    F: 'static + Fn(Column, Column) -> PolarsResult<Option<Column>> + Send + Sync,
    E: AsRef<[Expr]>,
{
    let mut exprs = exprs.as_ref().to_vec();
    exprs.push(acc);

    let function = new_column_udf(move |columns: &mut [Column]| {
        let mut columns = columns.to_vec();
        let mut acc = columns.pop().unwrap();

        let mut result = vec![];
        if include_init {
            result.push(acc.clone())
        }

        for c in columns {
            let name = c.name().clone();
            if let Some(a) = f(acc.clone(), c)? {
                acc = a;
                acc.rename(name);
                result.push(acc.clone());
            }
        }

        StructChunked::from_columns(acc.name().clone(), result[0].len(), &result)
            .map(|ca| Some(ca.into_column()))
    });

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: cum_fold_dtype(),
        options: FunctionOptions::aggregation()
            .with_fmt_str("cum_fold")
            .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),
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
