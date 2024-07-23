use super::*;

#[cfg(feature = "dtype-struct")]
fn cum_fold_dtype() -> GetOutput {
    GetOutput::map_fields(|fields| {
        let mut st = fields[0].dtype.clone();
        for fld in &fields[1..] {
            st = get_supertype(&st, &fld.dtype).unwrap();
        }
        Ok(Field::new(
            &fields[0].name,
            DataType::Struct(
                fields
                    .iter()
                    .map(|fld| Field::new(fld.name(), st.clone()))
                    .collect(),
            ),
        ))
    })
}

/// Accumulate over multiple columns horizontally / row wise.
pub fn fold_exprs<F, E>(acc: Expr, f: F, exprs: E) -> Expr
where
    F: 'static + Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
    E: AsRef<[Expr]>,
{
    let mut exprs = exprs.as_ref().to_vec();
    exprs.push(acc);

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut series = series.to_vec();
        let mut acc = series.pop().unwrap();

        for s in series {
            if let Some(a) = f(acc.clone(), s)? {
                acc = a
            }
        }
        Ok(Some(acc))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: GetOutput::super_type(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::RETURNS_SCALAR,
            fmt_str: "fold",
            ..Default::default()
        },
    }
}

/// Analogous to [`Iterator::reduce`](std::iter::Iterator::reduce).
///
/// An accumulator is initialized to the series given by the first expression in `exprs`, and then each subsequent value
/// of the accumulator is computed from `f(acc, next_expr_series)`. If `exprs` is empty, an error is returned when
/// `collect` is called.
pub fn reduce_exprs<F, E>(f: F, exprs: E) -> Expr
where
    F: 'static + Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref().to_vec();

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut s_iter = series.iter();

        match s_iter.next() {
            Some(acc) => {
                let mut acc = acc.clone();

                for s in s_iter {
                    if let Some(a) = f(acc.clone(), s.clone())? {
                        acc = a
                    }
                }
                Ok(Some(acc))
            },
            None => Err(polars_err!(ComputeError: "`reduce` did not have any expressions to fold")),
        }
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: GetOutput::super_type(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::RETURNS_SCALAR,
            fmt_str: "reduce",
            ..Default::default()
        },
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cum_reduce_exprs<F, E>(f: F, exprs: E) -> Expr
where
    F: 'static + Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
    E: AsRef<[Expr]>,
{
    let exprs = exprs.as_ref().to_vec();

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut s_iter = series.iter();

        match s_iter.next() {
            Some(acc) => {
                let mut acc = acc.clone();
                let mut result = vec![acc.clone()];

                for s in s_iter {
                    let name = s.name().to_string();
                    if let Some(a) = f(acc.clone(), s.clone())? {
                        acc = a;
                    }
                    acc.rename(&name);
                    result.push(acc.clone());
                }

                StructChunked::from_series(acc.name(), &result).map(|ca| Some(ca.into_series()))
            },
            None => Err(polars_err!(ComputeError: "`reduce` did not have any expressions to fold")),
        }
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: cum_fold_dtype(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::RETURNS_SCALAR,
            fmt_str: "cum_reduce",
            ..Default::default()
        },
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cum_fold_exprs<F, E>(acc: Expr, f: F, exprs: E, include_init: bool) -> Expr
where
    F: 'static + Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
    E: AsRef<[Expr]>,
{
    let mut exprs = exprs.as_ref().to_vec();
    exprs.push(acc);

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut series = series.to_vec();
        let mut acc = series.pop().unwrap();

        let mut result = vec![];
        if include_init {
            result.push(acc.clone())
        }

        for s in series {
            let name = s.name().to_string();
            if let Some(a) = f(acc.clone(), s)? {
                acc = a;
                acc.rename(&name);
                result.push(acc.clone());
            }
        }

        StructChunked::from_series(acc.name(), &result).map(|ca| Some(ca.into_series()))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: cum_fold_dtype(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::RETURNS_SCALAR,
            fmt_str: "cum_fold",
            ..Default::default()
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
    Ok(Expr::Function {
        input: exprs,
        function: FunctionExpr::Boolean(BooleanFunction::AllHorizontal),
        options: FunctionOptions {
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::ALLOW_EMPTY_INPUTS,
            ..Default::default()
        },
    })
}

/// Create a new column with the bitwise-or of the elements in each row.
///
/// The name of the resulting column will be "any"; use [`alias`](Expr::alias) to choose a different name.
pub fn any_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");
    // This will be reduced to `expr | expr` during conversion to IR.
    Ok(Expr::Function {
        input: exprs,
        function: FunctionExpr::Boolean(BooleanFunction::AnyHorizontal),
        options: FunctionOptions {
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::ALLOW_EMPTY_INPUTS,
            ..Default::default()
        },
    })
}

/// Create a new column with the maximum value per row.
///
/// The name of the resulting column will be `"max"`; use [`alias`](Expr::alias) to choose a different name.
pub fn max_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");

    Ok(Expr::Function {
        input: exprs,
        function: FunctionExpr::MaxHorizontal,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION & !FunctionFlags::RETURNS_SCALAR
                | FunctionFlags::ALLOW_RENAME,
            ..Default::default()
        },
    })
}

/// Create a new column with the minimum value per row.
///
/// The name of the resulting column will be `"min"`; use [`alias`](Expr::alias) to choose a different name.
pub fn min_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");

    Ok(Expr::Function {
        input: exprs,
        function: FunctionExpr::MinHorizontal,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION & !FunctionFlags::RETURNS_SCALAR
                | FunctionFlags::ALLOW_RENAME,
            ..Default::default()
        },
    })
}

/// Sum all values horizontally across columns.
pub fn sum_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");

    Ok(Expr::Function {
        input: exprs,
        function: FunctionExpr::SumHorizontal,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION & !FunctionFlags::RETURNS_SCALAR,
            cast_to_supertypes: None,
            ..Default::default()
        },
    })
}

/// Compute the mean of all values horizontally across columns.
pub fn mean_horizontal<E: AsRef<[Expr]>>(exprs: E) -> PolarsResult<Expr> {
    let exprs = exprs.as_ref().to_vec();
    polars_ensure!(!exprs.is_empty(), ComputeError: "cannot return empty fold because the number of output rows is unknown");

    Ok(Expr::Function {
        input: exprs,
        function: FunctionExpr::MeanHorizontal,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION & !FunctionFlags::RETURNS_SCALAR,
            cast_to_supertypes: None,
            ..Default::default()
        },
    })
}

/// Folds the expressions from left to right keeping the first non-null values.
///
/// It is an error to provide an empty `exprs`.
pub fn coalesce(exprs: &[Expr]) -> Expr {
    let input = exprs.to_vec();
    Expr::Function {
        input,
        function: FunctionExpr::Coalesce,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            cast_to_supertypes: Some(Default::default()),
            flags: FunctionFlags::default() | FunctionFlags::INPUT_WILDCARD_EXPANSION,
            ..Default::default()
        },
    }
}
