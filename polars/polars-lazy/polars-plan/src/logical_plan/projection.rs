//! this contains code used for rewriting projections, expanding wildcards, regex selection etc.
use polars_arrow::index::IndexToUsize;
use polars_core::utils::get_supertype;

use super::*;
use crate::prelude::function_expr::FunctionExpr;

/// This replace the wildcard Expr with a Column Expr. It also removes the Exclude Expr from the
/// expression chain.
pub(super) fn replace_wildcard_with_column(mut expr: Expr, column_name: Arc<str>) -> Expr {
    expr.mutate().apply(|e| {
        match e {
            Expr::Wildcard => {
                *e = Expr::Column(column_name.clone());
            }
            Expr::Exclude(input, _) => {
                *e = replace_wildcard_with_column(std::mem::take(input), column_name.clone());
            }
            _ => {}
        }
        // always keep iterating all inputs
        true
    });
    expr
}

fn rewrite_special_aliases(expr: Expr) -> PolarsResult<Expr> {
    // the blocks are added by cargo fmt
    #[allow(clippy::blocks_in_if_conditions)]
    if has_expr(&expr, |e| {
        matches!(e, Expr::KeepName(_) | Expr::RenameAlias { .. })
    }) {
        match expr {
            Expr::KeepName(expr) => {
                let roots = expr_to_leaf_column_names(&expr);
                let name = roots
                    .get(0)
                    .expect("expected root column to keep expression name");
                Ok(Expr::Alias(expr, name.clone()))
            }
            Expr::RenameAlias { expr, function } => {
                let name = get_single_leaf(&expr).unwrap();
                let name = function.call(&name)?;
                Ok(Expr::Alias(expr, Arc::from(name)))
            }
            _ => panic!("`keep_name`, `suffix`, `prefix` should be last expression"),
        }
    } else {
        Ok(expr)
    }
}

/// Take an expression with a root: col("*") and copies that expression for all columns in the schema,
/// with the exclusion of the `names` in the exclude expression.
/// The resulting expressions are written to result.
fn replace_wildcard(
    expr: &Expr,
    result: &mut Vec<Expr>,
    exclude: &[Arc<str>],
    schema: &Schema,
) -> PolarsResult<()> {
    for name in schema.iter_names() {
        if !exclude.iter().any(|excluded| &**excluded == name) {
            let new_expr = replace_wildcard_with_column(expr.clone(), Arc::from(name.as_str()));
            let new_expr = rewrite_special_aliases(new_expr)?;
            result.push(new_expr)
        }
    }
    Ok(())
}

fn replace_nth(expr: &mut Expr, schema: &Schema) {
    expr.mutate().apply(|e| match e {
        Expr::Nth(i) => {
            match i.negative_to_usize(schema.len()) {
                None => {
                    let name = if *i == 0 { "first" } else { "last" };
                    *e = Expr::Column(Arc::from(name));
                }
                Some(idx) => {
                    let (name, _dtype) = schema.get_index(idx).unwrap();
                    *e = Expr::Column(Arc::from(&**name))
                }
            }
            true
        }
        _ => true,
    })
}

#[cfg(feature = "regex")]
/// This function takes an expression containing a regex in `col("..")` and expands the columns
/// that are selected by that regex in `result`.
fn expand_regex(
    expr: &Expr,
    result: &mut Vec<Expr>,
    schema: &Schema,
    pattern: &str,
) -> PolarsResult<()> {
    let re = regex::Regex::new(pattern)
        .unwrap_or_else(|_| panic!("invalid regular expression in column: {pattern}"));
    for name in schema.iter_names() {
        if re.is_match(name) {
            let mut new_expr = expr.clone();

            new_expr.mutate().apply(|e| match &e {
                Expr::Column(pat) if pat.as_ref() == pattern => {
                    *e = Expr::Column(Arc::from(name.as_str()));
                    true
                }
                _ => true,
            });

            let new_expr = rewrite_special_aliases(new_expr)?;
            result.push(new_expr)
        }
    }
    Ok(())
}

pub(crate) fn is_regex_projection(name: &str) -> bool {
    name.starts_with('^') && name.ends_with('$')
}

#[cfg(feature = "regex")]
/// This function searches for a regex expression in `col("..")` and expands the columns
/// that are selected by that regex in `result`. The regex should start with `^` and end with `$`.
fn replace_regex(expr: &Expr, result: &mut Vec<Expr>, schema: &Schema) -> PolarsResult<()> {
    let roots = expr_to_leaf_column_names(expr);
    let mut regex = None;
    for name in &roots {
        if is_regex_projection(name) {
            match regex {
                None => {
                    regex = Some(name);
                    expand_regex(expr, result, schema, name)?
                }
                Some(r) => {
                    assert_eq!(
                        r, name,
                        "an expression is not allowed to have different regexes"
                    )
                }
            }
        }
    }
    if regex.is_none() {
        let expr = rewrite_special_aliases(expr.clone())?;
        result.push(expr)
    }
    Ok(())
}

/// replace `columns(["A", "B"])..` with `col("A")..`, `col("B")..`
fn expand_columns(expr: &Expr, result: &mut Vec<Expr>, names: &[String]) -> PolarsResult<()> {
    let mut is_valid = true;
    for name in names {
        let mut new_expr = expr.clone();
        new_expr.mutate().apply(|e| {
            if let Expr::Columns(members) = &e {
                // `col([a, b]) + col([c, d])`
                if members == names {
                    *e = Expr::Column(Arc::from(name.as_str()));
                } else {
                    is_valid = false;
                }
            }
            // always keep iterating all inputs
            true
        });

        let new_expr = rewrite_special_aliases(new_expr)?;
        result.push(new_expr)
    }
    if is_valid {
        Ok(())
    } else {
        Err(PolarsError::ComputeError(
            "Expanding more than one `col` is not yet allowed.".into(),
        ))
    }
}

/// This replaces the dtypes Expr with a Column Expr. It also removes the Exclude Expr from the
/// expression chain.
pub(super) fn replace_dtype_with_column(mut expr: Expr, column_name: Arc<str>) -> Expr {
    expr.mutate().apply(|e| {
        match e {
            Expr::DtypeColumn(_) => {
                *e = Expr::Column(column_name.clone());
            }
            Expr::Exclude(input, _) => {
                *e = replace_dtype_with_column(std::mem::take(input), column_name.clone());
            }
            _ => {}
        }
        // always keep iterating all inputs
        true
    });
    expr
}

/// replace `DtypeColumn` with `col("foo")..col("bar")`
fn expand_dtypes(
    expr: &Expr,
    result: &mut Vec<Expr>,
    schema: &Schema,
    dtypes: &[DataType],
    exclude: &[Arc<str>],
) -> PolarsResult<()> {
    // note: we loop over the schema to guarantee that we return a stable
    // field-order, irrespective of which dtypes are filtered against
    for field in schema.iter_fields().filter(|f| dtypes.contains(&f.dtype)) {
        let name = field.name();
        if exclude.iter().any(|excl| excl.as_ref() == name.as_str()) {
            continue; // skip excluded names
        }
        let new_expr = expr.clone();
        let new_expr = replace_dtype_with_column(new_expr, Arc::from(name.as_str()));
        let new_expr = rewrite_special_aliases(new_expr)?;
        result.push(new_expr)
    }
    Ok(())
}

// schema is not used if regex not activated
#[allow(unused_variables)]
fn prepare_excluded(expr: &Expr, schema: &Schema, keys: &[Expr]) -> PolarsResult<Vec<Arc<str>>> {
    let mut exclude = vec![];
    for e in expr {
        if let Expr::Exclude(_, to_exclude) = e {
            #[cfg(feature = "regex")]
            {
                // instead of matching the names for regex patterns
                // and expanding the matches in the schema we
                // reuse the `replace_regex` function. This is a bit
                // slower but DRY.
                let mut buf = vec![];
                for to_exclude_single in to_exclude {
                    match to_exclude_single {
                        Excluded::Name(name) => {
                            let e = Expr::Column(name.clone());
                            replace_regex(&e, &mut buf, schema)?;
                            // we cannot loop because of bchck
                            while let Some(col) = buf.pop() {
                                if let Expr::Column(name) = col {
                                    exclude.push(name)
                                }
                            }
                        }
                        Excluded::Dtype(dt) => {
                            for fld in schema.iter_fields() {
                                if fld.data_type() == dt {
                                    exclude.push(Arc::from(fld.name().as_ref()))
                                }
                            }
                        }
                    }
                }
            }

            #[cfg(not(feature = "regex"))]
            {
                for to_exclude_single in to_exclude {
                    match to_exclude_single {
                        Excluded::Name(name) => exclude.push(name.clone()),
                        Excluded::Dtype(dt) => {
                            for (name, dtype) in schema.iter() {
                                if matches!(dtype, dt) {
                                    exclude.push(Arc::from(name.as_str()))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for mut expr in keys.iter() {
        // Allow a number of aliases of a column expression, still exclude column from aggregation
        loop {
            match expr {
                Expr::Column(name) => {
                    exclude.push(name.clone());
                    break;
                }
                Expr::Alias(e, _) => {
                    expr = e;
                }
                _ => {
                    break;
                }
            }
        }
    }
    Ok(exclude)
}

// functions can have col(["a", "b"]) or col(Utf8) as inputs
fn expand_function_inputs(mut expr: Expr, schema: &Schema) -> Expr {
    expr.mutate().apply(|e| match e {
        Expr::AnonymousFunction { input, options, .. } | Expr::Function { input, options, .. }
            if options.input_wildcard_expansion =>
        {
            *input = rewrite_projections(input.clone(), schema, &[]).unwrap();
            // continue iteration, there might be more functions.
            true
        }
        _ => true,
    });
    expr
}

/// this is determined in type coercion
/// but checking a few types early can improve type stability (e.g. no need for unknown)
fn early_supertype(inputs: &[Expr], schema: &Schema) -> Option<DataType> {
    let mut arena = Arena::with_capacity(8);

    let mut st = None;
    for e in inputs {
        let dtype = e
            .to_field_amortized(schema, Context::Default, &mut arena)
            .ok()?
            .dtype;
        arena.clear();
        match st {
            None => {
                st = Some(dtype);
            }
            Some(st_val) => st = get_supertype(&st_val, &dtype),
        }
    }
    st
}

/// In case of single col(*) -> do nothing, no selection is the same as select all
/// In other cases replace the wildcard with an expression with all columns
pub(crate) fn rewrite_projections(
    exprs: Vec<Expr>,
    schema: &Schema,
    keys: &[Expr],
) -> PolarsResult<Vec<Expr>> {
    let mut result = Vec::with_capacity(exprs.len() + schema.len());

    for mut expr in exprs {
        let result_offset = result.len();

        // functions can have col(["a", "b"]) or col(Utf8) as inputs
        expr = expand_function_inputs(expr, schema);

        let mut multiple_columns = false;
        let mut has_nth = false;
        let mut has_wildcard = false;
        let mut replace_fill_null_type = false;

        // do a single pass and collect all flags at once.
        // supertypes/modification that can be done in place are also don e in that pass
        for expr in &expr {
            match expr {
                Expr::Columns(_) | Expr::DtypeColumn(_) => multiple_columns = true,
                Expr::Nth(_) => has_nth = true,
                Expr::Wildcard => has_wildcard = true,
                Expr::Function {
                    function: FunctionExpr::FillNull { .. },
                    ..
                } => replace_fill_null_type = true,
                _ => {}
            }
        }

        if has_nth {
            replace_nth(&mut expr, schema);
        }

        // has multiple column names
        // the expanded columns are added to the result
        if multiple_columns {
            if let Some(e) = expr
                .into_iter()
                .find(|e| matches!(e, Expr::Columns(_) | Expr::DtypeColumn(_)))
            {
                match &e {
                    Expr::Columns(names) => expand_columns(&expr, &mut result, names)?,
                    Expr::DtypeColumn(dtypes) => {
                        // keep track of column excluded from the dtypes
                        let exclude = prepare_excluded(&expr, schema, keys)?;
                        expand_dtypes(&expr, &mut result, schema, dtypes, &exclude)?
                    }
                    _ => {}
                }
            }
        }
        // has multiple column names due to wildcards
        else if has_wildcard {
            // keep track of column excluded from the wildcard
            let exclude = prepare_excluded(&expr, schema, keys)?;
            // this path prepares the wildcard as input for the Function Expr
            replace_wildcard(&expr, &mut result, &exclude, schema)?;
        }
        // can have multiple column names due to a regex
        else {
            #[allow(clippy::collapsible_else_if)]
            #[cfg(feature = "regex")]
            {
                replace_regex(&expr, &mut result, schema)?
            }
            #[cfg(not(feature = "regex"))]
            {
                let expr = rewrite_special_aliases(expr)?;
                result.push(expr)
            }
        }

        // this is done after all expansion (wildcard, column, dtypes)
        // have been done. This will ensure the conversion to aexpr does
        // not panic because of an unexpected wildcard etc.

        // the expanded expressions are written to result, so we pick
        // them up there.
        if replace_fill_null_type {
            for e in &mut result[result_offset..] {
                e.mutate().apply(|e| {
                    if let Expr::Function {
                        input,
                        function: FunctionExpr::FillNull { super_type },
                        ..
                    } = e
                    {
                        if let Some(new_st) = early_supertype(input, schema) {
                            *super_type = new_st;
                        }
                    }

                    // continue iteration
                    true
                })
            }
        }
    }
    Ok(result)
}
