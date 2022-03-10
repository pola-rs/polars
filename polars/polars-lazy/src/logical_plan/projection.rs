//! this contains code used for rewriting projections, expanding wildcards, regex selection etc.
use super::*;
use crate::utils::has_nth;
use polars_arrow::index::IndexToUsize;

/// This replace the wilcard Expr with a Column Expr. It also removes the Exclude Expr from the
/// expression chain.
pub(super) fn replace_wildcard_with_column(mut expr: Expr, column_name: Arc<str>) -> Expr {
    expr.mutate().apply(|e| {
        match &e {
            Expr::Wildcard => {
                *e = Expr::Column(column_name.clone());
            }
            Expr::Exclude(input, _) => {
                *e = replace_wildcard_with_column(*input.clone(), column_name.clone());
            }
            _ => {}
        }
        // always keep iterating all inputs
        true
    });
    expr
}

fn rewrite_special_aliases(expr: Expr) -> Expr {
    // the blocks are added by cargo fmt
    #[allow(clippy::blocks_in_if_conditions)]
    if has_expr(&expr, |e| {
        matches!(e, Expr::KeepName(_) | Expr::RenameAlias { .. })
    }) {
        match expr {
            Expr::KeepName(expr) => {
                let roots = expr_to_root_column_names(&expr);
                let name = roots
                    .get(0)
                    .expect("expected root column to keep expression name");
                Expr::Alias(expr, name.clone())
            }
            Expr::RenameAlias { expr, function } => {
                let name = get_single_root(&expr).unwrap();
                let name = function.call(&name);
                Expr::Alias(expr, Arc::from(name))
            }
            _ => panic!("`keep_name`, `suffix`, `prefix` should be last expression"),
        }
    } else {
        expr
    }
}

/// Take an expression with a root: col("*") and copies that expression for all columns in the schema,
/// with the exclusion of the `names` in the exclude expression.
/// The resulting expressions are written to result.
fn replace_wilcard(expr: &Expr, result: &mut Vec<Expr>, exclude: &[Arc<str>], schema: &Schema) {
    for name in schema.iter_names() {
        if !exclude.iter().any(|exluded| &**exluded == name) {
            let new_expr = replace_wildcard_with_column(expr.clone(), Arc::from(name.as_str()));
            let new_expr = rewrite_special_aliases(new_expr);
            result.push(new_expr)
        }
    }
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
fn expand_regex(expr: &Expr, result: &mut Vec<Expr>, schema: &Schema, pattern: &str) {
    let re = regex::Regex::new(pattern)
        .unwrap_or_else(|_| panic!("invalid regular expression in column: {}", pattern));
    for name in schema.iter_names() {
        if re.is_match(name) {
            let mut new_expr = expr.clone();

            new_expr.mutate().apply(|e| match &e {
                Expr::Column(_) => {
                    *e = Expr::Column(Arc::from(name.as_str()));
                    false
                }
                _ => true,
            });

            let new_expr = rewrite_special_aliases(new_expr);
            result.push(new_expr)
        }
    }
}

#[cfg(feature = "regex")]
/// This function searches for a regex expression in `col("..")` and expands the columns
/// that are selected by that regex in `result`. The regex should start with `^` and end with `$`.
fn replace_regex(expr: &Expr, result: &mut Vec<Expr>, schema: &Schema) {
    let roots = expr_to_root_column_names(expr);
    // only in simple expression (no binary expression)
    // we pattern match regex columns
    if roots.len() == 1 {
        let name = &*roots[0];
        if name.starts_with('^') && name.ends_with('$') {
            expand_regex(expr, result, schema, name)
        } else {
            let expr = rewrite_special_aliases(expr.clone());
            result.push(expr)
        }
    } else {
        let expr = rewrite_special_aliases(expr.clone());
        result.push(expr)
    }
}

/// replace `columns(["A", "B"])..` with `col("A")..`, `col("B")..`
fn expand_columns(expr: &Expr, result: &mut Vec<Expr>, names: &[String]) {
    for name in names {
        let mut new_expr = expr.clone();
        new_expr.mutate().apply(|e| {
            if let Expr::Columns(_) = &e {
                *e = Expr::Column(Arc::from(name.as_str()));
            }
            // always keep iterating all inputs
            true
        });

        let new_expr = rewrite_special_aliases(new_expr);
        result.push(new_expr)
    }
}

/// replace `DtypeColumn` with `col("foo")..col("bar")`
fn expand_dtypes(expr: &Expr, result: &mut Vec<Expr>, schema: &Schema, dtypes: &[DataType]) {
    for dtype in dtypes {
        for field in schema.iter_fields().filter(|f| f.data_type() == dtype) {
            let name = field.name();

            let mut new_expr = expr.clone();
            new_expr.mutate().apply(|e| {
                if let Expr::DtypeColumn(_) = &e {
                    *e = Expr::Column(Arc::from(name.as_str()));
                }
                // always keep iterating all inputs
                true
            });

            let new_expr = rewrite_special_aliases(new_expr);
            result.push(new_expr)
        }
    }
}

// schema is not used if regex not activated
#[allow(unused_variables)]
fn prepare_excluded(expr: &Expr, schema: &Schema, keys: &[Expr]) -> Vec<Arc<str>> {
    let mut exclude = vec![];
    expr.into_iter().for_each(|e| {
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
                            replace_regex(&e, &mut buf, schema);
                            for col in buf.drain(..) {
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
    });
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
    exclude
}

/// In case of single col(*) -> do nothing, no selection is the same as select all
/// In other cases replace the wildcard with an expression with all columns
pub(crate) fn rewrite_projections(exprs: Vec<Expr>, schema: &Schema, keys: &[Expr]) -> Vec<Expr> {
    let mut result = Vec::with_capacity(exprs.len() + schema.len());

    for mut expr in exprs {
        // has multiple column names
        if let Some(e) = expr
            .into_iter()
            .find(|e| matches!(e, Expr::Columns(_) | Expr::DtypeColumn(_)))
        {
            if let Expr::Columns(names) = e {
                expand_columns(&expr, &mut result, names)
            } else if let Expr::DtypeColumn(dtypes) = e {
                expand_dtypes(&expr, &mut result, schema, dtypes)
            }
            continue;
        }

        if has_nth(&expr) {
            replace_nth(&mut expr, schema);
        }

        if has_wildcard(&expr) {
            // keep track of column excluded from the wildcard
            let exclude = prepare_excluded(&expr, schema, keys);
            // this path prepares the wildcard as input for the Function Expr
            if has_expr(
                &expr,
                |e| matches!(e, Expr::Function { options,  .. } if options.input_wildcard_expansion),
            ) {
                expr.mutate().apply(|e| {
                    match e {
                        Expr::Function { input, options, .. }
                            if options.input_wildcard_expansion =>
                        {
                            let mut new_inputs = Vec::with_capacity(input.len());

                            input.iter_mut().for_each(|e| {
                                if has_wildcard(e) {
                                    replace_wilcard(e, &mut new_inputs, &exclude, schema)
                                } else {
                                    #[cfg(feature = "regex")]
                                    {
                                        replace_regex(e, &mut new_inputs, schema)
                                    }
                                    #[cfg(not(feature = "regex"))]
                                    {
                                        new_inputs.push(e.clone())
                                    }
                                };
                            });

                            *input = new_inputs;
                            // continue there can be more functions that require expansion
                            true
                        }
                        _ => true,
                    }
                });
                result.push(expr);
                continue;
            }
            replace_wilcard(&expr, &mut result, &exclude, schema);
        } else {
            #[allow(clippy::collapsible_else_if)]
            #[cfg(feature = "regex")]
            {
                replace_regex(&expr, &mut result, schema)
            }
            #[cfg(not(feature = "regex"))]
            {
                let expr = rewrite_special_aliases(expr);
                result.push(expr)
            }
        };
    }
    result
}
