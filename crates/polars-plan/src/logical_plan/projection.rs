//! this contains code used for rewriting projections, expanding wildcards, regex selection etc.
use arrow::legacy::index::IndexToUsize;
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
            },
            Expr::Exclude(input, _) => {
                *e = replace_wildcard_with_column(std::mem::take(input), column_name.clone());
            },
            _ => {},
        }
        // always keep iterating all inputs
        true
    });
    expr
}

pub fn remove_exclude(mut expr: Expr) -> Expr {
    expr.mutate().apply(|e| {
        if let Expr::Exclude(input, _) = e {
            *e = remove_exclude(std::mem::take(input));
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
            },
            Expr::RenameAlias { expr, function } => {
                let name = get_single_leaf(&expr).unwrap();
                let name = function.call(&name)?;
                Ok(Expr::Alias(expr, Arc::from(name)))
            },
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
    exclude: &PlHashSet<Arc<str>>,
    schema: &Schema,
) -> PolarsResult<()> {
    for name in schema.iter_names() {
        if !exclude.contains(name.as_str()) {
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
                },
                Some(idx) => {
                    let (name, _dtype) = schema.get_at_index(idx).unwrap();
                    *e = Expr::Column(Arc::from(&**name))
                },
            }
            true
        },
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
    exclude: &PlHashSet<Arc<str>>,
) -> PolarsResult<()> {
    let re =
        regex::Regex::new(pattern).map_err(|e| polars_err!(ComputeError: "invalid regex {}", e))?;
    for name in schema.iter_names() {
        if re.is_match(name) && !exclude.contains(name.as_str()) {
            let mut new_expr = remove_exclude(expr.clone());

            new_expr.mutate().apply(|e| match &e {
                Expr::Column(pat) if pat.as_ref() == pattern => {
                    *e = Expr::Column(Arc::from(name.as_str()));
                    true
                },
                _ => true,
            });

            let new_expr = rewrite_special_aliases(new_expr)?;
            result.push(new_expr);
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
fn replace_regex(
    expr: &Expr,
    result: &mut Vec<Expr>,
    schema: &Schema,
    exclude: &PlHashSet<Arc<str>>,
) -> PolarsResult<()> {
    let roots = expr_to_leaf_column_names(expr);
    let mut regex = None;
    for name in &roots {
        if is_regex_projection(name) {
            match regex {
                None => {
                    regex = Some(name);
                    expand_regex(expr, result, schema, name, exclude)?;
                },
                Some(r) => {
                    polars_ensure!(
                        r == name,
                        ComputeError:
                        "an expression is not allowed to have different regexes"
                    )
                },
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
fn expand_columns(
    expr: &Expr,
    result: &mut Vec<Expr>,
    names: &[String],
    schema: &Schema,
    exclude: &PlHashSet<Arc<str>>,
) -> PolarsResult<()> {
    let mut is_valid = true;
    for name in names {
        if !exclude.contains(name.as_str()) {
            let new_expr = expr.clone();
            let (new_expr, new_expr_valid) =
                replace_columns_with_column(new_expr, names, name.as_str());
            is_valid &= new_expr_valid;
            // we may have regex col in columns.
            #[allow(clippy::collapsible_else_if)]
            #[cfg(feature = "regex")]
            {
                replace_regex(&new_expr, result, schema, exclude)?;
            }
            #[cfg(not(feature = "regex"))]
            {
                let new_expr = rewrite_special_aliases(new_expr)?;
                result.push(new_expr)
            }
        }
    }
    polars_ensure!(is_valid, ComputeError: "expanding more than one `col` is not allowed");
    Ok(())
}

/// This replaces the dtypes Expr with a Column Expr. It also removes the Exclude Expr from the
/// expression chain.
pub(super) fn replace_dtype_with_column(mut expr: Expr, column_name: Arc<str>) -> Expr {
    expr.mutate().apply(|e| {
        match e {
            Expr::DtypeColumn(_) => {
                *e = Expr::Column(column_name.clone());
            },
            Expr::Exclude(input, _) => {
                *e = replace_dtype_with_column(std::mem::take(input), column_name.clone());
            },
            _ => {},
        }
        // always keep iterating all inputs
        true
    });
    expr
}

/// This replaces the columns Expr with a Column Expr. It also removes the Exclude Expr from the
/// expression chain.
pub(super) fn replace_columns_with_column(
    mut expr: Expr,
    names: &[String],
    column_name: &str,
) -> (Expr, bool) {
    let mut is_valid = true;
    expr.mutate().apply(|e| {
        match e {
            Expr::Columns(members) => {
                // `col([a, b]) + col([c, d])`
                if members == names {
                    *e = Expr::Column(Arc::from(column_name));
                } else {
                    is_valid = false;
                }
            },
            Expr::Exclude(input, _) => {
                let (new_expr, new_expr_valid) =
                    replace_columns_with_column(std::mem::take(input), names, column_name);
                *e = new_expr;
                is_valid &= new_expr_valid;
            },
            _ => {},
        }
        // always keep iterating all inputs
        true
    });
    (expr, is_valid)
}

fn dtypes_match(d1: &DataType, d2: &DataType) -> bool {
    match (d1, d2) {
        // note: allow Datetime "*" wildcard for timezones...
        (DataType::Datetime(tu_l, tz_l), DataType::Datetime(tu_r, tz_r)) => {
            tu_l == tu_r
                && (tz_l == tz_r
                    || tz_r.is_some() && (tz_l.as_deref().unwrap_or("") == "*")
                    || tz_l.is_some() && (tz_r.as_deref().unwrap_or("") == "*"))
        },
        // ...but otherwise require exact match
        _ => d1 == d2,
    }
}

/// replace `DtypeColumn` with `col("foo")..col("bar")`
fn expand_dtypes(
    expr: &Expr,
    result: &mut Vec<Expr>,
    schema: &Schema,
    dtypes: &[DataType],
    exclude: &PlHashSet<Arc<str>>,
) -> PolarsResult<()> {
    // note: we loop over the schema to guarantee that we return a stable
    // field-order, irrespective of which dtypes are filtered against
    for field in schema.iter_fields().filter(|f| {
        dtypes.iter().any(|dtype| dtypes_match(dtype, &f.dtype))
            && !exclude.contains(f.name().as_str())
    }) {
        let name = field.name();
        let new_expr = expr.clone();
        let new_expr = replace_dtype_with_column(new_expr, Arc::from(name.as_str()));
        let new_expr = rewrite_special_aliases(new_expr)?;
        result.push(new_expr)
    }
    Ok(())
}

// schema is not used if regex not activated
#[allow(unused_variables)]
fn prepare_excluded(
    expr: &Expr,
    schema: &Schema,
    keys: &[Expr],
    has_exclude: bool,
) -> PolarsResult<PlHashSet<Arc<str>>> {
    let mut exclude = PlHashSet::new();

    // explicit exclude branch
    if has_exclude {
        for e in expr {
            if let Expr::Exclude(_, to_exclude) = e {
                #[cfg(feature = "regex")]
                {
                    // instead of matching the names for regex patterns and
                    // expanding the matches in the schema we reuse the
                    // `replace_regex` func; this is a bit slower but DRY.
                    let mut buf = vec![];
                    for to_exclude_single in to_exclude {
                        match to_exclude_single {
                            Excluded::Name(name) => {
                                let e = Expr::Column(name.clone());
                                replace_regex(&e, &mut buf, schema, &Default::default())?;
                                // we cannot loop because of bchck
                                while let Some(col) = buf.pop() {
                                    if let Expr::Column(name) = col {
                                        exclude.insert(name);
                                    }
                                }
                            },
                            Excluded::Dtype(dt) => {
                                for fld in schema.iter_fields() {
                                    if dtypes_match(fld.data_type(), dt) {
                                        exclude.insert(Arc::from(fld.name().as_ref()));
                                    }
                                }
                            },
                        }
                    }
                }

                #[cfg(not(feature = "regex"))]
                {
                    for to_exclude_single in to_exclude {
                        match to_exclude_single {
                            Excluded::Name(name) => {
                                exclude.insert(name.clone());
                            },
                            Excluded::Dtype(dt) => {
                                for (name, dtype) in schema.iter() {
                                    if matches!(dtype, dt) {
                                        exclude.insert(Arc::from(name.as_str()));
                                    }
                                }
                            },
                        }
                    }
                }
            }
        }
    }

    // exclude group_by keys
    for mut expr in keys.iter() {
        // Allow a number of aliases of a column expression, still exclude column from aggregation
        loop {
            match expr {
                Expr::Column(name) => {
                    exclude.insert(name.clone());
                    break;
                },
                Expr::Alias(e, _) => {
                    expr = e;
                },
                _ => {
                    break;
                },
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
        },
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
            },
            Some(st_val) => st = get_supertype(&st_val, &dtype),
        }
    }
    st
}

#[derive(Copy, Clone)]
struct ExpansionFlags {
    multiple_columns: bool,
    has_nth: bool,
    has_wildcard: bool,
    replace_fill_null_type: bool,
    has_selector: bool,
    has_exclude: bool,
}

fn find_flags(expr: &Expr) -> ExpansionFlags {
    let mut multiple_columns = false;
    let mut has_nth = false;
    let mut has_wildcard = false;
    let mut replace_fill_null_type = false;
    let mut has_selector = false;
    let mut has_exclude = false;

    // do a single pass and collect all flags at once.
    // supertypes/modification that can be done in place are also don e in that pass
    for expr in expr {
        match expr {
            Expr::Columns(_) | Expr::DtypeColumn(_) => multiple_columns = true,
            Expr::Nth(_) => has_nth = true,
            Expr::Wildcard => has_wildcard = true,
            Expr::Selector(_) => has_selector = true,
            Expr::Function {
                function: FunctionExpr::FillNull { .. },
                ..
            } => replace_fill_null_type = true,
            Expr::Exclude(_, _) => has_exclude = true,
            _ => {},
        }
    }
    ExpansionFlags {
        multiple_columns,
        has_nth,
        has_wildcard,
        replace_fill_null_type,
        has_selector,
        has_exclude,
    }
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

        let mut flags = find_flags(&expr);
        if flags.has_selector {
            replace_selector(&mut expr, schema, keys)?;
            // the selector is replaced with Expr::Columns
            flags.multiple_columns = true;
        }

        replace_and_add_to_results(expr, flags, &mut result, schema, keys)?;

        // this is done after all expansion (wildcard, column, dtypes)
        // have been done. This will ensure the conversion to aexpr does
        // not panic because of an unexpected wildcard etc.

        // the expanded expressions are written to result, so we pick
        // them up there.
        if flags.replace_fill_null_type {
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

fn replace_and_add_to_results(
    mut expr: Expr,
    flags: ExpansionFlags,
    result: &mut Vec<Expr>,
    schema: &Schema,
    keys: &[Expr],
) -> PolarsResult<()> {
    if flags.has_nth {
        replace_nth(&mut expr, schema);
    }

    // has multiple column names
    // the expanded columns are added to the result
    if flags.multiple_columns {
        if let Some(e) = expr
            .into_iter()
            .find(|e| matches!(e, Expr::Columns(_) | Expr::DtypeColumn(_)))
        {
            match &e {
                Expr::Columns(names) => {
                    let exclude = prepare_excluded(&expr, schema, keys, flags.has_exclude)?;
                    expand_columns(&expr, result, names, schema, &exclude)?;
                },
                Expr::DtypeColumn(dtypes) => {
                    // keep track of column excluded from the dtypes
                    let exclude = prepare_excluded(&expr, schema, keys, flags.has_exclude)?;
                    expand_dtypes(&expr, result, schema, dtypes, &exclude)?
                },
                _ => {},
            }
        }
    }
    // has multiple column names due to wildcards
    else if flags.has_wildcard {
        // keep track of column excluded from the wildcard
        let exclude = prepare_excluded(&expr, schema, keys, flags.has_exclude)?;
        // this path prepares the wildcard as input for the Function Expr
        replace_wildcard(&expr, result, &exclude, schema)?;
    }
    // can have multiple column names due to a regex
    else {
        #[allow(clippy::collapsible_else_if)]
        #[cfg(feature = "regex")]
        {
            // keep track of column excluded from the dtypes
            let exclude = prepare_excluded(&expr, schema, keys, flags.has_exclude)?;
            replace_regex(&expr, result, schema, &exclude)?;
        }
        #[cfg(not(feature = "regex"))]
        {
            let expr = rewrite_special_aliases(expr)?;
            result.push(expr)
        }
    }
    Ok(())
}

fn replace_selector_inner(
    s: Selector,
    members: &mut PlIndexSet<Expr>,
    scratch: &mut Vec<Expr>,
    schema: &Schema,
    keys: &[Expr],
) -> PolarsResult<()> {
    match s {
        Selector::Root(expr) => {
            let local_flags = find_flags(&expr);
            replace_and_add_to_results(*expr, local_flags, scratch, schema, keys)?;
            members.extend(scratch.drain(..))
        },
        Selector::Add(lhs, rhs) => {
            replace_selector_inner(*lhs, members, scratch, schema, keys)?;
            let mut rhs_members: PlIndexSet<Expr> = Default::default();
            replace_selector_inner(*rhs, &mut rhs_members, scratch, schema, keys)?;
            members.extend(rhs_members)
        },
        Selector::Sub(lhs, rhs) => {
            // fill lhs
            replace_selector_inner(*lhs, members, scratch, schema, keys)?;

            // subtract rhs
            let mut rhs_members = Default::default();
            replace_selector_inner(*rhs, &mut rhs_members, scratch, schema, keys)?;

            let mut new_members = PlIndexSet::with_capacity(members.len());
            for e in members.drain(..) {
                if !rhs_members.contains(&e) {
                    new_members.insert(e);
                }
            }

            *members = new_members;
        },
        Selector::InterSect(lhs, rhs) => {
            // fill lhs
            replace_selector_inner(*lhs, members, scratch, schema, keys)?;

            // fill rhs
            let mut rhs_members = Default::default();
            replace_selector_inner(*rhs, &mut rhs_members, scratch, schema, keys)?;

            *members = members.intersection(&rhs_members).cloned().collect()
        },
    }
    Ok(())
}

fn replace_selector(expr: &mut Expr, schema: &Schema, keys: &[Expr]) -> PolarsResult<()> {
    // first pass we replace the selectors
    // with Expr::Columns
    // we expand the `to_add` columns
    // and then subtract the `to_subtract` columns
    expr.mutate().try_apply(|e| match e {
        Expr::Selector(s) => {
            let mut swapped = Selector::Root(Box::new(Expr::Wildcard));
            std::mem::swap(s, &mut swapped);

            let mut members = PlIndexSet::new();
            replace_selector_inner(swapped, &mut members, &mut vec![], schema, keys)?;

            *e = Expr::Columns(
                members
                    .into_iter()
                    .map(|e| {
                        let Expr::Column(name) = e else {
                            unreachable!()
                        };
                        name.to_string()
                    })
                    .collect(),
            );

            Ok(true)
        },
        _ => Ok(true),
    })?;
    Ok(())
}
