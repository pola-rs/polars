use super::*;
use crate::logical_plan::expr_expansion::{is_regex_projection, rewrite_projections};


fn expand_expressions(
    input: Node,
    exprs: Vec<Expr>,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Vec<ExprIR>> {
    let schema = lp_arena.get(input).schema(lp_arena);
    let exprs = rewrite_projections(exprs, &schema, &[])?;
    Ok(to_expr_irs(exprs, expr_arena))
}

/// converts LogicalPlan to IR
/// it adds expressions & lps to the respective arenas as it traverses the plan
/// finally it returns the top node of the logical plan
#[recursive]
pub fn to_alp(
    lp: DslPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<IR>,
) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;
    let v = match lp {
        DslPlan::Scan {
            file_info,
            paths,
            predicate,
            scan_type,
            file_options: options,
        } => IR::Scan {
            file_info,
            paths,
            output_schema: None,
            predicate: predicate.map(|expr| to_expr_ir(expr, expr_arena)),
            scan_type,
            file_options: options,
        },
        #[cfg(feature = "python")]
        DslPlan::PythonScan { options } => IR::PythonScan {
            options,
            predicate: None,
        },
        DslPlan::Union { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<_>>()?;
            IR::Union { inputs, options }
        },
        DslPlan::HConcat {
            inputs,
            schema,
            options,
        } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<_>>()?;
            IR::HConcat {
                inputs,
                schema,
                options,
            }
        },
        DslPlan::Filter { input, predicate } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            let predicate = expand_filter(predicate, input, lp_arena)?;
            let predicate = to_expr_ir(predicate, expr_arena);
            IR::Filter { input, predicate }
        },
        DslPlan::Slice { input, offset, len } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            IR::Slice { input, offset, len }
        },
        DslPlan::DataFrameScan {
            df,
            schema,
            output_schema,
            projection,
            selection,
        } => IR::DataFrameScan {
            df,
            schema,
            output_schema,
            projection,
            selection: selection.map(|expr| to_expr_ir(expr, expr_arena)),
        },
        DslPlan::Select {
            expr,
            input,
            schema,
            options,
        } => {
            let eirs = to_expr_irs(expr, expr_arena);
            let expr = eirs.into();
            let i = to_alp(owned(input), expr_arena, lp_arena)?;
            IR::Select {
                expr,
                input: i,
                schema,
                options,
            }
        },
        DslPlan::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            let by_column = expand_expressions(input, by_column, lp_arena, expr_arena)?;
            IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            }
        },
        DslPlan::Cache {
            input,
            id,
            cache_hits,
        } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            IR::Cache {
                input,
                id,
                cache_hits,
            }
        },
        DslPlan::GroupBy {
            input,
            keys,
            aggs,
            schema,
            apply,
            maintain_order,
            options,
        } => {
            let i = to_alp(owned(input), expr_arena, lp_arena)?;
            let aggs = to_expr_irs(aggs, expr_arena);
            let keys = keys.convert(|e| to_expr_ir(e.clone(), expr_arena));

            IR::GroupBy {
                input: i,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options,
            }
        },
        DslPlan::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } => {
            let input_left = to_alp(owned(input_left), expr_arena, lp_arena)?;
            let input_right = to_alp(owned(input_right), expr_arena, lp_arena)?;

            let left_on = to_expr_irs_ignore_alias(left_on, expr_arena);
            let right_on = to_expr_irs_ignore_alias(right_on, expr_arena);

            IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            }
        },
        DslPlan::HStack {
            input,
            exprs,
            options,
        } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            let (exprs, schema) = resolve_with_columns(exprs, input, lp_arena)?;
            let eirs = to_expr_irs(exprs, expr_arena);
            let exprs = eirs.into();
            IR::HStack {
                input,
                exprs,
                schema,
                options,
            }
        },
        DslPlan::Distinct { input, options } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            IR::Distinct { input, options }
        },
        DslPlan::MapFunction { input, function } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            IR::MapFunction { input, function }
        },
        DslPlan::Error { err, .. } => {
            // We just take the error. The LogicalPlan should not be used anymore once this
            // is taken.
            return Err(err.take());
        },
        DslPlan::ExtContext {
            input,
            contexts,
        } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            let contexts = contexts
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<Vec<_>>>()?;

            let mut schema = (**lp_arena.get(input).schema(lp_arena)).clone();
            for input in &contexts {
                let other_schema = lp_arena.get(*input).schema(lp_arena);
                for fld in other_schema.iter_fields() {
                    if schema.get(fld.name()).is_none() {
                        schema.with_column(fld.name, fld.dtype);
                    }
                }
            }

            IR::ExtContext {
                input,
                contexts,
                schema: Arc::new(schema),
            }
        },
        DslPlan::Sink { input, payload } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            IR::Sink { input, payload }
        },
    };
    Ok(lp_arena.add(v))
}

fn expand_filter(predicate: Expr, input: Node, lp_arena: &Arena<IR>) -> PolarsResult<Expr> {
    let schema = lp_arena.get(input).schema(lp_arena);
    let predicate = if has_expr(&predicate, |e| match e {
        Expr::Column(name) => is_regex_projection(name),
        Expr::Wildcard
        | Expr::Selector(_)
        | Expr::RenameAlias { .. }
        | Expr::Columns(_)
        | Expr::DtypeColumn(_)
        | Expr::Nth(_) => true,
        _ => false,
    }) {
        let mut rewritten = rewrite_projections(vec![predicate], &schema, &[])?;
        match rewritten.len() {
            1 => {
                // all good
                rewritten.pop().unwrap()
            },
            0 => {
                let msg = "The predicate expanded to zero expressions. \
                        This may for example be caused by a regex not matching column names or \
                        a column dtype match not hitting any dtypes in the DataFrame";
                polars_bail!(ComputeError: msg);
            },
            _ => {
                let mut expanded = String::new();
                for e in rewritten.iter().take(5) {
                    expanded.push_str(&format!("\t{e},\n"))
                }
                // pop latest comma
                expanded.pop();
                if rewritten.len() > 5 {
                    expanded.push_str("\t...\n")
                }

                let msg = if cfg!(feature = "python") {
                    format!("The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                            This is ambiguous. Try to combine the predicates with the 'all' or `any' expression.")
                } else {
                    format!("The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                            This is ambiguous. Try to combine the predicates with the 'all_horizontal' or `any_horizontal' expression.")
                };
                polars_bail!(ComputeError: msg)
            },
        }
    } else {
        predicate
    };
    expr_to_leaf_column_names_iter(&predicate)
        .try_for_each(|c| schema.try_index_of(&c).and(Ok(())))?;

    Ok(predicate)
}

fn resolve_with_columns(
    exprs: Vec<Expr>,
    input: Node,
    lp_arena: &Arena<IR>
) -> PolarsResult<(Vec<Expr>, SchemaRef)>{
    let schema = lp_arena.get(input).schema(lp_arena);
    let mut new_schema = (**schema).clone();
    let (exprs, _) = prepare_projection(exprs, &schema)?;
    let mut output_names = PlHashSet::with_capacity(exprs.len());

    let mut arena = Arena::with_capacity(8);
    for e in &exprs {
        let field = e
            .to_field_amortized(&schema, Context::Default, &mut arena)
            .unwrap();

        if !output_names.insert(field.name().clone()) {
            let msg = format!(
                "the name: '{}' passed to `LazyFrame.with_columns` is duplicate\n\n\
                    It's possible that multiple expressions are returning the same default column name. \
                    If this is the case, try renaming the columns with `.alias(\"new_name\")` to avoid \
                    duplicate column names.",
                field.name()
            );
            polars_bail!(ComputeError: msg)
        }
        new_schema.with_column(field.name().clone(), field.data_type().clone());
        arena.clear();
    }

    Ok((exprs, Arc::new(new_schema)))
}
