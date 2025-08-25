//! this contains code used for rewriting projections, expanding wildcards, regex selection etc.

use super::*;

pub fn prepare_projection(
    exprs: Vec<Expr>,
    schema: &Schema,
    opt_flags: &mut OptFlags,
) -> PolarsResult<(Vec<Expr>, Schema)> {
    let exprs = rewrite_projections(exprs, &PlHashSet::new(), schema, opt_flags)?;
    let schema = expressions_to_schema(&exprs, schema)?;
    Ok((exprs, schema))
}

pub fn is_regex_projection(name: &str) -> bool {
    name.starts_with('^') && name.ends_with('$')
}

pub fn expand_expression(
    expr: &Expr,
    ignored_selector_columns: &PlHashSet<PlSmallStr>,
    schema: &Schema,
    out: &mut Vec<Expr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<()> {
    if expr.into_iter().all(|e| !needs_expansion(e)) {
        out.push(expr.clone());
        return Ok(());
    }

    expand_expression_rec(expr, ignored_selector_columns, schema, out, opt_flags)?;
    Ok(())
}

/// In case of single col(*) -> do nothing, no selection is the same as select all
/// In other cases replace the wildcard with an expression with all columns
pub fn rewrite_projections(
    exprs: Vec<Expr>,
    ignored_selector_columns: &PlHashSet<PlSmallStr>,
    schema: &Schema,
    opt_flags: &mut OptFlags,
) -> PolarsResult<Vec<Expr>> {
    let mut result = Vec::with_capacity(exprs.len() + schema.len());
    for expr in &exprs {
        expand_expression(
            expr,
            ignored_selector_columns,
            schema,
            &mut result,
            opt_flags,
        )?;
    }

    Ok(result)
}

fn toggle_cse_for_structs(opt_flags: &mut OptFlags) {
    if opt_flags.contains(OptFlags::EAGER) && !opt_flags.contains(OptFlags::NEW_STREAMING) {
        use polars_core::config::verbose;
        if verbose() {
            eprintln!("CSE turned on because of struct expansion")
        }
        *opt_flags |= OptFlags::COMM_SUBEXPR_ELIM;
    }
}

struct FunctionExpansionFlags {
    expand_into_input: bool,
    allow_empty_input: bool,
}

fn function_input_wildcard_expansion(function: &FunctionExpr) -> FunctionExpansionFlags {
    use FunctionExpr as F;
    let mut expand_into_inputs = matches!(
        function,
        F::Boolean(BooleanFunction::AnyHorizontal | BooleanFunction::AllHorizontal)
            | F::Coalesce
            | F::ListExpr(ListFunction::Concat)
            | F::ConcatExpr(_)
            | F::MinHorizontal
            | F::MaxHorizontal
            | F::FoldHorizontal { .. }
            | F::ReduceHorizontal { .. }
            | F::SumHorizontal { .. }
            | F::MeanHorizontal { .. }
            | F::RowEncode(..)
    );
    let mut allow_empty_inputs = matches!(
        function,
        F::Boolean(BooleanFunction::AnyHorizontal | BooleanFunction::AllHorizontal) | F::DropNulls
    );
    #[cfg(feature = "dtype-array")]
    {
        expand_into_inputs |= matches!(function, F::ArrayExpr(ArrayFunction::Concat));
    }
    #[cfg(feature = "dtype-struct")]
    {
        expand_into_inputs |= matches!(function, F::AsStruct);
        expand_into_inputs |= matches!(function, F::StructExpr(StructFunction::WithFields));
        expand_into_inputs |= matches!(
            function,
            F::CumReduceHorizontal { .. } | F::CumFoldHorizontal { .. }
        );
    }
    #[cfg(feature = "ffi_plugin")]
    {
        expand_into_inputs |= matches!(function, F::FfiPlugin { flags, .. } if flags.flags.contains(FunctionFlags::INPUT_WILDCARD_EXPANSION));
        allow_empty_inputs |= matches!(function, F::FfiPlugin { flags, .. } if flags.flags.contains(FunctionFlags::ALLOW_EMPTY_INPUTS));
    }
    #[cfg(feature = "concat_str")]
    {
        expand_into_inputs |= matches!(
            function,
            F::StringExpr(StringFunction::ConcatHorizontal { .. })
        );
    }

    FunctionExpansionFlags {
        expand_into_input: expand_into_inputs,
        allow_empty_input: allow_empty_inputs,
    }
}

fn expand_expression_by_combination(
    exprs: &[Expr],
    ignored_selector_columns: &PlHashSet<PlSmallStr>,
    schema: &Schema,
    out: &mut Vec<Expr>,
    opt_flags: &mut OptFlags,
    f: impl Fn(&[Expr]) -> Expr,
) -> PolarsResult<usize> {
    let mut results = Vec::new();

    // Expand expressions until we find one that expands to more than 1 expression.
    let mut expansion_size = 0;
    for (i, expr) in exprs.iter().enumerate() {
        let start_len = out.len();
        let size = expand_expression_rec(expr, ignored_selector_columns, schema, out, opt_flags)?;
        if size != 1 {
            results.reserve(exprs.len() + 1);
            results.extend((0..i).map(|j| start_len - i + j));
            expansion_size = out.len() - start_len;
            results.push(start_len);
            break;
        }
        assert_eq!(out.len(), start_len + 1);
    }

    // Check if all expressions expanded to 1 expression.
    if results.is_empty() {
        let expr = f(&out[out.len() - exprs.len()..]);
        out.truncate(out.len() - exprs.len());
        out.push(expr);
        return Ok(1);
    }

    // Now do the remaining expression, and check if they match the size of the original expansion
    // (or 1)
    for expr in exprs.iter().skip(results.len()) {
        let start_len = out.len();
        let size = expand_expression_rec(expr, ignored_selector_columns, schema, out, opt_flags)?;
        polars_ensure!(
            size == 1 || size == expansion_size,
            InvalidOperation: "cannot combine selectors that produce a different number of columns ({size} != {expansion_size})"
        );
        results.push(start_len);
    }
    results.push(out.len());

    // Create actual output expressions.
    let mut scratch = Vec::with_capacity(exprs.len());
    let mut tmp_out = Vec::with_capacity(expansion_size);
    for i in 0..expansion_size {
        scratch.clear();
        for w in results.windows(2) {
            let start_offset = w[0];
            let size = w[1] - w[0];

            if size == 1 {
                scratch.push(out[start_offset].clone())
            } else {
                scratch.push(std::mem::take(&mut out[start_offset + i]))
            }
        }
        tmp_out.push(f(&scratch));
    }

    out.truncate(results[0]);
    let size = tmp_out.len();
    out.extend(tmp_out);

    Ok(size)
}

fn expand_single(
    subexpr: &Expr,
    ignored_selector_columns: &PlHashSet<PlSmallStr>,
    schema: &Schema,
    out: &mut Vec<Expr>,
    opt_flags: &mut OptFlags,
    f: impl Fn(Expr) -> Expr,
) -> PolarsResult<usize> {
    try_expand_single(
        subexpr,
        ignored_selector_columns,
        schema,
        out,
        opt_flags,
        |e| Ok(f(e)),
    )
}

fn try_expand_single(
    subexpr: &Expr,
    ignored_selector_columns: &PlHashSet<PlSmallStr>,
    schema: &Schema,
    out: &mut Vec<Expr>,
    opt_flags: &mut OptFlags,
    f: impl Fn(Expr) -> PolarsResult<Expr>,
) -> PolarsResult<usize> {
    let start_len = out.len();
    let did_expand =
        expand_expression_rec(subexpr, ignored_selector_columns, schema, out, opt_flags)?;
    for e in out[start_len..].iter_mut() {
        *e = f(std::mem::take(e))?;
    }
    Ok(did_expand)
}

fn needs_expansion(expr: &Expr) -> bool {
    expr.into_iter().any(|e| {
        let mut v = matches!(e, Expr::Selector(_))
            || matches!(e, Expr::Eval { evaluation, .. } if needs_expansion(evaluation.as_ref()));

        #[cfg(feature = "dtype-struct")]
        {
            v |= matches!(e, Expr::Field(s) if s.len() != 1)
                || matches!(
                    e,
                    Expr::Function {
                        function: FunctionExpr::StructExpr(
                            StructFunction::SelectFields(_) | StructFunction::FieldByName(_)
                        ),
                        ..
                    }
                );
        }

        v
    })
}

fn expand_expression_rec(
    expr: &Expr,
    ignored_selector_columns: &PlHashSet<PlSmallStr>,
    schema: &Schema,
    out: &mut Vec<Expr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<usize> {
    let start_len = out.len();
    match &expr {
        Expr::Alias(subexpr, name) => {
            _ = expand_single(
                subexpr.as_ref(),
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Alias(Arc::new(e), name.clone()),
            )?
        },
        Expr::Column(_) => out.push(expr.clone()),
        Expr::Selector(selector) => {
            let columns = selector.into_columns(schema, ignored_selector_columns)?;
            out.extend(columns.into_iter().map(Expr::Column));
        },
        Expr::Literal(_) => out.push(expr.clone()),
        Expr::BinaryExpr { left, op, right } => {
            _ = expand_expression_by_combination(
                &[left.as_ref().clone(), right.as_ref().clone()],
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::BinaryExpr {
                    left: Arc::new(e[0].clone()),
                    op: *op,
                    right: Arc::new(e[1].clone()),
                },
            )?
        },
        Expr::Cast {
            expr: subexpr,
            dtype,
            options,
        } => {
            _ = expand_single(
                subexpr.as_ref(),
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Cast {
                    expr: Arc::new(e),
                    dtype: dtype.clone(),
                    options: *options,
                },
            )?
        },
        Expr::Sort {
            expr: subexpr,
            options,
        } => {
            _ = expand_single(
                subexpr.as_ref(),
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Sort {
                    expr: Arc::new(e),
                    options: *options,
                },
            )?
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            _ = expand_expression_by_combination(
                &[expr.as_ref().clone(), idx.as_ref().clone()],
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Gather {
                    expr: Arc::new(e[0].clone()),
                    idx: Arc::new(e[1].clone()),
                    returns_scalar: *returns_scalar,
                },
            )?
        },
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            let mut exprs = Vec::with_capacity(1 + by.len());
            exprs.push(expr.as_ref().clone());
            exprs.extend(by.iter().cloned());
            _ = expand_expression_by_combination(
                &exprs,
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::SortBy {
                    expr: Arc::new(e[0].clone()),
                    by: e[1..].to_vec(),
                    sort_options: sort_options.clone(),
                },
            )?
        },
        Expr::Agg(AggExpr::Quantile {
            expr,
            quantile,
            method,
        }) => {
            _ = expand_expression_by_combination(
                &[expr.as_ref().clone(), quantile.as_ref().clone()],
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| {
                    Expr::Agg(AggExpr::Quantile {
                        expr: Arc::new(e[0].clone()),
                        quantile: Arc::new(e[1].clone()),
                        method: *method,
                    })
                },
            )?
        },
        Expr::Agg(agg) => {
            _ = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => expand_single(
                    input.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| {
                        Expr::Agg(AggExpr::Min {
                            input: Arc::new(e),
                            propagate_nans: *propagate_nans,
                        })
                    },
                )?,
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => expand_single(
                    input.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| {
                        Expr::Agg(AggExpr::Max {
                            input: Arc::new(e),
                            propagate_nans: *propagate_nans,
                        })
                    },
                )?,
                AggExpr::Median(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Median(Arc::new(e))),
                )?,
                AggExpr::NUnique(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::NUnique(Arc::new(e))),
                )?,
                AggExpr::First(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::First(Arc::new(e))),
                )?,
                AggExpr::Last(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Last(Arc::new(e))),
                )?,
                AggExpr::Mean(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Mean(Arc::new(e))),
                )?,
                AggExpr::Implode(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Implode(Arc::new(e))),
                )?,
                AggExpr::Count {
                    input,
                    include_nulls,
                } => expand_single(
                    input.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| {
                        Expr::Agg(AggExpr::Count {
                            input: Arc::new(e),
                            include_nulls: *include_nulls,
                        })
                    },
                )?,
                AggExpr::Sum(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Sum(Arc::new(e))),
                )?,
                AggExpr::AggGroups(expr) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::AggGroups(Arc::new(e))),
                )?,
                AggExpr::Std(expr, ddof) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Std(Arc::new(e), *ddof)),
                )?,
                AggExpr::Var(expr, ddof) => expand_single(
                    expr.as_ref(),
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::Agg(AggExpr::Var(Arc::new(e), *ddof)),
                )?,
                AggExpr::Quantile {
                    expr,
                    quantile,
                    method,
                } => expand_expression_by_combination(
                    &[expr.as_ref().clone(), quantile.as_ref().clone()],
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| {
                        Expr::Agg(AggExpr::Quantile {
                            expr: Arc::new(e[0].clone()),
                            quantile: Arc::new(e[1].clone()),
                            method: *method,
                        })
                    },
                )?,
            }
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            _ = expand_expression_by_combination(
                &[
                    predicate.as_ref().clone(),
                    truthy.as_ref().clone(),
                    falsy.as_ref().clone(),
                ],
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Ternary {
                    predicate: Arc::new(e[0].clone()),
                    truthy: Arc::new(e[1].clone()),
                    falsy: Arc::new(e[2].clone()),
                },
            )?
        },
        Expr::Function { input, function } => {
            let function_expansion = function_input_wildcard_expansion(function);
            if function_expansion.expand_into_input {
                let mut expanded_input = Vec::with_capacity(input.len());
                for e in input {
                    expand_expression_rec(
                        e,
                        ignored_selector_columns,
                        schema,
                        &mut expanded_input,
                        opt_flags,
                    )?;
                }
                if expanded_input.is_empty() && !function_expansion.allow_empty_input {
                    let expr = Expr::Function {
                        // Needed to visualize the error
                        input: vec![Expr::Literal(LiteralValue::Scalar(Scalar::null(
                            DataType::Null,
                        )))],
                        function: function.clone(),
                    };
                    polars_bail!(InvalidOperation: "expected at least 1 input in {expr}")
                }
                out.push(Expr::Function {
                    input: expanded_input,
                    function: function.clone(),
                });
            } else {
                if input.is_empty() && !function_expansion.allow_empty_input {
                    let expr = Expr::Function {
                        // Needed to visualize the error
                        input: vec![Expr::Literal(LiteralValue::Scalar(Scalar::null(
                            DataType::Null,
                        )))],
                        function: function.clone(),
                    };
                    polars_bail!(InvalidOperation: "expected at least 1 input in {expr}")
                }

                #[cfg(feature = "dtype-struct")]
                if matches!(
                    function,
                    FunctionExpr::StructExpr(
                        StructFunction::FieldByName(_) | StructFunction::SelectFields(_)
                    )
                ) {
                    toggle_cse_for_structs(opt_flags);
                }

                match function {
                    #[cfg(feature = "dtype-struct")]
                    FunctionExpr::StructExpr(StructFunction::SelectFields(selector)) => {
                        let mut tmp_out = Vec::new();
                        expand_single(
                            &input[0],
                            ignored_selector_columns,
                            schema,
                            &mut tmp_out,
                            opt_flags,
                            |e| e,
                        )?;
                        for e in tmp_out {
                            let dtype = e.to_field(schema)?.dtype;

                            let DataType::Struct(fields) = dtype else {
                                polars_bail!(op = "struct.field", &dtype);
                            };
                            let schema = Schema::from_iter(fields);
                            let fields = selector.into_columns(&schema, &Default::default())?;
                            out.extend(
                                fields
                                    .into_iter()
                                    .map(|f| e.clone().struct_().field_by_name(&f)),
                            );
                        }
                    },
                    _ => {
                        _ = expand_expression_by_combination(
                            input,
                            ignored_selector_columns,
                            schema,
                            out,
                            opt_flags,
                            |e| Expr::Function {
                                input: e.to_vec(),
                                function: function.clone(),
                            },
                        )?
                    },
                }
            }
        },
        Expr::Explode { input, skip_empty } => {
            _ = expand_single(
                input.as_ref(),
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Explode {
                    input: Arc::new(e),
                    skip_empty: *skip_empty,
                },
            )?
        },
        Expr::Filter { input, by } => {
            _ = expand_expression_by_combination(
                &[input.as_ref().clone(), by.as_ref().clone()],
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Filter {
                    input: Arc::new(e[0].clone()),
                    by: Arc::new(e[1].clone()),
                },
            )?
        },
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let mut exprs =
                Vec::with_capacity(partition_by.len() + 1 + usize::from(order_by.is_some()));
            exprs.push(function.as_ref().clone());
            exprs.extend(partition_by.iter().cloned());
            if let Some((e, _)) = &order_by {
                exprs.push(e.as_ref().clone());
            }
            _ = expand_expression_by_combination(
                &exprs,
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Window {
                    function: Arc::new(e[0].clone()),
                    partition_by: e[1..e.len() - usize::from(order_by.is_some())].to_vec(),
                    order_by: order_by
                        .as_ref()
                        .map(|(_, options)| (Arc::new(e.last().unwrap().clone()), *options)),
                    options: options.clone(),
                },
            )?
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => {
            _ = expand_expression_by_combination(
                &[
                    input.as_ref().clone(),
                    offset.as_ref().clone(),
                    length.as_ref().clone(),
                ],
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::Slice {
                    input: Arc::new(e[0].clone()),
                    offset: Arc::new(e[1].clone()),
                    length: Arc::new(e[2].clone()),
                },
            )?
        },
        Expr::KeepName(expr) => {
            _ = expand_single(
                expr.as_ref(),
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::KeepName(Arc::new(e)),
            )?
        },
        Expr::Len => out.push(Expr::Len),
        Expr::AnonymousFunction {
            input,
            function,
            options,
            fmt_str,
        } => {
            let function = function.clone().materialize()?;
            if options
                .flags
                .contains(FunctionFlags::INPUT_WILDCARD_EXPANSION)
            {
                let mut expanded_input = Vec::with_capacity(input.len());
                for e in input {
                    expand_expression_rec(
                        e,
                        ignored_selector_columns,
                        schema,
                        &mut expanded_input,
                        opt_flags,
                    )?;
                }
                out.push(Expr::AnonymousFunction {
                    input: expanded_input,
                    function: LazySerde::Deserialized(function.deep_clone()),
                    options: *options,
                    fmt_str: fmt_str.clone(),
                });
            } else {
                expand_expression_by_combination(
                    input,
                    ignored_selector_columns,
                    schema,
                    out,
                    opt_flags,
                    |e| Expr::AnonymousFunction {
                        input: e.to_vec(),
                        function: LazySerde::Deserialized(function.clone().deep_clone()),
                        options: *options,
                        fmt_str: fmt_str.clone(),
                    },
                )?;
            }
        },
        Expr::DataTypeFunction(v) => out.push(Expr::DataTypeFunction(v.clone())),
        Expr::Eval {
            expr,
            evaluation,
            variant,
        } => {
            // Perform this before schema resolution so that we can better error messages.
            for e in evaluation.as_ref().into_iter() {
                if let Expr::Column(name) = e {
                    polars_ensure!(
                        name.is_empty(),
                        ComputeError:
                        "named columns are not allowed in `eval` functions; consider using `element`"
                    );
                }
            }

            let mut tmp = Vec::with_capacity(1);
            expand_expression_rec(expr, ignored_selector_columns, schema, &mut tmp, opt_flags)?;

            for expr in tmp {
                let expr = Arc::new(expr);
                let expr_dtype = expr.to_field(schema)?.dtype;
                let element_dtype = variant.element_dtype(&expr_dtype)?;
                let evaluation_schema =
                    Schema::from_iter([(PlSmallStr::EMPTY, element_dtype.clone())]);

                let start_length = out.len();
                expand_expression_rec(
                    evaluation,
                    &Default::default(),
                    &evaluation_schema,
                    out,
                    opt_flags,
                )?;

                for e in out[start_length..].iter_mut() {
                    *e = Expr::Eval {
                        expr: expr.clone(),
                        evaluation: Arc::new(std::mem::take(e)),
                        variant: *variant,
                    };
                }
            }
        },
        Expr::RenameAlias { expr, function } => {
            _ = expand_single(
                expr.as_ref(),
                ignored_selector_columns,
                schema,
                out,
                opt_flags,
                |e| Expr::RenameAlias {
                    expr: Arc::new(e),
                    function: function.clone(),
                },
            )?
        },

        #[cfg(feature = "dtype-struct")]
        Expr::Field(names) => {
            toggle_cse_for_structs(opt_flags);
            out.extend(names.iter().cloned().map(|n| Expr::Field([n].into())));
        },

        // SQL only
        Expr::SubPlan(_, _) => unreachable!(),
    };
    Ok(out.len() - start_len)
}
