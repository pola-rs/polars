use arrow::datatypes::ArrowSchemaRef;
use either::Either;
use expr_expansion::rewrite_projections;
use hive::hive_partitions_from_paths;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::config::verbose;
use polars_utils::plpath::PlPath;
use polars_utils::unique_id::UniqueId;

use super::convert_utils::SplitPredicates;
use super::stack_opt::ConversionOptimizer;
use super::*;

mod concat;
mod datatype_fn_to_ir;
mod expr_expansion;
mod expr_to_ir;
mod functions;
mod join;
mod scans;
mod utils;
pub use expr_expansion::{expand_expression, is_regex_projection, prepare_projection};
pub use expr_to_ir::{ExprToIRContext, to_expr_ir};
use expr_to_ir::{to_expr_ir_materialized_lit, to_expr_irs};
use utils::DslConversionContext;

macro_rules! failed_here {
    ($($t:tt)*) => {
        format!("'{}'", stringify!($($t)*)).into()
    }
}
pub(super) use failed_here;

pub fn to_alp(
    lp: DslPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<IR>,
    // Only `SIMPLIFY_EXPR`, `TYPE_COERCION`, `TYPE_CHECK` are respected.
    opt_flags: &mut OptFlags,
) -> PolarsResult<Node> {
    let conversion_optimizer = ConversionOptimizer::new(
        opt_flags.contains(OptFlags::SIMPLIFY_EXPR),
        opt_flags.contains(OptFlags::TYPE_COERCION),
        opt_flags.contains(OptFlags::TYPE_CHECK),
    );

    let mut ctxt = DslConversionContext {
        expr_arena,
        lp_arena,
        conversion_optimizer,
        opt_flags,
        nodes_scratch: &mut unitvec![],
        cache_file_info: Default::default(),
        pushdown_maintain_errors: optimizer::pushdown_maintain_errors(),
        verbose: verbose(),
        seen_caches: Default::default(),
    };

    match to_alp_impl(lp, &mut ctxt) {
        Ok(out) => Ok(out),
        Err(err) => {
            if opt_flags.contains(OptFlags::EAGER) {
                // If we dispatched to the lazy engine from the eager API, we don't want to resolve
                // where in the query plan it went wrong. It is clear from the backtrace anyway.
                return Err(err.remove_context());
            };

            let Some(ir_until_then) = lp_arena.last_node() else {
                return Err(err);
            };

            let node_name = if let PolarsError::Context { msg, .. } = &err {
                msg
            } else {
                "THIS_NODE"
            };
            let plan = IRPlan::new(
                ir_until_then,
                std::mem::take(lp_arena),
                std::mem::take(expr_arena),
            );
            let location = format!("{}", plan.display());
            Err(err.wrap_msg(|msg| {
                format!("{msg}\n\nResolved plan until failure:\n\n\t---> FAILED HERE RESOLVING {node_name} <---\n{location}")
            }))
        },
    }
}

fn run_conversion(lp: IR, ctxt: &mut DslConversionContext, name: &str) -> PolarsResult<Node> {
    let lp_node = ctxt.lp_arena.add(lp);
    ctxt.conversion_optimizer
        .optimize_exprs(ctxt.expr_arena, ctxt.lp_arena, lp_node, false)
        .map_err(|e| e.context(format!("'{name}' failed").into()))?;

    Ok(lp_node)
}

/// converts LogicalPlan to IR
/// it adds expressions & lps to the respective arenas as it traverses the plan
/// finally it returns the top node of the logical plan
#[recursive]
pub fn to_alp_impl(lp: DslPlan, ctxt: &mut DslConversionContext) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;

    let v = match lp {
        DslPlan::Scan {
            sources,
            unified_scan_args,
            scan_type,
            cached_ir,
        } => scans::dsl_to_ir(sources, unified_scan_args, scan_type, cached_ir, ctxt)?,
        #[cfg(feature = "python")]
        DslPlan::PythonScan { options } => {
            use crate::dsl::python_dsl::PythonOptionsDsl;

            let schema = options.get_schema()?;

            let PythonOptionsDsl {
                scan_fn,
                schema_fn: _,
                python_source,
                validate_schema,
                is_pure,
            } = options;

            IR::PythonScan {
                options: PythonOptions {
                    scan_fn,
                    schema,
                    python_source,
                    validate_schema,
                    output_schema: Default::default(),
                    with_columns: Default::default(),
                    n_rows: Default::default(),
                    predicate: Default::default(),
                    is_pure,
                },
            }
        },
        DslPlan::Union { inputs, args } => {
            let mut inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_here!(vertical concat)))?;

            if args.diagonal {
                inputs = concat::convert_diagonal_concat(inputs, ctxt.lp_arena, ctxt.expr_arena)?;
            }

            if args.to_supertypes {
                concat::convert_st_union(
                    &mut inputs,
                    ctxt.lp_arena,
                    ctxt.expr_arena,
                    ctxt.opt_flags,
                )
                .map_err(|e| e.context(failed_here!(vertical concat)))?;
            }

            let first = *inputs.first().ok_or_else(
                || polars_err!(InvalidOperation: "expected at least one input in 'union'/'concat'"),
            )?;
            let schema = ctxt.lp_arena.get(first).schema(ctxt.lp_arena);
            for n in &inputs[1..] {
                let schema_i = ctxt.lp_arena.get(*n).schema(ctxt.lp_arena);
                // The first argument
                schema_i.matches_schema(schema.as_ref()).map_err(|_| polars_err!(InvalidOperation:  "'union'/'concat' inputs should all have the same schema,\
                    got\n{:?} and \n{:?}", schema, schema_i)
                )?;
            }

            let options = args.into();
            IR::Union { inputs, options }
        },
        DslPlan::HConcat { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_here!(horizontal concat)))?;

            let schema = concat::h_concat_schema(&inputs, ctxt.lp_arena)?;

            IR::HConcat {
                inputs,
                schema,
                options,
            }
        },
        DslPlan::Filter { input, predicate } => {
            let mut input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(filter)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            let mut out = Vec::with_capacity(1);
            expr_expansion::expand_expression(
                &predicate,
                &PlHashSet::default(),
                input_schema.as_ref().as_ref(),
                &mut out,
                ctxt.opt_flags,
            )?;

            let predicate = match out.len() {
                1 => {
                    // all good
                    out.pop().unwrap()
                },
                0 => {
                    let msg = "The predicate expanded to zero expressions. \
                            This may for example be caused by a regex not matching column names or \
                            a column dtype match not hitting any dtypes in the DataFrame";
                    polars_bail!(ComputeError: msg);
                },
                _ => {
                    let mut expanded = String::new();
                    for e in out.iter().take(5) {
                        expanded.push_str(&format!("\t{e:?},\n"))
                    }
                    // pop latest comma
                    expanded.pop();
                    if out.len() > 5 {
                        expanded.push_str("\t...\n")
                    }

                    let msg = if cfg!(feature = "python") {
                        format!(
                            "The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                                This is ambiguous. Try to combine the predicates with the 'all' or `any' expression."
                        )
                    } else {
                        format!(
                            "The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                                This is ambiguous. Try to combine the predicates with the 'all_horizontal' or `any_horizontal' expression."
                        )
                    };
                    polars_bail!(ComputeError: msg)
                },
            };
            let predicate_ae = to_expr_ir(
                predicate,
                &mut ExprToIRContext::new_with_opt_eager(
                    ctxt.expr_arena,
                    &input_schema,
                    ctxt.opt_flags,
                ),
            )?;

            if ctxt.opt_flags.predicate_pushdown() {
                ctxt.nodes_scratch.clear();

                if let Some(SplitPredicates { pushable, fallible }) = SplitPredicates::new(
                    predicate_ae.node(),
                    ctxt.expr_arena,
                    Some(ctxt.nodes_scratch),
                    ctxt.pushdown_maintain_errors,
                ) {
                    let mut update_input = |predicate: Node| -> PolarsResult<()> {
                        let predicate = ExprIR::from_node(predicate, ctxt.expr_arena);
                        ctxt.conversion_optimizer
                            .push_scratch(predicate.node(), ctxt.expr_arena);
                        let lp = IR::Filter { input, predicate };
                        input = run_conversion(lp, ctxt, "filter")?;

                        Ok(())
                    };

                    // Pushables first, then fallible.

                    for predicate in pushable {
                        update_input(predicate)?;
                    }

                    if let Some(node) = fallible {
                        update_input(node)?;
                    }

                    return Ok(input);
                };
            };

            ctxt.conversion_optimizer
                .push_scratch(predicate_ae.node(), ctxt.expr_arena);
            let lp = IR::Filter {
                input,
                predicate: predicate_ae,
            };
            return run_conversion(lp, ctxt, "filter");
        },
        DslPlan::Slice { input, offset, len } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(slice)))?;
            IR::Slice { input, offset, len }
        },
        DslPlan::DataFrameScan { df, schema } => IR::DataFrameScan {
            df,
            schema,
            output_schema: None,
        },
        DslPlan::Select {
            expr,
            input,
            options,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(select)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);
            let (exprs, schema) = prepare_projection(expr, &input_schema, ctxt.opt_flags)
                .map_err(|e| e.context(failed_here!(select)))?;

            if exprs.is_empty() {
                ctxt.lp_arena.replace(input, utils::empty_df());
                return Ok(input);
            }

            let eirs = to_expr_irs(
                exprs,
                &mut ExprToIRContext::new_with_opt_eager(
                    ctxt.expr_arena,
                    &input_schema,
                    ctxt.opt_flags,
                ),
            )?;
            ctxt.conversion_optimizer
                .fill_scratch(&eirs, ctxt.expr_arena);

            let schema = Arc::new(schema);
            let lp = IR::Select {
                expr: eirs,
                input,
                schema,
                options,
            };

            return run_conversion(lp, ctxt, "select").map_err(|e| e.context(failed_here!(select)));
        },
        DslPlan::Sort {
            input,
            by_column,
            slice,
            mut sort_options,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(select)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            // note: if given an Expr::Columns, count the individual cols
            let n_by_exprs = if by_column.len() == 1 {
                match &by_column[0] {
                    Expr::Selector(s) => s.into_columns(&input_schema, &Default::default())?.len(),
                    _ => 1,
                }
            } else {
                by_column.len()
            };
            let n_desc = sort_options.descending.len();
            polars_ensure!(
                n_desc == n_by_exprs || n_desc == 1,
                ComputeError: "the length of `descending` ({}) does not match the length of `by` ({})", n_desc, by_column.len()
            );
            let n_nulls_last = sort_options.nulls_last.len();
            polars_ensure!(
                n_nulls_last == n_by_exprs || n_nulls_last == 1,
                ComputeError: "the length of `nulls_last` ({}) does not match the length of `by` ({})", n_nulls_last, by_column.len()
            );

            let mut expanded_cols = Vec::new();
            let mut nulls_last = Vec::new();
            let mut descending = Vec::new();

            // note: nulls_last/descending need to be matched to expanded multi-output expressions.
            // when one of nulls_last/descending has not been updated from the default (single
            // value true/false), 'cycle' ensures that "by_column" iter is not truncated.
            for (c, (&n, &d)) in by_column.into_iter().zip(
                sort_options
                    .nulls_last
                    .iter()
                    .cycle()
                    .zip(sort_options.descending.iter().cycle()),
            ) {
                let exprs = utils::expand_expressions(
                    input,
                    vec![c],
                    ctxt.lp_arena,
                    ctxt.expr_arena,
                    ctxt.opt_flags,
                )
                .map_err(|e| e.context(failed_here!(sort)))?;

                nulls_last.extend(std::iter::repeat_n(n, exprs.len()));
                descending.extend(std::iter::repeat_n(d, exprs.len()));
                expanded_cols.extend(exprs);
            }
            sort_options.nulls_last = nulls_last;
            sort_options.descending = descending;

            ctxt.conversion_optimizer
                .fill_scratch(&expanded_cols, ctxt.expr_arena);
            let mut by_column = expanded_cols;

            // Remove null columns in multi-columns sort
            if by_column.len() > 1 {
                let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

                let mut null_columns = vec![];

                for (i, c) in by_column.iter().enumerate() {
                    if let DataType::Null = c.dtype(&input_schema, ctxt.expr_arena)? {
                        null_columns.push(i);
                    }
                }
                // All null columns, only take one.
                if null_columns.len() == by_column.len() {
                    by_column.truncate(1);
                    sort_options.nulls_last.truncate(1);
                    sort_options.descending.truncate(1);
                }
                // Remove the null columns
                else if !null_columns.is_empty() {
                    for i in null_columns.into_iter().rev() {
                        by_column.remove(i);
                        sort_options.nulls_last.remove(i);
                        sort_options.descending.remove(i);
                    }
                }
            }
            if by_column.is_empty() {
                return Ok(input);
            };

            let lp = IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            };

            return run_conversion(lp, ctxt, "sort").map_err(|e| e.context(failed_here!(sort)));
        },
        DslPlan::Cache { input, id } => {
            let input = match ctxt.seen_caches.get(&id) {
                Some(input) => *input,
                None => {
                    let input = to_alp_impl(owned(input), ctxt)
                        .map_err(|e| e.context(failed_here!(cache)))?;
                    let seen_before = ctxt.seen_caches.insert(id, input);
                    assert!(
                        seen_before.is_none(),
                        "Cache could not have been created in the mean time. That would make the DAG cyclic."
                    );
                    input
                },
            };

            IR::Cache { input, id }
        },
        DslPlan::GroupBy {
            input,
            keys,
            aggs,
            apply,
            maintain_order,
            options,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(group_by)))?;

            // Rolling + group-by sorts the whole table, so remove unneeded columns
            if ctxt.opt_flags.eager() && options.is_rolling() && !keys.is_empty() {
                ctxt.opt_flags.insert(OptFlags::PROJECTION_PUSHDOWN)
            }

            let (keys, aggs, schema) = resolve_group_by(
                input,
                keys,
                aggs,
                &options,
                ctxt.lp_arena,
                ctxt.expr_arena,
                ctxt.opt_flags,
            )
            .map_err(|e| e.context(failed_here!(group_by)))?;

            let (apply, schema) = if let Some((apply, schema)) = apply {
                (Some(apply), schema)
            } else {
                (None, schema)
            };

            ctxt.conversion_optimizer
                .fill_scratch(&keys, ctxt.expr_arena);
            ctxt.conversion_optimizer
                .fill_scratch(&aggs, ctxt.expr_arena);

            let lp = IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options,
            };

            return run_conversion(lp, ctxt, "group_by")
                .map_err(|e| e.context(failed_here!(group_by)));
        },
        DslPlan::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            predicates,
            options,
        } => {
            return join::resolve_join(
                Either::Left(input_left),
                Either::Left(input_right),
                left_on,
                right_on,
                predicates,
                JoinOptionsIR::from(Arc::unwrap_or_clone(options)),
                ctxt,
            )
            .map_err(|e| e.context(failed_here!(join)))
            .map(|t| t.0);
        },
        DslPlan::HStack {
            input,
            exprs,
            options,
        } => {
            let input = to_alp_impl(owned(input), ctxt)
                .map_err(|e| e.context(failed_here!(with_columns)))?;
            let (exprs, schema) =
                resolve_with_columns(exprs, input, ctxt.lp_arena, ctxt.expr_arena, ctxt.opt_flags)
                    .map_err(|e| e.context(failed_here!(with_columns)))?;

            ctxt.conversion_optimizer
                .fill_scratch(&exprs, ctxt.expr_arena);
            let lp = IR::HStack {
                input,
                exprs,
                schema,
                options,
            };
            return run_conversion(lp, ctxt, "with_columns");
        },
        DslPlan::MatchToSchema {
            input,
            match_schema,
            per_column,
            extra_columns,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(unique)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            assert_eq!(per_column.len(), match_schema.len());

            if input_schema.as_ref() == &match_schema {
                return Ok(input);
            }

            let mut exprs = Vec::with_capacity(match_schema.len());
            let mut found_missing_columns = Vec::new();
            let mut used_input_columns = 0;

            for ((column, dtype), per_column) in match_schema.iter().zip(per_column.iter()) {
                match input_schema.get(column) {
                    None => match &per_column.missing_columns {
                        MissingColumnsPolicyOrExpr::Raise => found_missing_columns.push(column),
                        MissingColumnsPolicyOrExpr::Insert => exprs.push(Expr::Alias(
                            Arc::new(Expr::Literal(LiteralValue::Scalar(Scalar::null(
                                dtype.clone(),
                            )))),
                            column.clone(),
                        )),
                        MissingColumnsPolicyOrExpr::InsertWith(expr) => {
                            exprs.push(Expr::Alias(Arc::new(expr.clone()), column.clone()))
                        },
                    },
                    Some(input_dtype) if dtype == input_dtype => {
                        used_input_columns += 1;
                        exprs.push(Expr::Column(column.clone()))
                    },
                    Some(input_dtype) => {
                        let from_dtype = input_dtype;
                        let to_dtype = dtype;

                        let policy = CastColumnsPolicy {
                            integer_upcast: per_column.integer_cast == UpcastOrForbid::Upcast,
                            float_upcast: per_column.float_cast == UpcastOrForbid::Upcast,
                            missing_struct_fields: per_column.missing_struct_fields,
                            extra_struct_fields: per_column.extra_struct_fields,

                            ..Default::default()
                        };

                        let should_cast =
                            policy.should_cast_column(column, to_dtype, from_dtype)?;

                        let mut expr = Expr::Column(PlSmallStr::from_str(column));
                        if should_cast {
                            expr = expr.cast_with_options(to_dtype.clone(), CastOptions::NonStrict);
                        }

                        used_input_columns += 1;
                        exprs.push(expr);
                    },
                }
            }

            // Report the error for missing columns
            if let Some(lst) = found_missing_columns.first() {
                use std::fmt::Write;
                let mut formatted = String::new();
                write!(&mut formatted, "\"{}\"", found_missing_columns[0]).unwrap();
                for c in &found_missing_columns[1..] {
                    write!(&mut formatted, ", \"{c}\"").unwrap();
                }

                write!(&mut formatted, "\"{lst}\"").unwrap();
                polars_bail!(SchemaMismatch: "missing columns in `match_to_schema`: {formatted}");
            }

            // Report the error for extra columns
            if used_input_columns != input_schema.len()
                && extra_columns == ExtraColumnsPolicy::Raise
            {
                let found_extra_columns = input_schema
                    .iter_names()
                    .filter(|n| !match_schema.contains(n))
                    .collect::<Vec<_>>();

                use std::fmt::Write;
                let mut formatted = String::new();
                write!(&mut formatted, "\"{}\"", found_extra_columns[0]).unwrap();
                for c in &found_extra_columns[1..] {
                    write!(&mut formatted, ", \"{c}\"").unwrap();
                }

                polars_bail!(SchemaMismatch: "extra columns in `match_to_schema`: {formatted}");
            }

            let exprs = to_expr_irs(
                exprs,
                &mut ExprToIRContext::new_with_opt_eager(
                    ctxt.expr_arena,
                    &input_schema,
                    ctxt.opt_flags,
                ),
            )?;

            ctxt.conversion_optimizer
                .fill_scratch(&exprs, ctxt.expr_arena);
            let lp = IR::Select {
                input,
                expr: exprs,
                schema: match_schema.clone(),
                options: ProjectionOptions {
                    run_parallel: true,
                    duplicate_check: false,
                    should_broadcast: true,
                },
            };
            return run_conversion(lp, ctxt, "match_to_schema");
        },
        DslPlan::PipeWithSchema { input, callback } => {
            let input_owned = owned(input);

            // Derive the schema from the input
            let input = to_alp_impl(input_owned.clone(), ctxt)
                .map_err(|e| e.context(failed_here!(pipe_with_schema)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            let input_owned = DslPlan::IR {
                dsl: Arc::new(input_owned),
                version: ctxt.lp_arena.version(),
                node: Some(input),
            };

            // Adjust the input and start conversion again
            let input_adjusted =
                callback.call((input_owned, Arc::unwrap_or_clone(input_schema.into_owned())))?;
            return to_alp_impl(input_adjusted, ctxt);
        },
        DslPlan::Distinct { input, options } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(unique)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            let subset = options
                .subset
                .map(|s| {
                    PolarsResult::Ok(
                        s.into_columns(input_schema.as_ref(), &Default::default())?
                            .into_iter()
                            .collect(),
                    )
                })
                .transpose()?;

            let options = DistinctOptionsIR {
                subset,
                maintain_order: options.maintain_order,
                keep_strategy: options.keep_strategy,
                slice: None,
            };

            IR::Distinct { input, options }
        },
        DslPlan::MapFunction { input, function } => {
            let input = to_alp_impl(owned(input), ctxt)
                .map_err(|e| e.context(failed_here!(format!("{}", function).to_lowercase())))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            match function {
                DslFunction::Explode {
                    columns,
                    allow_empty,
                } => {
                    let columns = columns.into_columns(&input_schema, &Default::default())?;
                    polars_ensure!(!columns.is_empty() || allow_empty, InvalidOperation: "no columns provided in explode");
                    if columns.is_empty() {
                        return Ok(input);
                    }
                    let function = FunctionIR::Explode {
                        columns: columns.into_iter().collect(),
                        schema: Default::default(),
                    };
                    let ir = IR::MapFunction { input, function };
                    return Ok(ctxt.lp_arena.add(ir));
                },
                DslFunction::FillNan(fill_value) => {
                    let exprs = input_schema
                        .iter()
                        .filter_map(|(name, dtype)| match dtype {
                            DataType::Float32 | DataType::Float64 => Some(
                                col(name.clone())
                                    .fill_nan(fill_value.clone())
                                    .alias(name.clone()),
                            ),
                            _ => None,
                        })
                        .collect::<Vec<_>>();

                    let (exprs, schema) = resolve_with_columns(
                        exprs,
                        input,
                        ctxt.lp_arena,
                        ctxt.expr_arena,
                        ctxt.opt_flags,
                    )
                    .map_err(|e| e.context(failed_here!(fill_nan)))?;

                    ctxt.conversion_optimizer
                        .fill_scratch(&exprs, ctxt.expr_arena);

                    let lp = IR::HStack {
                        input,
                        exprs,
                        schema,
                        options: ProjectionOptions {
                            duplicate_check: false,
                            ..Default::default()
                        },
                    };
                    return run_conversion(lp, ctxt, "fill_nan");
                },
                DslFunction::Stats(sf) => {
                    let exprs = match sf {
                        StatsFunction::Var { ddof } => stats_helper(
                            |dt| dt.is_primitive_numeric() || dt.is_bool(),
                            |name| col(name.clone()).var(ddof),
                            &input_schema,
                        ),
                        StatsFunction::Std { ddof } => stats_helper(
                            |dt| dt.is_primitive_numeric() || dt.is_bool(),
                            |name| col(name.clone()).std(ddof),
                            &input_schema,
                        ),
                        StatsFunction::Quantile { quantile, method } => stats_helper(
                            |dt| dt.is_primitive_numeric(),
                            |name| col(name.clone()).quantile(quantile.clone(), method),
                            &input_schema,
                        ),
                        StatsFunction::Mean => stats_helper(
                            |dt| {
                                dt.is_primitive_numeric()
                                    || dt.is_temporal()
                                    || dt == &DataType::Boolean
                            },
                            |name| col(name.clone()).mean(),
                            &input_schema,
                        ),
                        StatsFunction::Sum => stats_helper(
                            |dt| {
                                dt.is_primitive_numeric()
                                    || dt.is_decimal()
                                    || matches!(dt, DataType::Boolean | DataType::Duration(_))
                            },
                            |name| col(name.clone()).sum(),
                            &input_schema,
                        ),
                        StatsFunction::Min => stats_helper(
                            |dt| dt.is_ord(),
                            |name| col(name.clone()).min(),
                            &input_schema,
                        ),
                        StatsFunction::Max => stats_helper(
                            |dt| dt.is_ord(),
                            |name| col(name.clone()).max(),
                            &input_schema,
                        ),
                        StatsFunction::Median => stats_helper(
                            |dt| {
                                dt.is_primitive_numeric()
                                    || dt.is_temporal()
                                    || dt == &DataType::Boolean
                            },
                            |name| col(name.clone()).median(),
                            &input_schema,
                        ),
                    };
                    let schema = Arc::new(expressions_to_schema(&exprs, &input_schema)?);
                    let eirs = to_expr_irs(
                        exprs,
                        &mut ExprToIRContext::new_with_opt_eager(
                            ctxt.expr_arena,
                            &input_schema,
                            ctxt.opt_flags,
                        ),
                    )?;

                    ctxt.conversion_optimizer
                        .fill_scratch(&eirs, ctxt.expr_arena);

                    let lp = IR::Select {
                        input,
                        expr: eirs,
                        schema,
                        options: ProjectionOptions {
                            duplicate_check: false,
                            ..Default::default()
                        },
                    };
                    return run_conversion(lp, ctxt, "stats");
                },
                DslFunction::Rename {
                    existing,
                    new,
                    strict,
                } => {
                    assert_eq!(existing.len(), new.len());
                    if existing.is_empty() {
                        return Ok(input);
                    }

                    let existing_lut =
                        PlIndexSet::from_iter(existing.iter().map(PlSmallStr::as_str));

                    let mut schema = Schema::with_capacity(input_schema.len());
                    let mut num_replaced = 0;

                    // Turn the rename into a select.
                    let expr = input_schema
                        .iter()
                        .map(|(n, dtype)| {
                            Ok(match existing_lut.get_index_of(n.as_str()) {
                                None => {
                                    schema.try_insert(n.clone(), dtype.clone())?;
                                    Expr::Column(n.clone())
                                },
                                Some(i) => {
                                    num_replaced += 1;
                                    schema.try_insert(new[i].clone(), dtype.clone())?;
                                    Expr::Column(n.clone()).alias(new[i].clone())
                                },
                            })
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;

                    if strict && num_replaced != existing.len() {
                        let col = existing.iter().find(|c| !input_schema.contains(c)).unwrap();
                        polars_bail!(col_not_found = col);
                    }

                    // Nothing changed, make into a no-op.
                    if num_replaced == 0 {
                        return Ok(input);
                    }

                    let expr = to_expr_irs(
                        expr,
                        &mut ExprToIRContext::new_with_opt_eager(
                            ctxt.expr_arena,
                            &input_schema,
                            ctxt.opt_flags,
                        ),
                    )?;
                    ctxt.conversion_optimizer
                        .fill_scratch(&expr, ctxt.expr_arena);

                    IR::Select {
                        input,
                        expr,
                        schema: Arc::new(schema),
                        options: ProjectionOptions {
                            run_parallel: false,
                            duplicate_check: false,
                            should_broadcast: false,
                        },
                    }
                },
                _ => {
                    let function = function.into_function_ir(&input_schema)?;
                    IR::MapFunction { input, function }
                },
            }
        },
        DslPlan::ExtContext { input, contexts } => {
            let input = to_alp_impl(owned(input), ctxt)
                .map_err(|e| e.context(failed_here!(with_context)))?;
            let contexts = contexts
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_here!(with_context)))?;

            let mut schema = (**ctxt.lp_arena.get(input).schema(ctxt.lp_arena)).clone();
            for input in &contexts {
                let other_schema = ctxt.lp_arena.get(*input).schema(ctxt.lp_arena);
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
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(sink)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);
            let payload = match payload {
                SinkType::Memory => SinkTypeIR::Memory,
                SinkType::File(f) => SinkTypeIR::File(f),
                SinkType::Partition(f) => SinkTypeIR::Partition(PartitionSinkTypeIR {
                    base_path: f.base_path,
                    file_path_cb: f.file_path_cb,
                    file_type: f.file_type,
                    sink_options: f.sink_options,
                    variant: match f.variant {
                        PartitionVariant::MaxSize(max_size) => {
                            PartitionVariantIR::MaxSize(max_size)
                        },
                        PartitionVariant::Parted {
                            key_exprs,
                            include_key,
                        } => {
                            let eirs = to_expr_irs(
                                key_exprs,
                                &mut ExprToIRContext::new_with_opt_eager(
                                    ctxt.expr_arena,
                                    &input_schema,
                                    ctxt.opt_flags,
                                ),
                            )?;
                            ctxt.conversion_optimizer
                                .fill_scratch(&eirs, ctxt.expr_arena);

                            PartitionVariantIR::Parted {
                                key_exprs: eirs,
                                include_key,
                            }
                        },
                        PartitionVariant::ByKey {
                            key_exprs,
                            include_key,
                        } => {
                            let eirs = to_expr_irs(
                                key_exprs,
                                &mut ExprToIRContext::new_with_opt_eager(
                                    ctxt.expr_arena,
                                    &input_schema,
                                    ctxt.opt_flags,
                                ),
                            )?;
                            ctxt.conversion_optimizer
                                .fill_scratch(&eirs, ctxt.expr_arena);

                            PartitionVariantIR::ByKey {
                                key_exprs: eirs,
                                include_key,
                            }
                        },
                    },
                    cloud_options: f.cloud_options,
                    per_partition_sort_by: match f.per_partition_sort_by {
                        None => None,
                        Some(sort_by) => Some(
                            sort_by
                                .into_iter()
                                .map(|s| {
                                    let expr = to_expr_ir(
                                        s.expr,
                                        &mut ExprToIRContext::new_with_opt_eager(
                                            ctxt.expr_arena,
                                            &input_schema,
                                            ctxt.opt_flags,
                                        ),
                                    )?;
                                    ctxt.conversion_optimizer
                                        .push_scratch(expr.node(), ctxt.expr_arena);
                                    Ok(SortColumnIR {
                                        expr,
                                        descending: s.descending,
                                        nulls_last: s.nulls_last,
                                    })
                                })
                                .collect::<PolarsResult<Vec<_>>>()?,
                        ),
                    },
                    finish_callback: f.finish_callback,
                }),
            };

            let lp = IR::Sink { input, payload };
            return run_conversion(lp, ctxt, "sink");
        },
        DslPlan::SinkMultiple { inputs } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_here!(vertical concat)))?;
            IR::SinkMultiple { inputs }
        },
        #[cfg(feature = "merge_sorted")]
        DslPlan::MergeSorted {
            input_left,
            input_right,
            key,
        } => {
            let input_left = to_alp_impl(owned(input_left), ctxt)
                .map_err(|e| e.context(failed_here!(merge_sorted)))?;
            let input_right = to_alp_impl(owned(input_right), ctxt)
                .map_err(|e| e.context(failed_here!(merge_sorted)))?;

            let left_schema = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
            let right_schema = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

            left_schema
                .ensure_is_exact_match(&right_schema)
                .map_err(|err| err.context("merge_sorted".into()))?;

            left_schema
                .try_get(key.as_str())
                .map_err(|err| err.context("merge_sorted".into()))?;

            IR::MergeSorted {
                input_left,
                input_right,
                key,
            }
        },
        DslPlan::IR { node, dsl, version } => {
            return match node {
                Some(node)
                    if version == ctxt.lp_arena.version()
                        && ctxt.conversion_optimizer.used_arenas.insert(version) =>
                {
                    Ok(node)
                },
                _ => to_alp_impl(owned(dsl), ctxt),
            };
        },
    };
    Ok(ctxt.lp_arena.add(v))
}

fn resolve_with_columns(
    exprs: Vec<Expr>,
    input: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<(Vec<ExprIR>, SchemaRef)> {
    let input_schema = lp_arena.get(input).schema(lp_arena);
    let mut output_schema = (**input_schema).clone();
    let exprs = rewrite_projections(exprs, &PlHashSet::new(), &input_schema, opt_flags)?;
    let mut output_names = PlHashSet::with_capacity(exprs.len());

    let eirs = to_expr_irs(
        exprs,
        &mut ExprToIRContext::new_with_opt_eager(expr_arena, &input_schema, opt_flags),
    )?;
    for eir in eirs.iter() {
        let field = eir.field(&input_schema, expr_arena)?;

        if !output_names.insert(field.name().clone()) {
            let msg = format!(
                "the name '{}' passed to `LazyFrame.with_columns` is duplicate\n\n\
                    It's possible that multiple expressions are returning the same default column name. \
                    If this is the case, try renaming the columns with `.alias(\"new_name\")` to avoid \
                    duplicate column names.",
                field.name()
            );
            polars_bail!(ComputeError: msg)
        }
        output_schema.with_column(field.name, field.dtype.materialize_unknown(true)?);
    }

    Ok((eirs, Arc::new(output_schema)))
}

fn resolve_group_by(
    input: Node,
    keys: Vec<Expr>,
    aggs: Vec<Expr>,
    _options: &GroupbyOptions,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<(Vec<ExprIR>, Vec<ExprIR>, SchemaRef)> {
    let input_schema = lp_arena.get(input).schema(lp_arena);
    let input_schema = input_schema.as_ref();
    let mut keys = rewrite_projections(keys, &PlHashSet::default(), input_schema, opt_flags)?;

    // Initialize schema from keys
    let mut output_schema = expressions_to_schema(&keys, input_schema)?;
    let mut key_names: PlHashSet<PlSmallStr> = output_schema.iter_names().cloned().collect();

    #[allow(unused_mut)]
    let mut pop_keys = false;
    // Add dynamic groupby index column(s)
    // Also add index columns to keys for expression expansion.
    #[cfg(feature = "dynamic_group_by")]
    {
        if let Some(options) = _options.rolling.as_ref() {
            let name = options.index_column.clone();
            let dtype = input_schema.try_get(name.as_str())?;
            keys.push(col(name.clone()));
            key_names.insert(name.clone());
            pop_keys = true;
            output_schema.with_column(name.clone(), dtype.clone());
        } else if let Some(options) = _options.dynamic.as_ref() {
            let name = options.index_column.clone();
            keys.push(col(name.clone()));
            key_names.insert(name.clone());
            pop_keys = true;
            let dtype = input_schema.try_get(name.as_str())?;
            if options.include_boundaries {
                output_schema.with_column("_lower_boundary".into(), dtype.clone());
                output_schema.with_column("_upper_boundary".into(), dtype.clone());
            }
            output_schema.with_column(name.clone(), dtype.clone());
        }
    }
    let keys_index_len = output_schema.len();
    if pop_keys {
        let _ = keys.pop();
    }
    let keys = to_expr_irs(
        keys,
        &mut ExprToIRContext::new_with_opt_eager(expr_arena, input_schema, opt_flags),
    )?;

    // Add aggregation column(s)
    let aggs = rewrite_projections(aggs, &key_names, input_schema, opt_flags)?;
    let aggs = to_expr_irs(
        aggs,
        &mut ExprToIRContext::new_with_opt_eager(expr_arena, input_schema, opt_flags),
    )?;
    utils::validate_expressions(&keys, expr_arena, input_schema, "group by")?;
    utils::validate_expressions(&aggs, expr_arena, input_schema, "group by")?;

    let mut aggs_schema = expr_irs_to_schema(&aggs, input_schema, expr_arena);

    // Make sure aggregation columns do not contain duplicates
    if aggs_schema.len() < aggs.len() {
        let mut names = PlHashSet::with_capacity(aggs.len());
        for agg in aggs.iter() {
            let name = agg.output_name();
            polars_ensure!(names.insert(name.clone()), duplicate = name)
        }
    }

    // Coerce aggregation column(s) into List unless not needed (auto-implode)
    debug_assert!(aggs_schema.len() == aggs.len());
    for ((_name, dtype), expr) in aggs_schema.iter_mut().zip(&aggs) {
        if !expr.is_scalar(expr_arena) {
            *dtype = dtype.clone().implode();
        }
    }

    // Final output_schema
    output_schema.merge(aggs_schema);

    // Make sure aggregation columns do not contain keys or index columns
    if output_schema.len() < (keys_index_len + aggs.len()) {
        let mut names = PlHashSet::with_capacity(output_schema.len());
        for agg in aggs.iter().chain(keys.iter()) {
            let name = agg.output_name();
            polars_ensure!(names.insert(name.clone()), duplicate = name)
        }
    }

    Ok((keys, aggs, Arc::new(output_schema)))
}
fn stats_helper<F, E>(condition: F, expr: E, schema: &Schema) -> Vec<Expr>
where
    F: Fn(&DataType) -> bool,
    E: Fn(&PlSmallStr) -> Expr,
{
    schema
        .iter()
        .map(|(name, dt)| {
            if condition(dt) {
                expr(name)
            } else {
                lit(NULL).cast(dt.clone()).alias(name.clone())
            }
        })
        .collect()
}

pub(crate) fn maybe_init_projection_excluding_hive(
    reader_schema: &Either<ArrowSchemaRef, SchemaRef>,
    hive_parts: Option<&SchemaRef>,
) -> Option<Arc<[PlSmallStr]>> {
    // Update `with_columns` with a projection so that hive columns aren't loaded from the
    // file
    let hive_schema = hive_parts?;

    match &reader_schema {
        Either::Left(reader_schema) => hive_schema
            .iter_names()
            .any(|x| reader_schema.contains(x))
            .then(|| {
                reader_schema
                    .iter_names_cloned()
                    .filter(|x| !hive_schema.contains(x))
                    .collect::<Arc<[_]>>()
            }),
        Either::Right(reader_schema) => hive_schema
            .iter_names()
            .any(|x| reader_schema.contains(x))
            .then(|| {
                reader_schema
                    .iter_names_cloned()
                    .filter(|x| !hive_schema.contains(x))
                    .collect::<Arc<[_]>>()
            }),
    }
}
