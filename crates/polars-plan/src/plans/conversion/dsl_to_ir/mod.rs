use arrow::datatypes::ArrowSchemaRef;
use either::Either;
use expr_expansion::rewrite_projections;
use futures::stream::FuturesUnordered;
use hive::hive_partitions_from_paths;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::config::verbose;
use polars_io::ExternalCompression;
use polars_io::pl_async::get_runtime;
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;
use polars_utils::pl_path::PlRefPath;
use polars_utils::unique_id::UniqueId;

use super::convert_utils::SplitPredicates;
use super::stack_opt::ConversionOptimizer;
use super::*;
use crate::constants::get_pl_element_name;
use crate::dsl::PartitionedSinkOptions;
use crate::dsl::file_provider::{FileProviderType, HivePathProvider};
use crate::dsl::functions::{all_horizontal, col};
use crate::plans::conversion::dsl_to_ir::scans::SourcesToFileInfo;

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

async fn fetch_metadata(
    lp: &DslPlan,
    cache_file_info: SourcesToFileInfo,
    verbose: bool,
) -> PolarsResult<()> {
    use futures::stream::StreamExt;
    let mut futures = lp
        .into_iter()
        .filter_map(|dsl| {
            let DslPlan::Scan {
                sources,
                unified_scan_args,
                scan_type,
                cached_ir,
            } = dsl
            else {
                return None;
            };
            Some(scans::dsl_to_ir(
                sources.clone(),
                unified_scan_args.clone(),
                scan_type.clone(),
                cached_ir.clone(),
                cache_file_info.clone(),
                verbose,
            ))
        })
        .collect::<FuturesUnordered<_>>();

    while let Some(result) = futures.next().await {
        result?
    }
    Ok::<(), PolarsError>(())
}

/// converts LogicalPlan to IR
/// it adds expressions & lps to the respective arenas as it traverses the plan
/// finally it returns the top node of the logical plan
#[recursive]
pub fn to_alp_impl(lp: DslPlan, ctxt: &mut DslConversionContext) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;

    // First do a pass to collect all scans and fetch all metadata concurrently.
    {
        let verbose = ctxt.verbose;
        let cache_file_info = ctxt.cache_file_info.clone();
        use tokio::runtime::Handle;

        let fut = fetch_metadata(&lp, cache_file_info, verbose);
        if let Ok(_handle) = Handle::try_current() {
            get_runtime().block_in_place_on(fut)?;
        } else {
            get_runtime().block_on(fut)?;
        }
    }

    let v = match lp {
        DslPlan::Scan {
            sources: _,
            unified_scan_args: _,
            scan_type: _,
            cached_ir,
        } => cached_ir.lock().unwrap().clone().unwrap(),
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

            let first_n = *inputs.first().ok_or_else(
                || polars_err!(InvalidOperation: "expected at least one input in 'union'/'concat'"),
            )?;
            let schema = ctxt.lp_arena.get(first_n).schema(ctxt.lp_arena);
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
                    polars_bail!(
                        ComputeError:
                        "The predicate expanded to zero expressions. \
                        This may for example be caused by a regex not matching column names or \
                        a column dtype match not hitting any dtypes in the DataFrame"
                    );
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

                    if cfg!(feature = "python") {
                        polars_bail!(
                            ComputeError:
                            "The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                                This is ambiguous. Try to combine the predicates with the 'all' or `any' expression."
                        )
                    } else {
                        polars_bail!(
                            ComputeError:
                            "The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                                This is ambiguous. Try to combine the predicates with the 'all_horizontal' or `any_horizontal' expression."
                        )
                    };
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

            if len == 0 {
                let input_schema = ctxt
                    .lp_arena
                    .get(input)
                    .schema(ctxt.lp_arena)
                    .as_ref()
                    .clone();

                IR::DataFrameScan {
                    df: Arc::new(DataFrame::empty_with_schema(&input_schema)),
                    schema: input_schema.clone(),
                    output_schema: None,
                }
            } else {
                IR::Slice { input, offset, len }
            }
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
                slice: slice.map(|t| (t.0, t.1, None)),
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
            predicates,
            mut aggs,
            apply,
            maintain_order,
            options,
        } => {
            // If the group by contains any predicates, we update the plan by turning the
            // predicates into aggregations and filtering on them. Then, we recursively call
            // this function.
            if !predicates.is_empty() {
                let predicate_names = (0..predicates.len())
                    .map(|i| format_pl_smallstr!("__POLARS_HAVING_{i}"))
                    .collect::<Arc<[_]>>();
                let predicates = predicates
                    .into_iter()
                    .zip(predicate_names.iter())
                    .map(|(p, name)| p.alias(name.clone()))
                    .collect_vec();
                aggs.extend(predicates);

                let lp = DslPlan::GroupBy {
                    input,
                    keys,
                    predicates: vec![],
                    aggs,
                    apply,
                    maintain_order,
                    options,
                };
                let lp = DslBuilder::from(lp)
                    .filter(
                        all_horizontal(
                            predicate_names.iter().map(|n| col(n.clone())).collect_vec(),
                        )
                        .unwrap(),
                    )
                    .drop(Selector::ByName {
                        names: predicate_names,
                        strict: true,
                    })
                    .build();
                return to_alp_impl(lp, ctxt);
            }

            // NOTE: As we went into this branch, we know that no predicates are provided.
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
            // Derive the schema from the input
            let mut inputs = Vec::with_capacity(input.len());
            let mut input_schemas = Vec::with_capacity(input.len());

            for plan in input.as_ref() {
                let ir = to_alp_impl(plan.clone(), ctxt)?;
                let schema = ctxt.lp_arena.get(ir).schema(ctxt.lp_arena).into_owned();

                let dsl = DslPlan::IR {
                    dsl: Arc::new(plan.clone()),
                    version: ctxt.lp_arena.version(),
                    node: Some(ir),
                };
                inputs.push(dsl);
                input_schemas.push(schema);
            }

            // Adjust the input and start conversion again
            let input_adjusted = callback.call((inputs, input_schemas))?;
            return to_alp_impl(input_adjusted, ctxt);
        },
        #[cfg(feature = "pivot")]
        DslPlan::Pivot {
            input,
            on,
            on_columns,
            index,
            values,
            agg,
            maintain_order,
            separator,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(unique)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            let on = on.into_columns(input_schema.as_ref(), &Default::default())?;
            let index = index.into_columns(input_schema.as_ref(), &Default::default())?;
            let values = values.into_columns(input_schema.as_ref(), &Default::default())?;

            polars_ensure!(!on.is_empty(), InvalidOperation: "`pivot` called without `on` columns.");
            polars_ensure!(on.len() == on_columns.width(), InvalidOperation: "`pivot` expected `on` and `on_columns` to have the same amount of columns.");
            if on.len() > 1 {
                polars_ensure!(
                    on_columns.columns().iter().zip(on.iter()).all(|(c, o)| o == c.name()),
                    InvalidOperation: "`pivot` has mismatching column names between `on` and `on_columns`."
                );
            }
            polars_ensure!(!values.is_empty(), InvalidOperation: "`pivot` called without `values` columns.");

            let on_titles = if on_columns.width() == 1 {
                on_columns.columns()[0].cast(&DataType::String)?
            } else {
                on_columns
                    .as_ref()
                    .clone()
                    .into_struct(PlSmallStr::EMPTY)
                    .cast(&DataType::String)?
                    .into_column()
            };
            let on_titles = on_titles.str()?;

            let mut expr_schema = input_schema.as_ref().as_ref().clone();
            let mut out = Vec::with_capacity(1);
            let mut aggs = Vec::<ExprIR>::with_capacity(values.len() * on_columns.height());
            for value in values.iter() {
                out.clear();
                let value_dtype = input_schema.try_get(value)?;
                expr_schema.insert(get_pl_element_name(), value_dtype.clone());
                expand_expression(
                    &agg,
                    &Default::default(),
                    &expr_schema,
                    &mut out,
                    ctxt.opt_flags,
                )?;
                polars_ensure!(
                    out.len() == 1,
                    InvalidOperation: "Pivot expression are not allowed to expand to more than 1 expression"
                );
                let agg = out.pop().unwrap();
                let agg_ae = to_expr_ir(
                    agg,
                    &mut ExprToIRContext::new_with_opt_eager(
                        ctxt.expr_arena,
                        &expr_schema,
                        ctxt.opt_flags,
                    ),
                )?
                .node();

                polars_ensure!(
                    aexpr_to_leaf_names_iter(agg_ae, ctxt.expr_arena).count() == 0,
                    InvalidOperation: "explicit column references are not allowed in the `aggregate_function` of `pivot`"
                );

                for i in 0..on_columns.height() {
                    let mut name = String::new();
                    if values.len() > 1 {
                        name.push_str(value.as_str());
                        name.push_str(separator.as_str());
                    }

                    name.push_str(on_titles.get(i).unwrap_or("null"));

                    fn on_predicate(
                        on: &PlSmallStr,
                        on_column: &Column,
                        i: usize,
                        expr_arena: &mut Arena<AExpr>,
                    ) -> AExprBuilder {
                        let e = AExprBuilder::col(on.clone(), expr_arena);
                        e.eq(
                            AExprBuilder::lit_scalar(
                                Scalar::new(
                                    on_column.dtype().clone(),
                                    on_column.get(i).unwrap().into_static(),
                                ),
                                expr_arena,
                            ),
                            expr_arena,
                        )
                    }

                    let predicate = if on.len() == 1 {
                        on_predicate(&on[0], &on_columns.columns()[0], i, ctxt.expr_arena)
                    } else {
                        AExprBuilder::function(
                            on.iter()
                                .enumerate()
                                .map(|(j, on_col)| {
                                    on_predicate(
                                        on_col,
                                        &on_columns.columns()[j],
                                        i,
                                        ctxt.expr_arena,
                                    )
                                    .expr_ir(on_col.clone())
                                })
                                .collect::<Vec<_>>(),
                            IRFunctionExpr::Boolean(IRBooleanFunction::AllHorizontal),
                            ctxt.expr_arena,
                        )
                    };

                    let replacement_element = AExprBuilder::col(value.clone(), ctxt.expr_arena)
                        .filter(predicate, ctxt.expr_arena)
                        .node();

                    #[recursive::recursive]
                    fn deep_clone_element_replace(
                        ae: Node,
                        arena: &mut Arena<AExpr>,
                        replacement: Node,
                    ) -> Node {
                        let slf = arena.get(ae).clone();
                        if matches!(slf, AExpr::Element) {
                            return deep_clone_ae(replacement, arena);
                        } else if matches!(slf, AExpr::Len) {
                            // For backwards-compatibility, we support providing `pl.len()` to mean
                            // the length of the group here.
                            let element = deep_clone_ae(replacement, arena);
                            return AExprBuilder::new_from_node(element).len(arena).node();
                        }

                        let mut children = vec![];
                        slf.children_rev(&mut children);
                        for child in &mut children {
                            *child = deep_clone_element_replace(*child, arena, replacement);
                        }
                        children.reverse();

                        arena.add(slf.replace_children(&children))
                    }
                    aggs.push(ExprIR::new(
                        deep_clone_element_replace(agg_ae, ctxt.expr_arena, replacement_element),
                        OutputName::Alias(name.into()),
                    ));
                }
            }

            let keys: Vec<_> = index
                .into_iter()
                .map(|i| AExprBuilder::col(i.clone(), ctxt.expr_arena).expr_ir(i))
                .collect();

            let mut uniq_names = PlHashSet::new();
            for expr in keys.iter().chain(aggs.iter()) {
                let name = expr.output_name();
                let is_uniq = uniq_names.insert(name.clone());
                polars_ensure!(is_uniq, duplicate = name);
            }

            IRBuilder::new(input, ctxt.expr_arena, ctxt.lp_arena)
                .group_by(keys, aggs, None, maintain_order, Default::default())
                .build()
        },
        DslPlan::Distinct { input, options } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(unique)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena).into_owned();

            // "subset" param supports cols and/or arbitrary expressions
            let (input, subset, temp_cols) = if let Some(exprs) = options.subset {
                let exprs = rewrite_projections(
                    exprs,
                    &PlHashSet::default(),
                    &input_schema,
                    ctxt.opt_flags,
                )?;

                // identify cols and exprs in "subset" param
                let mut subset_colnames = vec![];
                let mut subset_exprs = vec![];
                for expr in &exprs {
                    match expr {
                        Expr::Column(name) => {
                            polars_ensure!(
                                input_schema.contains(name),
                                ColumnNotFound: "{name:?} not found"
                            );
                            subset_colnames.push(name.clone());
                        },
                        _ => subset_exprs.push(expr.clone()),
                    }
                }

                if subset_exprs.is_empty() {
                    // "subset" is a collection of basic cols (or empty)
                    (input, Some(subset_colnames.into_iter().collect()), vec![])
                } else {
                    // "subset" contains exprs; add them as temporary cols
                    let (aliased_exprs, temp_names): (Vec<_>, Vec<_>) = subset_exprs
                        .into_iter()
                        .enumerate()
                        .map(|(idx, expr)| {
                            let temp_name = format_pl_smallstr!("__POLARS_UNIQUE_SUBSET_{}", idx);
                            (expr.alias(temp_name.clone()), temp_name)
                        })
                        .unzip();

                    subset_colnames.extend_from_slice(&temp_names);

                    // integrate the temporary cols with the existing "input" node
                    let (temp_expr_irs, schema) = resolve_with_columns(
                        aliased_exprs,
                        input,
                        ctxt.lp_arena,
                        ctxt.expr_arena,
                        ctxt.opt_flags,
                    )?;
                    ctxt.conversion_optimizer
                        .fill_scratch(&temp_expr_irs, ctxt.expr_arena);

                    let input_with_exprs = ctxt.lp_arena.add(IR::HStack {
                        input,
                        exprs: temp_expr_irs,
                        schema,
                        options: ProjectionOptions {
                            run_parallel: false,
                            duplicate_check: false,
                            should_broadcast: true,
                        },
                    });
                    (
                        input_with_exprs,
                        Some(subset_colnames.into_iter().collect()),
                        temp_names,
                    )
                }
            } else {
                (input, None, vec![])
            };

            // `distinct` definition (will contain temporary cols if we have "subset" exprs)
            let distinct_node = ctxt.lp_arena.add(IR::Distinct {
                input,
                options: DistinctOptionsIR {
                    subset,
                    maintain_order: options.maintain_order,
                    keep_strategy: options.keep_strategy,
                    slice: None,
                },
            });

            // if no temporary cols (eg: we had no "subset" exprs), we're done...
            if temp_cols.is_empty() {
                return Ok(distinct_node);
            }

            // ...otherwise, drop them by projecting the original schema
            return Ok(ctxt.lp_arena.add(IR::SimpleProjection {
                input: distinct_node,
                columns: input_schema,
            }));
        },
        DslPlan::MapFunction { input, function } => {
            let input = to_alp_impl(owned(input), ctxt)
                .map_err(|e| e.context(failed_here!(format!("{}", function).to_lowercase())))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            match function {
                DslFunction::Explode {
                    columns,
                    options,
                    allow_empty,
                } => {
                    let columns = columns.into_columns(&input_schema, &Default::default())?;
                    polars_ensure!(!columns.is_empty() || allow_empty, InvalidOperation: "no columns provided in explode");
                    if columns.is_empty() {
                        return Ok(input);
                    }
                    let function = FunctionIR::Explode {
                        columns: columns.into_iter().collect(),
                        options,
                        schema: Default::default(),
                    };
                    let ir = IR::MapFunction { input, function };
                    return Ok(ctxt.lp_arena.add(ir));
                },
                DslFunction::FillNan(fill_value) => {
                    let exprs = input_schema
                        .iter()
                        .filter_map(|(name, dtype)| match dtype {
                            DataType::Float16 | DataType::Float32 | DataType::Float64 => Some(
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
                            |dt| dt.is_primitive_numeric() || dt.is_bool() || dt.is_decimal(),
                            |name| col(name.clone()).var(ddof),
                            &input_schema,
                        ),
                        StatsFunction::Std { ddof } => stats_helper(
                            |dt| dt.is_primitive_numeric() || dt.is_bool() || dt.is_decimal(),
                            |name| col(name.clone()).std(ddof),
                            &input_schema,
                        ),
                        StatsFunction::Quantile { quantile, method } => stats_helper(
                            |dt| dt.is_primitive_numeric() || dt.is_decimal() || dt.is_temporal(),
                            |name| col(name.clone()).quantile(quantile.clone(), method),
                            &input_schema,
                        ),
                        StatsFunction::Mean => stats_helper(
                            |dt| {
                                dt.is_primitive_numeric()
                                    || dt.is_temporal()
                                    || dt.is_bool()
                                    || dt.is_decimal()
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
                    let schema = Arc::new(expressions_to_schema(
                        &exprs,
                        &input_schema,
                        |duplicate_name: &str| duplicate_name.to_string(),
                    )?);
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
                SinkType::Callback(f) => SinkTypeIR::Callback(f),
                SinkType::File(options) => {
                    let mut compression_opt = None::<ExternalCompression>;

                    #[cfg(feature = "parquet")]
                    if let FileWriteFormat::Parquet(options) = &options.file_format
                        && let Some(arrow_schema) = &options.arrow_schema
                    {
                        validate_arrow_schema_conversion(
                            input_schema.as_ref(),
                            arrow_schema,
                            options.compat_level(),
                        )?;
                    }

                    #[cfg(feature = "csv")]
                    if let FileWriteFormat::Csv(csv_options) = &options.file_format
                        && csv_options.check_extension
                    {
                        compression_opt = Some(csv_options.compression);
                    }

                    #[cfg(feature = "json")]
                    if let FileWriteFormat::NDJson(ndjson_options) = &options.file_format
                        && ndjson_options.check_extension
                    {
                        compression_opt = Some(ndjson_options.compression);
                    }

                    if let Some(compression) = compression_opt {
                        if let SinkTarget::Path(path) = &options.target {
                            let extension = path.extension();

                            if let Some(suffix) = compression.file_suffix() {
                                polars_ensure!(
                                    extension.is_none_or(|extension| extension == suffix.strip_prefix(".").unwrap_or(suffix)),
                                    InvalidOperation: "the path ({}) does not conform to standard naming, expected suffix: ({}), set `check_extension` to `False` if you don't want this behavior", path, suffix
                                );
                            } else if ["gz", "zst", "zstd"].iter().any(|compression_extension| {
                                extension == Some(compression_extension)
                            }) {
                                polars_bail!(
                                    InvalidOperation: "use the compression parameter to control compression, or set `check_extension` to `False` if you want to suffix an uncompressed filename with an ending intended for compression"
                                );
                            }
                        }
                    }

                    SinkTypeIR::File(options)
                },
                SinkType::Partitioned(PartitionedSinkOptions {
                    base_path,
                    file_path_provider,
                    partition_strategy,
                    file_format,
                    unified_sink_args,
                    max_rows_per_file,
                    approximate_bytes_per_file,
                }) => {
                    let expr_to_ir_cx = &mut ExprToIRContext::new_with_opt_eager(
                        ctxt.expr_arena,
                        &input_schema,
                        ctxt.opt_flags,
                    );

                    let partition_strategy = match partition_strategy {
                        PartitionStrategy::Keyed {
                            keys,
                            include_keys,
                            keys_pre_grouped,
                        } => {
                            let keys = to_expr_irs(keys, expr_to_ir_cx)?;

                            polars_ensure!(
                                keys.iter().all(|e| is_elementwise_rec(e.node(), ctxt.expr_arena)),
                                InvalidOperation:
                                "cannot use non-elementwise expressions for PartitionBy keys"
                            );

                            PartitionStrategyIR::Keyed {
                                keys,
                                include_keys,
                                keys_pre_grouped,
                            }
                        },
                        PartitionStrategy::FileSize => PartitionStrategyIR::FileSize,
                    };

                    let options = PartitionedSinkOptionsIR {
                        base_path,
                        file_path_provider: file_path_provider.unwrap_or_else(|| {
                            FileProviderType::Hive(HivePathProvider {
                                extension: PlSmallStr::from_static(file_format.extension()),
                            })
                        }),
                        partition_strategy,
                        file_format,
                        unified_sink_args,
                        max_rows_per_file,
                        approximate_bytes_per_file,
                    };

                    #[cfg(feature = "parquet")]
                    if let FileWriteFormat::Parquet(parquet_options) = &options.file_format
                        && let Some(arrow_schema) = &parquet_options.arrow_schema
                    {
                        let file_schema =
                            options.file_output_schema(&input_schema, ctxt.expr_arena)?;

                        validate_arrow_schema_conversion(
                            file_schema.as_ref(),
                            arrow_schema,
                            parquet_options.compat_level(),
                        )?;
                    }

                    ctxt.conversion_optimizer
                        .fill_scratch(options.expr_irs_iter(), ctxt.expr_arena);

                    SinkTypeIR::Partitioned(options)
                },
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
            polars_bail!(
                ComputeError:
                "the name '{}' passed to `LazyFrame.with_columns` is duplicate\n\n\
                It's possible that multiple expressions are returning the same default column name. \
                If this is the case, try renaming the columns with `.alias(\"new_name\")` to avoid \
                duplicate column names.",
                field.name()
            )
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
    let mut output_schema = expressions_to_schema(&keys, input_schema, |duplicate_name: &str| {
        format!("group_by keys contained duplicate output name '{duplicate_name}'")
    })?;
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

    let mut aggs_schema = expr_irs_to_schema(&aggs, input_schema, expr_arena)?;

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
