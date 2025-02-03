use arrow::datatypes::ArrowSchemaRef;
use either::Either;
use expr_expansion::{is_regex_projection, rewrite_projections};
use hive::{hive_partitions_from_paths, HivePartitions};

use super::stack_opt::ConversionOptimizer;
use super::*;
use crate::plans::conversion::expr_expansion::expand_selectors;

fn expand_expressions(
    input: Node,
    exprs: Vec<Expr>,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<Vec<ExprIR>> {
    let schema = lp_arena.get(input).schema(lp_arena);
    let exprs = rewrite_projections(exprs, &schema, &[], opt_flags)?;
    to_expr_irs(exprs, expr_arena)
}

fn empty_df() -> IR {
    IR::DataFrameScan {
        df: Arc::new(Default::default()),
        schema: Arc::new(Default::default()),
        output_schema: None,
    }
}

fn validate_expression(
    node: Node,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    operation_name: &str,
) -> PolarsResult<()> {
    let iter = aexpr_to_leaf_names_iter(node, expr_arena);
    validate_columns_in_input(iter, input_schema, operation_name)
}

fn validate_expressions<N: Into<Node>, I: IntoIterator<Item = N>>(
    nodes: I,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    operation_name: &str,
) -> PolarsResult<()> {
    let nodes = nodes.into_iter();

    for node in nodes {
        validate_expression(node.into(), expr_arena, input_schema, operation_name)?
    }
    Ok(())
}

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
    };

    match to_alp_impl(lp, &mut ctxt) {
        Ok(out) => Ok(out),
        Err(err) => {
            if let Some(ir_until_then) = lp_arena.last_node() {
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
            } else {
                Err(err)
            }
        },
    }
}

pub(super) struct DslConversionContext<'a> {
    pub(super) expr_arena: &'a mut Arena<AExpr>,
    pub(super) lp_arena: &'a mut Arena<IR>,
    pub(super) conversion_optimizer: ConversionOptimizer,
    pub(super) opt_flags: &'a mut OptFlags,
}

pub(super) fn run_conversion(
    lp: IR,
    ctxt: &mut DslConversionContext,
    name: &str,
) -> PolarsResult<Node> {
    let lp_node = ctxt.lp_arena.add(lp);
    ctxt.conversion_optimizer
        .optimize_exprs(ctxt.expr_arena, ctxt.lp_arena, lp_node)
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
            file_info,
            file_options,
            scan_type,
            cached_ir,
        } => {
            // Note that the first metadata can still end up being `None` later if the files were
            // filtered from predicate pushdown.
            let mut cached_ir = cached_ir.lock().unwrap();

            if cached_ir.is_none() {
                let mut file_options = file_options.clone();
                let mut scan_type = scan_type.clone();

                if let Some(hive_schema) = file_options.hive_options.schema.as_deref() {
                    match file_options.hive_options.enabled {
                        // Enable hive_partitioning if it is unspecified but a non-empty hive_schema given
                        None if !hive_schema.is_empty() => {
                            file_options.hive_options.enabled = Some(true)
                        },
                        // hive_partitioning was explicitly disabled
                        Some(false) => polars_bail!(
                            ComputeError:
                            "a hive schema was given but hive_partitioning was disabled"
                        ),
                        Some(true) | None => {},
                    }
                }

                let sources = match &scan_type {
                    #[cfg(feature = "parquet")]
                    FileScan::Parquet { cloud_options, .. } => sources
                        .expand_paths_with_hive_update(&mut file_options, cloud_options.as_ref())?,
                    #[cfg(feature = "ipc")]
                    FileScan::Ipc { cloud_options, .. } => sources
                        .expand_paths_with_hive_update(&mut file_options, cloud_options.as_ref())?,
                    #[cfg(feature = "csv")]
                    FileScan::Csv { cloud_options, .. } => {
                        sources.expand_paths(&file_options, cloud_options.as_ref())?
                    },
                    #[cfg(feature = "json")]
                    FileScan::NDJson { cloud_options, .. } => {
                        sources.expand_paths(&file_options, cloud_options.as_ref())?
                    },
                    FileScan::Anonymous { .. } => sources,
                };

                let mut file_info = match &mut scan_type {
                    #[cfg(feature = "parquet")]
                    FileScan::Parquet {
                        options,
                        cloud_options,
                        metadata,
                    } => {
                        if let Some(schema) = &options.schema {
                            // We were passed a schema, we don't have to call `parquet_file_info`,
                            // but this does mean we don't have `row_estimation` and `first_metadata`.
                            FileInfo {
                                schema: schema.clone(),
                                reader_schema: Some(either::Either::Left(Arc::new(
                                    schema.to_arrow(CompatLevel::newest()),
                                ))),
                                row_estimation: (None, 0),
                            }
                        } else {
                            let (file_info, md) = scans::parquet_file_info(
                                &sources,
                                &file_options,
                                cloud_options.as_ref(),
                            )
                            .map_err(|e| e.context(failed_here!(parquet scan)))?;

                            *metadata = md;
                            file_info
                        }
                    },
                    #[cfg(feature = "ipc")]
                    FileScan::Ipc {
                        cloud_options,
                        metadata,
                        ..
                    } => {
                        let (file_info, md) =
                            scans::ipc_file_info(&sources, &file_options, cloud_options.as_ref())
                                .map_err(|e| e.context(failed_here!(ipc scan)))?;
                        *metadata = Some(Arc::new(md));
                        file_info
                    },
                    #[cfg(feature = "csv")]
                    FileScan::Csv {
                        options,
                        cloud_options,
                    } => scans::csv_file_info(
                        &sources,
                        &file_options,
                        options,
                        cloud_options.as_ref(),
                    )
                    .map_err(|e| e.context(failed_here!(csv scan)))?,
                    #[cfg(feature = "json")]
                    FileScan::NDJson {
                        options,
                        cloud_options,
                    } => scans::ndjson_file_info(
                        &sources,
                        &file_options,
                        options,
                        cloud_options.as_ref(),
                    )
                    .map_err(|e| e.context(failed_here!(ndjson scan)))?,
                    FileScan::Anonymous { .. } => {
                        file_info.expect("FileInfo should be set for AnonymousScan")
                    },
                };

                if file_options.hive_options.enabled.is_none() {
                    // We expect this to be `Some(_)` after this point. If it hasn't been auto-enabled
                    // we explicitly set it to disabled.
                    file_options.hive_options.enabled = Some(false);
                }

                let hive_parts = if file_options.hive_options.enabled.unwrap()
                    && file_info.reader_schema.is_some()
                {
                    let paths = sources.as_paths().ok_or_else(|| {
                        polars_err!(nyi = "Hive-partitioning of in-memory buffers")
                    })?;

                    #[allow(unused_assignments)]
                    let mut owned = None;

                    hive_partitions_from_paths(
                        paths,
                        file_options.hive_options.hive_start_idx,
                        file_options.hive_options.schema.clone(),
                        match file_info.reader_schema.as_ref().unwrap() {
                            Either::Left(v) => {
                                owned = Some(Schema::from_arrow_schema(v.as_ref()));
                                owned.as_ref().unwrap()
                            },
                            Either::Right(v) => v.as_ref(),
                        },
                        file_options.hive_options.try_parse_dates,
                    )?
                } else {
                    None
                };

                if let Some(ref hive_parts) = hive_parts {
                    let hive_schema = hive_parts[0].schema();
                    file_info.update_schema_with_hive_schema(hive_schema.clone());
                } else if let Some(hive_schema) = file_options.hive_options.schema.clone() {
                    // We hit here if we are passed the `hive_schema` to `scan_parquet` but end up with an empty file
                    // list during path expansion. In this case we still want to return an empty DataFrame with this
                    // schema.
                    file_info.update_schema_with_hive_schema(hive_schema);
                }

                file_options.include_file_paths =
                    file_options.include_file_paths.filter(|_| match scan_type {
                        #[cfg(feature = "parquet")]
                        FileScan::Parquet { .. } => true,
                        #[cfg(feature = "ipc")]
                        FileScan::Ipc { .. } => true,
                        #[cfg(feature = "csv")]
                        FileScan::Csv { .. } => true,
                        #[cfg(feature = "json")]
                        FileScan::NDJson { .. } => true,
                        FileScan::Anonymous { .. } => false,
                    });

                if let Some(ref file_path_col) = file_options.include_file_paths {
                    let schema = Arc::make_mut(&mut file_info.schema);

                    if schema.contains(file_path_col) {
                        polars_bail!(
                            Duplicate: r#"column name for file paths "{}" conflicts with column name from file"#,
                            file_path_col
                        );
                    }

                    schema.insert_at_index(
                        schema.len(),
                        file_path_col.clone(),
                        DataType::String,
                    )?;
                }

                file_options.with_columns = if file_info.reader_schema.is_some() {
                    maybe_init_projection_excluding_hive(
                        file_info.reader_schema.as_ref().unwrap(),
                        hive_parts.as_ref().map(|x| &x[0]),
                    )
                } else {
                    None
                };

                if let Some(row_index) = &file_options.row_index {
                    let schema = Arc::make_mut(&mut file_info.schema);
                    *schema = schema
                        .new_inserting_at_index(0, row_index.name.clone(), IDX_DTYPE)
                        .unwrap();
                }

                let ir = if sources.is_empty() && !matches!(scan_type, FileScan::Anonymous { .. }) {
                    IR::DataFrameScan {
                        df: Arc::new(DataFrame::empty_with_schema(&file_info.schema)),
                        schema: file_info.schema,
                        output_schema: None,
                    }
                } else {
                    IR::Scan {
                        sources,
                        file_info,
                        hive_parts,
                        predicate: None,
                        scan_type,
                        output_schema: None,
                        file_options,
                    }
                };

                cached_ir.replace(ir);
            }

            cached_ir.clone().unwrap()
        },
        #[cfg(feature = "python")]
        DslPlan::PythonScan { options } => IR::PythonScan { options },
        DslPlan::Union { inputs, args } => {
            let mut inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_here!(vertical concat)))?;

            if args.diagonal {
                inputs =
                    convert_utils::convert_diagonal_concat(inputs, ctxt.lp_arena, ctxt.expr_arena)?;
            }

            if args.to_supertypes {
                convert_utils::convert_st_union(&mut inputs, ctxt.lp_arena, ctxt.expr_arena)
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

            let schema = convert_utils::h_concat_schema(&inputs, ctxt.lp_arena)?;

            IR::HConcat {
                inputs,
                schema,
                options,
            }
        },
        DslPlan::Filter { input, predicate } => {
            let mut input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(filter)))?;
            let predicate = expand_filter(predicate, input, ctxt.lp_arena, ctxt.opt_flags)
                .map_err(|e| e.context(failed_here!(filter)))?;

            let predicate_ae = to_expr_ir(predicate.clone(), ctxt.expr_arena)?;

            // TODO: We could do better here by using `pushdown_eligibility()`
            return if ctxt.opt_flags.predicate_pushdown()
                && permits_filter_pushdown_rec(
                    ctxt.expr_arena.get(predicate_ae.node()),
                    ctxt.expr_arena,
                ) {
                // Split expression that are ANDed into multiple Filter nodes as the optimizer can then
                // push them down independently. Especially if they refer columns from different tables
                // this will be more performant.
                // So:
                // filter[foo == bar & ham == spam]
                // filter [foo == bar]
                // filter [ham == spam]
                let mut predicates = vec![];

                let mut stack = vec![predicate_ae.node()];
                while let Some(n) = stack.pop() {
                    if let AExpr::BinaryExpr {
                        left,
                        op: Operator::And | Operator::LogicalAnd,
                        right,
                    } = ctxt.expr_arena.get(n)
                    {
                        stack.push(*left);
                        stack.push(*right);
                    } else {
                        predicates.push(n)
                    }
                }

                for predicate in predicates {
                    let predicate = ExprIR::from_node(predicate, ctxt.expr_arena);
                    ctxt.conversion_optimizer
                        .push_scratch(predicate.node(), ctxt.expr_arena);
                    let lp = IR::Filter { input, predicate };
                    input = run_conversion(lp, ctxt, "filter")?;
                }

                Ok(input)
            } else {
                ctxt.conversion_optimizer
                    .push_scratch(predicate_ae.node(), ctxt.expr_arena);
                let lp = IR::Filter {
                    input,
                    predicate: predicate_ae,
                };
                run_conversion(lp, ctxt, "filter")
            };
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
            let schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);
            let (exprs, schema) = prepare_projection(expr, &schema, ctxt.opt_flags)
                .map_err(|e| e.context(failed_here!(select)))?;

            if exprs.is_empty() {
                ctxt.lp_arena.replace(input, empty_df());
            }

            let schema = Arc::new(schema);
            let eirs = to_expr_irs(exprs, ctxt.expr_arena)?;
            ctxt.conversion_optimizer
                .fill_scratch(&eirs, ctxt.expr_arena);

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
            // note: if given an Expr::Columns, count the individual cols
            let n_by_exprs = if by_column.len() == 1 {
                match &by_column[0] {
                    Expr::Columns(cols) => cols.len(),
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

            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(sort)))?;

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
                let exprs = expand_expressions(
                    input,
                    vec![c],
                    ctxt.lp_arena,
                    ctxt.expr_arena,
                    ctxt.opt_flags,
                )
                .map_err(|e| e.context(failed_here!(sort)))?;

                nulls_last.extend(std::iter::repeat(n).take(exprs.len()));
                descending.extend(std::iter::repeat(d).take(exprs.len()));
                expanded_cols.extend(exprs);
            }
            sort_options.nulls_last = nulls_last;
            sort_options.descending = descending;

            ctxt.conversion_optimizer
                .fill_scratch(&expanded_cols, ctxt.expr_arena);
            let by_column = expanded_cols;

            let lp = IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            };

            return run_conversion(lp, ctxt, "sort").map_err(|e| e.context(failed_here!(sort)));
        },
        DslPlan::Cache { input, id } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(cache)))?;
            IR::Cache {
                input,
                id,
                cache_hits: crate::constants::UNLIMITED_CACHE,
            }
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
                options,
                ctxt,
            )
            .map_err(|e| e.context(failed_here!(join)))
            .map(|t| t.0)
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
        DslPlan::Distinct { input, options } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(unique)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            let subset = options
                .subset
                .map(|s| {
                    let cols = expand_selectors(s, input_schema.as_ref(), &[])?;

                    // Checking if subset columns exist in the dataframe
                    for col in cols.iter() {
                        let _ = input_schema
                            .try_get(col)
                            .map_err(|_| polars_err!(col_not_found = col))?;
                    }

                    Ok::<_, PolarsError>(cols)
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
                    let columns = expand_selectors(columns, &input_schema, &[])?;
                    validate_columns_in_input(columns.as_ref(), &input_schema, "explode")?;
                    polars_ensure!(!columns.is_empty() || allow_empty, InvalidOperation: "no columns provided in explode");
                    if columns.is_empty() {
                        return Ok(input);
                    }
                    let function = FunctionIR::Explode {
                        columns,
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
                DslFunction::Drop(DropFunction { to_drop, strict }) => {
                    let to_drop = expand_selectors(to_drop, &input_schema, &[])?;
                    let to_drop = to_drop.iter().map(|s| s.as_ref()).collect::<PlHashSet<_>>();

                    if strict {
                        for col_name in to_drop.iter() {
                            polars_ensure!(
                                input_schema.contains(col_name),
                                col_not_found = col_name
                            );
                        }
                    }

                    let mut output_schema =
                        Schema::with_capacity(input_schema.len().saturating_sub(to_drop.len()));

                    for (col_name, dtype) in input_schema.iter() {
                        if !to_drop.contains(col_name.as_str()) {
                            output_schema.with_column(col_name.clone(), dtype.clone());
                        }
                    }

                    if output_schema.is_empty() {
                        ctxt.lp_arena.replace(input, empty_df());
                    }

                    IR::SimpleProjection {
                        input,
                        columns: Arc::new(output_schema),
                    }
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
                    let schema = Arc::new(expressions_to_schema(
                        &exprs,
                        &input_schema,
                        Context::Default,
                    )?);
                    let eirs = to_expr_irs(exprs, ctxt.expr_arena)?;

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
            IR::Sink { input, payload }
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

            IR::MergeSorted {
                input_left,
                input_right,
                key,
            }
        },
        DslPlan::IR { node, dsl, version } => {
            return if node.is_some()
                && version == ctxt.lp_arena.version()
                && ctxt.conversion_optimizer.used_arenas.insert(version)
            {
                Ok(node.unwrap())
            } else {
                to_alp_impl(owned(dsl), ctxt)
            }
        },
    };
    Ok(ctxt.lp_arena.add(v))
}

fn expand_filter(
    predicate: Expr,
    input: Node,
    lp_arena: &Arena<IR>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<Expr> {
    let schema = lp_arena.get(input).schema(lp_arena);
    let predicate = if has_expr(&predicate, |e| match e {
        Expr::Column(name) => is_regex_projection(name),
        Expr::Wildcard
        | Expr::Selector(_)
        | Expr::RenameAlias { .. }
        | Expr::Columns(_)
        | Expr::DtypeColumn(_)
        | Expr::IndexColumn(_)
        | Expr::Nth(_) => true,
        #[cfg(feature = "dtype-struct")]
        Expr::Function {
            function: FunctionExpr::StructExpr(StructFunction::FieldByIndex(_)),
            ..
        } => true,
        _ => false,
    }) {
        let mut rewritten = rewrite_projections(vec![predicate], &schema, &[], opt_flags)?;
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
                    expanded.push_str(&format!("\t{e:?},\n"))
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
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<(Vec<ExprIR>, SchemaRef)> {
    let schema = lp_arena.get(input).schema(lp_arena);
    let mut new_schema = (**schema).clone();
    let (exprs, _) = prepare_projection(exprs, &schema, opt_flags)?;
    let mut output_names = PlHashSet::with_capacity(exprs.len());

    let mut arena = Arena::with_capacity(8);
    for e in &exprs {
        let field = e
            .to_field_amortized(&schema, Context::Default, &mut arena)
            .unwrap();

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
        new_schema.with_column(field.name, field.dtype.materialize_unknown(true)?);
        arena.clear();
    }

    let eirs = to_expr_irs(exprs, expr_arena)?;
    Ok((eirs, Arc::new(new_schema)))
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
    let current_schema = lp_arena.get(input).schema(lp_arena);
    let current_schema = current_schema.as_ref();
    let mut keys = rewrite_projections(keys, current_schema, &[], opt_flags)?;

    // Initialize schema from keys
    let mut schema = expressions_to_schema(&keys, current_schema, Context::Default)?;

    #[allow(unused_mut)]
    let mut pop_keys = false;
    // Add dynamic groupby index column(s)
    // Also add index columns to keys for expression expansion.
    #[cfg(feature = "dynamic_group_by")]
    {
        if let Some(options) = _options.rolling.as_ref() {
            let name = options.index_column.clone();
            let dtype = current_schema.try_get(name.as_str())?;
            keys.push(col(name.clone()));
            pop_keys = true;
            schema.with_column(name.clone(), dtype.clone());
        } else if let Some(options) = _options.dynamic.as_ref() {
            let name = options.index_column.clone();
            keys.push(col(name.clone()));
            pop_keys = true;
            let dtype = current_schema.try_get(name.as_str())?;
            if options.include_boundaries {
                schema.with_column("_lower_boundary".into(), dtype.clone());
                schema.with_column("_upper_boundary".into(), dtype.clone());
            }
            schema.with_column(name.clone(), dtype.clone());
        }
    }
    let keys_index_len = schema.len();

    let aggs = rewrite_projections(aggs, current_schema, &keys, opt_flags)?;
    if pop_keys {
        let _ = keys.pop();
    }

    // Add aggregation column(s)
    let aggs_schema = expressions_to_schema(&aggs, current_schema, Context::Aggregation)?;
    schema.merge(aggs_schema);

    // Make sure aggregation columns do not contain keys or index columns
    if schema.len() < (keys_index_len + aggs.len()) {
        let mut names = PlHashSet::with_capacity(schema.len());
        for expr in aggs.iter().chain(keys.iter()) {
            let name = expr_output_name(expr)?;
            polars_ensure!(names.insert(name.clone()), duplicate = name)
        }
    }
    let keys = to_expr_irs(keys, expr_arena)?;
    let aggs = to_expr_irs(aggs, expr_arena)?;
    validate_expressions(&keys, expr_arena, current_schema, "group by")?;
    validate_expressions(&aggs, expr_arena, current_schema, "group by")?;

    Ok((keys, aggs, Arc::new(schema)))
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
    hive_parts: Option<&HivePartitions>,
) -> Option<Arc<[PlSmallStr]>> {
    // Update `with_columns` with a projection so that hive columns aren't loaded from the
    // file
    let hive_parts = hive_parts?;
    let hive_schema = hive_parts.schema();

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
