use std::path::PathBuf;

use arrow::datatypes::ArrowSchemaRef;
use either::Either;
use expr_expansion::{is_regex_projection, rewrite_projections};
use hive::{hive_partitions_from_paths, HivePartitions};
#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars_io::cloud::CloudOptions;
#[cfg(any(feature = "csv", feature = "json"))]
use polars_io::path_utils::expand_paths;
#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars_io::path_utils::{expand_paths_hive, expanded_from_single_directory};

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
        filter: None,
    }
}

macro_rules! failed_input {
    ($($t:tt)*) => {
        failed_input_args!(stringify!($($t)*))
    }
}
macro_rules! failed_input_args {
    ($name:expr) => {
        format!("'{}' input failed to resolve", $name).into()
    };
}

macro_rules! failed_here {
    ($($t:tt)*) => {
        format!("'{}' failed", stringify!($($t)*)).into()
    }
}

pub fn to_alp(
    lp: DslPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<IR>,
    // Only `SIMPLIFY_EXPR` and `TYPE_COERCION` are respected.
    opt_flags: &mut OptFlags,
) -> PolarsResult<Node> {
    let conversion_optimizer = ConversionOptimizer::new(
        opt_flags.contains(OptFlags::SIMPLIFY_EXPR),
        opt_flags.contains(OptFlags::TYPE_COERCION),
    );

    let mut ctxt = ConversionContext {
        expr_arena,
        lp_arena,
        conversion_optimizer,
        opt_flags,
    };

    to_alp_impl(lp, &mut ctxt)
}

struct ConversionContext<'a> {
    expr_arena: &'a mut Arena<AExpr>,
    lp_arena: &'a mut Arena<IR>,
    conversion_optimizer: ConversionOptimizer,
    opt_flags: &'a mut OptFlags,
}

/// converts LogicalPlan to IR
/// it adds expressions & lps to the respective arenas as it traverses the plan
/// finally it returns the top node of the logical plan
#[recursive]
pub fn to_alp_impl(lp: DslPlan, ctxt: &mut ConversionContext) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;

    fn run_conversion(lp: IR, ctxt: &mut ConversionContext, name: &str) -> PolarsResult<Node> {
        let lp_node = ctxt.lp_arena.add(lp);
        ctxt.conversion_optimizer
            .coerce_types(ctxt.expr_arena, ctxt.lp_arena, lp_node)
            .map_err(|e| e.context(format!("'{name}' failed").into()))?;

        Ok(lp_node)
    }

    let v = match lp {
        DslPlan::Scan {
            paths,
            file_info,
            hive_parts,
            predicate,
            mut file_options,
            mut scan_type,
        } => {
            let paths = expand_scan_paths(paths, &mut scan_type, &mut file_options)?;

            let file_info_read = file_info.read().unwrap();

            // leading `_` as clippy doesn't understand that you don't want to read from a lock guard
            // if you want to keep it alive.
            let mut _file_info_write: Option<_>;
            let mut resolved_file_info = if let Some(file_info) = &*file_info_read {
                _file_info_write = None;
                let out = file_info.clone();
                drop(file_info_read);
                out
            } else {
                // Lock so that we don't resolve the same schema in parallel.
                drop(file_info_read);

                // Set write lock and keep that lock until all fields in `file_info` are resolved.
                _file_info_write = Some(file_info.write().unwrap());

                match &mut scan_type {
                    #[cfg(feature = "parquet")]
                    FileScan::Parquet {
                        cloud_options,
                        metadata,
                        ..
                    } => {
                        let (file_info, md) =
                            scans::parquet_file_info(&paths, &file_options, cloud_options.as_ref())
                                .map_err(|e| e.context(failed_here!(parquet scan)))?;
                        *metadata = md;
                        file_info
                    },
                    #[cfg(feature = "ipc")]
                    FileScan::Ipc {
                        cloud_options,
                        metadata,
                        ..
                    } => {
                        let (file_info, md) =
                            scans::ipc_file_info(&paths, &file_options, cloud_options.as_ref())
                                .map_err(|e| e.context(failed_here!(ipc scan)))?;
                        *metadata = Some(md);
                        file_info
                    },
                    #[cfg(feature = "csv")]
                    FileScan::Csv {
                        options,
                        cloud_options,
                    } => {
                        scans::csv_file_info(&paths, &file_options, options, cloud_options.as_ref())
                            .map_err(|e| e.context(failed_here!(csv scan)))?
                    },
                    #[cfg(feature = "json")]
                    FileScan::NDJson {
                        options,
                        cloud_options,
                    } => scans::ndjson_file_info(
                        &paths,
                        &file_options,
                        options,
                        cloud_options.as_ref(),
                    )
                    .map_err(|e| e.context(failed_here!(ndjson scan)))?,
                    // FileInfo should be set.
                    FileScan::Anonymous { .. } => unreachable!(),
                }
            };

            let hive_parts = if hive_parts.is_some() {
                hive_parts
            } else if file_options.hive_options.enabled.unwrap_or(false)
                && resolved_file_info.reader_schema.is_some()
            {
                #[allow(unused_assignments)]
                let mut owned = None;

                hive_partitions_from_paths(
                    paths.as_ref(),
                    file_options.hive_options.hive_start_idx,
                    file_options.hive_options.schema.clone(),
                    match resolved_file_info.reader_schema.as_ref().unwrap() {
                        Either::Left(v) => {
                            owned = Some(Schema::from(v));
                            owned.as_ref().unwrap()
                        },
                        Either::Right(v) => v.as_ref(),
                    },
                    file_options.hive_options.try_parse_dates,
                )?
            } else {
                None
            };

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

            // Only if we have a writing file handle we must resolve hive partitions
            // update schema's etc.
            if let Some(lock) = &mut _file_info_write {
                if let Some(ref hive_parts) = hive_parts {
                    let hive_schema = hive_parts[0].schema();
                    resolved_file_info.update_schema_with_hive_schema(hive_schema.clone());
                }

                if let Some(ref file_path_col) = file_options.include_file_paths {
                    let schema = Arc::make_mut(&mut resolved_file_info.schema);

                    if schema.contains(file_path_col) {
                        polars_bail!(
                            Duplicate: r#"column name for file paths "{}" conflicts with column name from file"#,
                            file_path_col
                        );
                    }

                    schema.insert_at_index(
                        schema.len(),
                        file_path_col.as_ref().into(),
                        DataType::String,
                    )?;
                }

                **lock = Some(resolved_file_info.clone());
            }

            file_options.with_columns = if resolved_file_info.reader_schema.is_some() {
                maybe_init_projection_excluding_hive(
                    resolved_file_info.reader_schema.as_ref().unwrap(),
                    hive_parts.as_ref().map(|x| &x[0]),
                )
            } else {
                None
            };

            if let Some(row_index) = &file_options.row_index {
                let schema = Arc::make_mut(&mut resolved_file_info.schema);
                *schema = schema
                    .new_inserting_at_index(0, row_index.name.as_ref().into(), IDX_DTYPE)
                    .unwrap();
            }

            IR::Scan {
                paths,
                file_info: resolved_file_info,
                hive_parts,
                output_schema: None,
                predicate: predicate
                    .map(|expr| to_expr_ir(expr, ctxt.expr_arena))
                    .transpose()?,
                scan_type,
                file_options,
            }
        },
        #[cfg(feature = "python")]
        DslPlan::PythonScan { options } => IR::PythonScan { options },
        DslPlan::Union { inputs, args } => {
            let mut inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_input!(vertical concat)))?;

            if args.diagonal {
                inputs =
                    convert_utils::convert_diagonal_concat(inputs, ctxt.lp_arena, ctxt.expr_arena)?;
            }

            if args.to_supertypes {
                convert_utils::convert_st_union(&mut inputs, ctxt.lp_arena, ctxt.expr_arena)
                    .map_err(|e| e.context(failed_input!(vertical concat)))?;
            }
            let options = args.into();
            IR::Union { inputs, options }
        },
        DslPlan::HConcat { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, ctxt))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_input!(horizontal concat)))?;

            let schema = convert_utils::h_concat_schema(&inputs, ctxt.lp_arena)?;

            IR::HConcat {
                inputs,
                schema,
                options,
            }
        },
        DslPlan::Filter { input, predicate } => {
            let mut input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(filter)))?;
            let predicate = expand_filter(predicate, input, ctxt.lp_arena, ctxt.opt_flags)
                .map_err(|e| e.context(failed_here!(filter)))?;

            let predicate_ae = to_expr_ir(predicate.clone(), ctxt.expr_arena)?;

            return if is_streamable(predicate_ae.node(), ctxt.expr_arena, Context::Default) {
                // Split expression that are ANDed into multiple Filter nodes as the optimizer can then
                // push them down independently. Especially if they refer columns from different tables
                // this will be more performant.
                // So:
                // filter[foo == bar & ham == spam]
                // filter [foo == bar]
                // filter [ham == spam]
                let mut predicates = vec![];
                let mut stack = vec![predicate];
                while let Some(expr) = stack.pop() {
                    if let Expr::BinaryExpr {
                        left,
                        op: Operator::And | Operator::LogicalAnd,
                        right,
                    } = expr
                    {
                        stack.push(Arc::unwrap_or_clone(left));
                        stack.push(Arc::unwrap_or_clone(right));
                    } else {
                        predicates.push(expr)
                    }
                }

                for predicate in predicates {
                    let predicate = to_expr_ir(predicate, ctxt.expr_arena)?;
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
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(slice)))?;
            IR::Slice { input, offset, len }
        },
        DslPlan::DataFrameScan {
            df,
            schema,
            output_schema,
            filter: selection,
        } => IR::DataFrameScan {
            df,
            schema,
            output_schema,
            filter: selection
                .map(|expr| to_expr_ir(expr, ctxt.expr_arena))
                .transpose()?,
        },
        DslPlan::Select {
            expr,
            input,
            options,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(select)))?;
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

            return run_conversion(lp, ctxt, "select");
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
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(sort)))?;

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

            return run_conversion(lp, ctxt, "sort");
        },
        DslPlan::Cache {
            input,
            id,
            cache_hits,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(cache)))?;
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
            apply,
            maintain_order,
            options,
        } => {
            let input =
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(group_by)))?;

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

            return run_conversion(lp, ctxt, "group_by");
        },
        DslPlan::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            mut options,
        } => {
            if matches!(options.args.how, JoinType::Cross) {
                polars_ensure!(left_on.len() + right_on.len() == 0, InvalidOperation: "a 'cross' join doesn't expect any join keys");
            } else {
                let mut turn_off_coalesce = false;
                for e in left_on.iter().chain(right_on.iter()) {
                    if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
                        polars_bail!(
                            ComputeError:
                            "'alias' is not allowed in a join key, use 'with_columns' first",
                        )
                    }
                    // Any expression that is not a simple column expression will turn of coalescing.
                    turn_off_coalesce |= has_expr(e, |e| !matches!(e, Expr::Column(_)));
                }
                if turn_off_coalesce {
                    let options = Arc::make_mut(&mut options);
                    if matches!(options.args.coalesce, JoinCoalesce::CoalesceColumns) {
                        polars_warn!("coalescing join requested but not all join keys are column references, turning off key coalescing");
                    }
                    options.args.coalesce = JoinCoalesce::KeepColumns;
                }

                options.args.validation.is_valid_join(&options.args.how)?;

                polars_ensure!(
                    left_on.len() == right_on.len(),
                    ComputeError:
                        format!(
                            "the number of columns given as join key (left: {}, right:{}) should be equal",
                            left_on.len(),
                            right_on.len()
                        )
                );
            }

            let input_left = to_alp_impl(owned(input_left), ctxt)
                .map_err(|e| e.context(failed_input!(join left)))?;
            let input_right = to_alp_impl(owned(input_right), ctxt)
                .map_err(|e| e.context(failed_input!(join, right)))?;

            let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
            let schema_right = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

            let schema =
                det_join_schema(&schema_left, &schema_right, &left_on, &right_on, &options)
                    .map_err(|e| e.context(failed_here!(join schema resolving)))?;

            let left_on = to_expr_irs_ignore_alias(left_on, ctxt.expr_arena)?;
            let right_on = to_expr_irs_ignore_alias(right_on, ctxt.expr_arena)?;
            let mut joined_on = PlHashSet::new();
            for (l, r) in left_on.iter().zip(right_on.iter()) {
                polars_ensure!(
                    joined_on.insert((l.output_name(), r.output_name())),
                    InvalidOperation: "joining with repeated key names; already joined on {} and {}",
                    l.output_name(),
                    r.output_name()
                )
            }
            drop(joined_on);

            ctxt.conversion_optimizer
                .fill_scratch(&left_on, ctxt.expr_arena);
            ctxt.conversion_optimizer
                .fill_scratch(&right_on, ctxt.expr_arena);

            // Every expression must be elementwise so that we are
            // guaranteed the keys for a join are all the same length.
            let all_elementwise =
                |aexprs: &[ExprIR]| all_streamable(aexprs, &*ctxt.expr_arena, Context::Default);
            polars_ensure!(
                all_elementwise(&left_on) && all_elementwise(&right_on),
                InvalidOperation: "All join key expressions must be elementwise."
            );
            let lp = IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            };
            return run_conversion(lp, ctxt, "join");
        },
        DslPlan::HStack {
            input,
            exprs,
            options,
        } => {
            let input = to_alp_impl(owned(input), ctxt)
                .map_err(|e| e.context(failed_input!(with_columns)))?;
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
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(unique)))?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            let subset = options
                .subset
                .map(|s| expand_selectors(s, input_schema.as_ref(), &[]))
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
            let input = to_alp_impl(owned(input), ctxt).map_err(|e| {
                e.context(failed_input_args!(format!("{}", function).to_lowercase()))
            })?;
            let input_schema = ctxt.lp_arena.get(input).schema(ctxt.lp_arena);

            match function {
                DslFunction::Explode {
                    columns,
                    allow_empty,
                } => {
                    let columns = expand_selectors(columns, &input_schema, &[])?;
                    validate_columns_in_input(&columns, &input_schema, "explode")?;
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
                            DataType::Float32 | DataType::Float64 => {
                                Some(col(name).fill_nan(fill_value.clone()).alias(name))
                            },
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
                            |dt| dt.is_numeric() || dt.is_bool(),
                            |name| col(name).var(ddof),
                            &input_schema,
                        ),
                        StatsFunction::Std { ddof } => stats_helper(
                            |dt| dt.is_numeric() || dt.is_bool(),
                            |name| col(name).std(ddof),
                            &input_schema,
                        ),
                        StatsFunction::Quantile { quantile, interpol } => stats_helper(
                            |dt| dt.is_numeric(),
                            |name| col(name).quantile(quantile.clone(), interpol),
                            &input_schema,
                        ),
                        StatsFunction::Mean => stats_helper(
                            |dt| dt.is_numeric() || dt.is_temporal() || dt == &DataType::Boolean,
                            |name| col(name).mean(),
                            &input_schema,
                        ),
                        StatsFunction::Sum => stats_helper(
                            |dt| {
                                dt.is_numeric()
                                    || dt.is_decimal()
                                    || matches!(dt, DataType::Boolean | DataType::Duration(_))
                            },
                            |name| col(name).sum(),
                            &input_schema,
                        ),
                        StatsFunction::Min => {
                            stats_helper(|dt| dt.is_ord(), |name| col(name).min(), &input_schema)
                        },
                        StatsFunction::Max => {
                            stats_helper(|dt| dt.is_ord(), |name| col(name).max(), &input_schema)
                        },
                        StatsFunction::Median => stats_helper(
                            |dt| dt.is_numeric() || dt.is_temporal() || dt == &DataType::Boolean,
                            |name| col(name).median(),
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
                .map_err(|e| e.context(failed_input!(with_context)))?;
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
                to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_input!(sink)))?;
            IR::Sink { input, payload }
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

/// Expand scan paths if they were not already expanded.
#[allow(unused_variables)]
fn expand_scan_paths(
    paths: Arc<Mutex<(Arc<Vec<PathBuf>>, bool)>>,
    scan_type: &mut FileScan,
    file_options: &mut FileScanOptions,
) -> PolarsResult<Arc<Vec<PathBuf>>> {
    #[allow(unused_mut)]
    let mut lock = paths.lock().unwrap();

    // Return if paths are already expanded
    if lock.1 {
        return Ok(lock.0.clone());
    }

    {
        let paths_expanded = match &scan_type {
            #[cfg(feature = "parquet")]
            FileScan::Parquet { cloud_options, .. } => {
                expand_scan_paths_with_hive_update(&lock.0, file_options, cloud_options)?
            },
            #[cfg(feature = "ipc")]
            FileScan::Ipc { cloud_options, .. } => {
                expand_scan_paths_with_hive_update(&lock.0, file_options, cloud_options)?
            },
            #[cfg(feature = "csv")]
            FileScan::Csv { cloud_options, .. } => {
                expand_paths(&lock.0, file_options.glob, cloud_options.as_ref())?
            },
            #[cfg(feature = "json")]
            FileScan::NDJson { cloud_options, .. } => {
                expand_paths(&lock.0, file_options.glob, cloud_options.as_ref())?
            },
            FileScan::Anonymous { .. } => unreachable!(), // Invariant: Anonymous scans are already expanded.
        };

        #[allow(unreachable_code)]
        {
            *lock = (paths_expanded, true);

            Ok(lock.0.clone())
        }
    }
}

/// Expand scan paths and update the Hive partition information of `file_options`.
#[cfg(any(feature = "ipc", feature = "parquet"))]
fn expand_scan_paths_with_hive_update(
    paths: &[PathBuf],
    file_options: &mut FileScanOptions,
    cloud_options: &Option<CloudOptions>,
) -> PolarsResult<Arc<Vec<PathBuf>>> {
    let hive_enabled = file_options.hive_options.enabled;
    let (expanded_paths, hive_start_idx) = expand_paths_hive(
        paths,
        file_options.glob,
        cloud_options.as_ref(),
        hive_enabled.unwrap_or(false),
    )?;
    let inferred_hive_enabled = hive_enabled
        .unwrap_or_else(|| expanded_from_single_directory(paths, expanded_paths.as_ref()));

    file_options.hive_options.enabled = Some(inferred_hive_enabled);
    file_options.hive_options.hive_start_idx = hive_start_idx;

    Ok(expanded_paths)
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
        new_schema.with_column(field.name().clone(), field.data_type().clone());
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
            let name = &options.index_column;
            let dtype = current_schema.try_get(name)?;
            keys.push(col(name));
            pop_keys = true;
            schema.with_column(name.clone(), dtype.clone());
        } else if let Some(options) = _options.dynamic.as_ref() {
            let name = &options.index_column;
            keys.push(col(name));
            pop_keys = true;
            let dtype = current_schema.try_get(name)?;
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
    let aggs = to_expr_irs(aggs, expr_arena)?;
    let keys = to_expr_irs(keys, expr_arena)?;

    Ok((keys, aggs, Arc::new(schema)))
}
fn stats_helper<F, E>(condition: F, expr: E, schema: &Schema) -> Vec<Expr>
where
    F: Fn(&DataType) -> bool,
    E: Fn(&str) -> Expr,
{
    schema
        .iter()
        .map(|(name, dt)| {
            if condition(dt) {
                expr(name)
            } else {
                lit(NULL).cast(dt.clone()).alias(name)
            }
        })
        .collect()
}

pub(crate) fn maybe_init_projection_excluding_hive(
    reader_schema: &Either<ArrowSchemaRef, SchemaRef>,
    hive_parts: Option<&HivePartitions>,
) -> Option<Arc<[String]>> {
    // Update `with_columns` with a projection so that hive columns aren't loaded from the
    // file
    let hive_parts = hive_parts?;

    let hive_schema = hive_parts.schema();

    let (first_hive_name, _) = hive_schema.get_at_index(0)?;

    let names = match reader_schema {
        Either::Left(ref v) => {
            let names = v.get_names();
            names.contains(&first_hive_name.as_str()).then_some(names)
        },
        Either::Right(ref v) => v.contains(first_hive_name.as_str()).then(|| v.get_names()),
    };

    let names = names?;

    Some(
        names
            .iter()
            .filter(|x| !hive_schema.contains(x))
            .map(ToString::to_string)
            .collect::<Arc<[_]>>(),
    )
}
