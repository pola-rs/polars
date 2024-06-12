use expr_expansion::{is_regex_projection, rewrite_projections};

use super::stack_opt::ConversionOpt;
use super::*;

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

fn empty_df() -> IR {
    IR::DataFrameScan {
        df: Arc::new(Default::default()),
        schema: Arc::new(Default::default()),
        output_schema: None,
        projection: None,
        selection: None,
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
    simplify_expr: bool,
    type_coercion: bool,
) -> PolarsResult<Node> {
    let mut convert = ConversionOpt::new(simplify_expr, type_coercion);
    to_alp_impl(lp, expr_arena, lp_arena, &mut convert)
}

/// converts LogicalPlan to IR
/// it adds expressions & lps to the respective arenas as it traverses the plan
/// finally it returns the top node of the logical plan
#[recursive]
pub fn to_alp_impl(
    lp: DslPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<IR>,
    convert: &mut ConversionOpt,
) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;

    fn run_conversion(
        lp: IR,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        convert: &mut ConversionOpt,
        name: &str,
    ) -> PolarsResult<Node> {
        let lp_node = lp_arena.add(lp);
        convert
            .coerce_types(expr_arena, lp_arena, lp_node)
            .map_err(|e| e.context(format!("'{name}' failed").into()))?;

        Ok(lp_node)
    }

    let v = match lp {
        DslPlan::Scan {
            file_info,
            paths,
            predicate,
            mut scan_type,
            file_options,
        } => {
            let mut file_info = if let Some(file_info) = file_info {
                file_info
            } else {
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
                    // FileInfo should be set.
                    FileScan::Anonymous { .. } => unreachable!(),
                }
            };

            if let Some(row_index) = &file_options.row_index {
                let schema = Arc::make_mut(&mut file_info.schema);
                *schema = schema
                    .new_inserting_at_index(0, row_index.name.as_ref().into(), IDX_DTYPE)
                    .unwrap();
            }

            IR::Scan {
                file_info,
                paths,
                output_schema: None,
                predicate: predicate.map(|expr| to_expr_ir(expr, expr_arena)),
                scan_type,
                file_options,
            }
        },
        #[cfg(feature = "python")]
        DslPlan::PythonScan { options } => IR::PythonScan {
            options,
            predicate: None,
        },
        DslPlan::Union { inputs, args } => {
            let mut inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, expr_arena, lp_arena, convert))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_input!(vertical concat)))?;

            if args.diagonal {
                inputs = convert_utils::convert_diagonal_concat(inputs, lp_arena, expr_arena);
            }

            if args.to_supertypes {
                convert_utils::convert_st_union(&mut inputs, lp_arena, expr_arena)
                    .map_err(|e| e.context(failed_input!(vertical concat)))?;
            }
            let options = args.into();
            IR::Union { inputs, options }
        },
        DslPlan::HConcat { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp_impl(lp, expr_arena, lp_arena, convert))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_input!(horizontal concat)))?;

            let schema = convert_utils::h_concat_schema(&inputs, lp_arena)?;

            IR::HConcat {
                inputs,
                schema,
                options,
            }
        },
        DslPlan::Filter { input, predicate } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(filter)))?;
            let predicate = expand_filter(predicate, input, lp_arena)
                .map_err(|e| e.context(failed_here!(filter)))?;
            let predicate = to_expr_ir(predicate, expr_arena);

            convert.push_scratch(predicate.node(), expr_arena);

            let lp = IR::Filter { input, predicate };
            return run_conversion(lp, lp_arena, expr_arena, convert, "filter");
        },
        DslPlan::Slice { input, offset, len } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(slice)))?;
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
            options,
        } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(select)))?;
            let schema = lp_arena.get(input).schema(lp_arena);
            let (exprs, schema) =
                prepare_projection(expr, &schema).map_err(|e| e.context(failed_here!(select)))?;

            if exprs.is_empty() {
                lp_arena.replace(input, empty_df());
            }

            let schema = Arc::new(schema);
            let eirs = to_expr_irs(exprs, expr_arena);
            convert.fill_scratch(&eirs, expr_arena);

            let lp = IR::Select {
                expr: eirs,
                input,
                schema,
                options,
            };

            return run_conversion(lp, lp_arena, expr_arena, convert, "select");
        },
        DslPlan::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(sort)))?;
            let by_column = expand_expressions(input, by_column, lp_arena, expr_arena)
                .map_err(|e| e.context(failed_here!(sort)))?;

            convert.fill_scratch(&by_column, expr_arena);
            let lp = IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            };

            return run_conversion(lp, lp_arena, expr_arena, convert, "sort");
        },
        DslPlan::Cache {
            input,
            id,
            cache_hits,
        } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(cache)))?;
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
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(group_by)))?;

            let (keys, aggs, schema) =
                resolve_group_by(input, keys, aggs, &options, lp_arena, expr_arena)
                    .map_err(|e| e.context(failed_here!(group_by)))?;

            let (apply, schema) = if let Some((apply, schema)) = apply {
                (Some(apply), schema)
            } else {
                (None, schema)
            };

            convert.fill_scratch(&keys, expr_arena);
            convert.fill_scratch(&aggs, expr_arena);

            let lp = IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options,
            };

            return run_conversion(lp, lp_arena, expr_arena, convert, "group_by");
        },
        DslPlan::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            options,
        } => {
            for e in left_on.iter().chain(right_on.iter()) {
                if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
                    polars_bail!(
                        ComputeError:
                        "'alias' is not allowed in a join key, use 'with_columns' first",
                    )
                }
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

            let input_left = to_alp_impl(owned(input_left), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(join left)))?;
            let input_right = to_alp_impl(owned(input_right), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(join, right)))?;

            let schema_left = lp_arena.get(input_left).schema(lp_arena);
            let schema_right = lp_arena.get(input_right).schema(lp_arena);

            let schema =
                det_join_schema(&schema_left, &schema_right, &left_on, &right_on, &options)
                    .map_err(|e| e.context(failed_here!(join schema resolving)))?;

            let left_on = to_expr_irs_ignore_alias(left_on, expr_arena);
            let right_on = to_expr_irs_ignore_alias(right_on, expr_arena);
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

            convert.fill_scratch(&left_on, expr_arena);
            convert.fill_scratch(&right_on, expr_arena);

            let lp = IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            };
            return run_conversion(lp, lp_arena, expr_arena, convert, "join");
        },
        DslPlan::HStack {
            input,
            exprs,
            options,
        } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(with_columns)))?;
            let (exprs, schema) = resolve_with_columns(exprs, input, lp_arena, expr_arena)
                .map_err(|e| e.context(failed_here!(with_columns)))?;

            convert.fill_scratch(&exprs, expr_arena);
            let lp = IR::HStack {
                input,
                exprs,
                schema,
                options,
            };
            return run_conversion(lp, lp_arena, expr_arena, convert, "with_columns");
        },
        DslPlan::Distinct { input, options } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(unique)))?;
            IR::Distinct { input, options }
        },
        DslPlan::MapFunction { input, function } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert).map_err(|e| {
                e.context(failed_input_args!(format!("{}", function).to_lowercase()))
            })?;
            let input_schema = lp_arena.get(input).schema(lp_arena);

            match function {
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

                    let (exprs, schema) = resolve_with_columns(exprs, input, lp_arena, expr_arena)
                        .map_err(|e| e.context(failed_here!(fill_nan)))?;

                    convert.fill_scratch(&exprs, expr_arena);

                    let lp = IR::HStack {
                        input,
                        exprs,
                        schema,
                        options: ProjectionOptions {
                            duplicate_check: false,
                            ..Default::default()
                        },
                    };
                    return run_conversion(lp, lp_arena, expr_arena, convert, "fill_nan");
                },
                DslFunction::Drop(to_drop) => {
                    let mut output_schema =
                        Schema::with_capacity(input_schema.len().saturating_sub(to_drop.len()));

                    for (col_name, dtype) in input_schema.iter() {
                        if !to_drop.contains(col_name.as_str()) {
                            output_schema.with_column(col_name.clone(), dtype.clone());
                        }
                    }

                    if output_schema.is_empty() {
                        lp_arena.replace(input, empty_df());
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
                    let eirs = to_expr_irs(exprs, expr_arena);

                    convert.fill_scratch(&eirs, expr_arena);

                    let lp = IR::Select {
                        input,
                        expr: eirs,
                        schema,
                        options: ProjectionOptions {
                            duplicate_check: false,
                            ..Default::default()
                        },
                    };
                    return run_conversion(lp, lp_arena, expr_arena, convert, "stats");
                },
                _ => {
                    let function = function.into_function_node(&input_schema)?;
                    IR::MapFunction { input, function }
                },
            }
        },
        DslPlan::ExtContext { input, contexts } => {
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(with_context)))?;
            let contexts = contexts
                .into_iter()
                .map(|lp| to_alp_impl(lp, expr_arena, lp_arena, convert))
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(|e| e.context(failed_here!(with_context)))?;

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
            let input = to_alp_impl(owned(input), expr_arena, lp_arena, convert)
                .map_err(|e| e.context(failed_input!(sink)))?;
            IR::Sink { input, payload }
        },
        DslPlan::IR { node, dsl, version } => {
            return if let (true, Some(node)) = (version == lp_arena.version(), node) {
                Ok(node)
            } else {
                to_alp_impl(owned(dsl), expr_arena, lp_arena, convert)
            }
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
        | Expr::IndexColumn(_)
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
) -> PolarsResult<(Vec<ExprIR>, SchemaRef)> {
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

    let eirs = to_expr_irs(exprs, expr_arena);
    Ok((eirs, Arc::new(new_schema)))
}

fn resolve_group_by(
    input: Node,
    keys: Vec<Expr>,
    aggs: Vec<Expr>,
    _options: &GroupbyOptions,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<(Vec<ExprIR>, Vec<ExprIR>, SchemaRef)> {
    let current_schema = lp_arena.get(input).schema(lp_arena);
    let current_schema = current_schema.as_ref();
    let keys = rewrite_projections(keys, current_schema, &[])?;
    let aggs = rewrite_projections(aggs, current_schema, &keys)?;

    // Initialize schema from keys
    let mut schema = expressions_to_schema(&keys, current_schema, Context::Default)?;

    // Add dynamic groupby index column(s)
    #[cfg(feature = "dynamic_group_by")]
    {
        if let Some(options) = _options.rolling.as_ref() {
            let name = &options.index_column;
            let dtype = current_schema.try_get(name)?;
            schema.with_column(name.clone(), dtype.clone());
        } else if let Some(options) = _options.dynamic.as_ref() {
            let name = &options.index_column;
            let dtype = current_schema.try_get(name)?;
            if options.include_boundaries {
                schema.with_column("_lower_boundary".into(), dtype.clone());
                schema.with_column("_upper_boundary".into(), dtype.clone());
            }
            schema.with_column(name.clone(), dtype.clone());
        }
    }
    let keys_index_len = schema.len();

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
    let aggs = to_expr_irs(aggs, expr_arena);
    let keys = keys.convert(|e| to_expr_ir(e.clone(), expr_arena));

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
