use super::*;
use crate::logical_plan::expr_expansion::{is_regex_projection, rewrite_projections};

pub fn to_expr_ir(expr: Expr, arena: &mut Arena<AExpr>) -> ExprIR {
    let mut state = ConversionState::new();
    let node = to_aexpr_impl(expr, arena, &mut state);
    ExprIR::new(node, state.output_name)
}

fn to_expr_irs(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> Vec<ExprIR> {
    input.convert_owned(|e| to_expr_ir(e, arena))
}

pub fn to_expr_ir_ignore_alias(expr: Expr, arena: &mut Arena<AExpr>) -> ExprIR {
    let mut state = ConversionState::new();
    state.ignore_alias = true;
    let node = to_aexpr_impl(expr, arena, &mut state);
    ExprIR::new(node, state.output_name)
}

fn to_expr_irs_ignore_alias(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> Vec<ExprIR> {
    input.convert_owned(|e| to_expr_ir_ignore_alias(e, arena))
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
pub fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    to_aexpr_impl(
        expr,
        arena,
        &mut ConversionState {
            prune_alias: false,
            ..Default::default()
        },
    )
}

#[derive(Default)]
struct ConversionState {
    output_name: OutputName,
    /// Remove alias from the expressions and set as [`OutputName`].
    prune_alias: bool,
    /// If an `alias` is encountered prune and ignore it.
    ignore_alias: bool,
}

impl ConversionState {
    fn new() -> Self {
        Self {
            prune_alias: true,
            ..Default::default()
        }
    }
}

fn to_aexprs(input: Vec<Expr>, arena: &mut Arena<AExpr>, state: &mut ConversionState) -> Vec<Node> {
    input
        .into_iter()
        .map(|e| to_aexpr_impl(e, arena, state))
        .collect()
}

fn set_function_output_name<F>(e: &[ExprIR], state: &mut ConversionState, function_fmt: F)
where
    F: FnOnce() -> Cow<'static, str>,
{
    if state.output_name.is_none() {
        if e.is_empty() {
            state.output_name = OutputName::LiteralLhs(ColumnName::from(function_fmt().as_ref()));
        } else {
            state.output_name = e[0].output_name_inner().clone();
        }
    }
}

/// Converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation.
#[recursive]
fn to_aexpr_impl(expr: Expr, arena: &mut Arena<AExpr>, state: &mut ConversionState) -> Node {
    let owned = Arc::unwrap_or_clone;
    let v = match expr {
        Expr::Explode(expr) => AExpr::Explode(to_aexpr_impl(owned(expr), arena, state)),
        Expr::Alias(e, name) => {
            if state.prune_alias {
                if state.output_name.is_none() && !state.ignore_alias {
                    state.output_name = OutputName::Alias(name);
                }
                to_aexpr_impl(owned(e), arena, state);
                arena.pop().unwrap()
            } else {
                AExpr::Alias(to_aexpr_impl(owned(e), arena, state), name)
            }
        },
        Expr::Literal(lv) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(lv.output_column_name());
            }
            AExpr::Literal(lv)
        },
        Expr::Column(name) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::ColumnLhs(name.clone())
            }
            AExpr::Column(name)
        },
        Expr::BinaryExpr { left, op, right } => {
            let l = to_aexpr_impl(owned(left), arena, state);
            let r = to_aexpr_impl(owned(right), arena, state);
            AExpr::BinaryExpr {
                left: l,
                op,
                right: r,
            }
        },
        Expr::Cast {
            expr,
            data_type,
            strict,
        } => AExpr::Cast {
            expr: to_aexpr_impl(owned(expr), arena, state),
            data_type,
            strict,
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => AExpr::Gather {
            expr: to_aexpr_impl(owned(expr), arena, state),
            idx: to_aexpr_impl(owned(idx), arena, state),
            returns_scalar,
        },
        Expr::Sort { expr, options } => AExpr::Sort {
            expr: to_aexpr_impl(owned(expr), arena, state),
            options,
        },
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => AExpr::SortBy {
            expr: to_aexpr_impl(owned(expr), arena, state),
            by: by
                .into_iter()
                .map(|e| to_aexpr_impl(e, arena, state))
                .collect(),
            sort_options,
        },
        Expr::Filter { input, by } => AExpr::Filter {
            input: to_aexpr_impl(owned(input), arena, state),
            by: to_aexpr_impl(owned(by), arena, state),
        },
        Expr::Agg(agg) => {
            let a_agg = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => AAggExpr::Min {
                    input: to_aexpr_impl(owned(input), arena, state),
                    propagate_nans,
                },
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => AAggExpr::Max {
                    input: to_aexpr_impl(owned(input), arena, state),
                    propagate_nans,
                },
                AggExpr::Median(expr) => AAggExpr::Median(to_aexpr_impl(owned(expr), arena, state)),
                AggExpr::NUnique(expr) => {
                    AAggExpr::NUnique(to_aexpr_impl(owned(expr), arena, state))
                },
                AggExpr::First(expr) => AAggExpr::First(to_aexpr_impl(owned(expr), arena, state)),
                AggExpr::Last(expr) => AAggExpr::Last(to_aexpr_impl(owned(expr), arena, state)),
                AggExpr::Mean(expr) => AAggExpr::Mean(to_aexpr_impl(owned(expr), arena, state)),
                AggExpr::Implode(expr) => {
                    AAggExpr::Implode(to_aexpr_impl(owned(expr), arena, state))
                },
                AggExpr::Count(expr, include_nulls) => {
                    AAggExpr::Count(to_aexpr_impl(owned(expr), arena, state), include_nulls)
                },
                AggExpr::Quantile {
                    expr,
                    quantile,
                    interpol,
                } => AAggExpr::Quantile {
                    expr: to_aexpr_impl(owned(expr), arena, state),
                    quantile: to_aexpr_impl(owned(quantile), arena, state),
                    interpol,
                },
                AggExpr::Sum(expr) => AAggExpr::Sum(to_aexpr_impl(owned(expr), arena, state)),
                AggExpr::Std(expr, ddof) => {
                    AAggExpr::Std(to_aexpr_impl(owned(expr), arena, state), ddof)
                },
                AggExpr::Var(expr, ddof) => {
                    AAggExpr::Var(to_aexpr_impl(owned(expr), arena, state), ddof)
                },
                AggExpr::AggGroups(expr) => {
                    AAggExpr::AggGroups(to_aexpr_impl(owned(expr), arena, state))
                },
            };
            AExpr::Agg(a_agg)
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            // Truthy must be resolved first to get the lhs name first set.
            let t = to_aexpr_impl(owned(truthy), arena, state);
            let p = to_aexpr_impl(owned(predicate), arena, state);
            let f = to_aexpr_impl(owned(falsy), arena, state);
            AExpr::Ternary {
                predicate: p,
                truthy: t,
                falsy: f,
            }
        },
        Expr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => {
            let e = to_expr_irs(input, arena);
            set_function_output_name(&e, state, || Cow::Borrowed(options.fmt_str));
            AExpr::AnonymousFunction {
                input: e,
                function,
                output_type,
                options,
            }
        },
        Expr::Function {
            input,
            function,
            options,
        } => {
            let e = to_expr_irs(input, arena);

            if state.output_name.is_none() {
                // Handles special case functions like `struct.field`.
                if let Some(name) = function.output_name() {
                    state.output_name = OutputName::ColumnLhs(name.clone())
                } else {
                    set_function_output_name(&e, state, || Cow::Owned(format!("{}", &function)));
                }
            }
            AExpr::Function {
                input: e,
                function,
                options,
            }
        },
        Expr::Window {
            function,
            partition_by,
            options,
        } => AExpr::Window {
            function: to_aexpr_impl(owned(function), arena, state),
            partition_by: to_aexprs(partition_by, arena, state),
            options,
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => AExpr::Slice {
            input: to_aexpr_impl(owned(input), arena, state),
            offset: to_aexpr_impl(owned(offset), arena, state),
            length: to_aexpr_impl(owned(length), arena, state),
        },
        Expr::Len => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(get_len_name())
            }
            AExpr::Len
        },
        Expr::Nth(i) => AExpr::Nth(i),
        Expr::Wildcard => AExpr::Wildcard,
        Expr::SubPlan { .. } => panic!("no SQLSubquery expected at this point"),
        Expr::KeepName(_) => panic!("no `name.keep` expected at this point"),
        Expr::Exclude(_, _) => panic!("no `exclude` expected at this point"),
        Expr::RenameAlias { .. } => panic!("no `rename_alias` expected at this point"),
        Expr::Columns { .. } => panic!("no `columns` expected at this point"),
        Expr::DtypeColumn { .. } => panic!("no `dtype-columns` expected at this point"),
        Expr::Selector(_) => panic!("no `selector` expected at this point"),
    };
    arena.add(v)
}

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
            schema,
            options,
        } => {
            let eirs = to_expr_irs(exprs, expr_arena);
            let exprs = eirs.into();
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
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
            schema,
        } => {
            let input = to_alp(owned(input), expr_arena, lp_arena)?;
            let contexts = contexts
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<_>>()?;
            IR::ExtContext {
                input,
                contexts,
                schema,
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
