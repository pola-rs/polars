use arrow::legacy::error::PolarsResult;
use either::Either;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::feature_gated;
use polars_core::utils::{get_numeric_upcast_supertype_lossless, try_get_supertype};
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;

use super::*;
use crate::constants::POLARS_TMP_PREFIX;
use crate::dsl::Expr;
#[cfg(feature = "iejoin")]
use crate::plans::AExpr;

fn check_join_keys(keys: &[Expr]) -> PolarsResult<()> {
    for e in keys {
        if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
            polars_bail!(
                InvalidOperation:
                "'alias' is not allowed in a join key, use 'with_columns' first",
            )
        }
    }
    Ok(())
}

/// Returns: left: join_node, right: last_node (often both the same)
pub fn resolve_join(
    input_left: Either<Arc<DslPlan>, Node>,
    input_right: Either<Arc<DslPlan>, Node>,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    predicates: Vec<Expr>,
    mut options: JoinOptionsIR,
    ctxt: &mut DslConversionContext,
) -> PolarsResult<(Node, Node)> {
    if !predicates.is_empty() {
        feature_gated!("iejoin", {
            debug_assert!(left_on.is_empty() && right_on.is_empty());
            return resolve_join_where(
                input_left.unwrap_left(),
                input_right.unwrap_left(),
                predicates,
                options,
                ctxt,
            );
        })
    }

    let owned = Arc::unwrap_or_clone;
    let mut input_left = input_left.map_right(Ok).right_or_else(|input| {
        to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(join left)))
    })?;
    let mut input_right = input_right.map_right(Ok).right_or_else(|input| {
        to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(join right)))
    })?;

    let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
    let schema_right = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

    if options.args.how.is_cross() {
        polars_ensure!(left_on.len() + right_on.len() == 0, InvalidOperation: "a 'cross' join doesn't expect any join keys");
    } else {
        polars_ensure!(left_on.len() + right_on.len() > 0, InvalidOperation: "expected join keys/predicates");
        check_join_keys(&left_on)?;
        check_join_keys(&right_on)?;

        let mut turn_off_coalesce = false;
        for e in left_on.iter().chain(right_on.iter()) {
            // Any expression that is not a simple column expression will turn of coalescing.
            turn_off_coalesce |= has_expr(e, |e| !matches!(e, Expr::Column(_)));
        }
        if turn_off_coalesce {
            if matches!(options.args.coalesce, JoinCoalesce::CoalesceColumns) {
                polars_warn!(
                    "coalescing join requested but not all join keys are column references, turning off key coalescing"
                );
            }
            options.args.coalesce = JoinCoalesce::KeepColumns;
        }

        options.args.validation.is_valid_join(&options.args.how)?;

        #[cfg(feature = "asof_join")]
        if let JoinType::AsOf(options) = &options.args.how {
            match (&options.left_by, &options.right_by) {
                (None, None) => {},
                (Some(l), Some(r)) => {
                    polars_ensure!(l.len() == r.len(), InvalidOperation: "expected equal number of columns in 'by_left' and 'by_right' in 'asof_join'");
                    validate_columns_in_input(l, &schema_left, "asof_join")?;
                    validate_columns_in_input(r, &schema_right, "asof_join")?;
                },
                _ => {
                    polars_bail!(InvalidOperation: "expected both 'by_left' and 'by_right' to be set in 'asof_join'")
                },
            }
        }

        polars_ensure!(
            left_on.len() == right_on.len(),
            InvalidOperation:
                "the number of columns given as join key (left: {}, right:{}) should be equal",
                left_on.len(),
                right_on.len()
        );
    }

    let mut left_on = left_on
        .into_iter()
        .map(|e| {
            to_expr_ir_materialized_lit(
                e,
                &mut ExprToIRContext::new_with_opt_eager(
                    ctxt.expr_arena,
                    &schema_left,
                    ctxt.opt_flags,
                ),
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let mut right_on = right_on
        .into_iter()
        .map(|e| {
            to_expr_ir_materialized_lit(
                e,
                &mut ExprToIRContext::new_with_opt_eager(
                    ctxt.expr_arena,
                    &schema_right,
                    ctxt.opt_flags,
                ),
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let mut joined_on = PlHashSet::new();

    #[cfg(feature = "iejoin")]
    let check = !matches!(options.args.how, JoinType::IEJoin);
    #[cfg(not(feature = "iejoin"))]
    let check = true;
    if check {
        for (l, r) in left_on.iter().zip(right_on.iter()) {
            polars_ensure!(
                joined_on.insert((l.output_name(), r.output_name())),
                InvalidOperation: "joining with repeated key names; already joined on {} and {}",
                l.output_name(),
                r.output_name()
            )
        }
    }
    drop(joined_on);

    ctxt.conversion_optimizer
        .fill_scratch(&left_on, ctxt.expr_arena);
    ctxt.conversion_optimizer
        .optimize_exprs(ctxt.expr_arena, ctxt.lp_arena, input_left, true)
        .map_err(|e| e.context("'join' failed".into()))?;
    ctxt.conversion_optimizer
        .fill_scratch(&right_on, ctxt.expr_arena);
    ctxt.conversion_optimizer
        .optimize_exprs(ctxt.expr_arena, ctxt.lp_arena, input_right, true)
        .map_err(|e| e.context("'join' failed".into()))?;

    // Re-evaluate because of mutable borrows earlier.
    let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
    let schema_right = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

    // # Resolve scalars
    //
    // Scalars need to be expanded. We translate them to temporary columns added with
    // `with_columns` and remove them later with `project`
    // This way the backends don't have to expand the literals in the join implementation

    let has_scalars = left_on
        .iter()
        .chain(right_on.iter())
        .any(|e| e.is_scalar(ctxt.expr_arena));

    let (schema_left, schema_right) = if has_scalars {
        let mut as_with_columns_l = vec![];
        let mut as_with_columns_r = vec![];
        for (i, e) in left_on.iter().enumerate() {
            if e.is_scalar(ctxt.expr_arena) {
                as_with_columns_l.push((i, e.clone()));
            }
        }
        for (i, e) in right_on.iter().enumerate() {
            if e.is_scalar(ctxt.expr_arena) {
                as_with_columns_r.push((i, e.clone()));
            }
        }

        let mut count = 0;
        let get_tmp_name = |i| format_pl_smallstr!("{POLARS_TMP_PREFIX}{i}");

        // Early clone because of bck.
        let mut schema_right_new = if !as_with_columns_r.is_empty() {
            (**schema_right).clone()
        } else {
            Default::default()
        };
        if !as_with_columns_l.is_empty() {
            let mut schema_left_new = (**schema_left).clone();

            let mut exprs = Vec::with_capacity(as_with_columns_l.len());
            for (i, mut e) in as_with_columns_l {
                let tmp_name = get_tmp_name(count);
                count += 1;
                e.set_alias(tmp_name.clone());
                let dtype = e.dtype(&schema_left_new, ctxt.expr_arena)?;
                schema_left_new.with_column(tmp_name.clone(), dtype.clone());

                let col = ctxt.expr_arena.add(AExpr::Column(tmp_name));
                left_on[i] = ExprIR::from_node(col, ctxt.expr_arena);
                exprs.push(e);
            }
            input_left = ctxt.lp_arena.add(IR::HStack {
                input: input_left,
                exprs,
                schema: Arc::new(schema_left_new),
                options: ProjectionOptions::default(),
            })
        }
        if !as_with_columns_r.is_empty() {
            let mut exprs = Vec::with_capacity(as_with_columns_r.len());
            for (i, mut e) in as_with_columns_r {
                let tmp_name = get_tmp_name(count);
                count += 1;
                e.set_alias(tmp_name.clone());
                let dtype = e.dtype(&schema_right_new, ctxt.expr_arena)?;
                schema_right_new.with_column(tmp_name.clone(), dtype.clone());

                let col = ctxt.expr_arena.add(AExpr::Column(tmp_name));
                right_on[i] = ExprIR::from_node(col, ctxt.expr_arena);
                exprs.push(e);
            }
            input_right = ctxt.lp_arena.add(IR::HStack {
                input: input_right,
                exprs,
                schema: Arc::new(schema_right_new),
                options: ProjectionOptions::default(),
            })
        }

        (
            ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena),
            ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena),
        )
    } else {
        (schema_left, schema_right)
    };

    // Not a closure to avoid borrow issues because we mutate expr_arena as well.
    macro_rules! get_dtype {
        ($expr:expr, $schema:expr) => {
            ctxt.expr_arena
                .get($expr.node())
                .to_dtype(&ToFieldContext::new(ctxt.expr_arena, $schema))
        };
    }

    // As an optimization, when inserting casts for coalescing joins we only insert them beforehand for full-join.
    // This means for e.g. left-join, the LHS key preserves its dtype in the output even if it is joined
    // with an RHS key of wider type.
    let key_cols_coalesced =
        options.args.should_coalesce() && matches!(&options.args.how, JoinType::Full);
    let mut as_with_columns_l = vec![];
    let mut as_with_columns_r = vec![];
    for (lnode, rnode) in left_on.iter_mut().zip(right_on.iter_mut()) {
        let ltype = get_dtype!(lnode, &schema_left)?;
        let rtype = get_dtype!(rnode, &schema_right)?;

        if let Some(dtype) = get_numeric_upcast_supertype_lossless(&ltype, &rtype) {
            // We use overflowing cast to allow better optimization as we are casting to a known
            // lossless supertype.
            //
            // We have unique references to these nodes (they are created by this function),
            // so we can mutate in-place without causing side effects somewhere else.
            let casted_l = ctxt.expr_arena.add(AExpr::Cast {
                expr: lnode.node(),
                dtype: dtype.clone(),
                options: CastOptions::Overflowing,
            });
            let casted_r = ctxt.expr_arena.add(AExpr::Cast {
                expr: rnode.node(),
                dtype,
                options: CastOptions::Overflowing,
            });

            if key_cols_coalesced {
                let mut lnode = lnode.clone();
                let mut rnode = rnode.clone();

                let ae_l = ctxt.expr_arena.get(lnode.node());
                let ae_r = ctxt.expr_arena.get(rnode.node());

                polars_ensure!(
                    ae_l.is_col() && ae_r.is_col(),
                    SchemaMismatch: "can only 'coalesce' full join if join keys are column expressions",
                );

                lnode.set_node(casted_l);
                rnode.set_node(casted_r);

                as_with_columns_r.push(rnode);
                as_with_columns_l.push(lnode);
            } else {
                lnode.set_node(casted_l);
                rnode.set_node(casted_r);
            }
        } else {
            polars_ensure!(
                ltype == rtype,
                SchemaMismatch: "datatypes of join keys don't match - `{}`: {} on left does not match `{}`: {} on right (and no other type was available to cast to)",
                lnode.output_name(), ltype.pretty_format(), rnode.output_name(), rtype.pretty_format()
            );
        }
    }

    // Every expression must be elementwise so that we are
    // guaranteed the keys for a join are all the same length.

    polars_ensure!(
        all_elementwise(&left_on, ctxt.expr_arena) && all_elementwise(&right_on, ctxt.expr_arena),
        InvalidOperation: "all join key expressions must be elementwise."
    );

    #[cfg(feature = "asof_join")]
    if let JoinType::AsOf(options) = &mut options.args.how {
        use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;

        // prepare the tolerance
        // we must ensure that we use the right units
        if let Some(tol) = &options.tolerance_str {
            let duration = polars_time::Duration::try_parse(tol)?;
            polars_ensure!(
                duration.months() == 0,
                ComputeError: "cannot use month offset in timedelta of an asof join; \
                consider using 4 weeks"
            );
            use DataType::*;
            match ctxt
                .expr_arena
                .get(left_on[0].node())
                .to_dtype(&ToFieldContext::new(ctxt.expr_arena, &schema_left))?
            {
                Datetime(tu, _) | Duration(tu) => {
                    let tolerance = match tu {
                        TimeUnit::Nanoseconds => duration.duration_ns(),
                        TimeUnit::Microseconds => duration.duration_us(),
                        TimeUnit::Milliseconds => duration.duration_ms(),
                    };
                    options.tolerance = Some(Scalar::from(tolerance))
                },
                Date => {
                    let days = (duration.duration_ms() / MILLISECONDS_IN_DAY) as i32;
                    options.tolerance = Some(Scalar::from(days))
                },
                Time => {
                    let tolerance = duration.duration_ns();
                    options.tolerance = Some(Scalar::from(tolerance))
                },
                _ => {
                    panic!(
                        "can only use timedelta string language with Date/Datetime/Duration/Time dtypes"
                    )
                },
            }
        }
    }

    // These are Arc<Schema>, into_owned is free.
    let schema_left = schema_left.into_owned();
    let schema_right = schema_right.into_owned();

    let join_schema = det_join_schema(
        &schema_left,
        &schema_right,
        &left_on,
        &right_on,
        &options,
        ctxt.expr_arena,
    )
    .map_err(|e| e.context(failed_here!(join schema resolving)))?;

    if key_cols_coalesced {
        input_left = if as_with_columns_l.is_empty() {
            input_left
        } else {
            ctxt.lp_arena.add(IR::HStack {
                input: input_left,
                exprs: as_with_columns_l,
                schema: schema_left,
                options: ProjectionOptions::default(),
            })
        };

        input_right = if as_with_columns_r.is_empty() {
            input_right
        } else {
            ctxt.lp_arena.add(IR::HStack {
                input: input_right,
                exprs: as_with_columns_r,
                schema: schema_right,
                options: ProjectionOptions::default(),
            })
        };
    }

    let ir = IR::Join {
        input_left,
        input_right,
        schema: join_schema.clone(),
        left_on,
        right_on,
        options: Arc::new(options),
    };
    let join_node = ctxt.lp_arena.add(ir);

    if has_scalars {
        let names = join_schema
            .iter_names()
            .filter_map(|n| {
                if n.starts_with(POLARS_TMP_PREFIX) {
                    None
                } else {
                    Some(n.clone())
                }
            })
            .collect_vec();

        let builder = IRBuilder::new(join_node, ctxt.expr_arena, ctxt.lp_arena);
        let ir = builder.project_simple(names).map(|b| b.build())?;
        let select_node = ctxt.lp_arena.add(ir);

        Ok((select_node, join_node))
    } else {
        Ok((join_node, join_node))
    }
}

#[cfg(feature = "iejoin")]
impl From<InequalityOperator> for Operator {
    fn from(value: InequalityOperator) -> Self {
        match value {
            InequalityOperator::LtEq => Operator::LtEq,
            InequalityOperator::Lt => Operator::Lt,
            InequalityOperator::GtEq => Operator::GtEq,
            InequalityOperator::Gt => Operator::Gt,
        }
    }
}

#[cfg(feature = "iejoin")]
/// Returns: left: join_node, right: last_node (often both the same)
fn resolve_join_where(
    input_left: Arc<DslPlan>,
    input_right: Arc<DslPlan>,
    predicates: Vec<Expr>,
    mut options: JoinOptionsIR,
    ctxt: &mut DslConversionContext,
) -> PolarsResult<(Node, Node)> {
    // If not eager, respect the flag.
    if ctxt.opt_flags.eager() {
        ctxt.opt_flags.set(OptFlags::PREDICATE_PUSHDOWN, true);
    }
    check_join_keys(&predicates)?;
    let input_left = to_alp_impl(Arc::unwrap_or_clone(input_left), ctxt)
        .map_err(|e| e.context(failed_here!(join left)))?;
    let input_right = to_alp_impl(Arc::unwrap_or_clone(input_right), ctxt)
        .map_err(|e| e.context(failed_here!(join left)))?;

    let schema_left = ctxt
        .lp_arena
        .get(input_left)
        .schema(ctxt.lp_arena)
        .into_owned();

    options.args.how = JoinType::Cross;

    let (mut last_node, join_node) = resolve_join(
        Either::Right(input_left),
        Either::Right(input_right),
        vec![],
        vec![],
        vec![],
        options,
        ctxt,
    )?;

    let schema_merged = ctxt
        .lp_arena
        .get(last_node)
        .schema(ctxt.lp_arena)
        .into_owned();

    // Perform predicate validation.
    let mut upcast_exprs = Vec::<(Node, DataType)>::new();
    for e in predicates {
        let arena = &mut ctxt.expr_arena;
        let predicate = to_expr_ir_materialized_lit(
            e,
            &mut ExprToIRContext::new_with_opt_eager(arena, &schema_merged, ctxt.opt_flags),
        )?;
        let node = predicate.node();

        // Ensure the predicate dtype output of the root node is Boolean
        let ae = arena.get(node);
        let dt_out = ae.to_dtype(&ToFieldContext::new(arena, &schema_merged))?;
        polars_ensure!(
            dt_out == DataType::Boolean,
            ComputeError: "'join_where' predicates must resolve to boolean"
        );

        ensure_lossless_binary_comparisons(
            &node,
            &schema_left,
            &schema_merged,
            arena,
            &mut upcast_exprs,
        )?;

        ctxt.conversion_optimizer
            .push_scratch(predicate.node(), ctxt.expr_arena);

        let ir = IR::Filter {
            input: last_node,
            predicate,
        };

        last_node = ctxt.lp_arena.add(ir);
    }

    ctxt.conversion_optimizer
        .optimize_exprs(ctxt.expr_arena, ctxt.lp_arena, last_node, false)
        .map_err(|e| e.context("'join_where' failed".into()))?;

    Ok((last_node, join_node))
}

/// Locate nodes that are operands in a binary comparison involving both tables, and ensure that
/// these nodes are losslessly upcast to a safe dtype.
fn ensure_lossless_binary_comparisons(
    node: &Node,
    schema_left: &Schema,
    schema_merged: &Schema,
    expr_arena: &mut Arena<AExpr>,
    upcast_exprs: &mut Vec<(Node, DataType)>,
) -> PolarsResult<()> {
    // let mut upcast_exprs = Vec::<(Node, DataType)>::new();
    // Ensure that all binary comparisons that use both tables are lossless.
    build_upcast_node_list(node, schema_left, schema_merged, expr_arena, upcast_exprs)?;
    // Replace each node with its casted counterpart
    for (expr, dtype) in upcast_exprs.drain(..) {
        let old_expr = expr_arena.duplicate(expr);
        let new_aexpr = AExpr::Cast {
            expr: old_expr,
            dtype,
            options: CastOptions::Overflowing,
        };
        expr_arena.replace(expr, new_aexpr);
    }
    Ok(())
}

/// If we are dealing with a binary comparison involving columns from exclusively the left table
/// on the LHS and the right table on the RHS side, ensure that the cast is lossless.
/// Expressions involving binaries using either table alone we leave up to the user to verify
/// that they are valid, as they could theoretically be pushed outside of the join.
#[recursive]
fn build_upcast_node_list(
    node: &Node,
    schema_left: &Schema,
    schema_merged: &Schema,
    expr_arena: &Arena<AExpr>,
    to_replace: &mut Vec<(Node, DataType)>,
) -> PolarsResult<ExprOrigin> {
    let expr_origin = match expr_arena.get(*node) {
        AExpr::Column(name) => {
            if schema_left.contains(name) {
                ExprOrigin::Left
            } else if schema_merged.contains(name) {
                ExprOrigin::Right
            } else {
                polars_bail!(ColumnNotFound: "{name}");
            }
        },
        AExpr::Literal(..) => ExprOrigin::None,
        AExpr::Cast { expr: node, .. } => {
            build_upcast_node_list(node, schema_left, schema_merged, expr_arena, to_replace)?
        },
        AExpr::BinaryExpr {
            left: left_node,
            op,
            right: right_node,
        } => {
            // If left and right node has both, ensure the dtypes are valid.
            let left_origin = build_upcast_node_list(
                left_node,
                schema_left,
                schema_merged,
                expr_arena,
                to_replace,
            )?;
            let right_origin = build_upcast_node_list(
                right_node,
                schema_left,
                schema_merged,
                expr_arena,
                to_replace,
            )?;
            // We only update casts during comparisons if the operands are from different tables.
            if op.is_comparison() {
                match (left_origin, right_origin) {
                    (ExprOrigin::Left, ExprOrigin::Right)
                    | (ExprOrigin::Right, ExprOrigin::Left) => {
                        // Ensure our dtype casts are lossless
                        let left = expr_arena.get(*left_node);
                        let right = expr_arena.get(*right_node);
                        let dtype_left =
                            left.to_dtype(&ToFieldContext::new(expr_arena, schema_merged))?;
                        let dtype_right =
                            right.to_dtype(&ToFieldContext::new(expr_arena, schema_merged))?;
                        if dtype_left != dtype_right {
                            // Ensure that we have a lossless cast between the two types.
                            let dt = if dtype_left.is_primitive_numeric()
                                || dtype_right.is_primitive_numeric()
                            {
                                get_numeric_upcast_supertype_lossless(&dtype_left, &dtype_right)
                                    .ok_or(PolarsError::SchemaMismatch(
                                        format!(
                                            "'join_where' cannot compare {dtype_left:?} with {dtype_right:?}"
                                        )
                                        .into(),
                                    ))
                            } else {
                                try_get_supertype(&dtype_left, &dtype_right)
                            }?;

                            // Store the nodes and their replacements if a cast is required.
                            let replace_left = dt != dtype_left;
                            let replace_right = dt != dtype_right;
                            if replace_left && replace_right {
                                to_replace.push((*left_node, dt.clone()));
                                to_replace.push((*right_node, dt));
                            } else if replace_left {
                                to_replace.push((*left_node, dt));
                            } else if replace_right {
                                to_replace.push((*right_node, dt));
                            }
                        }
                    },
                    _ => (),
                }
            }
            left_origin | right_origin
        },
        _ => ExprOrigin::None,
    };
    Ok(expr_origin)
}
