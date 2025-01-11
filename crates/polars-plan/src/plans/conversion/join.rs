use arrow::legacy::error::PolarsResult;
use either::Either;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::feature_gated;
use polars_core::utils::get_numeric_upcast_supertype_lossless;
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
    mut options: Arc<JoinOptions>,
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
            let options = Arc::make_mut(&mut options);
            if matches!(options.args.coalesce, JoinCoalesce::CoalesceColumns) {
                polars_warn!("coalescing join requested but not all join keys are column references, turning off key coalescing");
            }
            options.args.coalesce = JoinCoalesce::KeepColumns;
        }

        options.args.validation.is_valid_join(&options.args.how)?;

        #[cfg(feature = "asof_join")]
        if let JoinType::AsOf(opt) = &options.args.how {
            match (&opt.left_by, &opt.right_by) {
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
                format!(
                    "the number of columns given as join key (left: {}, right:{}) should be equal",
                    left_on.len(),
                    right_on.len()
                )
        );
    }

    let mut left_on = to_expr_irs_ignore_alias(left_on, ctxt.expr_arena)?;
    let mut right_on = to_expr_irs_ignore_alias(right_on, ctxt.expr_arena)?;
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
        .coerce_types(ctxt.expr_arena, ctxt.lp_arena, input_left)
        .map_err(|e| e.context("'join' failed".into()))?;
    ctxt.conversion_optimizer
        .fill_scratch(&right_on, ctxt.expr_arena);
    ctxt.conversion_optimizer
        .coerce_types(ctxt.expr_arena, ctxt.lp_arena, input_right)
        .map_err(|e| e.context("'join' failed".into()))?;

    // Re-evaluate because of mutable borrows earlier.
    let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
    let schema_right = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

    // Not a closure to avoid borrow issues because we mutate expr_arena as well.
    macro_rules! get_dtype {
        ($expr:expr, $schema:expr) => {
            ctxt.expr_arena
                .get($expr.node())
                .get_type($schema, Context::Default, ctxt.expr_arena)
        };
    }
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
                let dtype = e.dtype(&schema_left_new, Context::Default, ctxt.expr_arena)?;
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
                let dtype = e.dtype(&schema_right_new, Context::Default, ctxt.expr_arena)?;
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

    // # Cast lossless
    //
    // If we do a full join and keys are coalesced, the cast keys must be added up front.
    let key_cols_coalesced =
        options.args.should_coalesce() && matches!(&options.args.how, JoinType::Full);
    let mut as_with_columns_l = vec![];
    let mut as_with_columns_r = vec![];
    for (lnode, rnode) in left_on.iter_mut().zip(right_on.iter_mut()) {
        //polars_ensure!(!lnode.is_scalar(&ctxt.expr_arena), InvalidOperation: "joining on scalars is not allowed, consider using 'join_where'");
        //polars_ensure!(!rnode.is_scalar(&ctxt.expr_arena), InvalidOperation: "joining on scalars is not allowed, consider using 'join_where'");

        let ltype = get_dtype!(lnode, &schema_left)?;
        let rtype = get_dtype!(rnode, &schema_right)?;

        if let Some(dtype) = get_numeric_upcast_supertype_lossless(&ltype, &rtype) {
            let casted_l = ctxt.expr_arena.add(AExpr::Cast {
                expr: lnode.node(),
                dtype: dtype.clone(),
                options: CastOptions::Strict,
            });
            let casted_r = ctxt.expr_arena.add(AExpr::Cast {
                expr: rnode.node(),
                dtype,
                options: CastOptions::Strict,
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
                SchemaMismatch: "datatypes of join keys don't match - `{}`: {} on left does not match `{}`: {} on right",
                lnode.output_name(), ltype, rnode.output_name(), rtype
            )
        }
    }

    // Every expression must be elementwise so that we are
    // guaranteed the keys for a join are all the same length.

    polars_ensure!(
        all_elementwise(&left_on, ctxt.expr_arena) && all_elementwise(&right_on, ctxt.expr_arena),
        InvalidOperation: "all join key expressions must be elementwise."
    );

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
        options,
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
    mut options: Arc<JoinOptions>,
    ctxt: &mut DslConversionContext,
) -> PolarsResult<(Node, Node)> {
    // If not eager, respect the flag.
    if ctxt.opt_flags.eager() {
        ctxt.opt_flags.set(OptFlags::PREDICATE_PUSHDOWN, true);
    }
    ctxt.opt_flags.set(OptFlags::COLLAPSE_JOINS, true);
    check_join_keys(&predicates)?;
    let input_left = to_alp_impl(Arc::unwrap_or_clone(input_left), ctxt)
        .map_err(|e| e.context(failed_here!(join left)))?;
    let input_right = to_alp_impl(Arc::unwrap_or_clone(input_right), ctxt)
        .map_err(|e| e.context(failed_here!(join left)))?;

    let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
    let schema_right = ctxt
        .lp_arena
        .get(input_right)
        .schema(ctxt.lp_arena)
        .into_owned();

    for expr in &predicates {
        fn all_in_schema(
            schema: &Schema,
            other: Option<&Schema>,
            left: &Expr,
            right: &Expr,
        ) -> bool {
            let mut iter =
                expr_to_leaf_column_names_iter(left).chain(expr_to_leaf_column_names_iter(right));
            iter.all(|name| {
                schema.contains(name.as_str()) && other.is_none_or(|s| !s.contains(name.as_str()))
            })
        }

        let valid = expr.into_iter().all(|e| match e {
            Expr::BinaryExpr { left, op, right } if op.is_comparison() => {
                !(all_in_schema(&schema_left, None, left, right)
                    || all_in_schema(&schema_right, Some(&schema_left), left, right))
            },
            _ => true,
        });
        polars_ensure!( valid, InvalidOperation: "'join_where' predicate only refers to columns from a single table")
    }

    let opts = Arc::make_mut(&mut options);
    opts.args.how = JoinType::Cross;

    let (mut last_node, join_node) = resolve_join(
        Either::Right(input_left),
        Either::Right(input_right),
        vec![],
        vec![],
        vec![],
        options.clone(),
        ctxt,
    )?;

    for e in predicates {
        let predicate = to_expr_ir_ignore_alias(e, ctxt.expr_arena)?;

        let ir = IR::Filter {
            input: last_node,
            predicate,
        };
        last_node = ctxt.lp_arena.add(ir);
    }
    Ok((last_node, join_node))
}
