use arrow::legacy::error::PolarsResult;
use either::Either;
use polars_core::error::feature_gated;

use super::*;
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
pub fn resolve_join(
    input_left: Either<Arc<DslPlan>, Node>,
    input_right: Either<Arc<DslPlan>, Node>,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    predicates: Vec<Expr>,
    mut options: Arc<JoinOptions>,
    ctxt: &mut DslConversionContext,
) -> PolarsResult<Node> {
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
    if matches!(options.args.how, JoinType::Cross) {
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

    let input_left = input_left.map_right(Ok).right_or_else(|input| {
        to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(join left)))
    })?;
    let input_right = input_right.map_right(Ok).right_or_else(|input| {
        to_alp_impl(owned(input), ctxt).map_err(|e| e.context(failed_here!(join right)))
    })?;

    let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
    let schema_right = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

    let schema = det_join_schema(&schema_left, &schema_right, &left_on, &right_on, &options)
        .map_err(|e| e.context(failed_here!(join schema resolving)))?;

    let left_on = to_expr_irs_ignore_alias(left_on, ctxt.expr_arena)?;
    let right_on = to_expr_irs_ignore_alias(right_on, ctxt.expr_arena)?;
    let mut joined_on = PlHashSet::new();

    #[cfg(feature = "iejoin")]
    let check = !matches!(options.args.how, JoinType::IEJoin(_));
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
    run_conversion(lp, ctxt, "join")
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
fn resolve_join_where(
    input_left: Arc<DslPlan>,
    input_right: Arc<DslPlan>,
    predicates: Vec<Expr>,
    mut options: Arc<JoinOptions>,
    ctxt: &mut DslConversionContext,
) -> PolarsResult<Node> {
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
    for e in &predicates {
        let no_binary_comparisons = e
            .into_iter()
            .filter(|e| match e {
                Expr::BinaryExpr { op, .. } => op.is_comparison(),
                _ => false,
            })
            .count();
        polars_ensure!(no_binary_comparisons == 1, InvalidOperation: "only 1 binary comparison allowed as join condition");

        fn all_in_schema(
            schema: &Schema,
            other: Option<&Schema>,
            left: &Expr,
            right: &Expr,
        ) -> bool {
            let mut iter =
                expr_to_leaf_column_names_iter(left).chain(expr_to_leaf_column_names_iter(right));
            iter.all(|name| {
                schema.contains(name.as_str()) && other.map_or(true, |s| !s.contains(name.as_str()))
            })
        }

        let valid = e.into_iter().all(|e| match e {
            Expr::BinaryExpr { left, op, right } if op.is_comparison() => {
                !(all_in_schema(&schema_left, None, left, right)
                    || all_in_schema(&schema_right, Some(&schema_left), left, right))
            },
            _ => true,
        });
        polars_ensure!( valid, InvalidOperation: "join predicate in 'join_where' only refers to columns of a single table")
    }

    let owned = |e: Arc<Expr>| (*e).clone();

    // We do a few things
    // First we partition to:
    // - IEjoin supported inequality predicates
    // - equality predicates
    // - remaining predicates
    // And then decide to which join we dispatch.
    // The remaining predicates will be applied as filter.

    // What make things a bit complicated is that duplicate join names
    // are referred to in the query with the name post-join, but on joins
    // we refer to the names pre-join (e.g. without suffix). So there is some
    // bookkeeping.
    //
    // - First we determine which side of the binary expression refers to the left and right table
    // and make sure that lhs of the binary expr, maps to the lhs of the join tables and vice versa.
    // Next we ensure the suffixes are removed when we partition.
    //
    // If a predicate has to be applied as post-join filter, we put the suffixes back if needed.
    let mut ie_left_on = vec![];
    let mut ie_right_on = vec![];
    let mut ie_op = vec![];

    let mut eq_left_on = vec![];
    let mut eq_right_on = vec![];

    let mut remaining_preds = vec![];

    fn to_inequality_operator(op: &Operator) -> Option<InequalityOperator> {
        match op {
            Operator::Lt => Some(InequalityOperator::Lt),
            Operator::LtEq => Some(InequalityOperator::LtEq),
            Operator::Gt => Some(InequalityOperator::Gt),
            Operator::GtEq => Some(InequalityOperator::GtEq),
            _ => None,
        }
    }

    fn rename_expr(e: Expr, old: &str, new: &str) -> Expr {
        e.map_expr(|e| match e {
            Expr::Column(name) if name.as_str() == old => Expr::Column(new.into()),
            e => e,
        })
    }

    fn determine_order_and_pre_join_names(
        left: Expr,
        op: Operator,
        right: Expr,
        schema_left: &Schema,
        schema_right: &Schema,
        suffix: &str,
    ) -> PolarsResult<(Expr, Operator, Expr)> {
        let left_names = expr_to_leaf_column_names_iter(&left).collect::<PlHashSet<_>>();
        let right_names = expr_to_leaf_column_names_iter(&right).collect::<PlHashSet<_>>();

        // All left should be in the left schema.
        let (left_names, right_names, left, op, mut right) =
            if !left_names.iter().all(|n| schema_left.contains(n)) {
                // If all right names are in left schema -> swap
                if right_names.iter().all(|n| schema_left.contains(n)) {
                    (right_names, left_names, right, op.swap_operands(), left)
                } else {
                    polars_bail!(InvalidOperation: "got ambiguous column names in 'join_where'")
                }
            } else {
                (left_names, right_names, left, op, right)
            };
        for name in &left_names {
            polars_ensure!(!right_names.contains(name.as_str()), InvalidOperation: "got ambiguous column names in 'join_where'\n\n\
            Note that you should refer to the column names as they are post-join operation.")
        }

        // Now we know left belongs to the left schema, rhs suffixes are dealt with.
        for post_join_name in right_names {
            if let Some(pre_join_name) = post_join_name.strip_suffix(suffix) {
                // Name is both sides, so a suffix will be added by the join.
                // We rename
                if schema_right.contains(pre_join_name) && schema_left.contains(pre_join_name) {
                    right = rename_expr(right, &post_join_name, pre_join_name);
                }
            }
        }
        Ok((left, op, right))
    }

    // Make it a binary comparison and ensure the columns refer to post join names.
    fn to_binary_post_join(
        l: Expr,
        op: Operator,
        mut r: Expr,
        schema_right: &Schema,
        suffix: &str,
    ) -> Expr {
        let names = expr_to_leaf_column_names_iter(&r).collect::<Vec<_>>();
        for pre_join_name in &names {
            if !schema_right.contains(pre_join_name) {
                let post_join_name = _join_suffix_name(pre_join_name, suffix);
                r = rename_expr(r, pre_join_name, post_join_name.as_str());
            }
        }

        Expr::BinaryExpr {
            left: Arc::from(l),
            op,
            right: Arc::from(r),
        }
    }

    let suffix = options.args.suffix().clone();
    for pred in predicates.into_iter() {
        let Expr::BinaryExpr { left, op, right } = pred.clone() else {
            polars_bail!(InvalidOperation: "can only join on binary expressions")
        };
        polars_ensure!(op.is_comparison(), InvalidOperation: "expected comparison in join predicate");
        let (left, op, right) = determine_order_and_pre_join_names(
            owned(left),
            op,
            owned(right),
            &schema_left,
            &schema_right,
            &suffix,
        )?;

        if let Some(ie_op_) = to_inequality_operator(&op) {
            fn is_numeric(e: &Expr, schema: &Schema) -> bool {
                expr_to_leaf_column_names_iter(e).any(|name| {
                    if let Some(dt) = schema.get(name.as_str()) {
                        dt.to_physical().is_numeric()
                    } else {
                        false
                    }
                })
            }

            // We fallback to remaining if:
            // - we already have an IEjoin or Inner join
            // - we already have an Inner join
            // - data is not numeric (our iejoin doesn't yet implement that)
            if ie_op.len() >= 2
                || !eq_right_on.is_empty()
                || !is_numeric(&left, &schema_left)
                || !is_numeric(&right, &schema_right)
            {
                remaining_preds.push(to_binary_post_join(left, op, right, &schema_right, &suffix))
            } else {
                ie_left_on.push(left);
                ie_right_on.push(right);
                ie_op.push(ie_op_)
            }
        } else if matches!(op, Operator::Eq) {
            eq_left_on.push(left);
            eq_right_on.push(right);
        } else {
            remaining_preds.push(to_binary_post_join(left, op, right, &schema_right, &suffix));
        }
    }

    // Now choose a primary join and do the remaining predicates as filters
    // Add the ie predicates to the remaining predicates buffer so that they will be executed in the
    // filter node.
    fn ie_predicates_to_remaining(
        remaining_preds: &mut Vec<Expr>,
        ie_left_on: Vec<Expr>,
        ie_right_on: Vec<Expr>,
        ie_op: Vec<InequalityOperator>,
        schema_right: &Schema,
        suffix: &str,
    ) {
        for ((l, op), r) in ie_left_on
            .into_iter()
            .zip(ie_op.into_iter())
            .zip(ie_right_on.into_iter())
        {
            remaining_preds.push(to_binary_post_join(l, op.into(), r, schema_right, suffix))
        }
    }

    let join_node = if !eq_left_on.is_empty() {
        // We found one or more  equality predicates. Go into a default equi join
        // as those are cheapest on avg.
        let join_node = resolve_join(
            Either::Right(input_left),
            Either::Right(input_right),
            eq_left_on,
            eq_right_on,
            vec![],
            options.clone(),
            ctxt,
        )?;

        ie_predicates_to_remaining(
            &mut remaining_preds,
            ie_left_on,
            ie_right_on,
            ie_op,
            &schema_right,
            &suffix,
        );
        join_node
    } else if ie_right_on.len() >= 2 {
        // Do an IEjoin.
        let opts = Arc::make_mut(&mut options);
        opts.args.how = JoinType::IEJoin(IEJoinOptions {
            operator1: ie_op[0],
            operator2: Some(ie_op[1]),
        });

        let join_node = resolve_join(
            Either::Right(input_left),
            Either::Right(input_right),
            ie_left_on[..2].to_vec(),
            ie_right_on[..2].to_vec(),
            vec![],
            options.clone(),
            ctxt,
        )?;

        // The surplus ie-predicates will be added to the remaining predicates so that
        // they will be applied in a filter node.
        while ie_right_on.len() > 2 {
            // Invariant: they all have equal length, so we can pop and unwrap all while len > 2.
            // The first 2 predicates are used in the
            let l = ie_right_on.pop().unwrap();
            let r = ie_left_on.pop().unwrap();
            let op = ie_op.pop().unwrap();

            remaining_preds.push(to_binary_post_join(l, op.into(), r, &schema_right, &suffix))
        }
        join_node
    } else if ie_right_on.len() == 1 {
        // For a single inequality comparison, we use the piecewise merge join algorithm
        let opts = Arc::make_mut(&mut options);
        opts.args.how = JoinType::IEJoin(IEJoinOptions {
            operator1: ie_op[0],
            operator2: None,
        });

        resolve_join(
            Either::Right(input_left),
            Either::Right(input_right),
            ie_left_on,
            ie_right_on,
            vec![],
            options.clone(),
            ctxt,
        )?
    } else {
        // No predicates found that are supported in a fast algorithm.
        // Do a cross join and follow up with filters.
        let opts = Arc::make_mut(&mut options);
        opts.args.how = JoinType::Cross;

        resolve_join(
            Either::Right(input_left),
            Either::Right(input_right),
            vec![],
            vec![],
            vec![],
            options.clone(),
            ctxt,
        )?
    };

    let IR::Join {
        input_left,
        input_right,
        ..
    } = ctxt.lp_arena.get(join_node)
    else {
        unreachable!()
    };
    let schema_right = ctxt
        .lp_arena
        .get(*input_right)
        .schema(ctxt.lp_arena)
        .into_owned();

    let schema_left = ctxt
        .lp_arena
        .get(*input_left)
        .schema(ctxt.lp_arena)
        .into_owned();

    let mut last_node = join_node;

    // Ensure that the predicates use the proper suffix
    for e in remaining_preds {
        let predicate = to_expr_ir_ignore_alias(e, ctxt.expr_arena)?;
        let AExpr::BinaryExpr { mut right, .. } = *ctxt.expr_arena.get(predicate.node()) else {
            unreachable!()
        };

        let original_right = right;

        for name in aexpr_to_leaf_names(right, ctxt.expr_arena) {
            polars_ensure!(schema_right.contains(name.as_str()), ColumnNotFound: "could not find column {name} in the right table during join operation");
            if schema_left.contains(name.as_str()) {
                let new_name = _join_suffix_name(name.as_str(), suffix.as_str());

                right = rename_matching_aexpr_leaf_names(
                    right,
                    ctxt.expr_arena,
                    name.as_str(),
                    new_name,
                );
            }
        }
        ctxt.expr_arena.swap(right, original_right);

        let ir = IR::Filter {
            input: last_node,
            predicate,
        };
        last_node = ctxt.lp_arena.add(ir);
    }
    Ok(last_node)
}
