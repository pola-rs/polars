#![allow(clippy::too_many_arguments)]
use std::borrow::Cow;

use super::*;
use crate::prelude::optimizer::join_utils::split_suffix;

fn add_keys_to_accumulated_state(
    expr: Node,
    acc_projections: &mut Vec<ColumnNode>,
    local_projection: &mut Vec<ColumnNode>,
    projected_names: &mut PlHashSet<PlSmallStr>,
    expr_arena: &mut Arena<AExpr>,
    // Only for left hand side table we add local names.
    add_local: bool,
) -> Option<PlSmallStr> {
    add_expr_to_accumulated(expr, acc_projections, projected_names, expr_arena);
    // The projections may do more than simply project.
    // e.g. col("foo").truncate() * col("bar")
    // that means we don't want to execute the projection as that is already done by
    // the JOIN executor
    if add_local {
        // return the left most name as output name
        let names = aexpr_to_leaf_names_iter(expr, expr_arena).collect::<Vec<_>>();
        let output_name = names.first().cloned();
        for name in names {
            let node = expr_arena.add(AExpr::Column(name));
            local_projection.push(ColumnNode(node));
        }
        output_name
    } else {
        None
    }
}

#[cfg(feature = "asof_join")]
pub(super) fn process_asof_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    options: Arc<JoinOptions>,
    ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    join_schema: &Schema,
) -> PolarsResult<IR> {
    // n = 0 if no projections, so we don't allocate unneeded
    let n = ctx.acc_projections.len() * 2;
    let mut pushdown_left = Vec::with_capacity(n);
    let mut pushdown_right = Vec::with_capacity(n);
    let mut names_left = PlHashSet::with_capacity(n);
    let mut names_right = PlHashSet::with_capacity(n);
    let mut local_projection = Vec::with_capacity(n);

    let JoinType::AsOf(asof_options) = &options.args.how else {
        unreachable!()
    };

    // if there are no projections we don't have to do anything (all columns are projected)
    // otherwise we build local projections to sort out proper column names due to the
    // join operation
    //
    // Joins on columns with different names, for example
    // left_on = "a", right_on = "b
    // will remove the name "b" (it is "a" now). That columns should therefore not
    // be added to a local projection.
    if ctx.has_pushed_down() {
        let schema_left = lp_arena.get(input_left).schema(lp_arena);
        let schema_right = lp_arena.get(input_right).schema(lp_arena);

        // make sure that the asof join 'by' columns are projected
        if let (Some(left_by), Some(right_by)) = (&asof_options.left_by, &asof_options.right_by) {
            for name in left_by {
                let add = ctx.projected_names.contains(name.as_str());

                let node = expr_arena.add(AExpr::Column(name.clone()));
                add_keys_to_accumulated_state(
                    node,
                    &mut pushdown_left,
                    &mut local_projection,
                    &mut names_left,
                    expr_arena,
                    add,
                );
            }
            for name in right_by {
                let node = expr_arena.add(AExpr::Column(name.clone()));
                add_keys_to_accumulated_state(
                    node,
                    &mut pushdown_right,
                    &mut local_projection,
                    &mut names_right,
                    expr_arena,
                    false,
                );
            }
        }

        // The join on keys can lead that columns are already added, we don't want to create
        // duplicates so store the names.
        let mut local_projected_names = PlHashSet::new();

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            let local_name = add_keys_to_accumulated_state(
                e.node(),
                &mut pushdown_left,
                &mut local_projection,
                &mut names_left,
                expr_arena,
                true,
            )
            .unwrap();
            local_projected_names.insert(local_name);
        }
        // this differs from normal joins, as in `asof_joins`
        // both columns remain. So `add_local=true` also for the right table
        for e in &right_on {
            if let Some(local_name) = add_keys_to_accumulated_state(
                e.node(),
                &mut pushdown_right,
                &mut local_projection,
                &mut names_right,
                expr_arena,
                true,
            ) {
                // insert the name.
                // if name was already added we pop the local projection
                // otherwise we would project duplicate columns
                if !local_projected_names.insert(local_name) {
                    local_projection.pop();
                }
            };
        }

        for proj in ctx.acc_projections {
            let add_local = if local_projected_names.is_empty() {
                true
            } else {
                let name = column_node_to_name(proj, expr_arena);
                !local_projected_names.contains(name)
            };

            process_projection(
                proj_pd,
                &schema_left,
                &schema_right,
                proj,
                &mut pushdown_left,
                &mut pushdown_right,
                &mut names_left,
                &mut names_right,
                expr_arena,
                &mut local_projection,
                add_local,
                &options,
                join_schema,
            );
        }
    }

    let ctx_left = ProjectionContext::new(pushdown_left, names_left, ctx.inner);
    let ctx_right = ProjectionContext::new(pushdown_right, names_right, ctx.inner);

    proj_pd.pushdown_and_assign(input_left, ctx_left, lp_arena, expr_arena)?;
    proj_pd.pushdown_and_assign(input_right, ctx_right, lp_arena, expr_arena)?;

    resolve_join_suffixes(
        input_left,
        input_right,
        left_on,
        right_on,
        options,
        lp_arena,
        expr_arena,
        &local_projection,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    mut options: Arc<JoinOptions>,
    ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    join_schema: &Schema,
) -> PolarsResult<IR> {
    #[cfg(feature = "asof_join")]
    if matches!(options.args.how, JoinType::AsOf(_)) {
        return process_asof_join(
            proj_pd,
            input_left,
            input_right,
            left_on,
            right_on,
            options,
            ctx,
            lp_arena,
            expr_arena,
            join_schema,
        );
    }

    // n = 0 if no projections, so we don't allocate unneeded
    let n = ctx.acc_projections.len() * 2;
    let mut pushdown_left = Vec::with_capacity(n);
    let mut pushdown_right = Vec::with_capacity(n);
    let mut names_left = PlHashSet::with_capacity(n);
    let mut names_right = PlHashSet::with_capacity(n);
    let mut local_projection = Vec::with_capacity(n);

    // If there are no projections we don't have to do anything (all columns are projected)
    // otherwise we build local projections to sort out proper column names due to the
    // join operation
    //
    // Joins on columns with different names, for example
    // left_on = "a", right_on = "b
    // will remove the name "b" (it is "a" now). That columns should therefore not
    // be added to a local projection.
    if ctx.has_pushed_down() {
        let schema_left = lp_arena.get(input_left).schema(lp_arena);
        let schema_right = lp_arena.get(input_right).schema(lp_arena);

        // The join on keys can lead that columns are already added, we don't want to create
        // duplicates so store the names.
        let mut local_projected_names = PlHashSet::new();

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            if !local_projected_names.insert(e.output_name().clone()) {
                // A join can have multiple leaf names, so we must still ensure all leaf names are projected.
                if options.args.how.is_ie() {
                    add_expr_to_accumulated(
                        e.node(),
                        &mut pushdown_left,
                        &mut names_left,
                        expr_arena,
                    );
                }

                continue;
            }

            let _ = add_keys_to_accumulated_state(
                e.node(),
                &mut pushdown_left,
                &mut local_projection,
                &mut names_left,
                expr_arena,
                true,
            );
        }

        // For left and inner joins we can set `coalesce` to `true` if the rhs key columns are not projected.
        // This saves a materialization.
        if !options.args.should_coalesce()
            && matches!(options.args.how, JoinType::Left | JoinType::Inner)
        {
            let mut allow_opt = true;
            let non_coalesced_key_is_used = right_on.iter().any(|e| {
                // Inline expressions other than col should not coalesce.
                if !matches!(expr_arena.get(e.node()), AExpr::Column(_)) {
                    allow_opt = false;
                    return true;
                }
                let key_name = e.output_name();

                // If the name is in the lhs table, a suffix is added.
                let key_name_after_join = if schema_left.contains(key_name) {
                    Cow::Owned(_join_suffix_name(key_name, options.args.suffix()))
                } else {
                    Cow::Borrowed(key_name)
                };

                ctx.projected_names.contains(key_name_after_join.as_ref())
            });

            // If they key is not used, coalesce the columns as that is often cheaper.
            if !non_coalesced_key_is_used && allow_opt {
                let options = Arc::make_mut(&mut options);
                options.args.coalesce = JoinCoalesce::CoalesceColumns;
            }
        }

        // In  both columns remain. So `add_local=true` also for the right table
        let add_local = !options.args.should_coalesce();
        for e in &right_on {
            // In case of full outer joins we also add the columns.
            // But before we do that we must check if the column wasn't already added by the lhs.
            let add_local = if add_local {
                !local_projected_names.contains(e.output_name())
            } else {
                false
            };

            let local_name = add_keys_to_accumulated_state(
                e.node(),
                &mut pushdown_right,
                &mut local_projection,
                &mut names_right,
                expr_arena,
                add_local,
            );

            if let Some(local_name) = local_name {
                local_projected_names.insert(local_name);
            }
        }

        for proj in ctx.acc_projections {
            let add_local = if local_projected_names.is_empty() {
                true
            } else {
                let name = column_node_to_name(proj, expr_arena);
                !local_projected_names.contains(name)
            };

            process_projection(
                proj_pd,
                &schema_left,
                &schema_right,
                proj,
                &mut pushdown_left,
                &mut pushdown_right,
                &mut names_left,
                &mut names_right,
                expr_arena,
                &mut local_projection,
                add_local,
                &options,
                join_schema,
            );
        }
    }

    let ctx_left = ProjectionContext::new(pushdown_left, names_left, ctx.inner);
    let ctx_right = ProjectionContext::new(pushdown_right, names_right, ctx.inner);

    proj_pd.pushdown_and_assign(input_left, ctx_left, lp_arena, expr_arena)?;
    proj_pd.pushdown_and_assign(input_right, ctx_right, lp_arena, expr_arena)?;

    resolve_join_suffixes(
        input_left,
        input_right,
        left_on,
        right_on,
        options,
        lp_arena,
        expr_arena,
        &local_projection,
    )
}

fn process_projection(
    proj_pd: &mut ProjectionPushDown,
    schema_left: &Schema,
    schema_right: &Schema,
    proj: ColumnNode,
    pushdown_left: &mut Vec<ColumnNode>,
    pushdown_right: &mut Vec<ColumnNode>,
    names_left: &mut PlHashSet<PlSmallStr>,
    names_right: &mut PlHashSet<PlSmallStr>,
    expr_arena: &mut Arena<AExpr>,
    local_projection: &mut Vec<ColumnNode>,
    add_local: bool,
    options: &JoinOptions,
    join_schema: &Schema,
) {
    // Path for renamed columns due to the join. The column name of the left table
    // stays as is, the column of the right will have the "_right" suffix.
    // Thus joining two tables with both a foo column leads to ["foo", "foo_right"]

    // try to push down projection in either of two tables
    let (pushed_at_least_once, already_projected) = proj_pd.join_push_down(
        schema_left,
        schema_right,
        proj,
        pushdown_left,
        pushdown_right,
        names_left,
        names_right,
        expr_arena,
    );

    if !(pushed_at_least_once || already_projected)
    // did not succeed push down in any tables.,
    // this might be due to the suffix in the projection name
    // this branch tries to pushdown the column without suffix
    {
        // Column name of the projection without any alias.
        let leaf_column_name = column_node_to_name(proj, expr_arena).clone();

        let suffix = options.args.suffix().as_str();
        // If _right suffix exists we need to push a projection down without this
        // suffix.
        if leaf_column_name.ends_with(suffix) && join_schema.contains(leaf_column_name.as_ref()) {
            // downwards name is the name without the _right i.e. "foo".
            let downwards_name = split_suffix(leaf_column_name.as_ref(), suffix);
            let downwards_name = PlSmallStr::from_str(downwards_name);

            let downwards_name_column = expr_arena.add(AExpr::Column(downwards_name.clone()));
            // project downwards and locally immediately alias to prevent wrong projections
            if names_right.insert(downwards_name) {
                pushdown_right.push(ColumnNode(downwards_name_column));
            }
            local_projection.push(proj);
        }
    }
    // did succeed pushdown at least in any of the two tables
    // if not already added locally we ensure we project local as well
    else if add_local && pushed_at_least_once {
        // always also do the projection locally, because the join columns may not be
        // included in the projection.
        // for instance:
        //
        // SELECT [COLUMN temp]
        // FROM
        // JOIN (["days", "temp"]) WITH (["days", "rain"]) ON (left: days right: days)
        //
        // should drop the days column after the join.
        local_projection.push(proj);
    }
}

// Because we do a projection pushdown
// We may influence the suffixes.
// For instance if a join would have created a schema
//
// "foo", "foo_right"
//
// but we only project the "foo_right" column, the join will not produce
// a "name_right" because we did not project its left name duplicate "foo"
//
// The code below checks if can do the suffixed projections on the schema that
// we have after the join. If we cannot then we modify the projection:
//
// col("foo_right")  to col("foo").alias("foo_right")
fn resolve_join_suffixes(
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    options: Arc<JoinOptions>,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    local_projection: &[ColumnNode],
) -> PolarsResult<IR> {
    let suffix = options.args.suffix().as_str();
    let alp = IRBuilder::new(input_left, expr_arena, lp_arena)
        .join(input_right, left_on, right_on, options.clone())
        .build();
    let schema_after_join = alp.schema(lp_arena);

    let mut all_columns = true;
    let projections = local_projection
        .iter()
        .map(|proj| {
            let name = column_node_to_name(*proj, expr_arena).clone();
            if name.ends_with(suffix) && schema_after_join.get(&name).is_none() {
                let downstream_name = &name.as_str()[..name.len() - suffix.len()];
                let col = AExpr::Column(downstream_name.into());
                let node = expr_arena.add(col);
                all_columns = false;
                ExprIR::new(node, OutputName::Alias(name.clone()))
            } else {
                ExprIR::new(proj.0, OutputName::ColumnLhs(name.clone()))
            }
        })
        .collect::<Vec<_>>();

    let builder = IRBuilder::from_lp(alp, expr_arena, lp_arena);
    Ok(if all_columns {
        builder
            .project_simple(projections.iter().map(|e| e.output_name().clone()))?
            .build()
    } else {
        builder.project(projections, Default::default()).build()
    })
}
