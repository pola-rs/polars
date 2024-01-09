#![allow(clippy::too_many_arguments)]
use std::collections::BTreeSet;

use super::*;

fn add_keys_to_accumulated_state(
    expr: Node,
    acc_projections: &mut Vec<Node>,
    local_projection: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
    expr_arena: &mut Arena<AExpr>,
    // only for left hand side table we add local names
    add_local: bool,
) -> Option<Arc<str>> {
    add_expr_to_accumulated(expr, acc_projections, projected_names, expr_arena);
    // the projections may do more than simply project.
    // e.g. col("foo").truncate() * col("bar")
    // that means we don't want to execute the projection as that is already done by
    // the JOIN executor
    if add_local {
        // take the left most name as output name
        let name = aexpr_to_leaf_name(expr, expr_arena);
        let node = expr_arena.add(AExpr::Column(name.clone()));
        local_projection.push(node);
        Some(name)
    } else {
        None
    }
}

#[cfg(feature = "asof_join")]
pub(super) fn process_asof_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<Node>,
    right_on: Vec<Node>,
    options: Arc<JoinOptions>,
    acc_projections: Vec<Node>,
    _projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    // n = 0 if no projections, so we don't allocate unneeded
    let n = acc_projections.len() * 2;
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
    if !acc_projections.is_empty() {
        let schema_left = lp_arena.get(input_left).schema(lp_arena);
        let schema_right = lp_arena.get(input_right).schema(lp_arena);

        // make sure that the asof join 'by' columns are projected
        if let (Some(left_by), Some(right_by)) = (&asof_options.left_by, &asof_options.right_by) {
            for name in left_by {
                let add = _projected_names.contains(name.as_str());

                let node = expr_arena.add(AExpr::Column(Arc::from(name.as_str())));
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
                let node = expr_arena.add(AExpr::Column(Arc::from(name.as_str())));
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
        let mut already_added_local_to_local_projected = BTreeSet::new();

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            let local_name = add_keys_to_accumulated_state(
                *e,
                &mut pushdown_left,
                &mut local_projection,
                &mut names_left,
                expr_arena,
                true,
            )
            .unwrap();
            already_added_local_to_local_projected.insert(local_name);
        }
        // this differs from normal joins, as in `asof_joins`
        // both columns remain. So `add_local=true` also for the right table
        for e in &right_on {
            if let Some(local_name) = add_keys_to_accumulated_state(
                *e,
                &mut pushdown_right,
                &mut local_projection,
                &mut names_right,
                expr_arena,
                true,
            ) {
                // insert the name.
                // if name was already added we pop the local projection
                // otherwise we would project duplicate columns
                if !already_added_local_to_local_projected.insert(local_name) {
                    local_projection.pop();
                }
            };
        }

        for proj in acc_projections {
            let mut add_local = if already_added_local_to_local_projected.is_empty() {
                true
            } else {
                let name = aexpr_to_leaf_name(proj, expr_arena);
                !already_added_local_to_local_projected.contains(&name)
            };

            add_local = process_alias(proj, &mut local_projection, expr_arena, add_local);
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
            );
        }
    }

    proj_pd.pushdown_and_assign(
        input_left,
        pushdown_left,
        names_left,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;
    proj_pd.pushdown_and_assign(
        input_right,
        pushdown_right,
        names_right,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;

    let alp = resolve_join_suffixes(
        input_left,
        input_right,
        left_on,
        right_on,
        options,
        lp_arena,
        expr_arena,
        &mut local_projection,
    );
    let root = lp_arena.add(alp);
    let builder = ALogicalPlanBuilder::new(root, expr_arena, lp_arena);

    Ok(proj_pd.finish_node(local_projection, builder))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<Node>,
    right_on: Vec<Node>,
    options: Arc<JoinOptions>,
    acc_projections: Vec<Node>,
    _projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    #[cfg(feature = "asof_join")]
    if matches!(options.args.how, JoinType::AsOf(_)) {
        return process_asof_join(
            proj_pd,
            input_left,
            input_right,
            left_on,
            right_on,
            options,
            acc_projections,
            _projected_names,
            projections_seen,
            lp_arena,
            expr_arena,
        );
    }

    // n = 0 if no projections, so we don't allocate unneeded
    let n = acc_projections.len() * 2;
    let mut pushdown_left = Vec::with_capacity(n);
    let mut pushdown_right = Vec::with_capacity(n);
    let mut names_left = PlHashSet::with_capacity(n);
    let mut names_right = PlHashSet::with_capacity(n);
    let mut local_projection = Vec::with_capacity(n);

    // if there are no projections we don't have to do anything (all columns are projected)
    // otherwise we build local projections to sort out proper column names due to the
    // join operation
    //
    // Joins on columns with different names, for example
    // left_on = "a", right_on = "b
    // will remove the name "b" (it is "a" now). That columns should therefore not
    // be added to a local projection.
    if !acc_projections.is_empty() {
        let schema_left = lp_arena.get(input_left).schema(lp_arena);
        let schema_right = lp_arena.get(input_right).schema(lp_arena);

        // The join on keys can lead that columns are already added, we don't want to create
        // duplicates so store the names.
        let mut already_added_local_to_local_projected = BTreeSet::new();

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            let local_name = add_keys_to_accumulated_state(
                *e,
                &mut pushdown_left,
                &mut local_projection,
                &mut names_left,
                expr_arena,
                true,
            )
            .unwrap();
            already_added_local_to_local_projected.insert(local_name);
        }
        // In outer joins both columns remain. So `add_local=true` also for the right table
        let add_local = matches!(options.args.how, JoinType::Outer { coalesce: false });
        for e in &right_on {
            // In case of outer joins we also add the columns.
            // But before we do that we must check if the column wasn't already added by the lhs.
            let add_local = if add_local {
                let name = aexpr_to_leaf_name(*e, expr_arena);
                !already_added_local_to_local_projected.contains(name.as_ref())
            } else {
                false
            };

            let local_name = add_keys_to_accumulated_state(
                *e,
                &mut pushdown_right,
                &mut local_projection,
                &mut names_right,
                expr_arena,
                add_local,
            );

            if let Some(local_name) = local_name {
                already_added_local_to_local_projected.insert(local_name);
            }
        }

        for proj in acc_projections {
            let mut add_local = if already_added_local_to_local_projected.is_empty() {
                true
            } else {
                let name = aexpr_to_leaf_name(proj, expr_arena);
                !already_added_local_to_local_projected.contains(&name)
            };

            add_local = process_alias(proj, &mut local_projection, expr_arena, add_local);
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
            );
        }
    }

    proj_pd.pushdown_and_assign(
        input_left,
        pushdown_left,
        names_left,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;
    proj_pd.pushdown_and_assign(
        input_right,
        pushdown_right,
        names_right,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;

    let alp = resolve_join_suffixes(
        input_left,
        input_right,
        left_on,
        right_on,
        options,
        lp_arena,
        expr_arena,
        &mut local_projection,
    );
    let root = lp_arena.add(alp);
    let builder = ALogicalPlanBuilder::new(root, expr_arena, lp_arena);
    Ok(proj_pd.finish_node(local_projection, builder))
}

fn process_projection(
    proj_pd: &mut ProjectionPushDown,
    schema_left: &Schema,
    schema_right: &Schema,
    proj: Node,
    pushdown_left: &mut Vec<Node>,
    pushdown_right: &mut Vec<Node>,
    names_left: &mut PlHashSet<Arc<str>>,
    names_right: &mut PlHashSet<Arc<str>>,
    expr_arena: &mut Arena<AExpr>,
    local_projection: &mut Vec<Node>,
    add_local: bool,
    options: &JoinOptions,
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
        let leaf_column_name = aexpr_to_leaf_names(proj, expr_arena).pop().unwrap();

        let suffix = options.args.suffix();
        // If _right suffix exists we need to push a projection down without this
        // suffix.
        if leaf_column_name.ends_with(suffix) {
            // downwards name is the name without the _right i.e. "foo".
            let (downwards_name, _) =
                leaf_column_name.split_at(leaf_column_name.len() - suffix.len());

            let downwards_name_column = expr_arena.add(AExpr::Column(Arc::from(downwards_name)));
            // project downwards and locally immediately alias to prevent wrong projections
            if names_right.insert(Arc::from(downwards_name)) {
                pushdown_right.push(downwards_name_column);
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

// if it is an alias we want to project the leaf column name downwards
// but we don't want to project it a this level, otherwise we project both
// the root and the alias, hence add_local = false.
pub(super) fn process_alias(
    proj: Node,
    local_projection: &mut Vec<Node>,
    expr_arena: &mut Arena<AExpr>,
    mut add_local: bool,
) -> bool {
    if let AExpr::Alias(expr, name) = expr_arena.get(proj).clone() {
        for root_name in aexpr_to_leaf_names(expr, expr_arena) {
            let node = expr_arena.add(AExpr::Column(root_name));
            let proj = expr_arena.add(AExpr::Alias(node, name.clone()));
            local_projection.push(proj)
        }
        // now we don't
        add_local = false;
    }
    add_local
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
    left_on: Vec<Node>,
    right_on: Vec<Node>,
    options: Arc<JoinOptions>,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    local_projection: &mut [Node],
) -> ALogicalPlan {
    let suffix = options.args.suffix();
    let alp = ALogicalPlanBuilder::new(input_left, expr_arena, lp_arena)
        .join(input_right, left_on, right_on, options.clone())
        .build();
    let schema_after_join = alp.schema(lp_arena);

    for proj in local_projection {
        for name in aexpr_to_leaf_names(*proj, expr_arena) {
            if name.contains(suffix) && schema_after_join.get(&name).is_none() {
                let new_name = &name.as_ref()[..name.len() - suffix.len()];

                let renamed = aexpr_assign_renamed_leaf(*proj, expr_arena, &name, new_name);

                let aliased = expr_arena.add(AExpr::Alias(renamed, name));
                *proj = aliased;
            }
        }
    }
    alp
}
