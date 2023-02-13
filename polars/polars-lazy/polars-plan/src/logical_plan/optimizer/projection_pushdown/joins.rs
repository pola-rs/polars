use super::*;

fn add_nodes_to_accumulated_state(
    expr: Node,
    acc_projections: &mut Vec<Node>,
    local_projection: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
    expr_arena: &mut Arena<AExpr>,
    // only for left hand side table we add local names
    add_local: bool,
) {
    add_expr_to_accumulated(expr, acc_projections, projected_names, expr_arena);
    // the projections may do more than simply project.
    // e.g. col("foo").truncate().alias("bar")
    // that means we don't want to execute the projection as that is already done by
    // the JOIN executor
    // we only want to add the `col` and the `alias` as two `col()` expressions.
    if add_local {
        for node in aexpr_to_leaf_nodes(expr, expr_arena) {
            if !local_projection.contains(&node) {
                local_projection.push(node)
            }
        }
        // TODO! I think we must remove this, aliases are not allowed in join keys anymore.
        if let AExpr::Alias(_, alias_name) = expr_arena.get(expr) {
            let mut add = true;
            for node in local_projection.as_slice() {
                if let AExpr::Column(col_name) = expr_arena.get(*node) {
                    if alias_name == col_name {
                        add = false;
                        break;
                    }
                }
            }
            if add {
                let node = expr_arena.add(AExpr::Column(alias_name.clone()));
                local_projection.push(node);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<Node>,
    right_on: Vec<Node>,
    options: JoinOptions,
    acc_projections: Vec<Node>,
    _projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    proj_pd.has_joins_or_unions = true;
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

        // make sure that the asof join 'by' columns are projected
        #[cfg(feature = "asof_join")]
        if let JoinType::AsOf(asof_options) = &options.how {
            if let (Some(left_by), Some(right_by)) = (&asof_options.left_by, &asof_options.right_by)
            {
                for name in left_by {
                    let add = _projected_names.contains(name.as_str());

                    let node = expr_arena.add(AExpr::Column(Arc::from(name.as_str())));
                    add_nodes_to_accumulated_state(
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
                    add_nodes_to_accumulated_state(
                        node,
                        &mut pushdown_right,
                        &mut local_projection,
                        &mut names_right,
                        expr_arena,
                        false,
                    );
                }
            }
        }

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            add_nodes_to_accumulated_state(
                *e,
                &mut pushdown_left,
                &mut local_projection,
                &mut names_left,
                expr_arena,
                true,
            );
        }
        for e in &right_on {
            add_nodes_to_accumulated_state(
                *e,
                &mut pushdown_right,
                &mut local_projection,
                &mut names_right,
                expr_arena,
                false,
            );
        }

        for proj in acc_projections {
            match options.how {
                #[cfg(feature = "asof_join")]
                JoinType::AsOf(_) => {
                    // Asof joins don't replace
                    // the right column name with the left one
                    // so the two join columns remain
                    let names = aexpr_to_leaf_names(proj, expr_arena);
                    if names.len() == 1
                        // we only add to local projection
                        // if the right join column differs from the left
                        && names_right.contains(&names[0])
                        && !names_left.contains(&names[0])
                        && !local_projection.contains(&proj)
                    {
                        local_projection.push(proj);
                        continue;
                    }
                }
                _ => {}
            };
            let mut add_local = true;

            // if it is an alias we want to project the leaf column name downwards
            // but we don't want to project it a this level, otherwise we project both
            // the root and the alias, hence add_local = false.
            if let AExpr::Alias(expr, name) = expr_arena.get(proj).clone() {
                for root_name in aexpr_to_leaf_names(expr, expr_arena) {
                    let node = expr_arena.add(AExpr::Column(root_name));
                    let proj = expr_arena.add(AExpr::Alias(node, name.clone()));
                    local_projection.push(proj)
                }
                // now we don't
                add_local = false;
            }

            // Path for renamed columns due to the join. The column name of the left table
            // stays as is, the column of the right will have the "_right" suffix.
            // Thus joining two tables with both a foo column leads to ["foo", "foo_right"]
            if !proj_pd.join_push_down(
                &schema_left,
                &schema_right,
                proj,
                &mut pushdown_left,
                &mut pushdown_right,
                &mut names_left,
                &mut names_right,
                expr_arena,
            ) {
                // Column name of the projection without any alias.
                let leaf_column_name = aexpr_to_leaf_names(proj, expr_arena).pop().unwrap();

                let suffix = options.suffix.as_ref();
                // If _right suffix exists we need to push a projection down without this
                // suffix.
                if leaf_column_name.ends_with(suffix) {
                    // downwards name is the name without the _right i.e. "foo".
                    let (downwards_name, _) =
                        leaf_column_name.split_at(leaf_column_name.len() - suffix.len());

                    let downwards_name_column =
                        expr_arena.add(AExpr::Column(Arc::from(downwards_name)));
                    // project downwards and locally immediately alias to prevent wrong projections
                    if names_right.insert(Arc::from(downwards_name)) {
                        pushdown_right.push(downwards_name_column);
                    }
                    local_projection.push(proj);
                }
            } else if add_local {
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
    let suffix = options.suffix.clone();

    let alp = ALogicalPlanBuilder::new(input_left, expr_arena, lp_arena)
        .join(input_right, left_on, right_on, options)
        .build();
    let schema_after_join = alp.schema(lp_arena);

    for proj in &mut local_projection {
        for name in aexpr_to_leaf_names(*proj, expr_arena) {
            if name.contains(suffix.as_ref()) && schema_after_join.get(&name).is_none() {
                let new_name = &name.as_ref()[..name.len() - suffix.len()];

                let renamed = aexpr_assign_renamed_leaf(*proj, expr_arena, &name, new_name);

                let aliased = expr_arena.add(AExpr::Alias(renamed, name));
                *proj = aliased;
            }
        }
    }
    let root = lp_arena.add(alp);
    let builder = ALogicalPlanBuilder::new(root, expr_arena, lp_arena);

    Ok(proj_pd.finish_node(local_projection, builder))
}
