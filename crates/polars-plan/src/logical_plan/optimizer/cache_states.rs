use std::collections::BTreeMap;

use super::*;

fn get_upper_projections(
    parent: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
    names_scratch: &mut Vec<ColumnName>,
) {
    let parent = lp_arena.get(parent);

    use ALogicalPlan::*;
    // During projection pushdown all accumulated.
    match parent {
        Projection { expr, .. } => {
            for e in expr {
                names_scratch.extend(aexpr_to_leaf_names_iter(e.node(), expr_arena));
            }
        },
        SimpleProjection { columns, .. } => {
            let iter = columns.iter_names().map(|s| ColumnName::from(s.as_str()));
            names_scratch.extend(iter);
        },
        // other
        _ => {},
    }
}

/// This will ensure that all equal caches communicate the amount of columns
/// they need to project.
pub(super) fn set_cache_states(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    has_caches: bool,
) -> PolarsResult<()> {
    let mut stack = Vec::with_capacity(4);
    let mut names_scratch = vec![];

    scratch.clear();
    stack.clear();

    #[derive(Default)]
    struct Value {
        // All the children of the cache per cache-id.
        children: Vec<Node>,
        // Union over projected names.
        names_union: PlHashSet<ColumnName>,
    }
    let mut cache_schema_and_children = BTreeMap::new();

    // Stack frame
    #[derive(Default, Copy, Clone)]
    struct Frame {
        current: Node,
        cache_id: Option<usize>,
        parent: [Option<Node>; 2],
        previous_cache: Option<usize>,
    }
    let init = Frame {
        current: root,
        ..Default::default()
    };

    stack.push(init);

    // # First traversal.
    // Collect the union of columns per cache id.
    // And find the cache parents.
    while let Some(mut frame) = stack.pop() {
        let lp = lp_arena.get(frame.current);
        lp.copy_inputs(scratch);

        use ALogicalPlan::*;
        match lp {
            // don't allow parallelism as caches need each others work
            // also self-referencing plans can deadlock on the files they lock
            Join { options, .. } if has_caches && options.allow_parallel => {
                if let Join { options, .. } = lp_arena.get_mut(frame.current) {
                    let options = Arc::make_mut(options);
                    options.allow_parallel = false;
                }
            },
            // don't allow parallelism as caches need each others work
            // also self-referencing plans can deadlock on the files they lock
            Union { options, .. } if has_caches && options.parallel => {
                if let Union { options, .. } = lp_arena.get_mut(frame.current) {
                    options.parallel = false;
                }
            },
            Cache { input, id, .. } => {
                if let Some(cache_id) = frame.cache_id {
                    frame.previous_cache = Some(cache_id)
                }
                if frame.parent[0].is_some() {
                    // Projection pushdown has already run and blocked on cache nodes
                    // the pushed down columns are projected just above this cache
                    // if there were no pushed down column, we just take the current
                    // nodes schema
                    // we never want to naively take parents, as a join or aggregate for instance
                    // change the schema

                    let v = cache_schema_and_children
                        .entry(*id)
                        .or_insert_with(Value::default);
                    v.children.push(*input);

                    let mut found_columns = false;
                    for &parent_node in &frame.parent {
                        if let Some(parent_node) = parent_node {
                            get_upper_projections(
                                parent_node,
                                lp_arena,
                                expr_arena,
                                &mut names_scratch,
                            );
                            if !names_scratch.is_empty() {
                                found_columns = true;
                                v.names_union.extend(names_scratch.drain(..));
                            }
                        }
                    }

                    // There was no explicit projection and we must take
                    // all columns
                    if !found_columns {
                        let schema = lp.schema(lp_arena);
                        v.names_union.extend(
                            schema
                                .iter_names()
                                .map(|name| ColumnName::from(name.as_str())),
                        );
                    }
                }
                frame.cache_id = Some(*id);
            },
            _ => {},
        }

        // Shift parents.
        frame.parent[1] = frame.parent[0];
        frame.parent[0] = Some(frame.current);
        for n in scratch.iter() {
            let mut new_frame = frame;
            new_frame.current = *n;
            stack.push(new_frame);
        }
        scratch.clear();
    }

    // # Second pass.
    // we create a subtree where we project the columns
    // just before the cache. Then we do another projection pushdown
    // and finally remove that last projection and stitch the subplan
    // back to the cache node again
    if !cache_schema_and_children.is_empty() {
        let mut proj_pd = ProjectionPushDown::new();
        let pred_pd = PredicatePushDown::new(Default::default());
        for (_cache_id, v) in cache_schema_and_children {
            if !v.names_union.is_empty() {
                for child in v.children {
                    let columns = &v.names_union;
                    let child_lp = lp_arena.take(child);

                    // Make sure we project in the order of the schema
                    // if we don't a union may fail as we would project by the
                    // order we discovered all values.
                    let child_schema = child_lp.schema(lp_arena);
                    let child_schema = child_schema.as_ref();
                    let projection: Vec<_> = child_schema
                        .iter_names()
                        .flat_map(|name| columns.get(name.as_str()).map(|name| name.as_ref()))
                        .collect();

                    let new_child = lp_arena.add(child_lp);

                    let lp = ALogicalPlanBuilder::new(new_child, expr_arena, lp_arena)
                        .project_simple(projection.iter().copied())
                        .unwrap()
                        .build();

                    let lp = proj_pd.optimize(lp, lp_arena, expr_arena)?;
                    let lp = pred_pd.optimize(lp, lp_arena, expr_arena)?;
                    // Remove the projection added by the optimization.
                    let lp = if let ALogicalPlan::Projection { input, .. }
                    | ALogicalPlan::SimpleProjection { input, .. } = lp
                    {
                        lp_arena.take(input)
                    } else {
                        lp
                    };
                    lp_arena.replace(child, lp);
                }
            }
        }
    }
    Ok(())
}
