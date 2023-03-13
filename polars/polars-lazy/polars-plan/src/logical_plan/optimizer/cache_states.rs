use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

fn get_upper_projections(
    parent: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<Arc<str>>> {
    let parent = lp_arena.get(parent);

    use ALogicalPlan::*;
    // during projection pushdown all accumulated
    match parent {
        Projection { expr, .. } => {
            let mut out = Vec::with_capacity(expr.len());
            for node in expr {
                out.extend(aexpr_to_leaf_names_iter(*node, expr_arena));
            }
            Some(out)
        }
        // other
        _ => None,
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
) {
    let mut loop_count = 0;
    let mut stack = Vec::with_capacity(4);

    // we loop because there can be nested caches and we must run the projection pushdown
    // optimization between cache nodes.
    loop {
        scratch.clear();
        stack.clear();

        // per cache id holds:
        // a Vec: with children of the node
        // a Set: with the union of projected column names.
        // a Set: with the union of hstack column names.
        let mut cache_schema_and_children = BTreeMap::new();

        stack.push((root, None, None, None, 0));

        // the depth of the caches in a single tree branch
        let mut max_cache_depth = 0;

        // first traversal
        // collect the union of columns per cache id.
        // and find the cache parents
        while let Some((
            current_node,
            mut cache_id,
            mut parent,
            mut previous_cache,
            mut caches_seen,
        )) = stack.pop()
        {
            let lp = lp_arena.get(current_node);
            lp.copy_inputs(scratch);

            use ALogicalPlan::*;
            match lp {
                // don't allow parallelism as caches need each others work
                // also self-referencing plans can deadlock on the files they lock
                Join { options, .. } if has_caches && options.allow_parallel => {
                    if let Join { options, .. } = lp_arena.get_mut(current_node) {
                        options.allow_parallel = false;
                    }
                }
                // don't allow parallelism as caches need each others work
                // also self-referencing plans can deadlock on the files they lock
                Union { options, .. } if has_caches && options.parallel => {
                    if let Union { options, .. } = lp_arena.get_mut(current_node) {
                        options.parallel = false;
                    }
                }
                Cache { input, id, .. } => {
                    caches_seen += 1;

                    // no need to run the same cache optimization twice
                    if loop_count > caches_seen {
                        continue;
                    }

                    max_cache_depth = std::cmp::max(caches_seen, max_cache_depth);
                    if let Some(cache_id) = cache_id {
                        previous_cache = Some(cache_id)
                    }
                    if let Some(parent_node) = parent {
                        // projection pushdown has already run and blocked on cache nodes
                        // the pushed down columns are projected just above this cache
                        // if there were no pushed down column, we just take the current
                        // nodes schema
                        // we never want to naively take parents, as a join or aggregate for instance
                        // change the schema

                        let (children, union_names) = cache_schema_and_children
                            .entry(*id)
                            .or_insert_with(|| (Vec::new(), PlHashSet::new()));
                        children.push(*input);

                        if let Some(names) =
                            get_upper_projections(parent_node, lp_arena, expr_arena)
                        {
                            union_names.extend(names);
                        }
                        // There was no explicit projection and we must take
                        // all columns
                        else {
                            let schema = lp.schema(lp_arena);
                            union_names
                                .extend(schema.iter_names().map(|name| Arc::from(name.as_str())));
                        }
                    }
                    cache_id = Some(*id);
                }
                _ => {}
            }

            parent = Some(current_node);
            for n in scratch.iter() {
                stack.push((*n, cache_id, parent, previous_cache, caches_seen))
            }
            scratch.clear();
        }

        // second pass
        // we create a subtree where we project the columns
        // just before the cache. Then we do another projection pushdown
        // and finally remove that last projection and stitch the subplan
        // back to the cache node again
        if !cache_schema_and_children.is_empty() {
            let mut pd = ProjectionPushDown::new();
            for (_cache_id, (children, columns)) in cache_schema_and_children {
                if !columns.is_empty() {
                    for child in children {
                        let columns = &columns;
                        let child_lp = lp_arena.get(child).clone();

                        // make sure we project in the order of the schema
                        // if we don't a union may fail as we would project by the
                        // order we discovered all values.
                        let child_schema = child_lp.schema(lp_arena);
                        let child_schema = child_schema.as_ref();
                        let projection: Vec<_> = child_schema
                            .iter_names()
                            .flat_map(|name| {
                                columns
                                    .get(name.as_str())
                                    .map(|name| expr_arena.add(AExpr::Column(name.clone())))
                            })
                            .collect();

                        let new_child = lp_arena.add(child_lp);
                        let lp = ALogicalPlanBuilder::new(new_child, expr_arena, lp_arena)
                            .project(projection.clone())
                            .build();

                        let lp = pd.optimize(lp, lp_arena, expr_arena).unwrap();
                        // remove the projection added by the optimization
                        let lp = if let ALogicalPlan::Projection { input, .. }
                        | ALogicalPlan::LocalProjection { input, .. } = lp
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

        if loop_count >= max_cache_depth {
            break;
        }
        loop_count += 1;
    }
}
