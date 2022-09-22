use std::collections::BTreeMap;
use std::sync::Arc;

use polars_core::prelude::PlIndexSet;

use crate::prelude::*;

fn get_upper_projections(
    parent: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<Arc<str>>> {
    let parent = lp_arena.get(parent);

    use ALogicalPlan::*;
    // during projection pushdown all accumulated p
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
pub(crate) fn set_cache_states(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    has_caches: bool,
) {
    scratch.clear();

    // per cache id holds:
    // a Vec: with (parent, child) pairs
    // a Set: with the union of column names.
    let mut cache_schema_and_children = BTreeMap::new();

    let mut stack = Vec::with_capacity(4);
    stack.push((root, None, None, None));

    // first traversal
    // collect the union of columns per cache id.
    // and find the cache parents
    while let Some((node, mut cache_id, mut parent, mut previous_cache)) = stack.pop() {
        let lp = lp_arena.get(node);
        lp.copy_inputs(scratch);

        use ALogicalPlan::*;
        match lp {
            // don't allow parallelism as caches need eachothers work
            // also self-referencing plans can deadlock on the files they lock
            Join { options, .. } if has_caches && options.allow_parallel => {
                if let Join { options, .. } = lp_arena.get_mut(node) {
                    options.allow_parallel = false;
                }
            }
            // don't allow parallelism as caches need eachothers work
            // also self-referencing plans can deadlock on the files they lock
            Union { options, .. } if has_caches && options.parallel => {
                if let Union { options, .. } = lp_arena.get_mut(node) {
                    options.parallel = false;
                }
            }
            Cache { input, id, .. } => {
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

                    let entry = cache_schema_and_children.entry(*id).or_insert_with(|| {
                        (
                            Vec::new(),
                            PlIndexSet::with_capacity_and_hasher(0, Default::default()),
                        )
                    });
                    entry.0.push(*input);

                    if let Some(names) = get_upper_projections(parent_node, lp_arena, expr_arena) {
                        entry.1.extend(names);
                    }
                    // if there is no projection above, it maybe that the
                    // cache is underneath another cache and projection pushdown never reached it.
                    // other trails may take care of that cache
                    // if there is no other cache above, then there was no projection and we must take
                    // all columns
                    else if previous_cache.is_none() {
                        let schema = lp.schema(lp_arena);
                        entry
                            .1
                            .extend(schema.iter_names().map(|name| Arc::from(name.as_str())));
                    }
                }
                cache_id = Some(*id);
            }
            _ => {}
        }

        parent = Some(node);
        for n in scratch.iter() {
            stack.push((*n, cache_id, parent, previous_cache))
        }
        scratch.clear();
    }

    // second pass
    // we create a subtree where we project the columns
    // just before the cache. Then we do another projection pushdown
    // and finally remove that last projection and stitch the subplan
    // back to the cache node again
    if !cache_schema_and_children.is_empty() {
        let pd = projection_pushdown::ProjectionPushDown {};
        for (_cache_id, (children, columns)) in cache_schema_and_children {
            if !columns.is_empty() {
                let projection = columns
                    .into_iter()
                    .map(|name| expr_arena.add(AExpr::Column(name)))
                    .collect::<Vec<_>>();

                for child in children {
                    let child_lp = lp_arena.get(child).clone();
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
}
