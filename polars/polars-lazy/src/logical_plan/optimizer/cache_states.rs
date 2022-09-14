use std::collections::BTreeMap;
use std::sync::Arc;

use polars_core::prelude::PlIndexSet;

use crate::prelude::*;

fn get_pushdown_names(
    parent: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<Arc<str>>> {
    let parent = lp_arena.get(parent);

    use ALogicalPlan::*;
    match parent {
        Projection { expr, .. } | HStack { exprs: expr, .. } => Some(
            expr.iter()
                .map(|node| aexpr_to_root_column_name(*node, expr_arena).unwrap())
                .collect(),
        ),
        Join {
            left_on, right_on, ..
        } => {
            let iter_left = left_on
                .iter()
                .map(|node| aexpr_to_root_column_name(*node, expr_arena).unwrap());
            let iter_right = right_on
                .iter()
                .map(|node| aexpr_to_root_column_name(*node, expr_arena).unwrap());
            Some(iter_left.chain(iter_right).collect())
        }
        Aggregate {
            keys,
            aggs,
            options,
            ..
        } => {
            let keys = keys
                .iter()
                .map(|node| aexpr_to_root_column_name(*node, expr_arena).unwrap());
            let aggs = aggs
                .iter()
                .map(|node| aexpr_to_root_column_name(*node, expr_arena).unwrap());
            let mut names = keys.chain(aggs).collect::<Vec<_>>();
            if let Some(opt) = &options.rolling {
                names.push(Arc::from(opt.index_column.as_str()))
            }
            if let Some(opt) = &options.dynamic {
                names.push(Arc::from(opt.index_column.as_str()))
            }
            Some(names)
        }
        // todo! add more
        _ => None,
    }
}

pub(crate) fn set_cache_states(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
) {
    scratch.clear();

    // per cache id holds:
    // a Vec: with (parent, child) pairs
    // a Set: with the union of column names.
    let mut cache_schema_and_children = BTreeMap::new();

    let mut stack = Vec::with_capacity(4);
    stack.push((root, None, None));

    // first traversal
    // collect the union of columns per cache id.
    // and find the cache parents
    while let Some((node, mut cache_id, mut parent)) = stack.pop() {
        let lp = lp_arena.get(node);
        lp.copy_inputs(scratch);

        use ALogicalPlan::*;
        match lp {
            // don't allow parallelism if underneath a cache
            Join { .. } if cache_id.is_some() => {
                if let Join { options, .. } = lp_arena.get_mut(node) {
                    options.allow_parallel = false;
                }
            }
            Cache { input, id, .. } => {
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

                    if let Some(names) = get_pushdown_names(parent_node, lp_arena, expr_arena) {
                        entry.1.extend(names);
                    } else {
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
            stack.push((*n, cache_id, parent))
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
