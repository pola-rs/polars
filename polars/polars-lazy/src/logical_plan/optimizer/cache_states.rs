use std::collections::BTreeMap;
use std::sync::Arc;
use polars_core::prelude::{PlHashMap, PlHashSet};
use crate::prelude::*;
use crate::prelude::LogicalPlan::{DataFrameScan, IpcScan};

pub(crate) fn set_cache_states(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>
)  {
    scratch.clear();

    // per cache id holds:
    // a Vec: with (parent, child) pairs
    // a Set: with the union of column names.
    let mut cache_schema_and_cache_family = BTreeMap::new();

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
            Join {..} if cache_id.is_some() => {
                if let Join{options,
                    ..} = lp_arena.get_mut(node) {
                    options.allow_parallel = false;
                }
            }
            Cache {input, id, ..} => {
                if let Some(parent_node) = parent {
                    let parent= lp_arena.get(parent_node);
                    let schema = parent.schema(lp_arena);

                    let entry = cache_schema_and_cache_family.entry(*id).or_insert_with(|| {
                        (Vec::new(), PlHashSet::new())
                    });
                    entry.0.push((parent_node, *input));
                    entry.1.extend(schema.iter_names().map(|name| Arc::from(name.as_str())));

                }
                cache_id = Some(*id);
            },
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
    if !cache_schema_and_cache_family.is_empty() {
        let pd = projection_pushdown::ProjectionPushDown{};
        for (_cache_id, (family, columns)) in cache_schema_and_cache_family {
            let projection = columns.into_iter().map(|name| {
                expr_arena.add(AExpr::Column(name))
            }).collect::<Vec<_>>();

            for (parent, child) in family {
                let child_lp = lp_arena.get(child).clone();
                let new_child = lp_arena.add(child_lp);

                let lp = ALogicalPlanBuilder::new(new_child, expr_arena, lp_arena)
                    .project(projection.clone())
                    .build();

                let lp = pd.optimize(lp, lp_arena, expr_arena).unwrap();
                lp_arena.replace(child, lp);
            }
        }
    }
}
