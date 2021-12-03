use super::*;
use crate::logical_plan::iterator::ArenaExprIter;
use crate::utils::{has_aexpr, aexpr_to_root_names, aexpr_to_root_column_name, aexpr_to_root_nodes};
use ahash::AHashMap;

pub(crate) struct SeparateWindowExprs {}

fn is_window(e: Node, expr_arena: &Arena<AExpr>) -> bool {
    has_aexpr(e, expr_arena, |ae| matches!(ae, AExpr::Window {..}))
}

fn has_window_exprs(expr: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    expr.iter().any(|e| {
        is_window(*e, expr_arena)
    })
}
fn cached_window(expr: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    expr.iter().any(|e| {
        has_aexpr(*e, expr_arena, |ae| {
            if let AExpr::Window {options, ..}  = ae {
                options.cache
            } else {
                false
            }
        })
    })
}

impl OptimizationRule for SeparateWindowExprs {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get(node);

        match lp {
            ALogicalPlan::Projection {
                input,
                expr,
                ..
            } => {
                // we do a double scan, otherwise we always reallocate a vec
                if has_window_exprs(expr, expr_arena) && !cached_window(expr, expr_arena) {
                    let mut other_e = Vec::with_capacity(expr.len());
                    let mut window_e = Vec::with_capacity(expr.len());

                    for e in expr {
                        if is_window(*e, expr_arena) {
                            window_e.push(*e)
                        } else {
                            other_e.push(*e)
                        }
                    }

                    let mut name_count = AHashMap::with_capacity(window_e.len());

                    /// fill the name_count
                    /// this is a hashmap that counts the partition columns
                    /// so the following expressions
                    ///
                    /// col(foo).over(bar)
                    /// col(bar).over(bar)
                    /// col(ham).over(foo)
                    ///
                    /// would lead to the following count
                    /// bar -> 2
                    /// foo -> 1
                    for e in &window_e {
                        (&*expr_arena).iter(*e).for_each(|(node, ae)| {
                            if let AExpr::Window {partition_by, ..} = ae {
                                partition_by.iter().for_each(|node| {
                                    let ae = expr_arena.get(*node);
                                    // if a simple expression we add the name
                                    if let Some(1) = ae.depth(expr_arena) {
                                        let name = aexpr_to_root_column_name(*node, expr_arena).unwrap();
                                        name_count.entry(name).and_modify(|v| *v += 1).or_insert(1usize);
                                    }
                                })
                            }
                        })
                    }

                    let mut to_cache = Vec::with_capacity(window_e.len());


                    // Replace the window functions that have duplicate groups
                    // with a cached variant
                    for e in &window_e {
                        (&*expr_arena).iter(*e).for_each(|(opt_window_node, opt_window_ae)| {
                            if let AExpr::Window {function, partition_by, order_by, options} = opt_window_ae {
                                partition_by.iter().for_each(|node| {
                                    let ae = expr_arena.get(*node);
                                    // if a simple expression we add the name
                                    if let Some(1) = ae.depth(expr_arena) {
                                        let name = aexpr_to_root_column_name(*node, expr_arena).unwrap();
                                        let count = *name_count.get(name.as_ref()).unwrap();

                                        if count > 1 {
                                            let mut options = options.clone();
                                            options.cache = true;
                                            to_cache.push((opt_window_node, AExpr::Window { function: *function, partition_by: partition_by.clone(), order_by: *order_by,  options}));
                                        }
                                    }
                                })
                            }
                        })
                    }
                    if to_cache.is_empty() {
                        return None
                    }

                    for (node, ae) in to_cache {
                        expr_arena.replace(node, ae)
                    }

                    let lp  = ALogicalPlanBuilder::new(*input, expr_arena, lp_arena)
                        .project(window_e)
                        .build();

                    let lp  = ALogicalPlanBuilder::new(*input, expr_arena, lp_arena)
                        .project(other_e)
                        .build();

                    ALogicalPlan::HStack {

                    }


                    Some(lp)

                }
                else {
                    None
                }

            }
            _ => None
        }
    }
}
