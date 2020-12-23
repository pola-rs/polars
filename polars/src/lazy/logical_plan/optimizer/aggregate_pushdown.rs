use crate::lazy::logical_plan::Context;
use crate::lazy::prelude::*;
use crate::lazy::utils::{aexpr_to_root_nodes, has_aexpr};
use crate::prelude::*;
use ahash::RandomState;
use std::collections::{hash_map::Entry, HashMap};
use std::sync::Arc;

pub(crate) struct AggregatePushdown {
    state: HashMap<Arc<String>, Node, RandomState>,
}

impl AggregatePushdown {
    fn drain_nodes(&mut self) -> impl Iterator<Item = Node> {
        let state = std::mem::take(&mut self.state);
        state.into_iter().map(|tpl| tpl.1)
    }

    fn pushdown_projection(
        &mut self,
        node: Node,
        expr: Vec<Node>,
        input: Node,
        schema: Schema,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Option<ALogicalPlan> {
        let new_expr: Vec<_> = expr
            .iter()
            .filter_map(|node| {
                let dummy = AExpr::Agg(AAggExpr::Min(Node(usize::max_value())));

                // has an aggregation
                if has_aexpr(*node, expr_arena, &dummy, false) {
                    let roots = aexpr_to_root_nodes(*node, expr_arena);

                    // aggregation can be pushed down
                    if roots.len() == 1 {
                        let root_node = roots[0];
                        if let AExpr::Column(name) = expr_arena.get(root_node) {
                            match self.state.entry(name.clone()) {
                                Entry::Occupied(_) => Some(*node),
                                Entry::Vacant(e) => {
                                    e.insert(*node);
                                    None
                                }
                            }
                        } else {
                            unreachable!()
                        }
                        // None
                        // aggregation can not be pushed down
                    } else {
                        Some(*node)
                    }
                    // has no aggregation
                } else {
                    Some(*node)
                }
            })
            .collect();

        // All expressions were aggregations
        if new_expr.is_empty() {
            // swap projection with the input node
            let lp = lp_arena.take(input);
            Some(lp)
            // nothing changed
        } else if expr.len() == new_expr.len() {
            // restore lp node
            lp_arena.assign(
                node,
                ALogicalPlan::Projection {
                    expr,
                    input,
                    schema,
                },
            );
            None
        } else {
            Some(ALogicalPlan::Projection {
                expr: new_expr,
                input,
                schema,
            })
        }
    }
}

impl OptimizationRule for AggregatePushdown {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.take(node);
        use ALogicalPlan::*;
        match lp {
            Projection {
                expr,
                input,
                schema,
            } => self.pushdown_projection(node, expr, input, schema, lp_arena, expr_arena),
            LocalProjection {
                expr,
                input,
                schema,
            } => self.pushdown_projection(node, expr, input, schema, lp_arena, expr_arena),
            Join { .. } | Aggregate { .. } => {
                if self.state.is_empty() {
                    None
                } else {
                    // we cannot pass a join or GroupBy so we do the projection here
                    let new_node = lp_arena.add(lp);
                    let input_schema = lp_arena.get(new_node).schema(lp_arena);

                    let nodes: Vec<_> = self.drain_nodes().collect();
                    let fields = nodes
                        .iter()
                        .map(|n| {
                            expr_arena
                                .get(*n)
                                .to_field(input_schema, Context::Aggregation, expr_arena)
                                .unwrap()
                        })
                        .collect();

                    Some(ALogicalPlan::Projection {
                        expr: nodes,
                        input: new_node,
                        schema: Schema::new(fields),
                    })
                }
            }
            CsvScan {
                ..
                // path,
                // schema,
                // has_header,
                // delimiter,
                // ignore_errors,
                // skip_rows,
                // stop_after_n_rows,
                // with_columns,
                // predicate,
                // cache,
            } => {
                todo!()
            }

            _ => None,
        }
    }
}
