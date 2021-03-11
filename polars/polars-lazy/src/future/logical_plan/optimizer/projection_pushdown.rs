use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_root_names, aexpr_to_root_nodes, check_down_node, has_aexpr};
use ahash::RandomState;
use polars_core::prelude::*;
use std::collections::HashSet;

/// utility function such that we can recurse all binary expressions in the expression tree
fn add_to_accumulated(
    expr: Node,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut HashSet<Arc<String>, RandomState>,
    expr_arena: &mut Arena<AExpr>,
) {
    for root_node in aexpr_to_root_nodes(expr, expr_arena) {
        for name in aexpr_to_root_names(root_node, expr_arena) {
            if projected_names.insert(name) {
                acc_projections.push(root_node)
            }
        }
    }
}

pub struct ProjectionPushDown {}

impl ProjectionPushDown {
    fn finish_node(
        &self,
        local_projections: Vec<Node>,
        builder: ALogicalPlanBuilder,
    ) -> ALogicalPlan {
        if !local_projections.is_empty() {
            builder.project(local_projections).build()
        } else {
            builder.build()
        }
    }

    /// Helper method. This pushes down current node and assigns the result to this node.
    fn pushdown_and_assign(
        &self,
        input: Node,
        mut acc_projections: Vec<Node>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<()> {
        let alp = lp_arena.take(input);
        let lp = self.push_down(
            alp,
            acc_projections,
            names,
            projections_seen,
            lp_arena,
            expr_arena,
        )?;
        lp_arena.replace(input, lp);
        Ok(())
    }

    /// Projection pushdown optimizer
    ///
    /// # Arguments
    ///
    /// * `AlogicalPlan` - Arena based logical plan tree representing the query.
    /// * `acc_projections` - The projections we accumulate during tree traversal.
    /// * `names` - We keep track of the names to ensure we don't do duplicate projections.
    /// * `projections_seen` - Count the number of projection operations during tree traversal.
    /// * `lp_arena` - The local memory arena for the logical plan.
    /// * `expr_arena` - The local memory arena for the expressions.
    ///
    fn push_down(
        &self,
        logical_plan: ALogicalPlan,
        mut acc_projections: Vec<Node>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Slice { input, offset, len } => {
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Slice { input, offset, len })
            }

            Projection { expr, input, .. } => {
                // A projection can consist of a chain of expressions followed by an alias.
                // We want to do the chain locally because it can have complicated side effects.
                // The only thing we push down is the root name of the projection.
                // So we:
                //      - add the root of the projections to accumulation,
                //      - also do the complete projection locally to keep the schema (column order) and the alias.
                for e in &expr {
                    // in this branch we check a double projection case
                    // df
                    //   .select(col("foo").alias("bar"))
                    //   .select(col("bar")
                    //
                    // In this query, bar cannot pass this projection, as it would not exist in DF.
                    if !acc_projections.is_empty() {
                        if let AExpr::Alias(_, name) = expr_arena.get(*e) {
                            if names.remove(name) {
                                acc_projections = acc_projections
                                    .into_iter()
                                    .filter(|expr| {
                                        !aexpr_to_root_names(*expr, expr_arena).contains(name)
                                    })
                                    .collect();
                            }
                        }
                    }

                    add_to_accumulated(*e, &mut acc_projections, &mut names, expr_arena);
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = lp_arena.get(input);

                let mut local_projection = Vec::with_capacity(expr.len());

                // the projections should all be done at the latest projection node to keep the same schema order
                if projections_seen == 0 {
                    for expr in expr {
                        // TODO! maybe we can remove this check?
                        // We check if we still can the projection here.
                        if expr_arena
                            .get(expr)
                            .to_field(lp.schema(lp_arena), Context::Other, expr_arena)
                            .is_ok()
                        {
                            local_projection.push(expr);
                        }
                    }
                    // only aliases should be projected locally
                } else {
                    for expr in expr {
                        if has_aexpr(
                            expr,
                            expr_arena,
                            &AExpr::Alias(Default::default(), Arc::new("".into())),
                            true,
                        ) {
                            local_projection.push(expr)
                        }
                    }
                }

                let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
                Ok(self.finish_node(local_projection, builder))
            }
            LocalProjection { expr, input, .. } => {
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = lp_arena.get(input);
                let schema = lp.schema(lp_arena);

                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let proj = expr
                    .into_iter()
                    .filter(|e| check_down_node(*e, schema, expr_arena))
                    .collect();
                Ok(ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                    .project_local(proj)
                    .build())
            }

            lp => Ok(lp),
        }
    }
}
