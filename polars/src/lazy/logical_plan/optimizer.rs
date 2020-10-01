use crate::lazy::prelude::*;
use crate::prelude::*;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> LogicalPlan;
}

pub struct ProjectionPushDown {}

impl ProjectionPushDown {
    fn finish_at_leaf(&self, lp: LogicalPlan, acc_projections: Vec<Expr>) -> LogicalPlan {
        match acc_projections.len() {
            // There was no Projection in the logical plan
            0 => lp,
            _ => LogicalPlanBuilder::from(lp)
                .project(acc_projections)
                .build(),
        }
    }

    // check if a projection can be done upstream or should be done in this level of the tree.
    fn check_down_node(&self, expr: &Expr, down_schema: &Schema) -> bool {
        expr.to_field(down_schema).is_ok()
    }

    // split in a projection vec that can be pushed down and a projection vec that should be used
    // in this node
    fn split_acc_projections(
        &self,
        acc_projections: Vec<Expr>,
        down_schema: &Schema,
    ) -> (Vec<Expr>, Vec<Expr>) {
        // If node above has as many columns as the projection there is nothing to pushdown.
        if down_schema.fields().len() == acc_projections.len() {
            let local_projections = acc_projections;
            (vec![], local_projections)
        } else {
            let (acc_projections, local_projections) = acc_projections
                .into_iter()
                .partition(|expr| self.check_down_node(expr, down_schema));
            (acc_projections, local_projections)
        }
    }

    fn finish_node(
        &self,
        local_projections: Vec<Expr>,
        builder: LogicalPlanBuilder,
    ) -> LogicalPlan {
        if local_projections.len() > 0 {
            builder.project(local_projections).build()
        } else {
            builder.build()
        }
    }

    // We recurrently traverse the logical plan and every projection we encounter we add to the accumulated
    // projections.
    // Every non projection operation we recurse and rebuild that operation on the output of the recursion.
    // The recursion stops at the nodes of the logical plan. These nodes IO or existing DataFrames. On top of
    // these nodes we apply the projection.
    // TODO: renaming operations and joins interfere with the schema. We need to keep track of the schema somehow.
    fn push_down(&self, logical_plan: LogicalPlan, mut acc_projections: Vec<Expr>) -> LogicalPlan {
        use LogicalPlan::*;
        match logical_plan {
            Projection { expr, input, .. } => {
                for e in expr {
                    acc_projections.push(e);
                }

                let (acc_projections, local_projections) =
                    self.split_acc_projections(acc_projections, input.schema());

                let lp = self.push_down(*input, acc_projections);
                let builder = LogicalPlanBuilder::from(lp);
                self.finish_node(local_projections, builder)
            }
            DataFrameScan { df, schema } => {
                let lp = DataFrameScan { df, schema };
                self.finish_at_leaf(lp, acc_projections)
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
            } => {
                let lp = CsvScan {
                    path,
                    schema,
                    has_header,
                    delimiter,
                };
                self.finish_at_leaf(lp, acc_projections)
            }
            Sort {
                input,
                column,
                reverse,
            } => LogicalPlanBuilder::from(self.push_down(*input, acc_projections))
                .sort(column, reverse)
                .build(),
            Selection { predicate, input } => {
                LogicalPlanBuilder::from(self.push_down(*input, acc_projections))
                    .filter(predicate)
                    .build()
            }
            Aggregate {
                input, keys, aggs, ..
            } => {
                // TODO: projections of resulting columns of gb, should be renamed and pushed down
                let (acc_projections, local_projections) =
                    self.split_acc_projections(acc_projections, input.schema());

                let lp = self.push_down(*input, acc_projections);
                let builder = LogicalPlanBuilder::from(lp).groupby(keys, aggs);
                self.finish_node(local_projections, builder)
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                how,
                ..
            } => {
                let schema_left = input_left.schema();
                let schema_right = input_right.schema();
                let mut pushdown_left = vec![];
                let mut pushdown_right = vec![];
                let mut local_projection = vec![];
                for proj in acc_projections {
                    let mut pushed_down = false;
                    if self.check_down_node(&proj, schema_left) {
                        pushdown_left.push(proj.clone());
                        pushed_down = true;
                    }
                    if self.check_down_node(&proj, schema_right) {
                        pushdown_right.push(proj.clone());
                        pushed_down = true;
                    }
                    if !pushed_down {
                        local_projection.push(proj)
                    }
                }
                let lp_left = self.push_down(*input_left, pushdown_left);
                let lp_right = self.push_down(*input_right, pushdown_right);
                let builder =
                    LogicalPlanBuilder::from(lp_left).join(lp_right, how, left_on, right_on);
                self.finish_node(local_projection, builder)
            }
        }
    }
}

impl Optimize for ProjectionPushDown {
    fn optimize(&self, logical_plan: LogicalPlan) -> LogicalPlan {
        self.push_down(logical_plan, Vec::default())
    }
}

pub struct PredicatePushDown {}

impl PredicatePushDown {
    fn finish_at_leaf(&self, lp: LogicalPlan, acc_predicates: Vec<Expr>) -> LogicalPlan {
        match acc_predicates.len() {
            // No filter in the logical plan
            0 => lp,
            _ => {
                let mut builder = LogicalPlanBuilder::from(lp);
                for expr in acc_predicates {
                    builder = builder.filter(expr);
                }
                builder.build()
            }
        }
    }

    fn push_down(&self, logical_plan: LogicalPlan, mut acc_predicates: Vec<Expr>) -> LogicalPlan {
        use LogicalPlan::*;
        match logical_plan {
            Selection { predicate, input } => {
                acc_predicates.push(predicate);
                self.push_down(*input, acc_predicates)
            }
            Projection { expr, input, .. } => {
                LogicalPlanBuilder::from(self.push_down(*input, acc_predicates))
                    .project(expr)
                    .build()
            }
            DataFrameScan { df, schema } => {
                let lp = DataFrameScan { df, schema };
                self.finish_at_leaf(lp, acc_predicates)
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
            } => {
                let lp = CsvScan {
                    path,
                    schema,
                    has_header,
                    delimiter,
                };
                self.finish_at_leaf(lp, acc_predicates)
            }
            Sort {
                input,
                column,
                reverse,
            } => LogicalPlanBuilder::from(self.push_down(*input, acc_predicates))
                .sort(column, reverse)
                .build(),
            Aggregate {
                input, keys, aggs, ..
            } => LogicalPlanBuilder::from(self.push_down(*input, acc_predicates))
                .groupby(keys, aggs)
                .build(),
            Join { .. } => todo!(),
        }
    }
}

impl Optimize for PredicatePushDown {
    fn optimize(&self, logical_plan: LogicalPlan) -> LogicalPlan {
        self.push_down(logical_plan, vec![])
    }
}

#[cfg(test)]
mod test {
    use crate::lazy::logical_plan::optimizer::Optimize;
    use crate::lazy::prelude::*;
    use crate::lazy::tests::get_df;

    #[test]
    fn test_logical_plan() {
        let df = get_df();

        // expensive order
        let lf = df
            .clone()
            .lazy()
            .sort("sepal.width", false)
            .select(&[col("sepal.width")]);

        let logical_plan = lf.logical_plan;
        let opt = ProjectionPushDown {};
        let logical_plan = opt.optimize(logical_plan);
        println!("{}", logical_plan.describe());
    }
}
