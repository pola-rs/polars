use crate::lazy::prelude::*;
use crate::prelude::*;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> LogicalPlan;
}

/// Take an expression and unwrap to Expr::Column() if it exists.
fn expr_to_root_column(expr: &Expr) -> Result<Expr> {
    use Expr::*;
    match expr {
        Column(name) => Ok(Column(name.clone())),
        Alias(expr, ..) => expr_to_root_column(expr),
        Literal(_) => Err(PolarsError::Other("no root column exits for lit".into())),
        // todo: return root columns? multiple?
        BinaryExpr { .. } => Err(PolarsError::Other(
            "no root column exits for binary expr".into(),
        )),
        Not(expr) => expr_to_root_column(expr),
        IsNotNull(expr) => expr_to_root_column(expr),
        IsNull(expr) => expr_to_root_column(expr),
        Sort { expr, .. } => expr_to_root_column(expr),
        AggMin(expr) => expr_to_root_column(expr),
    }
}

// Result<[Column("foo"), Column("bar")]>
fn expressions_to_root_columns(exprs: &[Expr]) -> Result<Vec<Expr>> {
    exprs.into_iter().map(|e| expr_to_root_column(e)).collect()
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
    fn check_upstream(&self, expr: &Expr, upstream_schema: &Schema) -> bool {
        expr.to_field(upstream_schema).is_ok()
    }

    // split in a projection vec that can be pushed upstream and a projection vec that should be used
    // in this node
    fn split_acc_projections(
        &self,
        acc_projections: Vec<Expr>,
        upstream_schema: &Schema,
    ) -> (Vec<Expr>, Vec<Expr>) {
        // If node above has as many columns as the projection there is nothing to pushdown.
        if upstream_schema.fields().len() == acc_projections.len() {
            let local_projections = acc_projections;
            (vec![], local_projections)
        } else {
            let (acc_projections, local_projections) = acc_projections
                .into_iter()
                .partition(|expr| self.check_upstream(expr, upstream_schema));
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
                let (acc_projections, local_projections) =
                    self.split_acc_projections(acc_projections, input.schema());

                let lp = self.push_down(*input, acc_projections);
                let builder = LogicalPlanBuilder::from(lp).groupby(keys, aggs);
                self.finish_node(local_projections, builder)
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
