use crate::lazy::logical_plan::optimizer::check_down_node;
use crate::lazy::prelude::*;
use crate::lazy::utils::{expr_to_root_column, rename_expr_root_name};
use crate::prelude::*;
use fnv::FnvHashMap;
use std::sync::Arc;

pub struct PredicatePushDown {}

impl PredicatePushDown {
    fn finish_at_leaf(
        &self,
        lp: LogicalPlan,
        acc_predicates: FnvHashMap<Arc<String>, Expr>,
    ) -> Result<LogicalPlan> {
        match acc_predicates.len() {
            // No filter in the logical plan
            0 => Ok(lp),
            _ => {
                // TODO: create a single predicate
                let mut builder = LogicalPlanBuilder::from(lp);
                for expr in acc_predicates.values() {
                    builder = builder.filter(expr.clone());
                }
                Ok(builder.build())
            }
        }
    }

    fn finish_node(
        &self,
        local_predicates: Vec<Expr>,
        mut builder: LogicalPlanBuilder,
    ) -> Result<LogicalPlan> {
        if local_predicates.len() > 0 {
            for expr in local_predicates {
                builder = builder.filter(expr);
            }
            Ok(builder.build())
        } else {
            Ok(builder.build())
        }
    }

    // acc predicates maps the root column names to predicates
    fn push_down(
        &self,
        logical_plan: LogicalPlan,
        mut acc_predicates: FnvHashMap<Arc<String>, Expr>,
    ) -> Result<LogicalPlan> {
        use LogicalPlan::*;

        match logical_plan {
            Selection { predicate, input } => {
                match expr_to_root_column(&predicate) {
                    Ok(name) => {
                        acc_predicates.insert(name, predicate);
                    }
                    Err(_) => panic!("implement logic for binary expr with 2 root columns"),
                }
                self.push_down(*input, acc_predicates)
            }
            Projection { expr, input, .. } => {
                // maybe update predicate name if a projection is an alias
                for e in &expr {
                    // check if there is an alias
                    if let Expr::Alias(e, name) = e {
                        // if this alias refers to one of the predicates in the upper nodes
                        // we rename the column of the predicate before we push it downwards.
                        if let Some(predicate) = acc_predicates.remove(name) {
                            let new_name = expr_to_root_column(e).unwrap();
                            let new_predicate =
                                rename_expr_root_name(&predicate, new_name.clone()).unwrap();
                            acc_predicates.insert(new_name, new_predicate);
                        }
                    }
                }
                Ok(
                    LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?)
                        .project(expr)
                        .build(),
                )
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
            } => Ok(
                LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?)
                    .sort(column, reverse)
                    .build(),
            ),
            Aggregate {
                input, keys, aggs, ..
            } => Ok(
                LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?)
                    .groupby(keys, aggs)
                    .build(),
            ),
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

                let mut pushdown_left = FnvHashMap::default();
                let mut pushdown_right = FnvHashMap::default();
                let mut local_predicates = vec![];

                for predicate in acc_predicates.values() {
                    if check_down_node(&predicate, schema_left) {
                        let name =
                            Arc::new(predicate.to_field(schema_left).unwrap().name().clone());
                        pushdown_left.insert(name, predicate.clone());
                    } else if check_down_node(&predicate, schema_right) {
                        let name =
                            Arc::new(predicate.to_field(schema_right).unwrap().name().clone());
                        pushdown_right.insert(name, predicate.clone());
                    } else {
                        local_predicates.push(predicate.clone())
                    }
                }

                let lp_left = self.push_down(*input_left, pushdown_left)?;
                let lp_right = self.push_down(*input_right, pushdown_right)?;

                let builder =
                    LogicalPlanBuilder::from(lp_left).join(lp_right, how, left_on, right_on);
                self.finish_node(local_predicates, builder)
            }
            HStack { input, exprs, .. } => {
                let (local, acc_predicates) =
                    self.split_pushdown_and_local(acc_predicates, input.schema());
                let mut lp_builder =
                    LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?)
                        .with_columns(exprs);

                for predicate in local {
                    lp_builder = lp_builder.filter(predicate);
                }
                Ok(lp_builder.build())
            }
        }
    }

    /// Check if a predicate can be pushed down or not. If it cannot remove it from the accumulated predicates.
    fn split_pushdown_and_local(
        &self,
        mut acc_predicates: FnvHashMap<Arc<String>, Expr>,
        schema: &Schema,
    ) -> (Vec<Expr>, FnvHashMap<Arc<String>, Expr>) {
        let mut local = Vec::with_capacity(acc_predicates.len());
        let mut local_keys = Vec::with_capacity(acc_predicates.len());
        for (key, predicate) in &acc_predicates {
            if !check_down_node(predicate, schema) {
                local_keys.push(key.clone());
            }
        }
        for key in local_keys {
            local.push(acc_predicates.remove(&key).unwrap());
        }
        (local, acc_predicates)
    }
}

impl Optimize for PredicatePushDown {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        self.push_down(logical_plan, FnvHashMap::default())
    }
}
