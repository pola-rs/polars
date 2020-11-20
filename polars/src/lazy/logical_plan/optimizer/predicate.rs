use crate::lazy::logical_plan::optimizer::check_down_node;
use crate::lazy::prelude::*;
use crate::lazy::utils::{count_downtree_projections, expr_to_root_column, rename_expr_root_name};
use crate::prelude::*;
use ahash::RandomState;
use std::collections::HashMap;
use std::sync::Arc;
// arbitrary constant to reduce reallocation.
// don't expect more than 100 predicates.
const HASHMAP_SIZE: usize = 100;

fn init_hashmap<K, V>() -> HashMap<K, V, RandomState> {
    HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new())
}

/// Don't overwrite predicates but combine them.
fn insert_and_combine_predicate(
    predicates_map: &mut HashMap<Arc<String>, Expr, RandomState>,
    name: Arc<String>,
    predicate: Expr,
) {
    let existing_predicate = predicates_map.entry(name).or_insert_with(|| lit(true));
    *existing_predicate = existing_predicate.clone().and(predicate)
}

pub struct PredicatePushDown {}

fn combine_predicates<'a, I>(iter: I) -> Expr
where
    I: Iterator<Item = &'a Expr>,
{
    let mut single_pred = None;
    for expr in iter {
        single_pred = match single_pred {
            None => Some(expr.clone()),
            Some(e) => Some(e.and(expr.clone())),
        };
    }
    single_pred.unwrap()
}

impl PredicatePushDown {
    fn finish_at_leaf(
        &self,
        lp: LogicalPlan,
        acc_predicates: HashMap<Arc<String>, Expr, RandomState>,
    ) -> Result<LogicalPlan> {
        match acc_predicates.len() {
            // No filter in the logical plan
            0 => Ok(lp),
            _ => {
                let mut builder = LogicalPlanBuilder::from(lp);

                let predicate = combine_predicates(acc_predicates.values());
                builder = builder.filter(predicate);
                Ok(builder.build())
            }
        }
    }

    fn finish_node(
        &self,
        local_predicates: Vec<Expr>,
        mut builder: LogicalPlanBuilder,
    ) -> Result<LogicalPlan> {
        if !local_predicates.is_empty() {
            let predicate = combine_predicates(local_predicates.iter());
            builder = builder.filter(predicate);
            Ok(builder.build())
        } else {
            Ok(builder.build())
        }
    }

    // acc predicates maps the root column names to predicates
    fn push_down(
        &self,
        logical_plan: LogicalPlan,
        mut acc_predicates: HashMap<Arc<String>, Expr, RandomState>,
    ) -> Result<LogicalPlan> {
        use LogicalPlan::*;

        match logical_plan {
            Selection { predicate, input } => {
                match expr_to_root_column(&predicate) {
                    Ok(name) => insert_and_combine_predicate(&mut acc_predicates, name, predicate),
                    Err(e) => {
                        if let Expr::BinaryExpr { left, right, .. } = &predicate {
                            let left_name = expr_to_root_column(&*left)?;
                            let right_name = expr_to_root_column(&*right)?;
                            let name = Arc::new(format!("{}-binary-{}", left_name, right_name));
                            insert_and_combine_predicate(&mut acc_predicates, name, predicate);
                        } else {
                            panic!(format!("{:?}", e))
                        }
                    }
                }
                self.push_down(*input, acc_predicates)
            }
            Projection { expr, input, .. } => {
                // don't filter before the last projection that is more expensive as projections are free
                if count_downtree_projections(&input, 0) == 0 {
                    let builder = LogicalPlanBuilder::from(self.push_down(
                        *input,
                        HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new()),
                    )?)
                    .project(expr);
                    // todo! write utility that takes hashmap values by value
                    self.finish_node(acc_predicates.values().cloned().collect(), builder)
                } else {
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
                                insert_and_combine_predicate(
                                    &mut acc_predicates,
                                    new_name,
                                    new_predicate,
                                );
                            }
                        }
                    }
                    Ok(
                        LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?)
                            .project(expr)
                            .build(),
                    )
                }
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
            DataFrameOp { input, operation } => {
                let input = self.push_down(*input, acc_predicates)?;
                Ok(DataFrameOp {
                    input: Box::new(input),
                    operation,
                })
            }
            Aggregate {
                input,
                keys,
                aggs,
                schema,
            } => {
                // dont push down predicates. An aggregation needs all rows
                let lp = Aggregate {
                    input: Box::new(self.push_down(*input, init_hashmap())?),
                    keys,
                    aggs,
                    schema,
                };
                self.finish_at_leaf(lp, acc_predicates)
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

                let mut pushdown_left = init_hashmap();
                let mut pushdown_right = init_hashmap();
                let mut local_predicates = vec![];

                for predicate in acc_predicates.values() {
                    // no else if. predicate can be in both tables.
                    if check_down_node(&predicate, schema_left) {
                        let name =
                            Arc::new(predicate.to_field(schema_left).unwrap().name().clone());
                        insert_and_combine_predicate(&mut pushdown_left, name, predicate.clone());
                    }
                    if check_down_node(&predicate, schema_right) {
                        let name =
                            Arc::new(predicate.to_field(schema_right).unwrap().name().clone());
                        insert_and_combine_predicate(&mut pushdown_right, name, predicate.clone());
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

                if !local.is_empty() {
                    let predicate = combine_predicates(local.iter());
                    lp_builder = lp_builder.filter(predicate);
                }
                Ok(lp_builder.build())
            }
        }
    }

    /// Check if a predicate can be pushed down or not. If it cannot remove it from the accumulated predicates.
    fn split_pushdown_and_local(
        &self,
        mut acc_predicates: HashMap<Arc<String>, Expr, RandomState>,
        schema: &Schema,
    ) -> (Vec<Expr>, HashMap<Arc<String>, Expr, RandomState>) {
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
        self.push_down(
            logical_plan,
            HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new()),
        )
    }
}
