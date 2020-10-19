use crate::lazy::logical_plan::optimizer::check_down_node;
use crate::lazy::prelude::*;
use crate::lazy::utils::{
    count_downtree_joins, count_downtree_projections, expr_to_root_column, rename_expr_root_name,
};
use crate::prelude::*;
use fnv::{FnvBuildHasher, FnvHashMap};
use std::collections::HashMap;
use std::sync::Arc;

// arbitrary constant to reduce reallocation.
// don't expect more than 100 predicates.
const HASHMAP_SIZE: usize = 100;

fn init_hashmap<K, V>() -> HashMap<K, V, FnvBuildHasher> {
    FnvHashMap::with_capacity_and_hasher(HASHMAP_SIZE, FnvBuildHasher::default())
}

fn init_bubble_up() -> Vec<Expr> {
    Vec::with_capacity(HASHMAP_SIZE)
}

pub struct PredicatePushDown {}

fn combine_predicates<'a, I>(mut iter: I) -> Expr
where
    I: Iterator<Item = &'a Expr>,
{
    let mut single_pred = None;
    while let Some(expr) = iter.next() {
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
        acc_predicates: FnvHashMap<Arc<String>, Expr>,
        bubble_up: Vec<Expr>,
    ) -> Result<(LogicalPlan, Vec<Expr>)> {
        match acc_predicates.len() {
            // No filter in the logical plan
            0 => Ok((lp, bubble_up)),
            _ => {
                let mut builder = LogicalPlanBuilder::from(lp);

                let predicate = combine_predicates(acc_predicates.values());
                builder = builder.filter(predicate);
                Ok((builder.build(), bubble_up))
            }
        }
    }

    fn finish_node(
        &self,
        local_predicates: Vec<Expr>,
        mut builder: LogicalPlanBuilder,
        bubble_up: Vec<Expr>,
    ) -> Result<(LogicalPlan, Vec<Expr>)> {
        if local_predicates.len() > 0 {
            for expr in local_predicates {
                builder = builder.filter(expr);
            }
            Ok((builder.build(), bubble_up))
        } else {
            Ok((builder.build(), bubble_up))
        }
    }

    // acc predicates maps the root column names to predicates
    fn push_downup(
        &self,
        logical_plan: LogicalPlan,
        mut acc_predicates: FnvHashMap<Arc<String>, Expr>,
        // Used to determine if we should bubble up predicates
        join_encountered: bool, // the result is the rewritten logical plan and the predicates that should bubble up to the join
    ) -> Result<(LogicalPlan, Vec<Expr>)> {
        use LogicalPlan::*;

        match logical_plan {
            Selection { predicate, input } => {
                // bubble up/ push up
                if join_encountered && count_downtree_joins(&*input, 0) > 0 {
                    let (lp, mut bubble_up) =
                        self.push_downup(*input, acc_predicates, join_encountered)?;
                    bubble_up.push(predicate);
                    Ok((lp, bubble_up))
                // push down
                } else {
                    match expr_to_root_column(&predicate) {
                        Ok(name) => {
                            acc_predicates.insert(name, predicate);
                        }
                        Err(_) => panic!("implement logic for binary expr with 2 root columns"),
                    }
                    self.push_downup(*input, acc_predicates, join_encountered)
                }
            }
            Projection { expr, input, .. } => {
                // don't filter before the last projection that is more expensive as projections are free
                if count_downtree_projections(&input, 0) == 0 {
                    let (lp, bubble_up) =
                        self.push_downup(*input, init_hashmap(), join_encountered)?;
                    let builder = LogicalPlanBuilder::from(lp).project(expr);
                    // todo! write utility that takes hashmap values by value
                    self.finish_node(
                        acc_predicates.values().cloned().collect(),
                        builder,
                        bubble_up,
                    )
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
                                acc_predicates.insert(new_name, new_predicate);
                            }
                        }
                    }

                    let (lp, bubble_up) =
                        self.push_downup(*input, acc_predicates, join_encountered)?;
                    Ok((
                        LogicalPlanBuilder::from(lp).project(expr).build(),
                        bubble_up,
                    ))
                }
            }
            DataFrameScan { df, schema } => {
                let lp = DataFrameScan { df, schema };
                self.finish_at_leaf(lp, acc_predicates, init_bubble_up())
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
                self.finish_at_leaf(lp, acc_predicates, init_bubble_up())
            }
            Sort {
                input,
                column,
                reverse,
            } => {
                let (lp, bubble_up) = self.push_downup(*input, acc_predicates, join_encountered)?;
                Ok((
                    LogicalPlanBuilder::from(lp).sort(column, reverse).build(),
                    bubble_up,
                ))
            }
            Aggregate {
                input,
                keys,
                aggs,
                schema,
            } => {
                // dont push down predicates. An aggregation needs all rows
                let (lp, bubble_up) = self.push_downup(*input, init_hashmap(), false)?;

                // do filter before aggregations as this influence the result
                let lp = if bubble_up.len() > 0 {
                    let predicate = combine_predicates(bubble_up.iter());
                    LogicalPlanBuilder::from(lp).filter(predicate).build()
                } else {
                    lp
                };
                let lp = Aggregate {
                    input: Box::new(lp),
                    keys,
                    aggs,
                    schema,
                };
                self.finish_at_leaf(lp, acc_predicates, init_bubble_up())
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                how,
                ..
            } => {
                // todo! if we have csv scan we push predicate down, otherwise we combine with join operation
                let csv_scan = false;

                // for csv_scan
                if csv_scan {
                    self.pushdown_from_join_for_csv(
                        input_left,
                        input_right,
                        left_on,
                        right_on,
                        how,
                        acc_predicates,
                    )
                } else {
                    // start with an empty predicate accumulation
                    let schema_left = input_left.schema().clone();
                    let schema_right = input_right.schema().clone();

                    let (lp_left, bubble_up_left) =
                        self.push_downup(*input_left, init_hashmap(), true)?;
                    let (lp_right, bubble_up_right) =
                        self.push_downup(*input_right, init_hashmap(), true)?;

                    let mut predicates_left = bubble_up_left;
                    let mut predicates_right = bubble_up_right;
                    let mut local_predicates = Vec::with_capacity(acc_predicates.len());
                    predicates_left.reserve(acc_predicates.len());
                    predicates_right.reserve(acc_predicates.len());

                    // check for which table the predicates are
                    for predicate in acc_predicates.values() {
                        if check_down_node(&predicate, &schema_left) {
                            predicates_left.push(predicate.clone());
                        } else if check_down_node(&predicate, &schema_right) {
                            predicates_right.push(predicate.clone());
                        } else {
                            local_predicates.push(predicate.clone())
                        }
                    }

                    let mut builder = LogicalPlanBuilder::from(lp_left).join(
                        lp_right,
                        how,
                        left_on,
                        right_on,
                        Some(predicates_left),
                        Some(predicates_right),
                    );

                    if local_predicates.len() > 0 {
                        let predicate = combine_predicates(local_predicates.iter());
                        builder = builder.filter(predicate);
                    }
                    Ok((builder.build(), init_bubble_up()))
                }
            }
            HStack { input, exprs, .. } => {
                let (local, acc_predicates) =
                    self.split_pushdown_and_local(acc_predicates, input.schema());

                let (lp, bubble_up) = self.push_downup(*input, acc_predicates, join_encountered)?;
                let mut lp_builder = LogicalPlanBuilder::from(lp).with_columns(exprs);

                if local.len() > 0 {
                    let predicate = combine_predicates(local.iter());
                    lp_builder = lp_builder.filter(predicate);
                }
                Ok((lp_builder.build(), bubble_up))
            }
        }
    }

    fn pushdown_from_join_for_csv(
        &self,
        input_left: Box<LogicalPlan>,
        input_right: Box<LogicalPlan>,
        left_on: Expr,
        right_on: Expr,
        how: JoinType,
        acc_predicates: FnvHashMap<Arc<String>, Expr>,
    ) -> Result<(LogicalPlan, Vec<Expr>)> {
        let schema_left = input_left.schema();
        let schema_right = input_right.schema();
        let mut pushdown_left = init_hashmap();
        let mut pushdown_right = init_hashmap();
        let mut local_predicates = vec![];

        for predicate in acc_predicates.values() {
            if check_down_node(&predicate, schema_left) {
                let name = Arc::new(predicate.to_field(schema_left).unwrap().name().clone());
                pushdown_left.insert(name, predicate.clone());
            } else if check_down_node(&predicate, schema_right) {
                let name = Arc::new(predicate.to_field(schema_right).unwrap().name().clone());
                pushdown_right.insert(name, predicate.clone());
            } else {
                local_predicates.push(predicate.clone())
            }
        }

        let (lp_left, _bubble_up) = self.push_downup(*input_left, pushdown_left, false)?;
        let (lp_right, _bubble_up) = self.push_downup(*input_right, pushdown_right, false)?;
        let builder =
            LogicalPlanBuilder::from(lp_left).join(lp_right, how, left_on, right_on, None, None);
        self.finish_node(local_predicates, builder, init_bubble_up())
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
        let (lp, bubble_up) = self.push_downup(logical_plan, init_hashmap(), false)?;
        assert_eq!(bubble_up.len(), 0);
        Ok(lp)
    }
}
