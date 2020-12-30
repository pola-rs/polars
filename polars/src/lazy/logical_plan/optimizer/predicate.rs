use crate::lazy::logical_plan::optimizer::{check_down_node, HASHMAP_SIZE};
use crate::lazy::logical_plan::Context;
use crate::lazy::prelude::*;
use crate::lazy::utils::{
    expr_to_root_column_name, expr_to_root_column_names, has_expr, rename_expr_root_name,
};
use crate::prelude::*;
use ahash::RandomState;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Don't overwrite predicates but combine them.
fn insert_and_combine_predicate(
    predicates_map: &mut HashMap<Arc<String>, Expr, RandomState>,
    name: Arc<String>,
    predicate: Expr,
) {
    let existing_predicate = predicates_map.entry(name).or_insert_with(|| lit(true));
    *existing_predicate = existing_predicate.clone().and(predicate)
}

fn predicate_at_scan(
    acc_predicates: HashMap<Arc<String>, Expr, RandomState>,
    predicate: Option<Expr>,
) -> Option<Expr> {
    if !acc_predicates.is_empty() {
        let mut new_predicate = combine_predicates(acc_predicates.into_iter().map(|t| t.1));
        if let Some(pred) = predicate {
            new_predicate = new_predicate.and(pred)
        }
        Some(new_predicate)
    } else {
        None
    }
}

pub struct PredicatePushDown {
    // used in has_expr check. This reduces box allocations
    unique_dummy: Expr,
    duplicated_dummy: Expr,
    binary_dummy: Expr,
    is_null_dummy: Expr,
    is_not_null_dummy: Expr,
    explode_dummy: Expr,
}

impl Default for PredicatePushDown {
    fn default() -> Self {
        PredicatePushDown {
            unique_dummy: lit("_").is_unique(),
            duplicated_dummy: lit("_").is_duplicated(),
            binary_dummy: lit("_").eq(lit("_")),
            is_null_dummy: lit("_").is_null(),
            is_not_null_dummy: lit("_").is_null(),
            explode_dummy: Expr::Explode(Box::new(Expr::Wildcard)),
        }
    }
}

fn roots_to_key(roots: &[Arc<String>]) -> Arc<String> {
    if roots.len() == 1 {
        roots[0].clone()
    } else {
        let mut new = String::with_capacity(32 * roots.len());
        for name in roots {
            new.push_str(name);
        }
        Arc::new(new)
    }
}

pub(crate) fn combine_predicates<I>(iter: I) -> Expr
where
    I: Iterator<Item = Expr>,
{
    let mut single_pred = None;
    for expr in iter {
        single_pred = match single_pred {
            None => Some(expr),
            Some(e) => Some(e.and(expr)),
        };
    }
    single_pred.expect("an empty iterator was passed")
}

impl PredicatePushDown {
    fn finish_at_leaf(
        &self,
        lp: LogicalPlan,
        acc_predicates: HashMap<Arc<String>, Expr, RandomState>,
    ) -> LogicalPlan {
        match acc_predicates.len() {
            // No filter in the logical plan
            0 => lp,
            _ => {
                let mut builder = LogicalPlanBuilder::from(lp);

                let predicate = combine_predicates(acc_predicates.into_iter().map(|t| t.1));
                builder = builder.filter(predicate);
                builder.build()
            }
        }
    }
    fn finish_node(
        &self,
        local_predicates: Vec<Expr>,
        mut builder: LogicalPlanBuilder,
    ) -> LogicalPlan {
        if !local_predicates.is_empty() {
            let predicate = combine_predicates(local_predicates.into_iter());
            builder = builder.filter(predicate);
            builder.build()
        } else {
            builder.build()
        }
    }

    // acc predicates maps the root column names to predicates
    fn push_down(
        &self,
        logical_plan: LogicalPlan,
        mut acc_predicates: HashMap<Arc<String>, Expr, RandomState>,
    ) -> Result<LogicalPlan> {
        use crate::lazy::logical_plan::optimizer;
        use LogicalPlan::*;

        match logical_plan {
            Slice { input, offset, len } => {
                let input = Box::new(self.push_down(*input, acc_predicates)?);
                Ok(Slice { input, offset, len })
            }
            Selection { predicate, input } => {
                let name = roots_to_key(&expr_to_root_column_names(&predicate));
                insert_and_combine_predicate(&mut acc_predicates, name, predicate);
                self.push_down(*input, acc_predicates)
            }
            Projection { expr, input, .. } => {
                let mut local_predicates = Vec::with_capacity(acc_predicates.len());
                // maybe update predicate name if a projection is an alias
                for e in &expr {
                    // check if there is an alias
                    if let Expr::Alias(e, name) = e {
                        // if this alias refers to one of the predicates in the upper nodes
                        // we rename the column of the predicate before we push it downwards.
                        if let Some(predicate) = acc_predicates.remove(name) {
                            let new_name = expr_to_root_column_name(e).unwrap();
                            let new_predicate =
                                rename_expr_root_name(&predicate, new_name.clone()).unwrap();
                            insert_and_combine_predicate(
                                &mut acc_predicates,
                                new_name,
                                new_predicate,
                            );
                        }
                    }

                    // remove filters based on the exploded column. Aliased predicates
                    if has_expr(e, &self.explode_dummy) {
                        let columns = expr_to_root_column_names(e);
                        assert_eq!(columns.len(), 1);

                        let mut remove_keys = Vec::with_capacity(acc_predicates.len());

                        for (key, predicate) in &acc_predicates {
                            let root_names = expr_to_root_column_names(predicate);
                            for name in root_names {
                                if columns.contains(&name) {
                                    remove_keys.push(key.clone());
                                    continue;
                                }
                            }
                        }
                        for key in remove_keys {
                            let pred = acc_predicates.remove(&*key).unwrap();
                            local_predicates.push(pred)
                        }
                    }
                }

                let mut builder =
                    LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?).project(expr);
                if !local_predicates.is_empty() {
                    let predicate = combine_predicates(local_predicates.into_iter());
                    builder = builder.filter(predicate)
                }
                Ok(builder.build())
            }
            LocalProjection { expr, input, .. } => {
                let input = self.push_down(*input, acc_predicates)?;
                let schema = input.schema();
                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let proj = expr
                    .into_iter()
                    .filter(|e| check_down_node(e, schema))
                    .collect();
                Ok(LogicalPlanBuilder::from(input).project_local(proj).build())
            }
            DataFrameScan {
                df,
                schema,
                projection,
                selection,
            } => {
                let selection = predicate_at_scan(acc_predicates, selection);
                let lp = DataFrameScan {
                    df,
                    schema,
                    projection,
                    selection,
                };
                Ok(lp)
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                with_columns,
                predicate,
                aggregate,
                stop_after_n_rows,
                cache,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate);

                let lp = ParquetScan {
                    path,
                    schema,
                    with_columns,
                    predicate,
                    aggregate,
                    stop_after_n_rows,
                    cache,
                };
                Ok(lp)
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                stop_after_n_rows,
                with_columns,
                predicate,
                aggregate,
                cache,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate);

                let lp = CsvScan {
                    path,
                    schema,
                    has_header,
                    delimiter,
                    ignore_errors,
                    skip_rows,
                    stop_after_n_rows,
                    with_columns,
                    predicate,
                    aggregate,
                    cache,
                };
                Ok(lp)
            }
            Sort {
                input,
                by_column,
                reverse,
            } => {
                let input = Box::new(self.push_down(*input, acc_predicates)?);
                Ok(Sort {
                    input,
                    by_column,
                    reverse,
                })
            }
            Explode { input, columns } => {
                // we remove predicates that are done in one of the exploded columns.

                let columns_set: HashSet<_, RandomState> = columns.iter().collect();
                let mut remove_keys = Vec::with_capacity(acc_predicates.len());

                for (key, predicate) in &acc_predicates {
                    let root_names = expr_to_root_column_names(predicate);
                    for name in root_names {
                        if columns_set.contains(&*name) {
                            remove_keys.push(key.clone());
                            continue;
                        }
                    }
                }
                let mut local_predicates = Vec::with_capacity(remove_keys.len());
                for key in remove_keys {
                    let pred = acc_predicates.remove(&*key).unwrap();
                    local_predicates.push(pred)
                }

                let input = Box::new(self.push_down(*input, acc_predicates)?);
                let lp = Explode { input, columns };
                let mut builder = LogicalPlanBuilder::from(lp);
                if !local_predicates.is_empty() {
                    let predicate = combine_predicates(local_predicates.into_iter());
                    builder = builder.filter(predicate)
                }
                Ok(builder.build())
            }
            Cache { input } => {
                let input = Box::new(self.push_down(*input, acc_predicates)?);
                Ok(Cache { input })
            }
            Distinct {
                input,
                subset,
                maintain_order,
            } => {
                // currently the distinct operation only keeps the first occurrences.
                // this may have influence on the pushed down predicates. If the pushed down predicates
                // contain a binary expression (thus depending on values in multiple columns) the final result may differ if it is pushed down.
                let mut local_pred = Vec::with_capacity(acc_predicates.len());

                let mut new_acc_predicates = optimizer::init_hashmap();
                for (name, predicate) in acc_predicates {
                    if has_expr(&predicate, &self.binary_dummy) {
                        local_pred.push(predicate)
                    } else {
                        new_acc_predicates.insert(name, predicate);
                    }
                }

                let input = self.push_down(*input, new_acc_predicates)?;
                let lp = Distinct {
                    input: Box::new(input),
                    maintain_order,
                    subset,
                };
                let mut builder = LogicalPlanBuilder::from(lp);
                if !local_pred.is_empty() {
                    let predicate = combine_predicates(local_pred.into_iter());
                    builder = builder.filter(predicate)
                }
                Ok(builder.build())
            }
            Aggregate {
                input,
                keys,
                aggs,
                schema,
            } => {
                // dont push down predicates. An aggregation needs all rows
                let lp = Aggregate {
                    input: Box::new(self.push_down(*input, optimizer::init_hashmap())?),
                    keys,
                    aggs,
                    schema,
                };
                Ok(self.finish_at_leaf(lp, acc_predicates))
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                how,
                allow_par,
                force_par,
                ..
            } => {
                let schema_left = input_left.schema();
                let schema_right = input_right.schema();

                let mut pushdown_left = optimizer::init_hashmap();
                let mut pushdown_right = optimizer::init_hashmap();
                let mut local_predicates = Vec::with_capacity(acc_predicates.len());

                for (_, predicate) in acc_predicates {
                    // unique and duplicated can be caused by joins
                    if has_expr(&predicate, &self.unique_dummy) {
                        local_predicates.push(predicate.clone());
                        continue;
                    }
                    if has_expr(&predicate, &self.duplicated_dummy) {
                        local_predicates.push(predicate.clone());
                        continue;
                    }
                    let mut filter_left = false;
                    let mut filter_right = false;

                    // no else if. predicate can be in both tables.
                    if check_down_node(&predicate, schema_left) {
                        let name = Arc::new(
                            predicate
                                .to_field(schema_left, Context::Other)
                                .unwrap()
                                .name()
                                .clone(),
                        );
                        insert_and_combine_predicate(&mut pushdown_left, name, predicate.clone());
                        filter_left = true;
                    }
                    if check_down_node(&predicate, schema_right) {
                        let name = Arc::new(
                            predicate
                                .to_field(schema_right, Context::Other)
                                .unwrap()
                                .name()
                                .clone(),
                        );
                        insert_and_combine_predicate(&mut pushdown_right, name, predicate.clone());
                        filter_right = true;
                    }
                    if !(filter_left & filter_right) {
                        local_predicates.push(predicate.clone());
                        continue;
                    }
                    // An outer join or left join may create null values.
                    // we also do it local
                    if (how == JoinType::Outer) | (how == JoinType::Left) {
                        if has_expr(&predicate, &self.is_not_null_dummy) {
                            local_predicates.push(predicate.clone());
                            continue;
                        }
                        if has_expr(&predicate, &self.is_null_dummy) {
                            local_predicates.push(predicate);
                            continue;
                        }
                    }
                }

                let lp_left = self.push_down(*input_left, pushdown_left)?;
                let lp_right = self.push_down(*input_right, pushdown_right)?;

                let builder = LogicalPlanBuilder::from(lp_left)
                    .join(lp_right, how, left_on, right_on, allow_par, force_par);
                Ok(self.finish_node(local_predicates, builder))
            }
            HStack { input, exprs, .. } => {
                let mut local = Vec::with_capacity(acc_predicates.len());
                let mut local_keys = Vec::with_capacity(acc_predicates.len());
                for (key, predicate) in &acc_predicates {
                    if !check_down_node(predicate, input.schema()) {
                        local_keys.push(key.clone());
                    }
                }

                // get all names of added columns in this HStack
                let mut added_cols =
                    HashSet::with_capacity_and_hasher(exprs.len(), RandomState::default());
                for e in &exprs {
                    for name in expr_to_root_column_names(e) {
                        added_cols.insert(name);
                    }
                }
                // remove predicates that are dependent on columns added in this HStack.
                for key in acc_predicates.keys() {
                    if added_cols.contains(key) {
                        local_keys.push(key.clone())
                    }
                }

                for key in local_keys {
                    if let Some(val) = acc_predicates.remove(&key) {
                        local.push(val);
                    }
                }

                let mut lp_builder =
                    LogicalPlanBuilder::from(self.push_down(*input, acc_predicates)?)
                        .with_columns(exprs);

                if !local.is_empty() {
                    let predicate = combine_predicates(local.into_iter());
                    lp_builder = lp_builder.filter(predicate);
                }
                Ok(lp_builder.build())
            }
        }
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
