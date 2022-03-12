mod utils;

use crate::logical_plan::{optimizer, Context};
use crate::prelude::*;
use crate::utils::{aexpr_to_root_names, aexprs_to_schema, check_input_node, has_aexpr};
use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;
use utils::*;

#[derive(Default)]
pub(crate) struct PredicatePushDown {}

impl PredicatePushDown {
    fn optional_apply_predicate(
        &self,
        lp: ALogicalPlan,
        local_predicates: Vec<Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> ALogicalPlan {
        if !local_predicates.is_empty() {
            let predicate = combine_predicates(local_predicates.into_iter(), expr_arena);
            let input = lp_arena.add(lp);

            ALogicalPlan::Selection { input, predicate }
        } else {
            lp
        }
    }

    fn pushdown_and_assign(
        &self,
        input: Node,
        acc_predicates: PlHashMap<Arc<str>, Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<()> {
        let alp = lp_arena.take(input);
        let lp = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;
        lp_arena.replace(input, lp);
        Ok(())
    }

    /// Filter will be pushed down.
    fn pushdown_and_continue(
        &self,
        lp: ALogicalPlan,
        mut acc_predicates: PlHashMap<Arc<str>, Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        has_projections: bool,
    ) -> Result<ALogicalPlan> {
        let inputs = lp.get_inputs();
        let exprs = lp.get_exprs();

        if has_projections {
            // we should not pass these projections
            if exprs
                .iter()
                .any(|e_n| other_column_is_pushdown_boundary(*e_n, expr_arena))
            {
                return self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena);
            }

            // projections should only have a single input.
            assert_eq!(inputs.len(), 1);
            let input = inputs[0];
            let (local_predicates, projections) =
                rewrite_projection_node(expr_arena, lp_arena, &mut acc_predicates, exprs, input);

            let alp = lp_arena.take(input);
            let alp = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;
            lp_arena.replace(input, alp);

            let lp = lp.with_exprs_and_input(projections, inputs);
            Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
        } else {
            let mut local_predicates = Vec::with_capacity(acc_predicates.len());

            // determine new inputs by pushing down predicates
            let new_inputs = inputs
                .iter()
                .map(|&node| {
                    // first we check if we are able to push down the predicate passed this node
                    // it could be that this node just added the column where we base the predicate on
                    let input_schema = lp_arena.get(node).schema(lp_arena);
                    let mut pushdown_predicates = optimizer::init_hashmap();
                    for (name, &predicate) in acc_predicates.iter() {
                        // we can pushdown the predicate
                        if check_input_node(predicate, input_schema, expr_arena) {
                            insert_and_combine_predicate(
                                &mut pushdown_predicates,
                                name.clone(),
                                predicate,
                                expr_arena,
                            )
                        }
                        // we cannot pushdown the predicate we do it here
                        else {
                            local_predicates.push(predicate);
                        }
                    }

                    let alp = lp_arena.take(node);
                    let alp = self.push_down(alp, pushdown_predicates, lp_arena, expr_arena)?;
                    lp_arena.replace(node, alp);
                    Ok(node)
                })
                .collect::<Result<Vec<_>>>()?;

            let lp = lp.with_exprs_and_input(exprs, new_inputs);
            Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
        }
    }

    /// Filter will be done at this node, but we continue optimization
    fn no_pushdown_restart_opt(
        &self,
        lp: ALogicalPlan,
        acc_predicates: PlHashMap<Arc<str>, Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        let inputs = lp.get_inputs();
        let exprs = lp.get_exprs();

        let new_inputs = inputs
            .iter()
            .map(|&node| {
                let alp = lp_arena.take(node);
                let alp = self.push_down(alp, init_hashmap(), lp_arena, expr_arena)?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<Result<Vec<_>>>()?;
        let lp = lp.with_exprs_and_input(exprs, new_inputs);

        // all predicates are done locally
        let local_predicates = acc_predicates.values().copied().collect::<Vec<_>>();
        Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
    }

    /// Predicate pushdown optimizer
    ///
    /// # Arguments
    ///
    /// * `AlogicalPlan` - Arena based logical plan tree representing the query.
    /// * `acc_predicates` - The predicates we accumulate during tree traversal.
    ///                      The hashmap maps from root-column name to predicates on that column.
    ///                      If the key is already taken we combine the predicate with a bitand operation.
    ///                      The `Node`s are indexes in the `expr_arena`
    /// * `lp_arena` - The local memory arena for the logical plan.
    /// * `expr_arena` - The local memory arena for the expressions.
    fn push_down(
        &self,
        lp: ALogicalPlan,
        mut acc_predicates: PlHashMap<Arc<str>, Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match lp {
            Selection { predicate, input } => {
                let name = roots_to_key(&aexpr_to_root_names(predicate, expr_arena));
                insert_and_combine_predicate(&mut acc_predicates, name, predicate, expr_arena);
                let alp = lp_arena.take(input);
                self.push_down(alp, acc_predicates, lp_arena, expr_arena)
            }
            DataFrameScan {
                df,
                schema,
                projection,
                selection,
            } => {
                let selection = predicate_at_scan(acc_predicates, selection, expr_arena);
                let lp = DataFrameScan {
                    df,
                    schema,
                    projection,
                    selection,
                };
                Ok(lp)
            }

            Melt {
                input,
                id_vars,
                value_vars,
                schema,
            } => {
                // predicates that will be done at this level
                let condition = |name: Arc<str>| {
                    let name = &*name;
                    name == "variable"
                        || name == "value"
                        || value_vars.iter().any(|s| s.as_str() == name)
                };
                let local_predicates =
                    transfer_to_local(expr_arena, &mut acc_predicates, condition);

                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;

                let lp = ALogicalPlan::Melt {
                    input,
                    id_vars,
                    value_vars,
                    schema,
                };
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            LocalProjection { expr, input, .. } => {
                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;

                let schema = lp_arena.get(input).schema(lp_arena);
                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let expr: Vec<_> = expr
                    .into_iter()
                    .filter(|e| check_input_node(*e, schema, expr_arena))
                    .collect();

                let schema = aexprs_to_schema(&expr, schema, Context::Default, expr_arena);
                Ok(ALogicalPlan::LocalProjection {
                    expr,
                    input,
                    schema: Arc::new(schema),
                })
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                output_schema,
                predicate,
                aggregate,
                options,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate, expr_arena);

                let lp = IpcScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    options,
                };
                Ok(lp)
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                output_schema,
                predicate,
                aggregate,
                options,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate, expr_arena);

                let lp = ParquetScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    options,
                };
                Ok(lp)
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
                output_schema,
                options,
                predicate,
                aggregate,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate, expr_arena);

                let lp = CsvScan {
                    path,
                    schema,
                    output_schema,
                    options,
                    predicate,
                    aggregate,
                };
                Ok(lp)
            }
            Explode { input, columns } => {
                let condition = |name: Arc<str>| columns.iter().any(|s| s.as_str() == &*name);
                let local_predicates =
                    transfer_to_local(expr_arena, &mut acc_predicates, condition);

                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;
                let lp = Explode { input, columns };
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            Distinct {
                input,
                options
            } => {
                // currently the distinct operation only keeps the first occurrences.
                // this may have influence on the pushed down predicates. If the pushed down predicates
                // contain a binary expression (thus depending on values in multiple columns)
                // the final result may differ if it is pushed down.

                let mut root_count = 0;

                // if this condition is called more than once, its a binary or ternary operation.
                let condition = |_| {
                    if root_count == 0 {
                        root_count += 1;
                        false
                    } else {
                        true
                    }
                };
                let local_predicates =
                    transfer_to_local(expr_arena, &mut acc_predicates, condition);

                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;
                let lp = Distinct {
                    input,
                    options
                };
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                schema,
                options,
            } => {
                let schema_left = lp_arena.get(input_left).schema(lp_arena);
                let schema_right = lp_arena.get(input_right).schema(lp_arena);

                let mut pushdown_left = optimizer::init_hashmap();
                let mut pushdown_right = optimizer::init_hashmap();
                let mut local_predicates = Vec::with_capacity(acc_predicates.len());

                for (_, predicate) in acc_predicates {
                    // unique and duplicated can be caused by joins
                    let matches =
                        |e: &AExpr| matches!(e, AExpr::IsUnique(_) | AExpr::Duplicated(_));

                    let checks_nulls =
                        |e: &AExpr| matches!(e, AExpr::IsNull(_) | AExpr::IsNotNull(_));
                    if has_aexpr(predicate, expr_arena, matches)
                        // join might create null values.
                        || has_aexpr(predicate, expr_arena, checks_nulls) && matches!(&options.how, JoinType::Left | JoinType::Outer | JoinType::Cross){
                        local_predicates.push(predicate);
                        continue;
                    }
                    // these indicate to which tables we are going to push down the predicate
                    let mut filter_left = false;
                    let mut filter_right = false;

                    // no else if. predicate can be in both tables.
                    if check_input_node(predicate, schema_left, expr_arena) {
                        let name = get_insertion_name(expr_arena, predicate, schema_left);
                        insert_and_combine_predicate(
                            &mut pushdown_left,
                            name,
                            predicate,
                            expr_arena,
                        );
                        filter_left = true;
                    }
                    if check_input_node(predicate, schema_right, expr_arena) {
                        let name = get_insertion_name(expr_arena, predicate, schema_right);
                        insert_and_combine_predicate(
                            &mut pushdown_right,
                            name,
                            predicate,
                            expr_arena,
                        );
                        filter_right = true;
                    }
                    match (filter_left, filter_right, &options.how) {
                        // if not pushed down on of the tables we have to do it locally.
                        (false, false, _) |
                        // if left join and predicate only available in right table,
                        // 'we should not filter right, because that would lead to
                        // invalid results.
                        // see: #2057
                        (false, true, JoinType::Left)
                        => {
                            local_predicates.push(predicate);
                            continue;
                        },
                        // business as usual
                        _ => {}
                    }
                    // An outer join or left join may create null values.
                    // we also do it local
                    let matches = |e: &AExpr| matches!(e, AExpr::IsNotNull(_) | AExpr::IsNull(_));
                    if (options.how == JoinType::Outer) | (options.how == JoinType::Left)
                        && has_aexpr(predicate, expr_arena, matches)
                    {
                        local_predicates.push(predicate);
                        continue;
                    }
                }

                self.pushdown_and_assign(input_left, pushdown_left, lp_arena, expr_arena)?;
                self.pushdown_and_assign(input_right, pushdown_right, lp_arena, expr_arena)?;

                let lp = Join {
                    input_left,
                    input_right,
                    left_on,
                    right_on,
                    schema,
                    options,
                };
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }

            lp @ Udf { .. } => {
                if let ALogicalPlan::Udf {
                    options: LogicalPlanUdfOptions {
                        predicate_pd: true, ..
                    }, ..
                } = lp
                {
                    self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
                } else {
                    self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
                }
            }
            // Pushed down passed these nodes
            lp @ Cache { .. } | lp @ Union { .. } | lp @ Sort { .. } => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
            }
            lp @ HStack {..} | lp @ Projection {..} => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, true)
            }
            // NOT Pushed down passed these nodes
            // predicates influence slice sizes
            lp @ Slice { .. }
            // dont push down predicates. An aggregation needs all rows
            | lp @ Aggregate {..} => {
                self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
            }
        }
    }

    pub fn optimize(
        &self,
        logical_plan: ALogicalPlan,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        let acc_predicates = PlHashMap::with_capacity(32);
        self.push_down(logical_plan, acc_predicates, lp_arena, expr_arena)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_insert_and_combine_predicate() {
        let mut acc_predicates = PlHashMap::with_capacity(32);
        let mut expr_arena = Arena::new();

        let predicate_expr = col("foo").gt(col("bar"));
        let predicate = to_aexpr(predicate_expr.clone(), &mut expr_arena);
        insert_and_combine_predicate(
            &mut acc_predicates,
            Arc::from("foo"),
            predicate,
            &mut expr_arena,
        );
        let root = *acc_predicates.get("foo").unwrap();
        let expr = node_to_expr(root, &expr_arena);
        assert_eq!(
            format!("{:?}", &expr),
            format!("{:?}", predicate_expr.and(lit(true)))
        );
    }
}
