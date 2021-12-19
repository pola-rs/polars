mod utils;

use crate::logical_plan::optimizer::ALogicalPlanBuilder;
use crate::logical_plan::{optimizer, Context};
use crate::prelude::*;
use crate::utils::{aexpr_to_root_names, aexprs_to_schema, check_input_node, has_aexpr};
use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;
use utils::*;

#[derive(Default)]
pub(crate) struct PredicatePushDown {}

impl PredicatePushDown {
    fn apply_predicate(
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

    fn finish_at_leaf(
        &self,
        lp: ALogicalPlan,
        acc_predicates: PlHashMap<Arc<str>, Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> ALogicalPlan {
        match acc_predicates.len() {
            // No filter in the logical plan
            0 => lp,
            _ => {
                let local_predicates = acc_predicates.into_iter().map(|t| t.1).collect();
                self.apply_predicate(lp, local_predicates, lp_arena, expr_arena)
            }
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
        logical_plan: ALogicalPlan,
        mut acc_predicates: PlHashMap<Arc<str>, Node>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Selection { predicate, input } => {
                let name = roots_to_key(&aexpr_to_root_names(predicate, expr_arena));
                insert_and_combine_predicate(&mut acc_predicates, name, predicate, expr_arena);
                let alp = lp_arena.take(input);
                self.push_down(alp, acc_predicates, lp_arena, expr_arena)
            }

            Projection {
                expr,
                input,
                schema,
            } => {
                for node in &expr {
                    if is_pushdown_boundary(*node, expr_arena) {
                        let lp = ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                            .project(expr)
                            .build();
                        // do all predicates here
                        let local_predicates = acc_predicates.into_iter().map(|(_, v)| v).collect();
                        return Ok(self.apply_predicate(
                            lp,
                            local_predicates,
                            lp_arena,
                            expr_arena,
                        ));
                    }
                }

                let (local_predicates, expr) =
                    rewrite_projection_node(expr_arena, lp_arena, &mut acc_predicates, expr, input);
                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;

                let lp = ALogicalPlan::Projection {
                    expr,
                    input,
                    schema,
                };
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
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
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
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
                with_columns,
                predicate,
                aggregate,
                n_rows,
                cache,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate, expr_arena);

                let lp = ParquetScan {
                    path,
                    schema,
                    output_schema,
                    with_columns,
                    predicate,
                    aggregate,
                    n_rows,
                    cache,
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
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            Distinct {
                input,
                subset,
                maintain_order,
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
                    maintain_order,
                    subset,
                };
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            Aggregate {
                input,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                dynamic_options,
            } => {
                self.pushdown_and_assign(input, optimizer::init_hashmap(), lp_arena, expr_arena)?;

                // dont push down predicates. An aggregation needs all rows
                let lp = Aggregate {
                    input,
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    dynamic_options,
                };
                Ok(self.finish_at_leaf(lp, acc_predicates, lp_arena, expr_arena))
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
                    if has_aexpr(predicate, expr_arena, matches) {
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
                    match (filter_left, filter_right, options.how) {
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
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            HStack { input, exprs, .. } => {
                // First we get all names of added columns in this HStack operation
                // and then we remove the predicates from the eligible container if they are
                // dependent on data we've added in this node.

                for node in &exprs {
                    if is_pushdown_boundary(*node, expr_arena) {
                        let lp = ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                            .with_columns(exprs)
                            .build();
                        // do all predicates here
                        let local_predicates = acc_predicates.into_iter().map(|(_, v)| v).collect();
                        return Ok(self.apply_predicate(
                            lp,
                            local_predicates,
                            lp_arena,
                            expr_arena,
                        ));
                    }
                }

                let (local_predicates, exprs) = rewrite_projection_node(
                    expr_arena,
                    lp_arena,
                    &mut acc_predicates,
                    exprs,
                    input,
                );

                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;
                let lp = ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                    .with_columns(exprs)
                    .build();

                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }

            Udf {
                input,
                function,
                predicate_pd,
                projection_pd,
                schema,
            } => {
                if predicate_pd {
                    let input_schema = lp_arena.get(input).schema(lp_arena);
                    let mut pushdown_predicates = optimizer::init_hashmap();
                    let mut local_predicates = Vec::with_capacity(acc_predicates.len());
                    for (_, predicate) in acc_predicates {
                        if check_input_node(predicate, input_schema, expr_arena) {
                            let name = get_insertion_name(expr_arena, predicate, input_schema);
                            insert_and_combine_predicate(
                                &mut pushdown_predicates,
                                name,
                                predicate,
                                expr_arena,
                            )
                        } else {
                            local_predicates.push(predicate);
                        }
                    }
                    self.pushdown_and_assign(input, pushdown_predicates, lp_arena, expr_arena)?;
                    let lp = Udf {
                        input,
                        function,
                        predicate_pd,
                        projection_pd,
                        schema,
                    };

                    return Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena));
                }
                Ok(Udf {
                    input,
                    function,
                    predicate_pd,
                    projection_pd,
                    schema,
                })
            }
            lp @ Slice { .. } | lp @ Cache { .. } | lp @ Union { .. } | lp @ Sort { .. } => {
                let inputs = lp.get_inputs();
                let exprs = lp.get_exprs();

                let new_inputs = if inputs.len() == 1 {
                    let node = inputs[0];
                    let alp = lp_arena.take(node);
                    let alp = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;
                    lp_arena.replace(node, alp);
                    vec![node]
                } else {
                    inputs
                        .iter()
                        .map(|&node| {
                            let alp = lp_arena.take(node);
                            let alp =
                                self.push_down(alp, acc_predicates.clone(), lp_arena, expr_arena)?;
                            lp_arena.replace(node, alp);
                            Ok(node)
                        })
                        .collect::<Result<Vec<_>>>()?
                };

                Ok(lp.from_exprs_and_input(exprs, new_inputs))
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
        let expr = node_to_exp(root, &expr_arena);
        assert_eq!(
            format!("{:?}", &expr),
            format!("{:?}", predicate_expr.and(lit(true)))
        );
    }
}
