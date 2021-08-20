use crate::logical_plan::optimizer::ALogicalPlanBuilder;
use crate::logical_plan::{optimizer, Context};
use crate::prelude::*;
use crate::utils::rename_aexpr_root_name;
use crate::utils::{
    aexpr_to_root_column_name, aexpr_to_root_names, aexprs_to_schema, check_down_node, has_aexpr,
};
use polars_core::datatypes::{PlHashMap, PlHashSet};
use polars_core::prelude::*;

trait Dsl {
    fn and(self, right: Node, arena: &mut Arena<AExpr>) -> Node;
}

impl Dsl for Node {
    fn and(self, right: Node, arena: &mut Arena<AExpr>) -> Node {
        arena.add(AExpr::BinaryExpr {
            left: self,
            op: Operator::And,
            right,
        })
    }
}

/// Don't overwrite predicates but combine them.
fn insert_and_combine_predicate(
    acc_predicates: &mut PlHashMap<Arc<String>, Node>,
    name: Arc<String>,
    predicate: Node,
    arena: &mut Arena<AExpr>,
) {
    let existing_predicate = acc_predicates
        .entry(name)
        .or_insert_with(|| arena.add(AExpr::Literal(LiteralValue::Boolean(true))));

    let node = arena.add(AExpr::BinaryExpr {
        left: *existing_predicate,
        op: Operator::And,
        right: predicate,
    });

    *existing_predicate = node;
}

pub fn combine_predicates<I>(iter: I, arena: &mut Arena<AExpr>) -> Node
where
    I: Iterator<Item = Node>,
{
    let mut single_pred = None;
    for node in iter {
        single_pred = match single_pred {
            None => Some(node),
            Some(left) => Some(arena.add(AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right: node,
            })),
        };
    }
    single_pred.expect("an empty iterator was passed")
}

fn predicate_at_scan(
    acc_predicates: PlHashMap<Arc<String>, Node>,
    predicate: Option<Node>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    if !acc_predicates.is_empty() {
        let mut new_predicate =
            combine_predicates(acc_predicates.into_iter().map(|t| t.1), expr_arena);
        if let Some(pred) = predicate {
            new_predicate = new_predicate.and(pred, expr_arena)
        }
        Some(new_predicate)
    } else {
        None
    }
}

/// Determine the hashmap key by combining all the root column names of a predicate
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

fn get_insertion_name(expr_arena: &Arena<AExpr>, predicate: Node, schema: &Schema) -> Arc<String> {
    Arc::new(
        expr_arena
            .get(predicate)
            .to_field(schema, Context::Default, expr_arena)
            .unwrap()
            .name()
            .clone(),
    )
}

pub(crate) struct PredicatePushDown {}

impl Default for PredicatePushDown {
    fn default() -> Self {
        Self {}
    }
}

fn no_pushdown_preds<F>(
    // node that is projected | hstacked
    node: Node,
    arena: &Arena<AExpr>,
    matches: F,
    // predicates that will be filtered at this node in the LP
    local_predicates: &mut Vec<Node>,
    acc_predicates: &mut PlHashMap<Arc<String>, Node>,
) where
    F: Fn(&AExpr) -> bool,
{
    // matching expr are typically explode, shift, etc. expressions that mess up predicates when pushed down
    if has_aexpr(node, arena, matches) {
        // columns that are projected. We check if we can push down the predicates past this projection
        let columns = aexpr_to_root_names(node, arena);

        let condition = |name: Arc<String>| columns.contains(&name);
        local_predicates.extend(transfer_to_local(arena, acc_predicates, condition));
    }
}

/// Transfer a predicate from `acc_predicates` that will be pushed down
/// to a local_predicates vec based on a condition.
fn transfer_to_local<F>(
    expr_arena: &Arena<AExpr>,
    acc_predicates: &mut PlHashMap<Arc<String>, Node>,
    mut condition: F,
) -> Vec<Node>
where
    F: FnMut(Arc<String>) -> bool,
{
    let mut remove_keys = Vec::with_capacity(acc_predicates.len());

    for (key, predicate) in &*acc_predicates {
        let root_names = aexpr_to_root_names(*predicate, expr_arena);
        for name in root_names {
            if condition(name) {
                remove_keys.push(key.clone());
                continue;
            }
        }
    }
    let mut local_predicates = Vec::with_capacity(remove_keys.len());
    for key in remove_keys {
        if let Some(pred) = acc_predicates.remove(&*key) {
            local_predicates.push(pred)
        }
    }
    local_predicates
}

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
        acc_predicates: PlHashMap<Arc<String>, Node>,
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
        acc_predicates: PlHashMap<Arc<String>, Node>,
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
        mut acc_predicates: PlHashMap<Arc<String>, Node>,
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
                let mut local_predicates = Vec::with_capacity(acc_predicates.len());

                // maybe update predicate name if a projection is an alias
                // aliases change the column names and because we push the predicates downwards
                // this may be problematic as the aliased column may not yet exist.
                for node in &expr {
                    let e = expr_arena.get(*node);

                    if let AExpr::Alias(e, name) = e {
                        // if this alias refers to one of the predicates in the upper nodes
                        // we rename the column of the predicate before we push it downwards.
                        if let Some(predicate) = acc_predicates.remove(&*name) {
                            match aexpr_to_root_column_name(*e, &*expr_arena) {
                                // we were able to rename the alias column with the root column name
                                // before pushing down the predicate
                                Ok(new_name) => {
                                    rename_aexpr_root_name(predicate, expr_arena, new_name.clone())
                                        .unwrap();

                                    insert_and_combine_predicate(
                                        &mut acc_predicates,
                                        new_name,
                                        predicate,
                                        expr_arena,
                                    );
                                }
                                // this may be a complex binary function. The predicate may only be valid
                                // on this projected column so we do filter locally.
                                Err(_) => local_predicates.push(predicate),
                            }
                        }
                    }

                    // remove predicates that are based on an exploded column
                    no_pushdown_preds(
                        *node,
                        expr_arena,
                        |e| {
                            matches!(e, AExpr::Explode(_))
                                || matches!(e, AExpr::Shift { .. })
                                || matches!(e, AExpr::Sort { .. })
                        },
                        &mut local_predicates,
                        &mut acc_predicates,
                    );
                }
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
                let condition = |name: Arc<String>| {
                    let name = &*name;
                    name == "variable" || name == "value" || value_vars.contains(name)
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
                    .filter(|e| check_down_node(*e, schema, expr_arena))
                    .collect();

                let schema = aexprs_to_schema(&expr, schema, Context::Default, expr_arena);
                Ok(ALogicalPlan::LocalProjection {
                    expr,
                    input,
                    schema: Arc::new(schema),
                })
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
                let predicate = predicate_at_scan(acc_predicates, predicate, expr_arena);

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
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
                options,
                predicate,
                aggregate,
            } => {
                let predicate = predicate_at_scan(acc_predicates, predicate, expr_arena);

                let lp = CsvScan {
                    path,
                    schema,
                    options,
                    predicate,
                    aggregate,
                };
                Ok(lp)
            }
            Explode { input, columns } => {
                let condition = |name: Arc<String>| columns.contains(&*name);
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
            } => {
                self.pushdown_and_assign(input, optimizer::init_hashmap(), lp_arena, expr_arena)?;

                // dont push down predicates. An aggregation needs all rows
                let lp = Aggregate {
                    input,
                    keys,
                    aggs,
                    schema,
                    apply,
                };
                Ok(self.finish_at_leaf(lp, acc_predicates, lp_arena, expr_arena))
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                how,
                allow_par,
                force_par,
                schema,
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
                    let mut filter_left = false;
                    let mut filter_right = false;

                    // no else if. predicate can be in both tables.
                    if check_down_node(predicate, schema_left, expr_arena) {
                        let name = get_insertion_name(expr_arena, predicate, schema_left);
                        insert_and_combine_predicate(
                            &mut pushdown_left,
                            name,
                            predicate,
                            expr_arena,
                        );
                        filter_left = true;
                    }
                    if check_down_node(predicate, schema_right, expr_arena) {
                        let name = get_insertion_name(expr_arena, predicate, schema_right);
                        insert_and_combine_predicate(
                            &mut pushdown_right,
                            name,
                            predicate,
                            expr_arena,
                        );
                        filter_right = true;
                    }
                    // if not pushed down on of the tables we have to do it locally.
                    if !(filter_left | filter_right) {
                        local_predicates.push(predicate);
                        continue;
                    }
                    // An outer join or left join may create null values.
                    // we also do it local
                    let matches = |e: &AExpr| matches!(e, AExpr::IsNotNull(_) | AExpr::IsNull(_));
                    if (how == JoinType::Outer) | (how == JoinType::Left)
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
                    how,
                    allow_par,
                    force_par,
                    schema,
                };
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }
            HStack { input, exprs, .. } => {
                // First we get all names of added columns in this HStack operation
                // and then we remove the predicates from the eligible container if they are
                // dependent on data we've added in this node.

                let mut added_cols = PlHashSet::with_capacity(exprs.len());
                for e in &exprs {
                    // shifts | sorts are influenced by a filter so we do all predicates before the shift | sort
                    let matches = |e: &AExpr| matches!(e, AExpr::Shift { .. } | AExpr::Sort { .. });
                    if has_aexpr(*e, expr_arena, matches) {
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

                    for name in aexpr_to_root_names(*e, expr_arena) {
                        added_cols.insert(name);
                    }
                }

                let condition = |name: Arc<String>| {
                    // remove predicates that are dependent on columns added in this HStack.
                    added_cols.contains(&name)
                        || lp_arena
                            .get(input)
                            .schema(lp_arena)
                            .field_with_name(&*name)
                            .is_err()
                };
                let local_predicates =
                    transfer_to_local(expr_arena, &mut acc_predicates, condition);

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
                        if check_down_node(predicate, input_schema, expr_arena) {
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
            lp => {
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
            Arc::new("foo".into()),
            predicate,
            &mut expr_arena,
        );
        let root = *acc_predicates.get(&String::from("foo")).unwrap();
        let expr = node_to_exp(root, &mut expr_arena);
        assert_eq!(
            format!("{:?}", &expr),
            format!("{:?}", &lit(true).and(predicate_expr))
        );
    }
}
