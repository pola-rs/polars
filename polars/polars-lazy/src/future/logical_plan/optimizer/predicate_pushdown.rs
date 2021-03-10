use crate::future::logical_plan::arena::{assign_alp, expr_arena_get, lp_arena_get, map_alp};
use crate::future::utils::{aexpr_to_root_column_name, aexpr_to_root_names};
use crate::logical_plan::optimizer::to_aexpr;
use crate::prelude::*;
use crate::utils::{
    expr_to_root_column_name, expr_to_root_column_names, has_aexpr, has_expr,
    rename_aexpr_root_name, rename_expr_root_name,
};
use ahash::RandomState;
use polars_core::prelude::*;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::rc::Rc;

trait DSL {
    fn and(self, right: Node, arena: &mut Arena<AExpr>) -> Node;
}

impl DSL for Node {
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
    predicates_map: &mut HashMap<Arc<String>, Node, RandomState>,
    name: Arc<String>,
    predicate: Node,
    arena: &mut Arena<AExpr>,
) {
    let existing_predicate = predicates_map
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
    acc_predicates: HashMap<Arc<String>, Node, RandomState>,
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

pub struct PredicatePushdown {}

impl Default for PredicatePushdown {
    fn default() -> Self {
        Self {}
    }
}

fn no_pushdown_preds(
    // node that is projected | hstacked
    node: Node,
    arena: &Arena<AExpr>,
    matches: &[AExpr],
    // predicates that will be filtered at this node in the LP
    local_predicates: &mut Vec<Node>,
    acc_predicates: &mut HashMap<Arc<String>, Node, RandomState>,
) {
    // matching expr are typically explode, shift, etc. expressions that mess up predicates when pushed down
    for matching_expr in matches {
        if has_aexpr(node, &arena, matching_expr, true) {
            // columns that are projected. We check if we can push down the predicates past this projection
            let columns = aexpr_to_root_names(node, &arena);
            debug_assert_eq!(columns.len(), 1);

            // keep track of the predicates that should be removed from pushed down predicates
            // these predicates will be added to local predicates
            let mut remove_keys = Vec::with_capacity(acc_predicates.len());

            for (key, predicate) in &*acc_predicates {
                let root_names = aexpr_to_root_names(*predicate, arena);

                for name in root_names {
                    if columns.contains(&name) {
                        remove_keys.push(key.clone());
                        continue;
                    }
                }
            }
            for key in remove_keys {
                let pred = acc_predicates.remove(&*key).expect("we know it exists");
                local_predicates.push(pred);
            }
        }
    }
}

impl PredicatePushdown {
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

    fn pushdown_and_assign(
        &self,
        input: Node,
        acc_predicates: HashMap<Arc<String>, Node, RandomState>,
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
    ///
    /// The returned `Node` is an expression in the Tree.
    fn push_down(
        &self,
        logical_plan: ALogicalPlan,
        mut acc_predicates: HashMap<Arc<String>, Node, RandomState>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Slice { input, offset, len } => {
                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;
                Ok(Slice { input, offset, len })
            }
            Selection { predicate, input } => {
                let name = roots_to_key(&aexpr_to_root_names(predicate, expr_arena));
                insert_and_combine_predicate(&mut acc_predicates, name, predicate, expr_arena);
                let alp = lp_arena.take(input);
                self.push_down(alp, acc_predicates, lp_arena, expr_arena)
            }

            Projection {
                expr,
                mut input,
                schema,
            } => {
                let mut local_predicates = Vec::with_capacity(acc_predicates.len());

                // maybe update predicate name if a projection is an alias
                for node in &expr {
                    let e = expr_arena.get(*node);

                    if let AExpr::Alias(e, name) = e {
                        // if this alias refers to one of the predicates in the upper nodes
                        // we rename the column of the predicate before we push it downwards.
                        if let Some(predicate) = acc_predicates.remove(&*name) {
                            let new_name = aexpr_to_root_column_name(predicate, &*expr_arena)
                                .expect("more than one root");
                            rename_aexpr_root_name(*node, expr_arena, new_name.clone()).unwrap();
                            insert_and_combine_predicate(
                                &mut acc_predicates,
                                new_name,
                                *node,
                                expr_arena,
                            );
                        }
                    }

                    // remove predicates that are based on an exploded column
                    // todo! add shift
                    no_pushdown_preds(
                        *node,
                        &expr_arena,
                        &[AExpr::Explode(Default::default())],
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
                let mut remove_keys = Vec::with_capacity(acc_predicates.len());

                for (key, predicate) in &acc_predicates {
                    let root_names = aexpr_to_root_names(*predicate, expr_arena);
                    for name in root_names {
                        if (&*name == "variable")
                            || (&*name == "value")
                            || value_vars.contains(&*name)
                        {
                            remove_keys.push(key.clone());
                        }
                    }
                }
                let mut local_predicates = Vec::with_capacity(remove_keys.len());
                for key in remove_keys {
                    let pred = acc_predicates.remove(&*key).unwrap();
                    local_predicates.push(pred)
                }

                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;

                let lp = ALogicalPlan::Melt {
                    input,
                    id_vars,
                    value_vars,
                    schema,
                };
                Ok(self.apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            }

            lp => Ok(lp),
        }
    }

    pub fn optimize(
        &self,
        logical_plan: ALogicalPlan,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        let mut acc_predicates = HashMap::with_capacity_and_hasher(100, RandomState::new());
        self.push_down(logical_plan, acc_predicates, lp_arena, expr_arena)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use polars_core::df;

    #[test]
    fn test_arena_pushdown() {
        let df = df! {
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        }
        .unwrap();

        let mut lps = Vec::with_capacity(10);

        let lp = LogicalPlanBuilder::from_existing_df(df.clone())
            .slice(0, 0)
            .project(vec![col("a"), col("b")])
            .build();
        lps.push(lp);

        let lp = LogicalPlanBuilder::from_existing_df(df)
            .slice(0, 0)
            .project(vec![col("a"), col("b")])
            .filter(col("a").gt(col("b")))
            .build();
        lps.push(lp);

        for lp in lps {
            println!("\n\n");

            let original_lp = lp.clone();

            let mut expr_arena = Arena::with_capacity(10);
            let mut lp_arena = Arena::with_capacity(10);
            let root = to_alp(lp, &mut expr_arena, &mut lp_arena);
            let alp = lp_arena.take(root);

            let opt = PredicatePushdown {};
            let lp = opt.optimize(alp, &mut lp_arena, &mut expr_arena).unwrap();
            let root = lp_arena.add(lp);
            let lp = node_to_lp(root, &mut expr_arena, &mut lp_arena);

            let opt = crate::prelude::PredicatePushDown::default();
            let lp_expected = opt.optimize(original_lp).unwrap();

            println!("lp:\n{:?}\n", lp);
            println!("lp expected:\n{:?}\n", lp_expected);
            assert_eq!(format!("{:?}", &lp), format!("{:?}", &lp_expected));
        }
    }

    #[test]
    fn test_insert_and_combine_predicate() {
        let mut acc_predicates = HashMap::with_capacity_and_hasher(10, RandomState::new());
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
