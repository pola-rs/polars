use crate::prelude::*;
use crate::utils::{expr_to_root_column_name, expr_to_root_column_names, has_expr, rename_expr_root_name, aexpr_to_root_column_name, rename_aexpr_root_name};
use ahash::RandomState;
use polars_core::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;
use crate::logical_plan::{
    optimizer::to_aexpr,
    EXPR_ARENA, map_aexpr, map_alp, expr_arena_get};
use std::borrow::BorrowMut;

trait DSL {
    fn and(self, right: Node) -> Node;
}

impl DSL for Node {
    fn and(self, right: Node) -> Node {
        let arena_ref = expr_arena_get();
        let mut arena = (*arena_ref).borrow_mut();
        arena.add(AExpr::BinaryExpr {
            left: self,
            op: Operator::And,
            right
        })
    }
}

/// Don't overwrite predicates but combine them.
fn insert_and_combine_predicate(
    predicates_map: &mut HashMap<Arc<String>, Node, RandomState>,
    name: Arc<String>,
    predicate: Node,

) {
    let existing_predicate = predicates_map.entry(name).or_insert_with(|| {
        let arena_ref = expr_arena_get();
        let mut arena = (*arena_ref).borrow_mut();
        to_aexpr(lit(true), &mut *arena)
    });

    *existing_predicate = existing_predicate.and(predicate);
}

pub struct PredicatePushdown2 {}

impl Default for PredicatePushdown2 {
    fn default() -> Self {
        Self {}
    }
}

impl PredicatePushdown2 {
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
        acc_predicates: &mut HashMap<Rc<String>, Node, RandomState>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Slice { input, offset, len } => {
                map_alp(input, |alp| self.push_down(alp, acc_predicates))?;
                Ok(Slice { input, offset, len })
            }
            Projection { expr, input, .. } => {
                // lp_arena.replace_with(input, |lp| {
                //     let mut local_predicates = Vec::with_capacity(acc_predicates.len());
                //     // maybe update predicate name if a projection is an alias
                //     for node in expr {
                //         let e = expr_arena.get(node);
                //
                //         if let AExpr::Alias(e, name) = e {
                //             // if this alias refers to one of the predicates in the upper nodes
                //             // we rename the column of the predicate before we push it downwards.
                //             if let Some(predicate) = acc_predicates.remove(name) {
                //                 let expr = expr_arena.get(predicate)
                //                 let new_name = aexpr_to_root_column_name(predicate, arena).expect("more than one root");
                //                 rename_aexpr_root_name(node, expr_arena, new_name).unwrap();
                //                 insert_and_combine_predicate(
                //                     &mut acc_predicates,
                //                     new_name,
                //                     node
                //                 );
                //             }
                //         }
                //     }
                //
                // });
                // Ok(input)
                todo!()
            }
            _ => {
                todo!()
            }
        }
    }

    pub fn optimize(
        &self,
        logical_plan: ALogicalPlan,
    ) -> Result<ALogicalPlan> {
        let mut acc_predicates = HashMap::with_capacity_and_hasher(100, RandomState::new());
        self.push_down(
            logical_plan,
            &mut acc_predicates
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use polars_core::df;

    #[test]
    fn test_arena_pushdown() {
        let mut arena = Arena::new();

        let df = df! {
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        }
            .unwrap();
        let schema = df.schema();
        let lp = ALogicalPlan::DataFrameScan {
            df: Arc::new(df),
            schema: Arc::new(schema),
            projection: None,
            selection: None,
        };
        let node = arena.add(lp);
        let lp = ALogicalPlan::Slice {
            input: node,
            offset: 0,
            len: 1,
        };

        let opt = PredicatePushdown2 {};
        let lp = opt.optimize(lp).unwrap();
    }
}
