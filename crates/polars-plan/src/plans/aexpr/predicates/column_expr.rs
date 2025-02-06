//! This module creates predicates splits predicates into partial per-column predicates.

use polars_core::schema::Schema;
use polars_io::predicates::SpecializedColumnPredicateExpr;
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use crate::dsl::Operator;
use crate::plans::{aexpr_to_leaf_names_iter, AExpr, MintermIter};

pub struct ColumnPredicates {
    pub predicates: PlHashMap<PlSmallStr, (Node, Option<SpecializedColumnPredicateExpr>)>,

    /// Are all column predicates AND-ed together the original predicate.
    pub is_sumwise_complete: bool,
}

pub fn aexpr_to_column_predicates(
    root: Node,
    expr_arena: &mut Arena<AExpr>,
    _schema: &Schema,
) -> ColumnPredicates {
    let mut predicates =
        PlHashMap::<PlSmallStr, (Node, Option<SpecializedColumnPredicateExpr>)>::default();
    let mut is_sumwise_complete = true;

    let minterms = MintermIter::new(root, expr_arena).collect::<Vec<_>>();

    let mut leaf_names = Vec::with_capacity(2);
    for minterm in minterms {
        leaf_names.clear();
        leaf_names.extend(aexpr_to_leaf_names_iter(minterm, expr_arena));

        if leaf_names.len() != 1 {
            is_sumwise_complete = false;
            continue;
        }

        let column = leaf_names.pop().unwrap();
        let entry = predicates.entry(column);

        entry
            .and_modify(|n| {
                let left = n.0;
                n.0 = expr_arena.add(AExpr::BinaryExpr {
                    left,
                    op: Operator::LogicalAnd,
                    right: minterm,
                });
                n.1 = None;
            })
            .or_insert((minterm, None));
    }

    ColumnPredicates {
        predicates,
        is_sumwise_complete,
    }
}
