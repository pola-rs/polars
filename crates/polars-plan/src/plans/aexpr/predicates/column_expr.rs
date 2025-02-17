//! This module creates predicates splits predicates into partial per-column predicates.

use polars_core::datatypes::DataType;
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_io::predicates::SpecializedColumnPredicateExpr;
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::get_binary_expr_col_and_lv;
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
    schema: &Schema,
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
        let Some(dtype) = schema.get(&column) else {
            is_sumwise_complete = false;
            continue;
        };

        // We really don't want to deal with these types.
        use DataType as D;
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            D::Enum(_, _) | D::Categorical(_, _) => {
                is_sumwise_complete = false;
                continue;
            },
            #[cfg(feature = "dtype-decimal")]
            D::Decimal(_, _) => {
                is_sumwise_complete = false;
                continue;
            },
            _ if dtype.is_nested() => {
                is_sumwise_complete = false;
                continue;
            },
            _ => {},
        }

        let dtype = dtype.clone();
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
            .or_insert_with(|| {
                (
                    minterm,
                    Some(()).and_then(|_| {
                        if std::env::var("POLARS_SPECIALIZED_COLUMN_PRED").as_deref() != Ok("1") {
                            return None;
                        }

                        let aexpr = expr_arena.get(minterm);

                        let AExpr::BinaryExpr { left, op, right } = aexpr else {
                            return None;
                        };
                        let ((_, _), (lv, _)) =
                            get_binary_expr_col_and_lv(*left, *right, expr_arena, schema)?;
                        let lv = lv?;
                        let av = lv.to_any_value()?;
                        if av.dtype() != dtype {
                            return None;
                        }
                        let scalar = Scalar::new(dtype, av.into_static());
                        use Operator as O;
                        match op {
                            O::Eq | O::EqValidity => {
                                Some(SpecializedColumnPredicateExpr::Eq(scalar))
                            },
                            _ => None,
                        }
                    }),
                )
            });
    }

    ColumnPredicates {
        predicates,
        is_sumwise_complete,
    }
}
