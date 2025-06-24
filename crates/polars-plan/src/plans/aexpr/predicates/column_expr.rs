//! This module creates predicates splits predicates into partial per-column predicates.

use polars_core::datatypes::DataType;
use polars_core::prelude::AnyValue;
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_io::predicates::SpecializedColumnPredicate;
use polars_ops::series::ClosedInterval;
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::get_binary_expr_col_and_lv;
use crate::dsl::Operator;
use crate::plans::aexpr::evaluate::{constant_evaluate, into_column};
use crate::plans::{
    AExpr, IRBooleanFunction, IRFunctionExpr, MintermIter, aexpr_to_leaf_names_iter,
};

pub struct ColumnPredicates {
    pub predicates: PlHashMap<PlSmallStr, (Node, Option<SpecializedColumnPredicate>)>,

    /// Are all column predicates AND-ed together the original predicate.
    pub is_sumwise_complete: bool,
}

pub fn aexpr_to_column_predicates(
    root: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> ColumnPredicates {
    let mut predicates =
        PlHashMap::<PlSmallStr, (Node, Option<SpecializedColumnPredicate>)>::default();
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
            #[cfg(feature = "object")]
            D::Object(_) => {
                is_sumwise_complete = false;
                continue;
            },
            D::Float32 | D::Float64 => {
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
                        let aexpr = expr_arena.get(minterm);

                        match aexpr {
                            AExpr::Function {
                                input,
                                function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNull),
                                options: _,
                            } => {
                                assert_eq!(input.len(), 1);
                                if into_column(input[0].node(), expr_arena, schema, 0)
                                    .is_some()
                                {
                                    Some(SpecializedColumnPredicate::Equal(Scalar::null(
                                        dtype,
                                    )))
                                } else {
                                    None
                                }
                            },
                            #[cfg(feature = "is_between")]
                            AExpr::Function {
                                input,
                                function: IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }),
                                options: _,
                            } => {
                                let (Some(l), Some(r)) = (
                                    constant_evaluate(
                                        input[0].node(),
                                        expr_arena,
                                        schema,
                                        0,
                                    )?,
                                    constant_evaluate(
                                        input[1].node(),
                                        expr_arena,
                                        schema,
                                        0,
                                    )?,
                                ) else {
                                    return None;
                                };

                                let l = l.to_any_value()?;
                                let r = r.to_any_value()?;
                                if l.dtype() != dtype || r.dtype() != dtype {
                                    return None;
                                }

                                let (low_closed, high_closed) = match closed {
                                    ClosedInterval::Both => (true, true),
                                    ClosedInterval::Left => (true, false),
                                    ClosedInterval::Right => (false, true),
                                    ClosedInterval::None => (false, false),
                                };

                                is_between(
                                    &dtype,
                                    Some(Scalar::new(dtype.clone(), l.into_static())),
                                    Some(Scalar::new(dtype.clone(), r.into_static())),
                                    low_closed,
                                    high_closed,
                                )
                            },
                            #[cfg(feature = "is_in")]
                            AExpr::Function {
                                input,
                                function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
                                options: _,
                            } => {
                                into_column(input[0].node(), expr_arena, schema, 0)?;

                                let values = constant_evaluate(
                                    input[1].node(),
                                    expr_arena,
                                    schema,
                                    0,
                                )??;
                                let values = values.to_any_value()?;

                                let values = match values {
                                    AnyValue::List(v) => v,
                                    #[cfg(feature = "dtype-array")]
                                    AnyValue::Array(v, _) => v,
                                    _ => return None,
                                };

                                if values.dtype() != &dtype {
                                    return None;
                                }
                                if !nulls_equal && values.has_nulls() {
                                    return None;
                                }

                                let values = values.rechunk();
                                let values = values.iter()
                                    .map(|av| {
                                        Scalar::new(dtype.clone(), av.into_static())
                                    })
                                    .collect();

                                Some(SpecializedColumnPredicate::EqualOneOf(values))
                            },
                            AExpr::Function {
                                input,
                                function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
                                options: _,
                            } => {
                                if !dtype.is_bool() {
                                    return None;
                                }

                                assert_eq!(input.len(), 1);
                                if into_column(input[0].node(), expr_arena, schema, 0)
                                    .is_some()
                                {
                                    Some(SpecializedColumnPredicate::Equal(false.into()))
                                } else {
                                    None
                                }
                            },
                            AExpr::BinaryExpr { left, op, right } => {
                                let ((_, _), (lv, lv_node)) =
                                    get_binary_expr_col_and_lv(*left, *right, expr_arena, schema)?;
                                let lv = lv?;
                                let av = lv.to_any_value()?;
                                if av.dtype() != dtype {
                                    return None;
                                }
                                let scalar = Scalar::new(dtype.clone(), av.into_static());
                                use Operator as O;
                                match (op, lv_node == *right) {
                                    (O::Eq, _) if scalar.is_null() => None,
                                    (O::Eq | O::EqValidity, _) => {
                                        Some(SpecializedColumnPredicate::Equal(scalar))
                                    },
                                    (O::Lt, true) | (O::Gt, false) => {
                                        is_between(&dtype, None, Some(scalar), false, false)
                                    },
                                    (O::Lt, false) | (O::Gt, true) => {
                                        is_between(&dtype, Some(scalar), None, false, false)
                                    },
                                    (O::LtEq, true) | (O::GtEq, false) => {
                                        is_between(&dtype, None, Some(scalar), false, true)
                                    },
                                    (O::LtEq, false) | (O::GtEq, true) => {
                                        is_between(&dtype, Some(scalar), None, true, false)
                                    },
                                    _ => None,
                                }
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

fn is_between(
    dtype: &DataType,
    low: Option<Scalar>,
    high: Option<Scalar>,
    mut low_closed: bool,
    mut high_closed: bool,
) -> Option<SpecializedColumnPredicate> {
    let dtype = dtype.to_physical();

    if !dtype.is_integer() {
        return None;
    }
    assert!(low.is_some() || high.is_some());

    low_closed &= low.is_some();
    high_closed &= high.is_some();

    let mut low = low.map_or_else(|| dtype.min().unwrap(), |sc| sc.to_physical());
    let mut high = high.map_or_else(|| dtype.max().unwrap(), |sc| sc.to_physical());

    macro_rules! ints {
        ($($t:ident),+) => {
            match (low.any_value_mut(), high.any_value_mut()) {
                $(
                (AV::$t(l), AV::$t(h)) => {
                    if !low_closed {
                        *l = l.checked_add(1)?;
                    }
                    if !high_closed {
                        *h = h.checked_sub(1)?;
                    }
                    assert!(*l <= *h);
                },
                )+
                _ => return None,
            }
        };
    }

    use AnyValue as AV;
    ints!(
        Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64
    );

    Some(SpecializedColumnPredicate::Between(low, high))
}
