//! Join keys and `join_where` predicates over operands of different dtypes, which a SQL
//! comparison compares in a common type but the engine join only upcasts losslessly.
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_numeric_upcast_supertype_lossless, try_get_supertype};
use polars_lazy::prelude::*;
use polars_plan::plans::{AExpr, ExprToIRContext, ToFieldContext, node_to_expr, to_expr_ir};
use polars_utils::arena::{Arena, Node};

use crate::sql_expr::is_decimal_literal_valued;

/// `l` and `r` cast to the type a SQL comparison of them uses, for a key pair that the
/// engine join doesn't cast itself. `merging` allows casts that map distinct values to one
/// key, such as integers and floats to a float. `None` when no cast is needed or none is
/// known.
pub(crate) fn sql_join_key_pair(
    l: Expr,
    l_dtype: &DataType,
    r: Expr,
    r_dtype: &DataType,
    merging: bool,
) -> Option<(Expr, Expr)> {
    use DataType::*;
    // The join computes a key that isn't a column, which gives an untyped literal its type.
    let l_dtype = &l_dtype.clone().materialize_unknown(true).ok()?;
    let r_dtype = &r_dtype.clone().materialize_unknown(true).ok()?;
    if l_dtype == r_dtype || get_numeric_upcast_supertype_lossless(l_dtype, r_dtype).is_some() {
        return None;
    }
    // A decimal compares exactly with another decimal or an integer at the larger scale.
    let decimal = |scale: usize| {
        let dtype = Decimal(polars_compute::decimal::DEC128_MAX_PREC, scale);
        Some((
            l.clone().strict_cast(dtype.clone()),
            r.clone().strict_cast(dtype),
        ))
    };
    let both = |dtype: DataType| Some((l.clone().cast(dtype.clone()), r.clone().cast(dtype)));
    match (l_dtype, r_dtype) {
        (Decimal(_, l_scale), Decimal(_, r_scale)) => decimal(*l_scale.max(r_scale)),
        (Decimal(_, scale), dtype) | (dtype, Decimal(_, scale)) if dtype.is_integer() => {
            decimal(*scale)
        },
        (Null, dtype) => Some((l.cast(dtype.clone()), r)),
        (dtype, Null) => Some((l, r.cast(dtype.clone()))),
        _ if !merging => None,
        // A decimal literal takes the float's type, as in a SQL comparison.
        (Decimal(..), dtype) if dtype.is_float() && is_decimal_literal_valued(&l) => {
            Some((l.cast(dtype.clone()), r))
        },
        (dtype, Decimal(..)) if dtype.is_float() && is_decimal_literal_valued(&r) => {
            Some((l, r.cast(dtype.clone())))
        },
        (Decimal(..), dtype) | (dtype, Decimal(..)) if dtype.is_float() => both(Float64),
        _ if l_dtype.is_primitive_numeric() && r_dtype.is_primitive_numeric() => {
            both(try_get_supertype(l_dtype, r_dtype).ok()?)
        },
        _ => None,
    }
}

/// `predicate` with each comparison of a left and a right operand that `join_where`
/// rejects cast to their supertype: a primitive numeric pair without a lossless upcast,
/// such as an integer and a float, or a Null operand. Columns `left_schema` has are left
/// ones; `joined_schema` holds both inputs as `join_where` resolves them. The predicate
/// itself when nothing is cast.
pub(crate) fn sql_join_where_predicate(
    predicate: Expr,
    left_schema: &Schema,
    joined_schema: &Schema,
) -> Expr {
    // `join_where` checks the lowered predicate, where SQL arithmetic has become binary
    // expressions or functions.
    let mut arena = Arena::new();
    let mut ctx = ExprToIRContext::new(&mut arena, joined_schema);
    let Ok(ir) = to_expr_ir(predicate.clone(), &mut ctx) else {
        return predicate;
    };
    let mut casts = Vec::new();
    if comparison_casts(ir.node(), left_schema, joined_schema, &arena, &mut casts).is_err()
        || casts.is_empty()
    {
        return predicate;
    }
    for (node, dtype) in casts {
        let input = arena.duplicate(node);
        arena.replace(
            node,
            AExpr::Cast {
                expr: input,
                dtype,
                options: CastOptions::NonStrict,
            },
        );
    }
    node_to_expr(ir.node(), &arena)
}

/// Which join inputs an expression reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    None,
    Left,
    Right,
    Both,
}

impl Origin {
    fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, o) | (o, Self::None) => o,
            (a, b) if a == b => a,
            _ => Self::Both,
        }
    }
}

// The walk of the `join_where` dtype check (`build_upcast_node_list` in polars-plan),
// recording the operands of the comparisons it rejects with the dtype to cast them to.
fn comparison_casts(
    node: Node,
    left_schema: &Schema,
    joined_schema: &Schema,
    arena: &Arena<AExpr>,
    casts: &mut Vec<(Node, DataType)>,
) -> PolarsResult<Origin> {
    Ok(match arena.get(node) {
        AExpr::Column(name) if left_schema.contains(name) => Origin::Left,
        AExpr::Column(_) => Origin::Right,
        AExpr::Cast { expr, .. } => {
            comparison_casts(*expr, left_schema, joined_schema, arena, casts)?
        },
        AExpr::BinaryExpr { left, op, right } => {
            let (left, op, right) = (*left, *op, *right);
            let left_origin = comparison_casts(left, left_schema, joined_schema, arena, casts)?;
            let right_origin = comparison_casts(right, left_schema, joined_schema, arena, casts)?;
            if op.is_comparison()
                && matches!(
                    (left_origin, right_origin),
                    (Origin::Left, Origin::Right) | (Origin::Right, Origin::Left)
                )
            {
                let ctx = ToFieldContext::new(arena, joined_schema);
                let left_dtype = arena.get(left).to_dtype(&ctx)?;
                let right_dtype = arena.get(right).to_dtype(&ctx)?;
                let rejected = left_dtype != right_dtype
                    && !left_dtype.is_decimal()
                    && !right_dtype.is_decimal()
                    && (left_dtype.is_primitive_numeric() || right_dtype.is_primitive_numeric())
                    && get_numeric_upcast_supertype_lossless(&left_dtype, &right_dtype).is_none();
                if rejected {
                    let dtype = try_get_supertype(&left_dtype, &right_dtype)?;
                    if dtype != left_dtype {
                        casts.push((left, dtype.clone()));
                    }
                    if dtype != right_dtype {
                        casts.push((right, dtype));
                    }
                }
            }
            left_origin.union(right_origin)
        },
        _ => Origin::None,
    })
}
