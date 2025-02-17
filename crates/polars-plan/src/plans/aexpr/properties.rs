use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::*;

impl AExpr {
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self, AExpr::Column(_) | AExpr::Literal(_) | AExpr::Len)
    }

    pub(crate) fn is_col(&self) -> bool {
        matches!(self, AExpr::Column(_))
    }

    /// Checks whether this expression is elementwise. This only checks the top level expression.
    pub(crate) fn is_elementwise_top_level(&self) -> bool {
        use AExpr::*;

        match self {
            AnonymousFunction { options, .. } => options.is_elementwise(),

            // Non-strict strptime must be done in-memory to ensure the format
            // is consistent across the entire dataframe.
            #[cfg(all(feature = "strings", feature = "temporal"))]
            Function {
                options,
                function: FunctionExpr::StringExpr(StringFunction::Strptime(_, opts)),
                ..
            } => {
                assert!(options.is_elementwise());
                opts.strict
            },

            Function { options, .. } => options.is_elementwise(),

            Literal(v) => v.is_scalar(),

            Alias(_, _) | BinaryExpr { .. } | Column(_) | Ternary { .. } | Cast { .. } => true,

            Agg { .. }
            | Explode(_)
            | Filter { .. }
            | Gather { .. }
            | Len
            | Slice { .. }
            | Sort { .. }
            | SortBy { .. }
            | Window { .. } => false,
        }
    }
}

/// Checks if the top-level expression node is elementwise. If this is the case, then `stack` will
/// be extended further with any nested expression nodes.
pub fn is_elementwise(stack: &mut UnitVec<Node>, ae: &AExpr, expr_arena: &Arena<AExpr>) -> bool {
    use AExpr::*;

    if !ae.is_elementwise_top_level() {
        return false;
    }

    match ae {
        // Literals that aren't being projected are allowed to be non-scalar, so we don't add them
        // for inspection. (e.g. `is_in(<literal>)`).
        #[cfg(feature = "is_in")]
        Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsIn),
            input,
            ..
        } => (|| {
            if let Some(rhs) = input.get(1) {
                assert_eq!(input.len(), 2); // A.is_in(B)
                let rhs = rhs.node();

                if matches!(expr_arena.get(rhs), AExpr::Literal { .. }) {
                    stack.extend([input[0].node()]);
                    return;
                }
            };

            ae.inputs_rev(stack);
        })(),
        _ => ae.inputs_rev(stack),
    }

    true
}

pub fn all_elementwise<'a, N>(nodes: &'a [N], expr_arena: &Arena<AExpr>) -> bool
where
    Node: From<&'a N>,
{
    nodes
        .iter()
        .all(|n| is_elementwise_rec(expr_arena.get(n.into()), expr_arena))
}

/// Recursive variant of `is_elementwise`
pub fn is_elementwise_rec<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>) -> bool {
    let mut stack = unitvec![];

    loop {
        if !is_elementwise(&mut stack, ae, expr_arena) {
            return false;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}

/// Recursive variant of `is_elementwise` that also forbids casting to categoricals. This function
/// is used to determine if an expression evaluation can be vertically parallelized.
pub fn is_elementwise_rec_no_cat_cast<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>) -> bool {
    let mut stack = unitvec![];

    loop {
        if !is_elementwise(&mut stack, ae, expr_arena) {
            return false;
        }

        #[cfg(feature = "dtype-categorical")]
        {
            if let AExpr::Cast {
                dtype: DataType::Categorical(..),
                ..
            } = ae
            {
                return false;
            }
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}

/// Check whether filters can be pushed past this expression.
///
/// A query, `with_columns(C).filter(P)` can be re-ordered as `filter(P).with_columns(C)`, iff
/// both P and C permit filter pushdown.
///
/// If filter pushdown is permitted, `stack` is extended with any input expression nodes that this
/// expression may have.
///
/// Note that this  function is not recursive - the caller should repeatedly
/// call this function with the `stack` to perform a recursive check.
pub(crate) fn permits_filter_pushdown(
    stack: &mut UnitVec<Node>,
    ae: &AExpr,
    expr_arena: &Arena<AExpr>,
) -> bool {
    // This is a subset of an `is_elementwise` check that also blocks exprs that raise errors
    // depending on the data. The idea is that, although the success value of these functions
    // are elementwise, their error behavior is non-elementwise. Their error behavior is essentially
    // performing an aggregation `ANY(evaluation_result_was_error)`, and if this is the case then
    // the query result should be an error.
    match ae {
        // Rows that go OOB on get/gather may be filtered out in earlier operations,
        // so we don't push these down.
        AExpr::Function {
            function: FunctionExpr::ListExpr(ListFunction::Get(false)),
            ..
        } => false,
        #[cfg(feature = "list_gather")]
        AExpr::Function {
            function: FunctionExpr::ListExpr(ListFunction::Gather(false)),
            ..
        } => false,
        #[cfg(feature = "dtype-array")]
        AExpr::Function {
            function: FunctionExpr::ArrayExpr(ArrayFunction::Get(false)),
            ..
        } => false,
        // TODO: There are a lot more functions that should be caught here.
        ae => is_elementwise(stack, ae, expr_arena),
    }
}

pub fn permits_filter_pushdown_rec<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>) -> bool {
    let mut stack = unitvec![];

    loop {
        if !permits_filter_pushdown(&mut stack, ae, expr_arena) {
            return false;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}

pub fn can_pre_agg_exprs(
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    _input_schema: &Schema,
) -> bool {
    exprs
        .iter()
        .all(|e| can_pre_agg(e.node(), expr_arena, _input_schema))
}

/// Checks whether an expression can be pre-aggregated in a group-by. Note that this also must be
/// implemented physically, so this isn't a complete list.
pub fn can_pre_agg(agg: Node, expr_arena: &Arena<AExpr>, _input_schema: &Schema) -> bool {
    let aexpr = expr_arena.get(agg);

    match aexpr {
        AExpr::Len => true,
        AExpr::Column(_) | AExpr::Literal(_) => false,
        // We only allow expressions that end with an aggregation.
        AExpr::Agg(_) => {
            let has_aggregation =
                |node: Node| has_aexpr(node, expr_arena, |ae| matches!(ae, AExpr::Agg(_)));

            // check if the aggregation type is partitionable
            // only simple aggregation like col().sum
            // that can be divided in to the aggregation of their partitions are allowed
            let can_partition = (expr_arena).iter(agg).all(|(_, ae)| {
                use AExpr::*;
                match ae {
                    // struct is needed to keep both states
                    #[cfg(feature = "dtype-struct")]
                    Agg(IRAggExpr::Mean(_)) => {
                        // only numeric means for now.
                        // logical types seem to break because of casts to float.
                        matches!(
                            expr_arena
                                .get(agg)
                                .get_type(_input_schema, Context::Default, expr_arena)
                                .map(|dt| { dt.is_primitive_numeric() }),
                            Ok(true)
                        )
                    },
                    // only allowed expressions
                    Agg(agg_e) => {
                        matches!(
                            agg_e,
                            IRAggExpr::Min { .. }
                                | IRAggExpr::Max { .. }
                                | IRAggExpr::Sum(_)
                                | IRAggExpr::Last(_)
                                | IRAggExpr::First(_)
                                | IRAggExpr::Count(_, true)
                        )
                    },
                    Function { input, options, .. } => {
                        matches!(options.collect_groups, ApplyOptions::ElementWise)
                            && input.len() == 1
                            && !has_aggregation(input[0].node())
                    },
                    BinaryExpr { left, right, .. } => {
                        !has_aggregation(*left) && !has_aggregation(*right)
                    },
                    Ternary {
                        truthy,
                        falsy,
                        predicate,
                        ..
                    } => {
                        !has_aggregation(*truthy)
                            && !has_aggregation(*falsy)
                            && !has_aggregation(*predicate)
                    },
                    Literal(lv) => lv.is_scalar(),
                    Column(_) | Len | Cast { .. } => true,
                    _ => false,
                }
            });

            #[cfg(feature = "object")]
            {
                for name in aexpr_to_leaf_names(agg, expr_arena) {
                    let dtype = _input_schema.get(&name).unwrap();

                    if let DataType::Object(_, _) = dtype {
                        return false;
                    }
                }
            }
            can_partition
        },
        _ => false,
    }
}
