use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::*;
use crate::constants::MAP_LIST_NAME;

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

            Function { options, .. } => options.is_elementwise(),

            Literal(v) => v.is_scalar(),

            Alias(_, _) | BinaryExpr { .. } | Column(_) | Ternary { .. } | Cast { .. } => true,

            Agg { .. }
            | Explode { .. }
            | Filter { .. }
            | Gather { .. }
            | Len
            | Slice { .. }
            | Sort { .. }
            | SortBy { .. }
            | Window { .. } => false,
        }
    }

    pub(crate) fn does_not_modify_top_level(&self) -> bool {
        match self {
            AExpr::Column(_) => true,
            AExpr::Function { function, .. } => matches!(function, FunctionExpr::SetSortedFlag(_)),
            _ => false,
        }
    }
}

// Traversal utilities
fn property_and_traverse<F>(stack: &mut UnitVec<Node>, ae: &AExpr, property: F) -> bool
where
    F: Fn(&AExpr) -> bool,
{
    if !property(ae) {
        return false;
    }
    ae.inputs_rev(stack);
    true
}

fn property_rec<F>(node: Node, expr_arena: &Arena<AExpr>, property: F) -> bool
where
    F: Fn(&mut UnitVec<Node>, &AExpr, &Arena<AExpr>) -> bool,
{
    let mut stack = unitvec![];
    let mut ae = expr_arena.get(node);

    loop {
        if !property(&mut stack, ae, expr_arena) {
            return false;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}

/// Checks if the top-level expression node does not modify. If this is the case, then `stack` will
/// be extended further with any nested expression nodes.
fn does_not_modify(stack: &mut UnitVec<Node>, ae: &AExpr, _expr_arena: &Arena<AExpr>) -> bool {
    property_and_traverse(stack, ae, |ae| ae.does_not_modify_top_level())
}

pub fn does_not_modify_rec(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    property_rec(node, expr_arena, does_not_modify)
}

// Properties

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
            function: FunctionExpr::Boolean(BooleanFunction::IsIn { .. }),
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
        .all(|n| is_elementwise_rec(n.into(), expr_arena))
}

/// Recursive variant of `is_elementwise`
pub fn is_elementwise_rec(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    property_rec(node, expr_arena, is_elementwise)
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

#[derive(Debug, Clone)]
pub enum ExprPushdownGroup {
    /// Can be pushed. (elementwise, infallible)
    ///
    /// e.g. non-strict cast
    Pushable,
    /// Cannot be pushed, but doesn't block pushables. (elementwise, fallible)
    ///
    /// Fallible expressions are categorized into this group rather than the Barrier group. The
    /// effect of this means we push more predicates, but the expression may no longer error
    /// if the problematic rows are filtered out.
    ///
    /// e.g. strict-cast, list.get(null_on_oob=False), to_datetime(strict=True)
    Fallible,
    /// Cannot be pushed, and blocks all expressions at the current level. (non-elementwise)
    ///
    /// e.g. sort()
    Barrier,
}

impl ExprPushdownGroup {
    /// Note:
    /// * `stack` is not extended with any nodes if a barrier expression is seen.
    /// * This function is not recursive - the caller should repeatedly
    ///   call this function with the `stack` to perform a recursive check.
    pub fn update_with_expr(
        &mut self,
        stack: &mut UnitVec<Node>,
        ae: &AExpr,
        expr_arena: &Arena<AExpr>,
    ) -> &mut Self {
        match self {
            ExprPushdownGroup::Pushable | ExprPushdownGroup::Fallible => {
                // Downgrade to unpushable if fallible
                if match ae {
                    // Rows that go OOB on get/gather may be filtered out in earlier operations,
                    // so we don't push these down.
                    AExpr::Function {
                        function: FunctionExpr::ListExpr(ListFunction::Get(false)),
                        ..
                    } => true,

                    #[cfg(feature = "list_gather")]
                    AExpr::Function {
                        function: FunctionExpr::ListExpr(ListFunction::Gather(false)),
                        ..
                    } => true,

                    #[cfg(feature = "dtype-array")]
                    AExpr::Function {
                        function: FunctionExpr::ArrayExpr(ArrayFunction::Get(false)),
                        ..
                    } => true,

                    #[cfg(all(feature = "strings", feature = "temporal"))]
                    AExpr::Function {
                        input,
                        function:
                            FunctionExpr::StringExpr(StringFunction::Strptime(_, strptime_options)),
                        ..
                    } => {
                        debug_assert!(input.len() <= 2);

                        // `ambiguous` parameter to `to_datetime()`. Should always be a literal.
                        debug_assert!(matches!(
                            input.get(1).map(|x| expr_arena.get(x.node())),
                            Some(AExpr::Literal(_)) | None
                        ));

                        match input.first().map(|x| expr_arena.get(x.node())) {
                            Some(AExpr::Literal(_)) | None => false,
                            _ => strptime_options.strict,
                        }
                    },
                    #[cfg(feature = "python")]
                    // This is python `map_elements`. This is a hack because that function breaks
                    // the Polars model. It should be elementwise. This must be fixed.
                    AExpr::AnonymousFunction { options, .. }
                        if options.flags.contains(FunctionFlags::APPLY_LIST)
                            && options.fmt_str == MAP_LIST_NAME =>
                    {
                        return self;
                    },

                    AExpr::Cast {
                        expr,
                        dtype: _,
                        options: CastOptions::Strict,
                    } => !matches!(expr_arena.get(*expr), AExpr::Literal(_)),

                    _ => false,
                } {
                    *self = ExprPushdownGroup::Fallible;
                }

                // Downgrade to barrier if non-elementwise
                if !is_elementwise(stack, ae, expr_arena) {
                    *self = ExprPushdownGroup::Barrier
                }
            },

            ExprPushdownGroup::Barrier => {},
        }

        self
    }

    pub fn update_with_expr_rec<'a>(
        &mut self,
        mut ae: &'a AExpr,
        expr_arena: &'a Arena<AExpr>,
        scratch: Option<&mut UnitVec<Node>>,
    ) -> &mut Self {
        let mut local_scratch = unitvec![];
        let stack = scratch.unwrap_or(&mut local_scratch);

        loop {
            self.update_with_expr(stack, ae, expr_arena);

            if let ExprPushdownGroup::Barrier = self {
                return self;
            }

            let Some(node) = stack.pop() else {
                break;
            };

            ae = expr_arena.get(node);
        }

        self
    }

    pub fn blocks_pushdown(&self, maintain_errors: bool) -> bool {
        match self {
            ExprPushdownGroup::Barrier => true,
            ExprPushdownGroup::Fallible => maintain_errors,
            ExprPushdownGroup::Pushable => false,
        }
    }
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
                        options.is_elementwise()
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

                    if let DataType::Object(_) = dtype {
                        return false;
                    }
                }
            }
            can_partition
        },
        _ => false,
    }
}
