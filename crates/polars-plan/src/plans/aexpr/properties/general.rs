use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::super::*;

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

            Eval { variant, .. } => match variant {
                EvalVariant::List => true,
                EvalVariant::Cumulative { min_samples: _ } => false,
            },

            BinaryExpr { .. } | Column(_) | Ternary { .. } | Cast { .. } => true,

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

    /// Checks whether this expression is row-separable. This only checks the top level expression.
    pub(crate) fn is_row_separable_top_level(&self) -> bool {
        use AExpr::*;

        match self {
            AnonymousFunction { options, .. } => options.is_row_separable(),
            Function { options, .. } => options.is_row_separable(),
            Literal(v) => v.is_scalar(),
            Explode { .. } | Filter { .. } => true,
            _ => self.is_elementwise_top_level(),
        }
    }

    pub(crate) fn does_not_modify_top_level(&self) -> bool {
        match self {
            AExpr::Column(_) => true,
            AExpr::Function { function, .. } => {
                matches!(function, IRFunctionExpr::SetSortedFlag(_))
            },
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

pub fn is_prop<P: Fn(&AExpr) -> bool>(
    stack: &mut UnitVec<Node>,
    ae: &AExpr,
    expr_arena: &Arena<AExpr>,
    prop_top_level: P,
) -> bool {
    use AExpr::*;

    if !prop_top_level(ae) {
        return false;
    }

    match ae {
        // Literals that aren't being projected are allowed to be non-scalar, so we don't add them
        // for inspection. (e.g. `is_in(<literal>)`).
        #[cfg(feature = "is_in")]
        Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }),
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

/// Checks if the top-level expression node is elementwise. If this is the case, then `stack` will
/// be extended further with any nested expression nodes.
pub fn is_elementwise(stack: &mut UnitVec<Node>, ae: &AExpr, expr_arena: &Arena<AExpr>) -> bool {
    is_prop(stack, ae, expr_arena, |ae| ae.is_elementwise_top_level())
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

/// Checks if the top-level expression node is row-separable. If this is the case, then `stack` will
/// be extended further with any nested expression nodes.
pub fn is_row_separable(stack: &mut UnitVec<Node>, ae: &AExpr, expr_arena: &Arena<AExpr>) -> bool {
    is_prop(stack, ae, expr_arena, |ae| ae.is_row_separable_top_level())
}

pub fn all_row_separable<'a, N>(nodes: &'a [N], expr_arena: &Arena<AExpr>) -> bool
where
    Node: From<&'a N>,
{
    nodes
        .iter()
        .all(|n| is_row_separable_rec(n.into(), expr_arena))
}

/// Recursive variant of `is_row_separable`
pub fn is_row_separable_rec(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    property_rec(node, expr_arena, is_row_separable)
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
                        function: IRFunctionExpr::ListExpr(IRListFunction::Get(false)),
                        ..
                    } => true,

                    #[cfg(feature = "list_gather")]
                    AExpr::Function {
                        function: IRFunctionExpr::ListExpr(IRListFunction::Gather(false)),
                        ..
                    } => true,

                    #[cfg(feature = "dtype-array")]
                    AExpr::Function {
                        function: IRFunctionExpr::ArrayExpr(IRArrayFunction::Get(false)),
                        ..
                    } => true,

                    #[cfg(all(feature = "strings", feature = "temporal"))]
                    AExpr::Function {
                        input,
                        function:
                            IRFunctionExpr::StringExpr(IRStringFunction::Strptime(_, strptime_options)),
                        ..
                    } => {
                        debug_assert!(input.len() <= 2);

                        let ambiguous_arg_is_infallible_scalar = input
                            .get(1)
                            .map(|x| expr_arena.get(x.node()))
                            .is_some_and(|ae| match ae {
                                AExpr::Literal(lv) => {
                                    lv.extract_str().is_some_and(|ambiguous| match ambiguous {
                                        "earliest" | "latest" | "null" => true,
                                        "raise" => false,
                                        v => {
                                            if cfg!(debug_assertions) {
                                                panic!("unhandled parameter to ambiguous: {v}")
                                            }
                                            false
                                        },
                                    })
                                },
                                _ => false,
                            });

                        let ambiguous_is_fallible = !ambiguous_arg_is_infallible_scalar;

                        strptime_options.strict || ambiguous_is_fallible
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
                                .get_dtype(_input_schema, expr_arena)
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
                                | IRAggExpr::Count {
                                    input: _,
                                    include_nulls: true
                                }
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

/// Identifies columns that are guaranteed to be non-NULL after applying this filter.
///
/// This is conservative in that it will not give false positives, but may not identify all columns.
///
/// Note, this must be called with the root node of filter expressions (the root nodes after splitting
/// with MintermIter is also allowed).
pub(crate) fn predicate_non_null_column_outputs(
    predicate_node: Node,
    expr_arena: &Arena<AExpr>,
    non_null_column_callback: &mut dyn FnMut(&PlSmallStr),
) {
    let mut minterm_iter = MintermIter::new(predicate_node, expr_arena);
    let stack: &mut UnitVec<Node> = &mut unitvec![];

    /// Only traverse the first input, e.g. `A.is_in(B)` we don't consider B.
    macro_rules! traverse_first_input {
        // &[ExprIR]
        ($inputs:expr) => {{
            if let Some(expr_ir) = $inputs.first() {
                stack.push(expr_ir.node())
            }

            false
        }};
    }

    loop {
        use AExpr::*;

        let node = if let Some(node) = stack.pop() {
            node
        } else if let Some(minterm_node) = minterm_iter.next() {
            // Some additional leaf exprs can be pruned.
            match expr_arena.get(minterm_node) {
                Function {
                    input,
                    function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull),
                    options: _,
                } if !input.is_empty() => input.first().unwrap().node(),

                Function {
                    input,
                    function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
                    options: _,
                } if !input.is_empty() => match expr_arena.get(input.first().unwrap().node()) {
                    Function {
                        input,
                        function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNull),
                        options: _,
                    } if !input.is_empty() => input.first().unwrap().node(),

                    _ => minterm_node,
                },

                _ => minterm_node,
            }
        } else {
            break;
        };

        let ae = expr_arena.get(node);

        // This match we traverse a subset of the operations that are guaranteed to maintain NULLs.
        //
        // This must not catch any operations that materialize NULLs, as otherwise e.g.
        // `e.fill_null(False) >= False` will include NULLs
        let traverse_all_inputs = match ae {
            BinaryExpr {
                left: _,
                op,
                right: _,
            } => {
                use Operator::*;

                match op {
                    Eq | NotEq | Lt | LtEq | Gt | GtEq | Plus | Minus | Multiply | Divide
                    | TrueDivide | FloorDivide | Modulus | Xor => true,

                    // These can turn NULLs into true/false. E.g.:
                    // * (L & False) >= False becomes True
                    // * L | True becomes True
                    EqValidity | NotEqValidity | Or | LogicalOr | And | LogicalAnd => false,
                }
            },

            Cast { dtype, .. } => {
                // Forbid nested types, it's currently buggy:
                // >>> pl.select(a=pl.lit(None), b=pl.lit(None).cast(pl.Struct({})))
                // | a    | b         |
                // | ---  | ---       |
                // | null | struct[0] |
                // |------|-----------|
                // | null | {}        |
                //
                // (issue at https://github.com/pola-rs/polars/issues/23276)
                !dtype.is_nested()
            },

            Function {
                input,
                function: _,
                options,
            } => {
                if options
                    .flags
                    .contains(FunctionFlags::PRESERVES_NULL_FIRST_INPUT)
                {
                    traverse_first_input!(input)
                } else {
                    options
                        .flags
                        .contains(FunctionFlags::PRESERVES_NULL_ALL_INPUTS)
                }
            },

            Column(name) => {
                non_null_column_callback(name);
                false
            },

            _ => false,
        };

        if traverse_all_inputs {
            ae.inputs_rev(stack);
        }
    }
}
