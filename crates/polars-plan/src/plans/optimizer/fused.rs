use std::collections::BTreeSet;

use super::stack_opt::OptimizeExprContext;
use super::*;

pub struct FusedArithmetic {}

pub struct FusedSelectSlice {
    processed: BTreeSet<Node>,
}

impl FusedSelectSlice {
    pub fn new() -> Self {
        Self {
            processed: Default::default(),
        }
    }
}

fn get_literal_int(node: Node, expr_arena: &Arena<AExpr>) -> Option<i64> {
    match expr_arena.get(node) {
        AExpr::Literal(LiteralValue::Scalar(s)) => s.value().try_extract::<i64>().ok(),
        _ => None,
    }
}

impl OptimizationRule for FusedSelectSlice {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        if self.processed.contains(&node) {
            return Ok(None);
        }

        let lp = lp_arena.get(node);

        use IR::*;
        if let Select {
            input,
            expr,
            schema,
            options,
        } = lp
        {
            if expr.is_empty() {
                self.processed.insert(node);
                return Ok(None);
            }

            #[derive(PartialEq, Eq, Debug, Clone, Copy)]
            enum Range {
                None,
                Val(i64, IdxSize),
            }

            let mut common_range = Range::None;

            for e in expr {
                match expr_arena.get(e.node()) {
                    AExpr::Agg(agg) => {
                        let inner_input = match agg {
                            IRAggExpr::First(input) => {
                                let r = Range::Val(0, 1);
                                if common_range == Range::None {
                                    common_range = r;
                                } else if common_range != r {
                                    self.processed.insert(node);
                                    return Ok(None);
                                }
                                *input
                            },
                            IRAggExpr::Last(input) => {
                                let r = Range::Val(-1, 1);
                                if common_range == Range::None {
                                    common_range = r;
                                } else if common_range != r {
                                    self.processed.insert(node);
                                    return Ok(None);
                                }
                                *input
                            },
                            _ => {
                                self.processed.insert(node);
                                return Ok(None);
                            },
                        };
                        if !expr_arena.get(inner_input).is_length_preserving(expr_arena) {
                            self.processed.insert(node);
                            return Ok(None);
                        }
                    },
                    AExpr::Slice {
                        input: inner_input,
                        offset,
                        length,
                    } => {
                        let o = get_literal_int(*offset, expr_arena);
                        let l = get_literal_int(*length, expr_arena);

                        if let (Some(o), Some(l)) = (o, l) {
                            let r = Range::Val(o, l as IdxSize);
                            if common_range == Range::None {
                                common_range = r;
                            } else if common_range != r {
                                self.processed.insert(node);
                                return Ok(None);
                            }
                        } else {
                            self.processed.insert(node);
                            return Ok(None);
                        }

                        if !expr_arena.get(*inner_input).is_length_preserving(expr_arena) {
                            self.processed.insert(node);
                            return Ok(None);
                        }
                    },
                    _ => {
                        self.processed.insert(node);
                        return Ok(None);
                    },
                };
            }

            if let Range::Val(offset, len) = common_range {
                let input_node = *input;

                if matches!(lp_arena.get(input_node), IR::Slice { .. }) {
                    self.processed.insert(node);
                    return Ok(None);
                }

                // Clone data before mutable borrow
                let expr = expr.clone();
                let schema = schema.clone();
                let options = *options;

                let slice_node = lp_arena.add(IR::Slice {
                    input: input_node,
                    offset,
                    len,
                });

                self.processed.insert(node);
                return Ok(Some(IR::Select {
                    input: slice_node,
                    expr,
                    schema,
                    options,
                }));
            }
        }

        self.processed.insert(node);
        Ok(None)
    }
}


fn get_expr(input: &[Node], op: FusedOperator, expr_arena: &Arena<AExpr>) -> AExpr {

    let input = input
        .iter()
        .copied()
        .map(|n| ExprIR::from_node(n, expr_arena))
        .collect();
    let mut options =
        FunctionOptions::elementwise().with_casting_rules(CastingRules::cast_to_supertypes());
    // order of operations change because of FMA
    // so we must toggle this check off
    // it is still safe as it is a trusted operation
    unsafe { options.no_check_lengths() }
    AExpr::Function {
        input,
        function: IRFunctionExpr::Fused(op),
        options,
    }
}

fn check_eligible(
    left: &Node,
    right: &Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<bool> {
    let field_left = expr_arena
        .get(*left)
        .to_field(&ToFieldContext::new(expr_arena, schema))?;
    let type_right = expr_arena
        .get(*right)
        .to_dtype(&ToFieldContext::new(expr_arena, schema))?;
    let type_left = &field_left.dtype;
    // Exclude literals for now as these will not benefit from fused operations downstream #9857
    // This optimization would also interfere with the `col -> lit` type-coercion rules
    // And it might also interfere with constant folding which is a more suitable optimizations here
    if type_left.is_primitive_numeric()
        && type_right.is_primitive_numeric()
        && !has_aexpr_literal(*left, expr_arena)
        && !has_aexpr_literal(*right, expr_arena)
    {
        Ok(true)
    } else {
        Ok(false)
    }
}

impl OptimizationRule for FusedArithmetic {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        schema: &Schema,
        ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        // We don't want to fuse arithmetic that we send to pyarrow.
        if ctx.in_pyarrow_scan || ctx.in_io_plugin {
            return Ok(None);
        }

        let expr = expr_arena.get(expr_node);

        use AExpr::*;
        match expr {
            BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } => {
                // FUSED MULTIPLY ADD
                // For fma the plus is always the out as the multiply takes prevalence
                match expr_arena.get(*left) {
                    // Argument order is a + b * c
                    // so we must swap operands
                    //
                    // input
                    // (a * b) + c
                    // swapped as
                    // c + (a * b)
                    BinaryExpr {
                        left: a,
                        op: Operator::Multiply,
                        right: b,
                    } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                        let input = &[*right, *a, *b];
                        get_expr(input, FusedOperator::MultiplyAdd, expr_arena)
                    })),
                    _ => match expr_arena.get(*right) {
                        // input
                        // (a + (b * c)
                        // kept as input
                        BinaryExpr {
                            left: a,
                            op: Operator::Multiply,
                            right: b,
                        } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                            let input = &[*left, *a, *b];
                            get_expr(input, FusedOperator::MultiplyAdd, expr_arena)
                        })),
                        _ => Ok(None),
                    },
                }
            },

            BinaryExpr {
                left,
                op: Operator::Minus,
                right,
            } => {
                // FUSED SUB MULTIPLY
                match expr_arena.get(*right) {
                    // input
                    // (a - (b * c)
                    // kept as input
                    BinaryExpr {
                        left: a,
                        op: Operator::Multiply,
                        right: b,
                    } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                        let input = &[*left, *a, *b];
                        get_expr(input, FusedOperator::SubMultiply, expr_arena)
                    })),
                    _ => {
                        // FUSED MULTIPLY SUB
                        match expr_arena.get(*left) {
                            // input
                            // (a * b) - c
                            // kept as input
                            BinaryExpr {
                                left: a,
                                op: Operator::Multiply,
                                right: b,
                            } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                                let input = &[*a, *b, *right];
                                get_expr(input, FusedOperator::MultiplySub, expr_arena)
                            })),
                            _ => Ok(None),
                        }
                    },
                }
            },
            _ => Ok(None),
        }
    }
}
