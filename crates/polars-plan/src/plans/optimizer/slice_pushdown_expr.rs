use std::borrow::BorrowMut;

use super::*;

/// Returns whether slice was pushed
fn pushdown<T>(inputs: &mut [T], offset: Node, length: Node, arena: &mut Arena<AExpr>) -> bool
where
    T: BorrowMut<Node>,
{
    if inputs.is_empty() {
        return false;
    }

    let mut has_column_height_projection = false;

    macro_rules! is_column_height {
        ($node:expr) => {{
            let node = $node;
            is_length_preserving_ae(node, arena)
                && aexpr_to_leaf_names_iter(node, arena).next().is_some()
        }};
    }

    for node in inputs.iter().map(|x| x.borrow()).copied() {
        if is_scalar_ae(node, arena) {
            continue;
        }

        let column_height = is_column_height!(node);

        if column_height {
            has_column_height_projection = true;
        } else {
            // Unknown non-scalar height
            // TODO: Can technically still push slices with offset >=0.
            return false;
        }
    }

    if !has_column_height_projection {
        return false;
    }

    for node in inputs {
        let n = *node.borrow();

        if is_scalar_ae(n, arena) {
            continue;
        }

        if is_column_height!(n) {
            *node.borrow_mut() = arena.add(AExpr::Slice {
                input: n,
                offset,
                length,
            })
        }
    }

    true
}

impl OptimizationRule for SlicePushDown {
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _schema: &Schema,
        _ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        if let AExpr::Slice {
            input,
            offset,
            length,
        } = expr_arena.get(expr_node)
        {
            let offset = *offset;
            let length = *length;

            use AExpr::*;
            let out = match expr_arena.get(*input) {
                ae @ Cast { .. } => {
                    let ae = ae.clone();
                    let scratch = self.empty_nodes_scratch_mut();

                    ae.inputs_rev(scratch);
                    assert_eq!(scratch.len(), 1);

                    pushdown(scratch, offset, length, expr_arena)
                        .then(|| ae.replace_inputs(scratch))
                },
                BinaryExpr { left, right, op } => {
                    let left = *left;
                    let right = *right;
                    let op = *op;

                    let mut inputs = [left, right];

                    pushdown(&mut inputs[..], offset, length, expr_arena).then(|| BinaryExpr {
                        left: inputs[0],
                        op,
                        right: inputs[1],
                    })
                },
                Ternary {
                    truthy,
                    falsy,
                    predicate,
                } => {
                    let mut inputs = [*truthy, *falsy, *predicate];

                    pushdown(&mut inputs[..], offset, length, expr_arena).then(|| Ternary {
                        truthy: inputs[0],
                        falsy: inputs[1],
                        predicate: inputs[2],
                    })
                },
                m @ AnonymousFunction { options, .. } if options.is_elementwise() => {
                    if let AnonymousFunction {
                        mut input,
                        function,
                        options,
                        fmt_str,
                    } = m.clone()
                    {
                        pushdown(input.as_mut_slice(), offset, length, expr_arena).then(|| {
                            AnonymousFunction {
                                input,
                                function,
                                options,
                                fmt_str,
                            }
                        })
                    } else {
                        unreachable!()
                    }
                },
                m @ Function { options, .. } if options.is_elementwise() => {
                    if let Function {
                        mut input,
                        function,
                        options,
                    } = m.clone()
                    {
                        pushdown(input.as_mut_slice(), offset, length, expr_arena).then(|| {
                            Function {
                                input,
                                function,
                                options,
                            }
                        })
                    } else {
                        unreachable!()
                    }
                },
                _ => None,
            };
            Ok(out)
        } else {
            Ok(None)
        }
    }
}
