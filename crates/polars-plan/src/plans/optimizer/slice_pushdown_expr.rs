use super::*;

fn pushdown(input: Node, offset: Node, length: Node, arena: &mut Arena<AExpr>) -> Node {
    arena.add(AExpr::Slice {
        input,
        offset,
        length,
    })
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
                    let input = scratch[0];
                    let new_input = pushdown(input, offset, length, expr_arena);
                    Some(ae.replace_inputs(&[new_input]))
                },
                BinaryExpr { left, right, op } => {
                    let left = *left;
                    let right = *right;
                    let op = *op;

                    let left = pushdown(left, offset, length, expr_arena);
                    let right = pushdown(right, offset, length, expr_arena);
                    Some(BinaryExpr { left, op, right })
                },
                Ternary {
                    truthy,
                    falsy,
                    predicate,
                } => {
                    let truthy = *truthy;
                    let falsy = *falsy;
                    let predicate = *predicate;

                    let truthy = pushdown(truthy, offset, length, expr_arena);
                    let falsy = pushdown(falsy, offset, length, expr_arena);
                    let predicate = pushdown(predicate, offset, length, expr_arena);
                    Some(Ternary {
                        truthy,
                        falsy,
                        predicate,
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
                        input.iter_mut().for_each(|e| {
                            let n = pushdown(e.node(), offset, length, expr_arena);
                            e.set_node(n);
                        });

                        Some(AnonymousFunction {
                            input,
                            function,
                            options,
                            fmt_str,
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
                        input.iter_mut().for_each(|e| {
                            let n = pushdown(e.node(), offset, length, expr_arena);
                            e.set_node(n);
                        });

                        Some(Function {
                            input,
                            function,
                            options,
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
