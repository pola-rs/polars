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
        _lp_arena: &Arena<ALogicalPlan>,
        _lp_node: Node,
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
                ae @ Alias(..) | ae @ Cast { .. } => {
                    let ae = ae.clone();
                    self.scratch.clear();
                    ae.nodes(&mut self.scratch);
                    let input = self.scratch[0];
                    let new_input = pushdown(input, offset, length, expr_arena);
                    Some(ae.replace_inputs(&[new_input]))
                },
                Literal(lv) => {
                    match lv {
                        LiteralValue::Series(_) => None,
                        LiteralValue::Range { .. } => None,
                        // no need to slice a literal value of unit length
                        lv => Some(Literal(lv.clone())),
                    }
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
                m @ AnonymousFunction { options, .. }
                    if matches!(options.collect_groups, ApplyOptions::ElementWise) =>
                {
                    if let AnonymousFunction {
                        input,
                        function,
                        output_type,
                        options,
                    } = m.clone()
                    {
                        let input = input
                            .iter()
                            .map(|n| pushdown(*n, offset, length, expr_arena))
                            .collect();

                        Some(AnonymousFunction {
                            input,
                            function,
                            output_type,
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
