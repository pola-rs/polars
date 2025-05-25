use std::collections::HashMap;
use polars_core::error::PolarsResult;
use crate::prelude::*;
use crate::plans::aexpr::AExpr;
use crate::plans::optimizer::OptimizationRule;
use polars_utils::pl_str::PlSmallStr;

/// Rule that flattens Struct([Struct([a, b]), c]) → Struct([a, b, c])
pub struct FlattenStructRule {
    already_applied: bool,
}

impl FlattenStructRule {
    pub fn new() -> Self {
        Self { already_applied: false }
    }
}

impl OptimizationRule for FlattenStructRule {
    fn optimize_expr(
        &mut self,
        arena: &mut Arena<AExpr>,
        node: Node,
        _lp_arena: &Arena<IR>,
        _lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        if self.already_applied {
            return Ok(None);
        }
        self.already_applied = true;
        eprintln!("Applying FlattenStructRule");

        let mut memo = HashMap::new();
        let new = flatten_struct(node, arena, &mut memo);

        if new != node {
            Ok(Some(arena.get(new).clone()))
        } else {
            Ok(None)
        }
    }
}

const MAX_NESTING_DEPTH: usize = 20;

fn flatten_struct(
    node: Node,
    arena: &mut Arena<AExpr>,
    memo: &mut HashMap<Node, Node>,
) -> Node {
    flatten_struct_inner(node, arena, memo, 0)
}

fn flatten_struct_inner(
    node: Node,
    arena: &mut Arena<AExpr>,
    memo: &mut HashMap<Node, Node>,
    depth: usize,
) -> Node {
    if depth > MAX_NESTING_DEPTH {
        eprintln!(
            "FlattenStructRule: nesting depth exceeded limit of {MAX_NESTING_DEPTH}; skipping further flattening at node {:?}",
            node
        );
        return node;
    }

    if let Some(&cached) = memo.get(&node) {
        return cached;
    }

    let expr = arena.get(node).clone();

    let new_node = match expr {
        AExpr::Function {
            function: FunctionExpr::StructExpr(StructFunction::WithFields),
            input,
            options,
        } => {
            let mut stack = input.into_iter().rev().collect::<Vec<_>>();
            let mut flat = Vec::new();
            let mut changed = false;

            while let Some(ir) = stack.pop() {
                let n = ir.node();
                match arena.get(n) {
                    AExpr::Function {
                        function: FunctionExpr::StructExpr(StructFunction::WithFields),
                        input: nested,
                        ..
                    } => {
                        if depth + 1 <= MAX_NESTING_DEPTH {
                            stack.extend(nested.iter().rev().cloned());
                            changed = true;
                        } else {
                            eprintln!(
                                "FlattenStructRule: nested Struct skipped at depth {} to avoid deep recursion",
                                depth + 1
                            );
                            flat.push(ir);
                        }
                    }
                    _ => {
                        let new_n = flatten_struct_inner(n, arena, memo, depth + 1);
                        changed |= new_n != n;
                        flat.push(ExprIR::new(new_n, OutputName::LiteralLhs(PlSmallStr::from_static("_"))));
                    }
                }
            }

            if changed {
                arena.add(AExpr::Function {
                    input: flat,
                    function: FunctionExpr::StructExpr(StructFunction::WithFields),
                    options,
                })
            } else {
                node
            }
        }

        AExpr::Function {
            input,
            function,
            options,
        } => {
            let mut changed = false;
            let new_input = input
                .into_iter()
                .map(|ir| {
                    let new_n = flatten_struct_inner(ir.node(), arena, memo, depth + 1);
                    changed |= new_n != ir.node();
                    ExprIR::new(new_n, OutputName::LiteralLhs(PlSmallStr::from_static("_")))
                })
                .collect::<Vec<_>>();

            if changed {
                arena.add(AExpr::Function {
                    input: new_input,
                    function,
                    options,
                })
            } else {
                node
            }
        }

        AExpr::Ternary { predicate, truthy, falsy } => {
            let p = flatten_struct_inner(predicate, arena, memo, depth + 1);
            let t = flatten_struct_inner(truthy, arena, memo, depth + 1);
            let f = flatten_struct_inner(falsy, arena, memo, depth + 1);
            if p != predicate || t != truthy || f != falsy {
                arena.add(AExpr::Ternary {
                    predicate: p,
                    truthy: t,
                    falsy: f,
                })
            } else {
                node
            }
        }

        _ => node,
    };

    memo.insert(node, new_node);
    new_node
}
