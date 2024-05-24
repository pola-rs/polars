use polars_utils::unitvec;

use super::*;
use crate::prelude::*;

#[derive(Copy, Clone, Debug)]
pub struct IRNode {
    node: Node,
}

impl IRNode {
    pub fn new(node: Node) -> Self {
        Self { node }
    }

    pub fn node(&self) -> Node {
        self.node
    }

    pub fn replace_node(&mut self, node: Node) {
        self.node = node;
    }

    /// Replace the current `Node` with a new `IR`.
    pub fn replace(&mut self, ae: IR, arena: &mut Arena<IR>) {
        let node = self.node;
        arena.replace(node, ae);
    }

    pub fn to_alp<'a>(&self, arena: &'a Arena<IR>) -> &'a IR {
        arena.get(self.node)
    }

    pub fn to_alp_mut<'a>(&mut self, arena: &'a mut Arena<IR>) -> &'a mut IR {
        arena.get_mut(self.node)
    }

    pub fn assign(&mut self, ir_node: IR, arena: &mut Arena<IR>) {
        let node = arena.add(ir_node);
        self.node = node;
    }
}

pub type IRNodeArena = (Arena<IR>, Arena<AExpr>);

impl TreeWalker for IRNode {
    type Arena = IRNodeArena;

    fn apply_children<F: FnMut(&Self, &Self::Arena) -> PolarsResult<VisitRecursion>>(
        &self,
        op: &mut F,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = unitvec![];

        self.to_alp(&arena.0).copy_inputs(&mut scratch);
        for &node in scratch.as_slice() {
            let lp_node = IRNode::new(node);
            match op(&lp_node, arena)? {
                // let the recursion continue
                VisitRecursion::Continue | VisitRecursion::Skip => {},
                // early stop
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children<F: FnMut(Self, &mut Self::Arena) -> PolarsResult<Self>>(
        self,
        op: &mut F,
        arena: &mut Self::Arena,
    ) -> PolarsResult<Self> {
        let mut inputs = vec![];
        let mut exprs = vec![];

        let lp = arena.0.take(self.node);
        lp.copy_inputs(&mut inputs);
        lp.copy_exprs(&mut exprs);

        // rewrite the nodes
        for node in &mut inputs {
            let lp_node = IRNode::new(*node);
            *node = op(lp_node, arena)?.node;
        }

        let lp = lp.with_exprs_and_input(exprs, inputs);
        arena.0.replace(self.node, lp);
        Ok(self)
    }
}

#[cfg(feature = "cse")]
pub(crate) fn with_ir_arena<F: FnOnce(&mut IRNodeArena) -> T, T>(
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    func: F,
) -> T {
    try_with_ir_arena(lp_arena, expr_arena, |a| Ok(func(a))).unwrap()
}

#[cfg(feature = "cse")]
pub(crate) fn try_with_ir_arena<F: FnOnce(&mut IRNodeArena) -> PolarsResult<T>, T>(
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    func: F,
) -> PolarsResult<T> {
    let owned_lp_arena = std::mem::take(lp_arena);
    let owned_expr_arena = std::mem::take(expr_arena);

    let mut arena = (owned_lp_arena, owned_expr_arena);
    let out = func(&mut arena);
    std::mem::swap(lp_arena, &mut arena.0);
    std::mem::swap(expr_arena, &mut arena.1);
    out
}
