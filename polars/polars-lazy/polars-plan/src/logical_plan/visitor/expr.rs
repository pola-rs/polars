use super::*;
use crate::prelude::*;
use crate::push_expr;

impl TreeNode for Expr {
    fn apply_children<'a>(
        &'a self,
        op: &mut dyn FnMut(&Self) -> PolarsResult<VisitRecursion>,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = vec![];

        let mut push = |e: &'a Expr| scratch.push(e);
        push_expr!(self, push, iter);

        for child in scratch {
            match op(child)? {
                VisitRecursion::Continue => {}
                // early stop
                VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }
}

pub(crate) struct AexprNode {
    node: Node,
    arena: *mut Arena<AExpr>,
}

impl AexprNode {
    pub fn new(node: Node, arena: &mut Arena<AExpr>) -> Self {
        Self { node, arena }
    }

    pub fn node(&self) -> Node {
        self.node
    }

    pub fn with_arena<'a, F, T>(&'a self, op: F) -> T
    where
        F: Fn(&'a Arena<AExpr>) -> T,
    {
        let arena = unsafe { &(*self.arena) };

        op(arena)
    }

    pub fn to_aexpr(&self) -> &AExpr {
        self.with_arena(|arena| arena.get(self.node))
    }

    pub fn to_expr(&self) -> Expr {
        self.with_arena(|arena| node_to_expr(self.node, arena))
    }
}

impl TreeNode for AexprNode {
    fn apply_children<'a>(
        &'a self,
        op: &mut dyn FnMut(&Self) -> PolarsResult<VisitRecursion>,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = vec![];

        self.to_aexpr().nodes(&mut scratch);
        for node in scratch {
            let aenode = AexprNode {
                node,
                arena: self.arena,
            };
            match op(&aenode)? {
                VisitRecursion::Continue => {}
                // early stop
                VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }
}
