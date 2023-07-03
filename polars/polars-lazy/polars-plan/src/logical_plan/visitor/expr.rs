use super::*;
use crate::prelude::*;
use crate::push_expr;

impl TreeWalker for Expr {
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

    fn map_children(self, _op: &mut dyn FnMut(Self) -> PolarsResult<Self>) -> PolarsResult<Self> {
        todo!()
    }
}

pub struct AexprNode {
    node: Node,
    arena: *mut Arena<AExpr>,
}

impl AexprNode {
    /// Don't use this directly, use [`Self::with_context`]
    ///
    /// # Safety
    /// This will keep a pointer to `arena`. The caller must ensure it stays alive.
    unsafe fn new(node: Node, arena: &mut Arena<AExpr>) -> Self {
        Self { node, arena }
    }

    /// Safe interface. Take the `&mut Arena` only for the duration of `op`.
    pub fn with_context<F, T>(node: Node, arena: &mut Arena<AExpr>, mut op: F) -> T
    where
        F: FnMut(AexprNode) -> T,
    {
        // safety: we drop this context before arena is out of scope
        unsafe { op(Self::new(node, arena)) }
    }

    pub fn node(&self) -> Node {
        self.node
    }

    pub fn with_arena<'a, F, T>(&self, op: F) -> T
    where
        F: Fn(&'a Arena<AExpr>) -> T,
    {
        let arena = unsafe { &(*self.arena) };

        op(arena)
    }

    pub fn with_arena_mut<'a, F, T>(&mut self, op: F) -> T
    where
        F: FnOnce(&'a mut Arena<AExpr>) -> T,
    {
        let arena = unsafe { &mut (*self.arena) };

        op(arena)
    }

    pub fn to_aexpr(&self) -> &AExpr {
        self.with_arena(|arena| arena.get(self.node))
    }

    pub fn to_expr(&self) -> Expr {
        self.with_arena(|arena| node_to_expr(self.node, arena))
    }
}

impl TreeWalker for AexprNode {
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

    fn map_children(
        mut self,
        op: &mut dyn FnMut(Self) -> PolarsResult<Self>,
    ) -> PolarsResult<Self> {
        let mut scratch = vec![];

        let ae = self.to_aexpr();
        ae.nodes(&mut scratch);

        // rewrite the nodes
        for node in &mut scratch {
            let aenode = AexprNode {
                node: *node,
                arena: self.arena,
            };
            *node = op(aenode)?.node;
        }

        let ae = ae.clone().replace_inputs(&scratch);
        let node = self.with_arena_mut(move |arena| arena.add(ae));
        self.node = node;
        Ok(self)
    }
}
