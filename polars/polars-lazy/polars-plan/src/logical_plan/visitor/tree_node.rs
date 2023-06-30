use super::*;

/// A visitable and rewritable tree node.
/// Implemented for `[Expr]`.
pub(crate) trait TreeNode: Sized {
    fn apply_children(
        &self,
        op: &mut dyn FnMut(&Self) -> PolarsResult<VisitRecursion>,
    ) -> PolarsResult<VisitRecursion>;

    fn visit(
        &self,
        visitor: &mut dyn TreeNodeVisitor<Node = Self>,
    ) -> PolarsResult<VisitRecursion> {
        match visitor.pre_visit(self)? {
            VisitRecursion::Continue => {}
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        };

        match self.apply_children(&mut |node| node.visit(visitor))? {
            VisitRecursion::Continue => {}
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        }

        visitor.post_visit(self)
    }
}

pub(crate) trait TreeNodeVisitor {
    type Node;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion>;

    /// Invoked after all children of `node` are visited. Default
    /// implementation does nothing.
    fn post_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        Ok(VisitRecursion::Continue)
    }
}
