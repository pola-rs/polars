use super::*;

/// An implementor of this trait decides how and in which order its nodes get traversed
/// Implemented for [`crate::dsl::Expr`] and [`AexprNode`].
pub trait TreeWalker: Sized {
    fn apply_children(
        &self,
        op: &mut dyn FnMut(&Self) -> PolarsResult<VisitRecursion>,
    ) -> PolarsResult<VisitRecursion>;

    fn map_children(self, op: &mut dyn FnMut(Self) -> PolarsResult<Self>) -> PolarsResult<Self>;

    /// Walks all nodes in depth-first-order.
    fn visit(&self, visitor: &mut dyn Visitor<Node = Self>) -> PolarsResult<VisitRecursion> {
        match visitor.pre_visit(self)? {
            VisitRecursion::Continue => {},
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        };

        match self.apply_children(&mut |node| node.visit(visitor))? {
            // let the recursion continue
            VisitRecursion::Continue | VisitRecursion::Skip => {},
            // If the recursion should stop, no further post visit will be performed
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        }

        visitor.post_visit(self)
    }

    fn rewrite(self, rewriter: &mut dyn RewritingVisitor<Node = Self>) -> PolarsResult<Self> {
        let mutate_this_node = match rewriter.pre_visit(&self)? {
            RewriteRecursion::MutateAndStop => return rewriter.mutate(self),
            RewriteRecursion::Stop => return Ok(self),
            RewriteRecursion::MutateAndContinue => true,
            RewriteRecursion::NoMutateAndContinue => false,
        };

        let after_applied_children = self.map_children(&mut |node| node.rewrite(rewriter))?;

        if mutate_this_node {
            rewriter.mutate(after_applied_children)
        } else {
            Ok(after_applied_children)
        }
    }
}

pub trait Visitor {
    type Node;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        Ok(VisitRecursion::Continue)
    }

    /// Invoked after all children of `node` are visited. Default
    /// implementation does nothing.
    fn post_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        Ok(VisitRecursion::Continue)
    }
}

pub trait RewritingVisitor {
    type Node;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<RewriteRecursion> {
        Ok(RewriteRecursion::MutateAndContinue)
    }

    fn mutate(&mut self, node: Self::Node) -> PolarsResult<Self::Node>;
}
