use recursive::recursive;

use super::*;

/// An implementor of this trait decides how and in which order its nodes get traversed
/// Implemented for [`crate::dsl::Expr`] and [`AexprNode`].
pub trait TreeWalker: Sized {
    type Arena;
    fn apply_children<F: FnMut(&Self, &Self::Arena) -> PolarsResult<VisitRecursion>>(
        &self,
        op: &mut F,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion>;

    fn map_children<F: FnMut(Self, &mut Self::Arena) -> PolarsResult<Self>>(
        self,
        op: &mut F,
        arena: &mut Self::Arena,
    ) -> PolarsResult<Self>;

    /// Walks all nodes in depth-first-order.
    #[recursive]
    fn visit<V: Visitor<Node = Self, Arena = Self::Arena>>(
        &self,
        visitor: &mut V,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        match visitor.pre_visit(self, arena)? {
            VisitRecursion::Continue => {},
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        };

        match self.apply_children(&mut |node, arena| node.visit(visitor, arena), arena)? {
            // let the recursion continue
            VisitRecursion::Continue | VisitRecursion::Skip => {},
            // If the recursion should stop, no further post visit will be performed
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        }

        visitor.post_visit(self, arena)
    }

    #[recursive]
    fn rewrite<R: RewritingVisitor<Node = Self, Arena = Self::Arena>>(
        self,
        rewriter: &mut R,
        arena: &mut Self::Arena,
    ) -> PolarsResult<Self> {
        let mutate_this_node = match rewriter.pre_visit(&self, arena)? {
            RewriteRecursion::MutateAndStop => return rewriter.mutate(self, arena),
            RewriteRecursion::Stop => return Ok(self),
            RewriteRecursion::MutateAndContinue => true,
            RewriteRecursion::NoMutateAndContinue => false,
        };

        let after_applied_children =
            self.map_children(&mut |node, arena| node.rewrite(rewriter, arena), arena)?;

        if mutate_this_node {
            rewriter.mutate(after_applied_children, arena)
        } else {
            Ok(after_applied_children)
        }
    }
}

pub trait Visitor {
    type Node;
    type Arena;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(
        &mut self,
        _node: &Self::Node,
        _arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        Ok(VisitRecursion::Continue)
    }

    /// Invoked after all children of `node` are visited. Default
    /// implementation does nothing.
    fn post_visit(
        &mut self,
        _node: &Self::Node,
        _arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        Ok(VisitRecursion::Continue)
    }
}

pub trait RewritingVisitor {
    type Node;
    type Arena;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(
        &mut self,
        _node: &Self::Node,
        _arena: &mut Self::Arena,
    ) -> PolarsResult<RewriteRecursion> {
        Ok(RewriteRecursion::MutateAndContinue)
    }

    fn mutate(&mut self, node: Self::Node, _arena: &mut Self::Arena) -> PolarsResult<Self::Node>;
}
