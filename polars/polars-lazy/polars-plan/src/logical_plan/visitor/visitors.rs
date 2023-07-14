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

    fn rewrite(self, rewriter: &mut dyn RewritingVisitor<Node = Self>) -> PolarsResult<Self> {
        let mutate_this_node = match rewriter.pre_visit(&self)? {
            RewriteRecursion::MutateAndStop => return rewriter.mutate(self),
            RewriteRecursion::Stop => return Ok(self),
            RewriteRecursion::Continue => true,
            RewriteRecursion::Skip => false,
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
        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, node: Self::Node) -> PolarsResult<Self::Node>;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_visitor() {
        struct VisitPath {
            pre_idx: usize,
            pre_stack: Vec<usize>,
            #[allow(dead_code)]
            post_idx: usize,
            post_stack: Vec<usize>,
        }

        impl VisitPath {
            fn new() -> Self {
                Self {
                    pre_idx: 0,
                    pre_stack: vec![],
                    post_idx: 0,
                    post_stack: vec![],
                }
            }
        }

        impl Visitor for VisitPath {
            type Node = AexprNode;

            fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
                self.pre_idx += 1;
                self.pre_stack.push(self.pre_idx);
                Ok(VisitRecursion::Continue)
            }

            fn post_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
                // self.post_idx += 1;
                let idx = self.pre_stack.pop().unwrap();
                self.post_stack.push(idx);
                Ok(VisitRecursion::Continue)
            }
        }

        let e = (col("f00").sum() * col("bar")).sum() + col("f00").sum();
        let mut arena = Arena::new();
        let node = to_aexpr(e, &mut arena);
        let mut visitor = VisitPath::new();

        AexprNode::with_context(node, &mut arena, |node| node.visit(&mut visitor).unwrap());

        dbg!(visitor.pre_stack);
        dbg!(visitor.post_stack);
    }
}
