use std::borrow::Cow;

use polars_core::schema::SchemaRef;
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
        arena.replace(node, ae)
    }

    pub fn to_alp<'a>(&self, arena: &'a Arena<IR>) -> &'a IR {
         arena.get(self.node)
    }

    pub fn to_alp_mut<'a>(&mut self, arena: &'a mut Arena<IR>) -> &'a mut IR {
        arena.get_mut(self.node)
    }

    // /// Take a [`Node`] and convert it an [`IRNode`] and call
    // /// `F` with `self` and the new created [`IRNode`]
    // pub fn binary<F, T>(&self, other: Node, op: F) -> T
    // where
    //     F: FnOnce(&IRNode, &IRNode) -> T,
    // {
    //     // this is safe as we remain in context
    //     let other = unsafe { IRNode::from_raw(other, self.arena) };
    //     op(self, &other)
    // }
}

impl TreeWalker for IRNode {
    type Arena = Arena<IR>;

    fn apply_children(
        &self,
        op: &mut dyn FnMut(&Self, &mut Self::Arena) -> PolarsResult<VisitRecursion>,
        arena: &mut Self::Arena
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = unitvec![];

        self.to_alp(arena).copy_inputs(&mut scratch);
        for &node in scratch.as_slice() {
            let lp_node = IRNode::new(
                node
            );
            match op(&lp_node, arena)? {
                // let the recursion continue
                VisitRecursion::Continue | VisitRecursion::Skip => {},
                // early stop
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children(
        mut self,
        op: &mut dyn FnMut(Self, &mut Self::Arena) -> PolarsResult<Self>,
        arena: &mut Self::Arena
    ) -> PolarsResult<Self> {
        let mut inputs = vec![];
        let mut exprs = vec![];

        let lp = arena.take(self.node);
        lp.copy_inputs(&mut inputs);
        lp.copy_exprs(&mut exprs);

        // rewrite the nodes
        for node in &mut inputs {
            let lp_node = IRNode::new(
                *node,
            );
            *node = op(lp_node, arena)?.node;
        }

        let lp = lp.with_exprs_and_input(exprs, inputs);
        arena.replace(self.node, lp);
        Ok(self)
    }
}
