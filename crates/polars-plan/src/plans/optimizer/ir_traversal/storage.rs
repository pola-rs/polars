use std::ops::{Deref, DerefMut};

use polars_utils::arena::{Arena, Node};

use crate::plans::IR;
use crate::traversal::tree_traversal::GetNodeInputs;

pub struct IRTraversalStorage<'a> {
    pub arena: &'a mut Arena<IR>,
    pub skip_subtree: fn(&IR) -> bool,
}

impl IRTraversalStorage<'_> {
    pub fn skip_subtree(&self, ir: &IR) -> bool {
        (self.skip_subtree)(ir)
    }
}

impl GetNodeInputs<Node> for IRTraversalStorage<'_> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        if !self.skip_subtree(self.get(key)) {
            self.arena.get_node_inputs(key, push_fn)
        }
    }
}

impl<'a> Deref for IRTraversalStorage<'a> {
    type Target = Arena<IR>;

    fn deref(&self) -> &Self::Target {
        self.arena
    }
}

impl<'a> DerefMut for IRTraversalStorage<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arena
    }
}
