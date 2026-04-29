use std::ops::{Deref, DerefMut};

use polars_utils::arena::{Arena, Node};

use crate::plans::IR;
use crate::traversal::tree_traversal::GetNodeInputs;

pub struct IRTraversalStorageMut<'a> {
    arena: &'a mut Arena<IR>,
    skip_subtree: Option<fn(&IR) -> bool>,
}

impl<'a> IRTraversalStorageMut<'a> {
    pub fn new(arena: &'a mut Arena<IR>) -> Self {
        Self {
            arena,
            skip_subtree: None,
        }
    }

    pub fn new_with_skip(arena: &'a mut Arena<IR>, skip_subtree: fn(&IR) -> bool) -> Self {
        Self {
            arena,
            skip_subtree: Some(skip_subtree),
        }
    }

    #[inline]
    pub fn skip_subtree(&self, ir: &IR) -> bool {
        self.skip_subtree.is_some_and(|f| (f)(ir))
    }
}

impl GetNodeInputs<Node> for IRTraversalStorageMut<'_> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        let ir = self.get(key);
        if !self.skip_subtree(ir) {
            for node in ir.inputs() {
                push_fn(node);
            }
        }
    }
}

impl<'a> Deref for IRTraversalStorageMut<'a> {
    type Target = Arena<IR>;

    fn deref(&self) -> &Self::Target {
        self.arena
    }
}

impl<'a> DerefMut for IRTraversalStorageMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arena
    }
}

pub struct IRTraversalStorage<'a> {
    arena: &'a Arena<IR>,
    skip_subtree: Option<fn(&IR) -> bool>,
}

impl<'a> IRTraversalStorage<'a> {
    pub fn new(arena: &'a Arena<IR>) -> Self {
        Self {
            arena,
            skip_subtree: None,
        }
    }

    pub fn new_with_skip(arena: &'a Arena<IR>, skip_subtree: fn(&IR) -> bool) -> Self {
        Self {
            arena,
            skip_subtree: Some(skip_subtree),
        }
    }

    #[inline]
    pub fn skip_subtree(&self, ir: &IR) -> bool {
        self.skip_subtree.is_some_and(|f| (f)(ir))
    }
}

impl GetNodeInputs<Node> for IRTraversalStorage<'_> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        let ir = self.get(key);
        if !self.skip_subtree(ir) {
            for node in ir.inputs() {
                push_fn(node);
            }
        }
    }
}

impl<'a> Deref for IRTraversalStorage<'a> {
    type Target = Arena<IR>;

    fn deref(&self) -> &Self::Target {
        self.arena
    }
}
