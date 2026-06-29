#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use crate::error::*;
use crate::relaxed_cell::RelaxedCell;

unsafe fn index_of_unchecked<T>(slice: &[T], item: &T) -> usize {
    (item as *const _ as usize - slice.as_ptr() as usize) / size_of::<T>()
}

fn index_of<T>(slice: &[T], item: &T) -> Option<usize> {
    debug_assert!(size_of::<T>() > 0);
    let ptr = item as *const T;
    unsafe {
        if slice.as_ptr() < ptr && slice.as_ptr().add(slice.len()) > ptr {
            Some(index_of_unchecked(slice, item))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct Node(pub usize);

impl Default for Node {
    fn default() -> Self {
        Node(usize::MAX)
    }
}

static ARENA_VERSION: RelaxedCell<u32> = RelaxedCell::new_u32(0);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct Arena<T> {
    version: u32,
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Arena implementation
/// Allocates memory and stores item in a Vec. Only deallocates when being dropped itself.
impl<T> Arena<T> {
    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn add(&mut self, val: T) -> Node {
        let idx = self.items.len();
        self.items.push(val);
        Node(idx)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub fn last_node(&mut self) -> Option<Node> {
        if self.is_empty() {
            None
        } else {
            Some(Node(self.items.len() - 1))
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn new() -> Self {
        Arena {
            items: vec![],
            version: ARENA_VERSION.fetch_add(1),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Arena {
            items: Vec::with_capacity(cap),
            version: ARENA_VERSION.fetch_add(1),
        }
    }

    pub fn get_node(&self, val: &T) -> Option<Node> {
        index_of(&self.items, val).map(Node)
    }

    pub fn swap(&mut self, idx_a: Node, idx_b: Node) {
        self.items.swap(idx_a.0, idx_b.0)
    }

    #[inline]
    pub fn get(&self, idx: Node) -> &T {
        self.items.get(idx.0).unwrap()
    }

    /// # Safety
    /// Doesn't do any bound checks
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: Node) -> &T {
        unsafe { self.items.get_unchecked(idx.0) }
    }

    #[inline]
    pub fn get_mut(&mut self, idx: Node) -> &mut T {
        self.items.get_mut(idx.0).unwrap()
    }

    /// Get mutable references to multiple disjoint items of the Arena.
    ///
    /// # Panics
    /// Panics if indices are out of bounds or overlapping.
    #[inline]
    pub fn get_disjoint_mut<const N: usize>(&mut self, nodes: [Node; N]) -> [&mut T; N] {
        self.items.get_disjoint_mut(nodes.map(|n| n.0)).unwrap()
    }

    #[inline]
    pub fn replace(&mut self, idx: Node, val: T) -> T {
        let x = self.get_mut(idx);
        std::mem::replace(x, val)
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.version = ARENA_VERSION.fetch_add(1);
    }
}

impl<T: Clone> Arena<T> {
    pub fn duplicate(&mut self, node: Node) -> Node {
        let item = self.items[node.0].clone();
        self.add(item)
    }
}

impl<T: Default> Arena<T> {
    #[inline]
    pub fn take(&mut self, idx: Node) -> T {
        std::mem::take(self.get_mut(idx))
    }

    pub fn replace_with<F>(&mut self, idx: Node, f: F)
    where
        F: FnOnce(T) -> T,
    {
        let val = self.take(idx);
        self.replace(idx, f(val));
    }

    pub fn try_replace_with<F>(&mut self, idx: Node, mut f: F) -> Result<()>
    where
        F: FnMut(T) -> Result<T>,
    {
        let val = self.take(idx);
        self.replace(idx, f(val)?);
        Ok(())
    }
}
