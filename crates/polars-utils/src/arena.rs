use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use crate::error::*;
use crate::slice::GetSaferUnchecked;

unsafe fn index_of_unchecked<T>(slice: &[T], item: &T) -> usize {
    (item as *const _ as usize - slice.as_ptr() as usize) / std::mem::size_of::<T>()
}

fn index_of<T>(slice: &[T], item: &T) -> Option<usize> {
    debug_assert!(std::mem::size_of::<T>() > 0);
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

static ARENA_VERSION: AtomicU32 = AtomicU32::new(0);

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

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn new() -> Self {
        Arena {
            items: vec![],
            version: ARENA_VERSION.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Arena {
            items: Vec::with_capacity(cap),
            version: ARENA_VERSION.fetch_add(1, Ordering::Relaxed),
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

    #[inline]
    /// # Safety
    /// Doesn't do any bound checks
    pub unsafe fn get_unchecked(&self, idx: Node) -> &T {
        self.items.get_unchecked_release(idx.0)
    }

    #[inline]
    pub fn get_mut(&mut self, idx: Node) -> &mut T {
        self.items.get_mut(idx.0).unwrap()
    }

    #[inline]
    /// Get mutable references to several items of the Arena
    ///
    /// The `idxs` is asserted to contain unique `Node` elements which are preferably (not
    /// necessarily) in order.
    pub fn get_many_mut<const N: usize>(&mut self, indices: [Node; N]) -> [&mut T; N] {
        // @NOTE: This implementation is adapted from the Rust Nightly Standard Library. When
        // `get_many_mut` gets stabilized we should use that.

        let len = self.items.len();

        // NB: The optimizer should inline the loops into a sequence
        // of instructions without additional branching.
        let mut valid = true;
        for (i, &idx) in indices.iter().enumerate() {
            valid &= idx.0 < len;
            for &idx2 in &indices[..i] {
                valid &= idx != idx2;
            }
        }

        assert!(valid, "Duplicate index or out-of-bounds index");

        // NB: This implementation is written as it is because any variation of
        // `indices.map(|i| self.get_unchecked_mut(i))` would make miri unhappy,
        // or generate worse code otherwise. This is also why we need to go
        // through a raw pointer here.
        let slice: *mut [T] = &mut self.items[..] as *mut _;
        let mut arr: std::mem::MaybeUninit<[&mut T; N]> = std::mem::MaybeUninit::uninit();
        let arr_ptr = arr.as_mut_ptr();

        // SAFETY: We expect `indices` to contain disjunct values that are
        // in bounds of `self`.
        unsafe {
            for i in 0..N {
                let idx = *indices.get_unchecked(i);
                *(*arr_ptr).get_unchecked_mut(i) = (*slice).get_unchecked_mut(idx.0);
            }
            arr.assume_init()
        }
    }

    #[inline]
    pub fn replace(&mut self, idx: Node, val: T) -> T {
        let x = self.get_mut(idx);
        std::mem::replace(x, val)
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.version = ARENA_VERSION.fetch_add(1, Ordering::Relaxed);
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
