//! This module implements an order statistic multiset, which is implemented
//! as a weight-balanced tree (WBT).
//! It is based on the weight-balanced tree based on the following papers:
//!
//!   * <https://doi.org/10.1017/S0956796811000104>
//!   * <https://doi.org/10.1137/1.9781611976007.13>
//!
//! Each of the nodes in the tree contains a UnitVec of values to store
//! multiple values with the same key.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::RangeInclusive;

use slotmap::{Key as SlotMapKey, SlotMap, new_key_type};

use crate::UnitVec;

const DELTA: usize = 3;
const GAMMA: usize = 2;

type CompareFn<T> = fn(&T, &T) -> Ordering;

new_key_type! {
    struct Key;
}

#[derive(Debug)]
struct Node<T> {
    values: UnitVec<T>,
    left: Key,
    right: Key,
    weight: u32,
    num_elems: u32,
}

#[derive(Debug)]
pub struct OrderStatisticTree<T> {
    nodes: SlotMap<Key, Node<T>>,
    root: Key,
    compare: CompareFn<T>,
}

impl<T> OrderStatisticTree<T> {
    #[inline]
    pub fn new(compare: CompareFn<T>) -> Self {
        OrderStatisticTree {
            nodes: SlotMap::with_key(),
            root: Key::null(),
            compare,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize, compare: CompareFn<T>) -> Self {
        OrderStatisticTree {
            nodes: SlotMap::with_capacity_and_key(capacity),
            root: Key::null(),
            compare,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_elems(self.root)
    }

    #[inline]
    pub fn unique_len(&self) -> usize {
        self.tree_weight(self.root)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = Key::null();
    }

    /// Returns the total number of elements in the tree rooted at `tree`.
    fn num_elems(&self, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        unsafe { self.nodes.get_unchecked(tree) }.num_elems as usize
    }

    /// Returns the number of tree nodes, which is equal to the number of unique
    /// elements, in the tree rooted at `tree`.
    fn tree_weight(&self, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        unsafe { self.nodes.get_unchecked(tree) }.weight as usize
    }

    #[must_use]
    fn new_tree_node(&mut self, left: Key, values: UnitVec<T>, right: Key) -> Key {
        let weight = self.tree_weight(left) + self.tree_weight(right) + 1;
        let num_elems = self.num_elems(left) + self.num_elems(right) + values.len();
        let n = Node {
            values,
            left,
            right,
            weight: weight as u32,
            num_elems: num_elems as u32,
        };
        self.nodes.insert(n)
    }

    #[must_use]
    fn new_leaf(&mut self, value: T) -> Key {
        let mut uv = UnitVec::new();
        uv.push(value);
        self.new_tree_node(Key::null(), uv, Key::null())
    }

    #[must_use]
    unsafe fn drop_tree_node(&mut self, tree: Key) -> Node<T> {
        unsafe { self.nodes.remove(tree).unwrap_unchecked() }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        self._get(idx, self.root)
    }

    fn _get(&self, idx: usize, tree: Key) -> Option<&T> {
        if tree.is_null() {
            return None;
        }

        let n = unsafe { self.nodes.get_unchecked(tree) };
        let own_elems = self.num_elems(tree);
        let left_elems = self.num_elems(n.left);
        let right_elems = self.num_elems(n.right);

        if idx < left_elems {
            self._get(idx, n.left)
        } else if idx >= own_elems - right_elems {
            self._get(idx - (own_elems - right_elems), n.right)
        } else {
            n.values.get(idx - left_elems)
        }
    }

    #[inline]
    pub fn insert(&mut self, value: T) {
        (self.root, _) = self._insert(value, self.root);
    }

    #[must_use]
    fn _insert(&mut self, value: T, tree: Key) -> (Key, bool) {
        if tree.is_null() {
            return (self.new_leaf(value), true);
        }

        let n = unsafe { self.nodes.get_unchecked(tree) };
        match (self.compare)(&value, &n.values[0]) {
            Ordering::Less => {
                let (left, node_added) = self._insert(value, n.left);
                let n = unsafe { self.nodes.get_unchecked_mut(tree) };
                n.left = left;
                n.weight += node_added as u32;
                n.num_elems += 1;
                (self.balance_r(tree), node_added)
            },
            Ordering::Equal => {
                let n = unsafe { self.nodes.get_unchecked_mut(tree) };
                n.values.push(value);
                n.num_elems += 1;
                (tree, false)
            },
            Ordering::Greater => {
                let (right, node_added) = self._insert(value, n.right);
                let n = unsafe { self.nodes.get_unchecked_mut(tree) };
                n.right = right;
                n.weight += node_added as u32;
                n.num_elems += 1;
                (self.balance_l(tree), node_added)
            },
        }
    }

    #[inline]
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let deleted;
        (deleted, self.root, _) = self._remove(value, self.root);
        deleted
    }

    #[must_use]
    fn _remove(&mut self, value: &T, tree: Key) -> (Option<T>, Key, bool) {
        if tree.is_null() {
            return (None, tree, false);
        }

        let n = unsafe { self.nodes.get_unchecked(tree) };
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => {
                let (deleted, left, node_removed) = self._remove(value, n.left);
                let n = unsafe { self.nodes.get_unchecked_mut(tree) };
                n.left = left;
                n.weight -= node_removed as u32;
                n.num_elems -= deleted.is_some() as u32;
                (deleted, self.balance_l(tree), node_removed)
            },
            Ordering::Greater => {
                let (deleted, right, node_removed) = self._remove(value, n.right);
                let n = unsafe { self.nodes.get_unchecked_mut(tree) };
                n.right = right;
                n.weight -= node_removed as u32;
                n.num_elems -= deleted.is_some() as u32;
                (deleted, self.balance_r(tree), node_removed)
            },
            Ordering::Equal if n.values.len() > 1 => {
                let n = unsafe { self.nodes.get_unchecked_mut(tree) };
                let popped_value = unsafe { n.values.pop().unwrap_unchecked() };
                n.num_elems -= 1;
                (Some(popped_value), tree, false)
            },
            Ordering::Equal => {
                let mut n = unsafe { self.drop_tree_node(tree) };
                (
                    Some(unsafe { n.values.pop().unwrap_unchecked() }),
                    self.glue(n.left, n.right),
                    true,
                )
            },
        }
    }

    #[must_use]
    fn glue(&mut self, left: Key, right: Key) -> Key {
        if left.is_null() {
            right
        } else if right.is_null() {
            left
        } else if self.tree_weight(left) > self.tree_weight(right) {
            let (deleted, left) = self.remove_max(left);
            let tree = self.new_tree_node(left, deleted, right);
            self.balance_r(tree)
        } else {
            let (deleted, right) = self.remove_min(right);
            let tree = self.new_tree_node(left, deleted, right);
            self.balance_l(tree)
        }
    }

    #[must_use]
    fn remove_min(&mut self, tree: Key) -> (UnitVec<T>, Key) {
        debug_assert!(!tree.is_null());
        let n = unsafe { self.nodes.get_unchecked(tree) };
        if n.left.is_null() {
            let n = unsafe { self.drop_tree_node(tree) };
            return (n.values, n.right);
        }
        let (deleted, left) = self.remove_min(n.left);
        let n = unsafe { self.nodes.get_unchecked_mut(tree) };
        n.left = left;
        n.weight -= 1;
        n.num_elems -= deleted.len() as u32;
        (deleted, self.balance_l(tree))
    }

    #[must_use]
    fn remove_max(&mut self, tree: Key) -> (UnitVec<T>, Key) {
        debug_assert!(!tree.is_null());
        let n = unsafe { self.nodes.get_unchecked(tree) };
        if n.right.is_null() {
            let n = unsafe { self.drop_tree_node(tree) };
            return (n.values, n.left);
        }
        let (deleted, right) = self.remove_max(n.right);
        let n = unsafe { self.nodes.get_unchecked_mut(tree) };
        n.right = right;
        n.weight -= 1;
        n.num_elems -= deleted.len() as u32;
        (deleted, self.balance_r(tree))
    }

    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self._contains(value, self.root)
    }

    fn _contains(&self, value: &T, tree: Key) -> bool {
        if tree.is_null() {
            return false;
        }
        let n = unsafe { self.nodes.get_unchecked(tree) };
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._contains(value, n.left),
            Ordering::Equal => true,
            Ordering::Greater => self._contains(value, n.right),
        }
    }

    #[must_use]
    fn balance_l(&mut self, tree: Key) -> Key {
        let n = unsafe { self.nodes.get_unchecked(tree) };
        if self.pair_is_balanced(n.left, n.right) {
            return tree;
        }
        self.rotate_l(tree)
    }

    #[must_use]
    fn rotate_l(&mut self, tree: Key) -> Key {
        let n = unsafe { self.nodes.get_unchecked(tree) };
        let r = unsafe { self.nodes.get_unchecked(n.right) };
        if self.is_single(r.left, r.right) {
            self.single_l(tree)
        } else {
            self.double_l(tree)
        }
    }

    #[must_use]
    fn single_l(&mut self, tree: Key) -> Key {
        let n = unsafe { self.drop_tree_node(tree) };
        let r = unsafe { self.drop_tree_node(n.right) };
        let new_left = self.new_tree_node(n.left, n.values, r.left);
        self.new_tree_node(new_left, r.values, r.right)
    }

    #[must_use]
    fn double_l(&mut self, tree: Key) -> Key {
        let n = unsafe { self.drop_tree_node(tree) };
        let r = unsafe { self.drop_tree_node(n.right) };
        let rl = unsafe { self.drop_tree_node(r.left) };
        let new_left = self.new_tree_node(n.left, n.values, rl.left);
        let new_right = self.new_tree_node(rl.right, r.values, r.right);
        self.new_tree_node(new_left, rl.values, new_right)
    }

    #[must_use]
    fn balance_r(&mut self, tree: Key) -> Key {
        let n = unsafe { self.nodes.get_unchecked(tree) };
        if self.pair_is_balanced(n.right, n.left) {
            return tree;
        }
        self.rotate_r(tree)
    }

    #[must_use]
    fn rotate_r(&mut self, tree: Key) -> Key {
        let n = unsafe { self.nodes.get_unchecked(tree) };
        let l = unsafe { self.nodes.get_unchecked(n.left) };
        if self.is_single(l.right, l.left) {
            self.single_r(tree)
        } else {
            self.double_r(tree)
        }
    }

    #[must_use]
    fn single_r(&mut self, tree: Key) -> Key {
        let n = unsafe { self.drop_tree_node(tree) };
        let l = unsafe { self.drop_tree_node(n.left) };
        let new_right = self.new_tree_node(l.right, n.values, n.right);
        self.new_tree_node(l.left, l.values, new_right)
    }

    #[must_use]
    fn double_r(&mut self, tree: Key) -> Key {
        let n = unsafe { self.drop_tree_node(tree) };
        let l = unsafe { self.drop_tree_node(n.left) };
        let lr = unsafe { self.drop_tree_node(l.right) };
        let new_right = self.new_tree_node(lr.right, n.values, n.right);
        let new_left = self.new_tree_node(l.left, l.values, lr.left);
        self.new_tree_node(new_left, lr.values, new_right)
    }

    #[doc(hidden)]
    pub fn is_balanced(&self) -> bool {
        self.tree_is_balanced(self.root)
    }

    fn tree_is_balanced(&self, tree: Key) -> bool {
        if tree.is_null() {
            return true;
        }
        let n = unsafe { self.nodes.get_unchecked(tree) };
        self.pair_is_balanced(n.left, n.right)
            && self.pair_is_balanced(n.right, n.left)
            && self.tree_is_balanced(n.left)
            && self.tree_is_balanced(n.right)
    }

    fn pair_is_balanced(&self, left: Key, right: Key) -> bool {
        let a = self.tree_weight(left);
        let b = self.tree_weight(right);
        DELTA * (a + 1) >= (b + 1) && DELTA * (b + 1) >= (a + 1)
    }

    fn is_single(&self, left: Key, right: Key) -> bool {
        let a = self.tree_weight(left);
        let b = self.tree_weight(right);
        a + 1 < GAMMA * (b + 1)
    }

    #[inline]
    pub fn rank_range(&self, bound: &T) -> Result<RangeInclusive<usize>, usize> {
        self._rank_range(bound, self.root)
    }

    fn _rank_range(&self, value: &T, tree: Key) -> Result<RangeInclusive<usize>, usize> {
        if tree.is_null() {
            return Err(0);
        }
        let n = unsafe { self.nodes.get_unchecked(tree) };
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._rank_range(value, n.left),
            Ordering::Equal => {
                let lo = self.num_elems(n.left);
                let hi = lo + n.values.len() - 1;
                Ok(lo..=hi)
            },
            Ordering::Greater => {
                let update_rank = |r| self.num_elems(tree) - self.num_elems(n.right) + r;
                self._rank_range(value, n.right)
                    .map(|rank| update_rank(*rank.start())..=update_rank(*rank.end()))
                    .map_err(update_rank)
            },
        }
    }

    #[inline]
    pub fn rank_unique(&self, value: &T) -> Result<usize, usize> {
        self._rank_unique(value, self.root)
    }

    fn _rank_unique(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(0);
        }
        let n = unsafe { self.nodes.get_unchecked(tree) };
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._rank_unique(value, n.left),
            Ordering::Equal => Ok(self.tree_weight(n.left)),
            Ordering::Greater => self
                ._rank_unique(value, n.right)
                .map(|rank| self.tree_weight(tree) - self.tree_weight(n.right) + rank)
                .map_err(|rank| self.tree_weight(tree) - self.tree_weight(n.right) + rank),
        }
    }

    #[inline]
    pub fn count(&self, value: &T) -> usize {
        self._count(value, self.root)
    }

    fn _count(&self, value: &T, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        let n = unsafe { self.nodes.get_unchecked(tree) };
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._count(value, n.left),
            Ordering::Equal => n.values.len(),
            Ordering::Greater => self._count(value, n.right),
        }
    }
}

impl<T> Extend<T> for OrderStatisticTree<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        for element in iterator {
            self.insert(element);
        }
    }
}

#[cfg(test)]
mod test {

    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::TestRunner;

    use super::*;

    #[test]
    fn test_insert() {
        let mut runner = TestRunner::default();
        runner
            .run(&vec((0i32..100, 0i32..100), 0..100), test_insert_inner)
            .unwrap()
    }

    fn test_insert_inner(items: Vec<(i32, i32)>) -> Result<(), TestCaseError> {
        let cmp = |a: &(i32, i32), b: &(i32, i32)| i32::cmp(&a.0, &b.0);
        let mut ost = OrderStatisticTree::new(cmp);
        for item in &items {
            ost.insert(*item);
            assert!(ost.is_balanced());
        }
        assert_eq!(ost.len(), items.len());
        let mut sorted_items = items.clone();
        sorted_items.sort();
        let mut collected_items = Vec::new();
        let mut i = 0;
        while let Some(v) = ost.get(i) {
            collected_items.push(*v);
            i += 1;
        }
        collected_items.sort();
        assert_eq!(ost.len(), items.len());
        assert_eq!(&collected_items, &sorted_items);
        Ok(())
    }

    #[test]
    fn test_remove() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &(vec(0i32..100, 0..100), vec(0i32..100, 0..100)),
                test_remove_inner,
            )
            .unwrap();
    }

    fn test_remove_inner(input: (Vec<i32>, Vec<i32>)) -> Result<(), TestCaseError> {
        let (mut items, to_remove) = input;
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in &items {
            ost.insert(*item);
            assert!(ost.is_balanced());
        }
        items.sort();
        for item in &to_remove {
            let v = ost.remove(item);
            assert!(ost.is_balanced());
            let idx = items.binary_search(item);
            assert_eq!(v.is_some(), idx.is_ok());
            if let Ok(idx) = idx {
                items.remove(idx);
            }
            assert_eq!(ost.len(), items.len());
        }
        assert_eq!(ost.len(), items.len());
        for item in 0..100 {
            assert_eq!(ost.contains(&item), items.contains(&item));
        }
        Ok(())
    }

    #[test]
    fn test_rank() {
        let mut runner = TestRunner::default();
        runner
            .run(&vec(0i32..100, 0..100), test_rank_inner)
            .unwrap();
    }

    fn test_rank_inner(mut items: Vec<i32>) -> Result<(), TestCaseError> {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in &items {
            ost.insert(*item);
        }
        items.sort();
        for item in 0..100 {
            let rank = ost.rank_range(&item);

            let expected_rank = if items.contains(&item) {
                let expected_rank_lower = items.iter().filter(|&x| *x < item).count();
                let expected_rank_upper = items.iter().filter(|&x| *x <= item).count() - 1;
                Ok(expected_rank_lower..=expected_rank_upper)
            } else {
                Err(items.iter().filter(|&x| *x < item).count())
            };

            assert_eq!(rank, expected_rank);
        }
        Ok(())
    }

    #[test]
    fn test_unique_rank() {
        let mut runner = TestRunner::default();
        runner
            .run(&vec(0i32..50, 0..100), test_unique_rank_inner)
            .unwrap();
    }

    fn test_unique_rank_inner(mut items: Vec<i32>) -> Result<(), TestCaseError> {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in &items {
            ost.insert(*item);
        }
        assert_eq!(ost.len(), items.len());
        items.sort();
        items.dedup();
        assert_eq!(ost.unique_len(), items.len());
        for item in 0..50 {
            let unique_rank = ost.rank_unique(&item);
            let expected_unique_rank = if items.contains(&item) {
                Ok(items.iter().filter(|&x| *x < item).count())
            } else {
                Err(items.iter().filter(|&x| *x < item).count())
            };
            assert_eq!(unique_rank, expected_unique_rank);
        }
        Ok(())
    }

    #[test]
    fn test_empty() {
        let ost = OrderStatisticTree::<i32>::new(i32::cmp);
        assert!(ost.is_empty());
        assert_eq!(ost.len(), 0);
        assert_eq!(ost.unique_len(), 0);
        assert!(ost.is_balanced());
        assert!(!ost.contains(&1));
        assert_eq!(ost.rank_range(&1), Err(0));
        assert_eq!(ost.rank_unique(&1), Err(0));
    }

    #[test]
    fn test_clear() {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in 0..10 {
            ost.insert(item);
        }
        assert_eq!(ost.len(), 10);
        assert_eq!(ost.unique_len(), 10);
        ost.clear();
        assert!(ost.is_empty());
    }

    #[test]
    fn test_extend() {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        ost.extend(0..10);
        assert_eq!(ost.len(), 10);
        assert_eq!(ost.unique_len(), 10);
        for item in 0..10 {
            assert!(ost.contains(&item));
        }
    }

    #[test]
    fn test_count() {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in &[1, 2, 2, 3, 3, 3] {
            ost.insert(*item);
        }
        assert_eq!(ost.count(&1), 1);
        assert_eq!(ost.count(&2), 2);
        assert_eq!(ost.count(&3), 3);
        assert_eq!(ost.count(&4), 0);
    }

    #[test]
    fn test_get() {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        let mut items = [3, 1, 4, 1, 5, 9, 2, 6, 5];
        for item in items {
            ost.insert(item);
        }
        items.sort();
        for (i, item) in items.iter().enumerate() {
            assert_eq!(ost.get(i), Some(item));
        }
        assert_eq!(ost.get(items.len()), None);
    }
}
