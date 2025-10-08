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
        self.nodes[tree].num_elems as usize
    }

    /// Returns the number of tree nodes, which is equal to the number of unique
    /// elements, in the tree rooted at `tree`.
    fn tree_weight(&self, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        self.nodes[tree].weight as usize
    }

    fn update_weights(&mut self, tree: Key) {
        debug_assert!(!tree.is_null());
        let n = &self.nodes[tree];
        let weight = (self.tree_weight(n.left) + self.tree_weight(n.right) + 1)
            .try_into()
            .unwrap();
        let num_elems = (self.num_elems(n.left) + self.num_elems(n.right) + n.values.len())
            .try_into()
            .unwrap();
        let n = &mut self.nodes[tree];
        n.weight = weight;
        n.num_elems = num_elems;
    }

    #[must_use]
    fn new_tree_node(&mut self, left: Key, values: UnitVec<T>, right: Key) -> Key {
        let n = Node {
            values,
            left,
            right,
            weight: Default::default(),
            num_elems: Default::default(),
        };
        let tree = self.nodes.insert(n);
        self.update_weights(tree);
        tree
    }

    #[must_use]
    fn new_leaf(&mut self, value: T) -> Key {
        let mut uv = UnitVec::new();
        uv.push(value);
        self.new_tree_node(Key::null(), uv, Key::null())
    }

    #[must_use]
    fn drop_tree_node(&mut self, tree: Key) -> Node<T> {
        self.nodes.remove(tree).unwrap()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        self._get(idx, self.root)
    }

    fn _get(&self, idx: usize, tree: Key) -> Option<&T> {
        if tree.is_null() {
            return None;
        }

        let n = &self.nodes[tree];
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
        self.root = self._insert(value, self.root);
    }

    #[must_use]
    fn _insert(&mut self, value: T, tree: Key) -> Key {
        if tree.is_null() {
            return self.new_leaf(value);
        }

        let n = &self.nodes[tree];
        match (self.compare)(&value, &n.values[0]) {
            Ordering::Less => {
                let left = self._insert(value, n.left);
                self.nodes[tree].left = left;
                self.update_weights(tree);
                self.balance_r(tree)
            },
            Ordering::Equal => {
                let n = &mut self.nodes[tree];
                n.values.push(value);
                n.num_elems += 1;
                tree
            },
            Ordering::Greater => {
                let right = self._insert(value, n.right);
                self.nodes[tree].right = right;
                self.update_weights(tree);
                self.balance_l(tree)
            },
        }
    }

    #[inline]
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let deleted;
        (deleted, self.root) = self._remove(value, self.root);
        deleted
    }

    #[must_use]
    fn _remove(&mut self, value: &T, tree: Key) -> (Option<T>, Key) {
        if tree.is_null() {
            return (None, tree);
        }

        let n = &self.nodes[tree];
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => {
                let (deleted, left) = self._remove(value, n.left);
                let n = &mut self.nodes[tree];
                n.left = left;
                self.update_weights(tree);
                (deleted, self.balance_l(tree))
            },
            Ordering::Greater => {
                let (deleted, right) = self._remove(value, n.right);
                let n = &mut self.nodes[tree];
                n.right = right;
                self.update_weights(tree);
                (deleted, self.balance_r(tree))
            },
            Ordering::Equal if n.values.len() > 1 => {
                let n = &mut self.nodes[tree];
                let popped_value = n.values.pop().unwrap();
                n.num_elems -= 1;
                (Some(popped_value), tree)
            },
            Ordering::Equal => {
                let mut n = self.drop_tree_node(tree);
                (Some(n.values.pop().unwrap()), self.glue(n.left, n.right))
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
            let tree = self.new_tree_node(left, deleted.unwrap(), right);
            self.balance_r(tree)
        } else {
            let (deleted, right) = self.remove_min(right);
            let tree = self.new_tree_node(left, deleted.unwrap(), right);
            self.balance_l(tree)
        }
    }

    #[must_use]
    fn remove_min(&mut self, tree: Key) -> (Option<UnitVec<T>>, Key) {
        debug_assert!(!tree.is_null());
        if self.nodes[tree].left.is_null() {
            let n = self.drop_tree_node(tree);
            return (Some(n.values), n.right);
        }
        let (deleted, left) = self.remove_min(self.nodes[tree].left);
        self.nodes[tree].left = left;
        self.update_weights(tree);
        (deleted, self.balance_l(tree))
    }

    #[must_use]
    fn remove_max(&mut self, tree: Key) -> (Option<UnitVec<T>>, Key) {
        debug_assert!(!tree.is_null());
        if self.nodes[tree].right.is_null() {
            let n = self.drop_tree_node(tree);
            return (Some(n.values), n.left);
        }
        let (deleted, right) = self.remove_max(self.nodes[tree].right);
        self.nodes[tree].right = right;
        self.update_weights(tree);
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
        let n = &self.nodes[tree];
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._contains(value, n.left),
            Ordering::Equal => true,
            Ordering::Greater => self._contains(value, n.right),
        }
    }

    #[must_use]
    fn balance_l(&mut self, tree: Key) -> Key {
        let n = &self.nodes[tree];
        if self.pair_is_balanced(n.left, n.right) {
            return tree;
        }
        self.rotate_l(tree)
    }

    #[must_use]
    fn rotate_l(&mut self, tree: Key) -> Key {
        let n = &self.nodes[tree];
        let r = &self.nodes[n.right];
        if self.is_single(r.left, r.right) {
            self.single_l(tree)
        } else {
            self.double_l(tree)
        }
    }

    #[must_use]
    fn single_l(&mut self, tree: Key) -> Key {
        let n = self.drop_tree_node(tree);
        let r = self.drop_tree_node(n.right);
        let new_left = self.new_tree_node(n.left, n.values, r.left);
        self.new_tree_node(new_left, r.values, r.right)
    }

    #[must_use]
    fn double_l(&mut self, tree: Key) -> Key {
        let n = self.drop_tree_node(tree);
        let r = self.drop_tree_node(n.right);
        let rl = self.drop_tree_node(r.left);
        let new_left = self.new_tree_node(n.left, n.values, rl.left);
        let new_right = self.new_tree_node(rl.right, r.values, r.right);
        self.new_tree_node(new_left, rl.values, new_right)
    }

    #[must_use]
    fn balance_r(&mut self, tree: Key) -> Key {
        let n = &self.nodes[tree];
        if self.pair_is_balanced(n.right, n.left) {
            return tree;
        }
        self.rotate_r(tree)
    }

    #[must_use]
    fn rotate_r(&mut self, tree: Key) -> Key {
        let n = &self.nodes[tree];
        let l = &self.nodes[n.left];
        if self.is_single(l.right, l.left) {
            self.single_r(tree)
        } else {
            self.double_r(tree)
        }
    }

    #[must_use]
    fn single_r(&mut self, tree: Key) -> Key {
        let n = self.drop_tree_node(tree);
        let l = self.drop_tree_node(n.left);
        let new_right = self.new_tree_node(l.right, n.values, n.right);
        self.new_tree_node(l.left, l.values, new_right)
    }

    #[must_use]
    fn double_r(&mut self, tree: Key) -> Key {
        let n = self.drop_tree_node(tree);
        let l = self.drop_tree_node(n.left);
        let lr = self.drop_tree_node(l.right);
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
        let n = &self.nodes[tree];
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
    pub fn rank_lower(&self, bound: &T) -> Result<usize, usize> {
        self._rank_lower(bound, self.root)
    }

    fn _rank_lower(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(0);
        }
        let n = &self.nodes[tree];
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._rank_lower(value, n.left),
            Ordering::Equal => Ok(self.num_elems(n.left)),
            Ordering::Greater => self
                ._rank_lower(value, n.right)
                .map(|rank| self.num_elems(tree) - self.num_elems(n.right) + rank)
                .map_err(|rank| self.num_elems(tree) - self.num_elems(n.right) + rank),
        }
    }

    #[inline]
    pub fn rank_upper(&self, bound: &T) -> Result<usize, usize> {
        self._rank_upper(bound, self.root)
    }

    fn _rank_upper(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(self.num_elems(tree));
        }
        let n = &self.nodes[tree];
        match (self.compare)(value, &n.values[0]) {
            Ordering::Less => self._rank_upper(value, n.left),
            Ordering::Equal => Ok(self.num_elems(tree) - self.num_elems(n.right) - 1),
            Ordering::Greater => self
                ._rank_upper(value, n.right)
                .map(|rank| self.num_elems(tree) - self.num_elems(n.right) + rank)
                .map_err(|rank| self.num_elems(tree) - self.num_elems(n.right) + rank),
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
        let n = &self.nodes[tree];
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
        let n = &self.nodes[tree];
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
            let rank_lower = ost.rank_lower(&item);
            let rank_upper = ost.rank_upper(&item);

            let expected_rank_lower;
            let expected_rank_upper;
            if items.contains(&item) {
                expected_rank_lower = Ok(items.iter().filter(|&x| *x < item).count());
                expected_rank_upper = Ok(items.iter().filter(|&x| *x <= item).count() - 1);
            } else {
                expected_rank_lower = Err(items.iter().filter(|&x| *x < item).count());
                expected_rank_upper = Err(items.iter().filter(|&x| *x <= item).count());
            }

            assert_eq!(rank_lower, expected_rank_lower);
            assert_eq!(rank_upper, expected_rank_upper);
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
        assert_eq!(ost.rank_lower(&1), Err(0));
        assert_eq!(ost.rank_upper(&1), Err(0));
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
