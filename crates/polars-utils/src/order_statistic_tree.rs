//! This module implements an order statistic multiset, which is implemented
//! as a weight-balanced tree (WBT).
//! It is based on the weight-balanced tree based on the following papers:
//!
//!   * https://doi.org/10.1017/S0956796811000104
//!   * https://doi.org/10.1137/1.9781611976007.13
//!
//! Each of the nodes in the tree contains a linked list of values to store
//! multiple values with the same key.

use std::cmp::Ordering;
use std::fmt::Debug;

use slotmap::{Key, SlotMap, new_key_type};

const DELTA: usize = 3;
const GAMMA: usize = 2;

type CompareFn<T> = fn(&T, &T) -> Ordering;

new_key_type! {
    struct TreeKey;
}

new_key_type! {
    struct ValueKey;
}

#[derive(Debug)]
struct ValueNode<T> {
    value: T,
    next: ValueKey,
}

#[derive(Debug)]
struct ValueList {
    head: ValueKey,
    len: usize,
}

#[derive(Debug)]
struct TreeNode {
    values: ValueList,
    left: TreeKey,
    right: TreeKey,
    tree_weight: u32,
    unique_weight: u32,
}

#[derive(Debug)]
pub struct OrderStatisticTree<T> {
    tree_nodes: SlotMap<TreeKey, TreeNode>,
    value_nodes: SlotMap<ValueKey, ValueNode<T>>,
    root: TreeKey,
    compare: CompareFn<T>,
}

impl<'a, T> OrderStatisticTree<T> {
    #[inline]
    pub fn new(compare: CompareFn<T>) -> Self {
        OrderStatisticTree {
            tree_nodes: SlotMap::with_key(),
            value_nodes: SlotMap::with_key(),
            root: TreeKey::null(),
            compare,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize, compare: CompareFn<T>) -> Self {
        OrderStatisticTree {
            tree_nodes: SlotMap::with_capacity_and_key(capacity),
            value_nodes: SlotMap::with_capacity_and_key(capacity),
            root: TreeKey::null(),
            compare,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tree_weight(self.root).try_into().unwrap()
    }

    #[inline]
    pub fn unique_len(&self) -> usize {
        self.tree_unique_weight(self.root).try_into().unwrap()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.tree_nodes.clear();
        self.value_nodes.clear();
        self.root = Key::null();
    }

    fn tree_weight(&self, tree: TreeKey) -> usize {
        if tree.is_null() {
            return 0;
        }
        self.tree_nodes[tree].tree_weight as usize
    }

    fn tree_unique_weight(&self, tree: TreeKey) -> usize {
        if tree.is_null() {
            return 0;
        }
        self.tree_nodes[tree].unique_weight as usize
    }

    #[must_use]
    fn new_value_node(&mut self, value: T, next: ValueKey) -> ValueKey {
        let vn = ValueNode { value, next };
        self.value_nodes.insert(vn)
    }

    #[must_use]
    fn new_value_list(&mut self, value: T) -> ValueList {
        let vl = ValueList {
            head: ValueKey::null(),
            len: 0,
        };
        self.value_list_push(vl, value)
    }

    #[must_use]
    fn value_list_push(&mut self, mut vl: ValueList, value: T) -> ValueList {
        let new_head = self.new_value_node(value, vl.head);
        vl.head = new_head;
        vl.len += 1;
        vl
    }

    #[must_use]
    fn value_list_pop(&mut self, mut vl: ValueList) -> (T, ValueList) {
        let vn = self.drop_value_node(vl.head);
        vl.head = vn.next;
        vl.len -= 1;
        (vn.value, vl)
    }

    #[must_use]
    fn new_tree_node(&mut self, left: TreeKey, values: ValueList, right: TreeKey) -> TreeKey {
        let tree_weight = self.tree_weight(left) + self.tree_weight(right) + values.len;
        let unique_weight = self.tree_unique_weight(left) + self.tree_unique_weight(right) + 1;
        let tn = TreeNode {
            values,
            left,
            right,
            tree_weight: tree_weight.try_into().unwrap(),
            unique_weight: unique_weight.try_into().unwrap(),
        };
        self.tree_nodes.insert(tn)
    }

    #[must_use]
    fn new_leaf(&mut self, value: T) -> TreeKey {
        let vl = self.new_value_list(value);
        self.new_tree_node(TreeKey::null(), vl, TreeKey::null())
    }

    #[must_use]
    fn drop_tree_node(&mut self, tree: TreeKey) -> TreeNode {
        self.tree_nodes.remove(tree).unwrap()
    }

    #[must_use]
    fn drop_value_node(&mut self, value: ValueKey) -> ValueNode<T> {
        self.value_nodes.remove(value).unwrap()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.get_inner(idx, self.root)
    }

    fn get_inner(&self, idx: usize, tree: TreeKey) -> Option<&T> {
        if tree.is_null() {
            return None;
        }

        let tn = &self.tree_nodes[tree];
        let ow = self.tree_weight(tree);
        let lw = self.tree_weight(tn.left);
        let rw = self.tree_weight(tn.right);

        if idx < lw {
            return self.get_inner(idx, tn.left);
        } else if idx >= ow - rw {
            return self.get_inner(idx - (ow - rw), tn.right);
        }

        let mut vn = &self.value_nodes[tn.values.head];
        for _ in 0..(idx - lw) {
            let next_vn = vn.next;
            if next_vn.is_null() {
                return None;
            }
            vn = &self.value_nodes[next_vn];
        }
        Some(&vn.value)
    }

    #[inline]
    pub fn insert(&mut self, value: T) {
        self.root = self.insert_inner(value, self.root);
    }

    #[must_use]
    fn insert_inner(&mut self, value: T, tree: TreeKey) -> TreeKey {
        if tree.is_null() {
            return self.new_leaf(value);
        }

        let tn = self.drop_tree_node(tree);
        let vn = &self.value_nodes[tn.values.head];
        match (self.compare)(&value, &vn.value) {
            Ordering::Less => {
                let left = self.insert_inner(value, tn.left);
                self.balance_r(left, tn.values, tn.right)
            },
            Ordering::Equal => {
                let values = self.value_list_push(tn.values, value);
                self.new_tree_node(tn.left, values, tn.right)
            },
            Ordering::Greater => {
                let right = self.insert_inner(value, tn.right);
                self.balance_l(tn.left, tn.values, right)
            },
        }
    }

    #[inline]
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let deleted;
        (deleted, self.root) = self.remove_inner(value, self.root);
        deleted
    }

    #[must_use]
    fn remove_inner(&mut self, value: &T, tree: TreeKey) -> (Option<T>, TreeKey) {
        if tree.is_null() {
            return (None, tree);
        }

        let tn = self.drop_tree_node(tree);
        let vn = &self.value_nodes[tn.values.head];
        match (self.compare)(value, &vn.value) {
            Ordering::Less => {
                let (deleted, left) = self.remove_inner(value, tn.left);
                (deleted, self.balance_l(left, tn.values, tn.right))
            },
            Ordering::Greater => {
                let (deleted, right) = self.remove_inner(value, tn.right);
                (deleted, self.balance_r(tn.left, tn.values, right))
            },
            Ordering::Equal if !vn.next.is_null() => {
                let (stored_value, new_vl) = self.value_list_pop(tn.values);
                let new_tn = self.new_tree_node(tn.left, new_vl, tn.right);
                (Some(stored_value), new_tn)
            },
            Ordering::Equal => {
                let (stored_value, _) = self.value_list_pop(tn.values);
                (Some(stored_value), self.glue(tn.left, tn.right))
            },
        }
    }

    #[must_use]
    fn glue(&mut self, left: TreeKey, right: TreeKey) -> TreeKey {
        if left.is_null() {
            right
        } else if right.is_null() {
            left
        } else if self.tree_unique_weight(left) > self.tree_unique_weight(right) {
            let (deleted, left) = self.remove_max(left);
            self.balance_r(left, deleted.unwrap(), right)
        } else {
            let (deleted, right) = self.remove_min(right);
            self.balance_l(left, deleted.unwrap(), right)
        }
    }

    #[must_use]
    fn remove_min(&mut self, tree: TreeKey) -> (Option<ValueList>, TreeKey) {
        if tree.is_null() {
            return (None, tree);
        }
        let tn = self.drop_tree_node(tree);
        if tn.left.is_null() {
            return (Some(tn.values), tn.right);
        }
        let (deleted, left) = self.remove_min(tn.left);
        (deleted, self.balance_l(left, tn.values, tn.right))
    }

    #[must_use]
    fn remove_max(&mut self, tree: TreeKey) -> (Option<ValueList>, TreeKey) {
        if tree.is_null() {
            return (None, tree);
        }
        let tn = self.drop_tree_node(tree);
        if tn.right.is_null() {
            return (Some(tn.values), tn.left);
        }
        let (deleted, right) = self.remove_max(tn.right);
        (deleted, self.balance_r(tn.left, tn.values, right))
    }

    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.contains_inner(value, self.root)
    }

    fn contains_inner(&self, value: &T, tree: TreeKey) -> bool {
        if tree.is_null() {
            return false;
        }
        let tn = &self.tree_nodes[tree];
        let vn = &self.value_nodes[tn.values.head];
        match (self.compare)(value, &vn.value) {
            Ordering::Less => self.contains_inner(value, tn.left),
            Ordering::Equal => true,
            Ordering::Greater => self.contains_inner(value, tn.right),
        }
    }

    #[must_use]
    fn balance_l(&mut self, left: TreeKey, values: ValueList, right: TreeKey) -> TreeKey {
        if self.pair_is_balanced(left, right) {
            return self.new_tree_node(left, values, right);
        }
        self.rotate_l(left, values, right)
    }

    #[must_use]
    fn rotate_l(&mut self, left: TreeKey, values: ValueList, right: TreeKey) -> TreeKey {
        let r = &self.tree_nodes[right];
        if self.is_single(r.left, r.right) {
            self.single_l(left, values, right)
        } else {
            self.double_l(left, values, right)
        }
    }

    #[must_use]
    fn single_l(&mut self, left: TreeKey, values: ValueList, right: TreeKey) -> TreeKey {
        let r = self.drop_tree_node(right);
        let new_left = self.new_tree_node(left, values, r.left);
        self.new_tree_node(new_left, r.values, r.right)
    }

    #[must_use]
    fn double_l(&mut self, left: TreeKey, value: ValueList, right: TreeKey) -> TreeKey {
        let r = self.drop_tree_node(right);
        let rl = self.drop_tree_node(r.left);
        let new_left = self.new_tree_node(left, value, rl.left);
        let new_right = self.new_tree_node(rl.right, r.values, r.right);
        self.new_tree_node(new_left, rl.values, new_right)
    }

    #[must_use]
    fn balance_r(&mut self, left: TreeKey, value: ValueList, right: TreeKey) -> TreeKey {
        if self.pair_is_balanced(right, left) {
            return self.new_tree_node(left, value, right);
        }
        self.rotate_r(left, value, right)
    }

    #[must_use]
    fn rotate_r(&mut self, left: TreeKey, value: ValueList, right: TreeKey) -> TreeKey {
        let l = &self.tree_nodes[left];
        if self.is_single(l.right, l.left) {
            self.single_r(left, value, right)
        } else {
            self.double_r(left, value, right)
        }
    }

    #[must_use]
    fn single_r(&mut self, left: TreeKey, value: ValueList, right: TreeKey) -> TreeKey {
        let l = self.drop_tree_node(left);
        let new_right = self.new_tree_node(l.right, value, right);
        self.new_tree_node(l.left, l.values, new_right)
    }

    #[must_use]
    fn double_r(&mut self, left: TreeKey, value: ValueList, right: TreeKey) -> TreeKey {
        let l = self.drop_tree_node(left);
        let lr = self.drop_tree_node(l.right);
        let new_right = self.new_tree_node(lr.right, value, right);
        let new_left = self.new_tree_node(l.left, l.values, lr.left);
        self.new_tree_node(new_left, lr.values, new_right)
    }

    #[doc(hidden)]
    pub fn is_balanced(&self) -> bool {
        self.tree_is_balanced(self.root)
    }

    fn tree_is_balanced(&self, tree: TreeKey) -> bool {
        if tree.is_null() {
            return true;
        }
        let tn = &self.tree_nodes[tree];
        self.pair_is_balanced(tn.left, tn.right)
            && self.pair_is_balanced(tn.right, tn.left)
            && self.tree_is_balanced(tn.left)
            && self.tree_is_balanced(tn.right)
    }

    fn pair_is_balanced(&self, left: TreeKey, right: TreeKey) -> bool {
        let a = self.tree_unique_weight(left);
        let b = self.tree_unique_weight(right);
        DELTA * (a + 1) >= (b + 1) && DELTA * (b + 1) >= (a + 1)
    }

    fn is_single(&self, left: TreeKey, right: TreeKey) -> bool {
        let a = self.tree_unique_weight(left);
        let b = self.tree_unique_weight(right);
        a + 1 < GAMMA * (b + 1)
    }

    #[inline]
    pub fn rank_lower(&self, bound: &T) -> Result<usize, usize> {
        self.rank_lower_inner(bound, self.root)
    }

    fn rank_lower_inner(&self, value: &T, tree: TreeKey) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(0);
        }
        let tn = &self.tree_nodes[tree];
        let vn = &self.value_nodes[tn.values.head];
        match (self.compare)(value, &vn.value) {
            Ordering::Less => self.rank_lower_inner(value, tn.left),
            Ordering::Equal => Ok(self.tree_weight(tn.left)),
            Ordering::Greater => self
                .rank_lower_inner(value, tn.right)
                .map(|rank| self.tree_weight(tree) - self.tree_weight(tn.right) + rank)
                .map_err(|rank| self.tree_weight(tree) - self.tree_weight(tn.right) + rank),
        }
    }

    #[inline]
    pub fn rank_upper(&self, bound: &T) -> Result<usize, usize> {
        self._rank_upper(bound, self.root)
    }

    fn _rank_upper(&self, value: &T, tree: TreeKey) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(self.tree_weight(tree));
        }
        let tn = &self.tree_nodes[tree];
        let vn = &self.value_nodes[tn.values.head];
        match (self.compare)(value, &vn.value) {
            Ordering::Less => self._rank_upper(value, tn.left),
            Ordering::Equal => Ok(self.tree_weight(tree) - self.tree_weight(tn.right) - 1),
            Ordering::Greater => self
                ._rank_upper(value, tn.right)
                .map(|rank| self.tree_weight(tree) - self.tree_weight(tn.right) + rank)
                .map_err(|rank| self.tree_weight(tree) - self.tree_weight(tn.right) + rank),
        }
    }

    #[inline]
    pub fn rank_unique(&self, value: &T) -> Result<usize, usize> {
        self.rank_unique_inner(value, self.root)
    }

    fn rank_unique_inner(&self, value: &T, tree: TreeKey) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(0);
        }
        let tn = &self.tree_nodes[tree];
        let vn = &self.value_nodes[tn.values.head];
        match (self.compare)(value, &vn.value) {
            Ordering::Less => self.rank_unique_inner(value, tn.left),
            Ordering::Equal => Ok(self.tree_unique_weight(tn.left)),
            Ordering::Greater => self
                .rank_unique_inner(value, tn.right)
                .map(|rank| {
                    self.tree_unique_weight(tree) - self.tree_unique_weight(tn.right) + rank
                })
                .map_err(|rank| {
                    self.tree_unique_weight(tree) - self.tree_unique_weight(tn.right) + rank
                }),
        }
    }

    #[inline]
    pub fn count(&self, value: &T) -> usize {
        let Ok(lo) = self.rank_lower(value) else {
            return 0;
        };
        let hi = self.rank_upper(value).unwrap();
        hi + 1 - lo
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
        for i in 0..items.len() {
            assert_eq!(ost.get(i), Some(&items[i]));
        }
        assert_eq!(ost.get(items.len()), None);
    }
}
