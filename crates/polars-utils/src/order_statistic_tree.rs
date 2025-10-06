//! This module implements an order statistic multiset, which is implemented
//! as a weight-balanced tree (WBT).
//! It is based on the weight-balanced tree as described in:
//!
//! > Yoichi Hirai, Kazuhiko Yamamoto, "Balancing weight-balanced trees",
//! > JFP 21(3): 287–307, 2011, Cambridge University Press,
//! > https://doi.org/10.1017/S0956796811000104

use std::cmp::Ordering;
use std::fmt::Debug;

use slotmap::{Key as SlotMapKey, SlotMap, new_key_type};

// Balance parameters suggested by https://doi.org/10.1137/1.9781611976007.13
const DELTA: usize = 3;
const GAMMA: usize = 2;

type CompareFn<T> = fn(&T, &T) -> Ordering;

new_key_type! {
    struct Key;
}

#[derive(Debug)]
struct Node<T> {
    value: T,
    left: Key,
    right: Key,
    weight: usize,
    unique_weight: usize,
}

pub struct OrderStatisticTree<T> {
    sm: SlotMap<Key, Node<T>>,
    root: Key,
    compare: CompareFn<T>,
}

impl<'a, T> std::fmt::Debug for OrderStatisticTree<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OrderStatisticTree {{")?;
        writeln!(f, "  root: {:?}", self.root)?;
        for (k, v) in self.sm.iter() {
            writeln!(f, "  key: {:?}, value: {:?}", k, v)?;
        }
        Ok(())
    }
}

impl<'a, T: Debug> OrderStatisticTree<T> {
    pub fn new(compare: CompareFn<T>) -> Self {
        OrderStatisticTree {
            sm: SlotMap::<Key, Node<T>>::with_key(),
            root: Key::null(),
            compare,
        }
    }

    pub fn with_capacity(capacity: usize, compare: CompareFn<T>) -> Self {
        OrderStatisticTree {
            sm: SlotMap::<Key, Node<T>>::with_capacity_and_key(capacity),
            root: Key::null(),
            compare,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.weight(self.root)
    }

    pub fn unique_count(&self) -> usize {
        self.unique_weight(self.root)
    }

    fn eq(&self, lhs: &T, rhs: &T) -> bool {
        (self.compare)(lhs, rhs) == Ordering::Equal
    }

    fn weight(&self, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        self.sm[tree].weight
    }

    fn unique_weight(&self, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        self.sm[tree].unique_weight
    }

    fn new_node(&mut self, left: Key, value: T, right: Key) -> Key {
        let weight = self.weight(left) + 1 + self.weight(right);
        let leftmax = self.maximum(left);
        let rightmin = self.minimum(right);
        let left_contains_v = leftmax.map_or(false, |lm| self.eq(lm, &value));
        let right_contains_v = rightmin.map_or(false, |rm| self.eq(rm, &value));
        let unique_weight = if left_contains_v && right_contains_v {
            self.unique_weight(left) + self.unique_weight(right) - 1
        } else if left_contains_v || right_contains_v {
            self.unique_weight(left) + self.unique_weight(right)
        } else {
            self.unique_weight(left) + self.unique_weight(right) + 1
        };
        let n = Node {
            value,
            left,
            right,
            weight,
            unique_weight,
        };
        self.sm.insert(n)
    }

    fn new_leaf(&mut self, value: T) -> Key {
        self.new_node(Key::null(), value, Key::null())
    }

    fn drop_node(&mut self, tree: Key) -> Node<T> {
        self.sm.remove(tree).unwrap()
    }

    fn minimum(&self, tree: Key) -> Option<&T> {
        if tree.is_null() {
            return None;
        }
        let n = &self.sm[tree];
        if n.left.is_null() {
            return Some(&n.value);
        }
        self.minimum(n.left)
    }

    fn maximum(&self, tree: Key) -> Option<&T> {
        if tree.is_null() {
            return None;
        }
        let n = &self.sm[tree];
        if n.right.is_null() {
            return Some(&n.value);
        }
        self.maximum(n.right)
    }

    pub fn insert(&mut self, value: T) {
        self.root = self.insert_inner(value, self.root);
    }

    fn insert_inner(&mut self, value: T, tree: Key) -> Key {
        if tree.is_null() {
            return self.new_leaf(value);
        }

        let n = self.drop_node(tree);
        match (self.compare)(&value, &n.value) {
            Ordering::Less => {
                let left = self.insert_inner(value, n.left);
                self.balance_r(left, n.value, n.right)
            },
            _ => {
                let right = self.insert_inner(value, n.right);
                self.balance_l(n.left, n.value, right)
            },
        }
    }

    pub fn remove(&mut self, value: &T) -> Option<T> {
        let deleted;
        (deleted, self.root) = self.remove_inner(value, self.root);
        deleted
    }

    fn remove_inner(&mut self, value: &T, tree: Key) -> (Option<T>, Key) {
        if tree.is_null() {
            return (None, tree);
        }
        let n = self.drop_node(tree);
        match (self.compare)(&value, &n.value) {
            Ordering::Less => {
                let (deleted, left) = self.remove_inner(value, n.left);
                (deleted, self.balance_l(left, n.value, n.right))
            },
            Ordering::Greater => {
                let (deleted, right) = self.remove_inner(value, n.right);
                (deleted, self.balance_r(n.left, n.value, right))
            },
            Ordering::Equal => (Some(n.value), self.glue(n.left, n.right)),
        }
    }

    fn glue(&mut self, left: Key, right: Key) -> Key {
        if left.is_null() {
            return right;
        } else if right.is_null() {
            return left;
        } else if self.weight(left) > self.weight(right) {
            let (deleted, left) = self.remove_max(left);
            self.balance_r(left, deleted.unwrap(), right)
        } else {
            let (deleted, right) = self.remove_min(right);
            self.balance_l(left, deleted.unwrap(), right)
        }
    }

    fn remove_min(&mut self, tree: Key) -> (Option<T>, Key) {
        if tree.is_null() {
            return (None, tree);
        }
        let n = self.drop_node(tree);
        if n.left.is_null() {
            return (Some(n.value), n.right);
        }
        let (deleted, left) = self.remove_min(n.left);
        (deleted, self.balance_l(left, n.value, n.right))
    }

    fn remove_max(&mut self, tree: Key) -> (Option<T>, Key) {
        if tree.is_null() {
            return (None, tree);
        }
        let n = self.drop_node(tree);
        if n.right.is_null() {
            return (Some(n.value), n.left);
        }
        let (deleted, right) = self.remove_max(n.right);
        (deleted, self.balance_r(n.left, n.value, right))
    }

    pub fn contains(&self, value: &T) -> bool {
        self._contains(value, self.root)
    }

    fn _contains(&self, value: &T, tree: Key) -> bool {
        if tree.is_null() {
            return false;
        }
        let n = &self.sm[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self._contains(value, n.left),
            Ordering::Equal => true,
            Ordering::Greater => self._contains(value, n.right),
        }
    }

    fn balance_l(&mut self, left: Key, value: T, right: Key) -> Key {
        if self.pair_is_balanced(left, right) {
            return self.new_node(left, value, right);
        }
        self.rotate_l(left, value, right)
    }

    fn rotate_l(&mut self, left: Key, value: T, right: Key) -> Key {
        let r = &self.sm[right];
        if self.is_single(r.left, r.right) {
            self.single_l(left, value, right)
        } else {
            self.double_l(left, value, right)
        }
    }

    fn single_l(&mut self, left: Key, value: T, right: Key) -> Key {
        let r = self.drop_node(right);
        let new_left = self.new_node(left, value, r.left);
        self.new_node(new_left, r.value, r.right)
    }

    fn double_l(&mut self, left: Key, value: T, right: Key) -> Key {
        let r = self.drop_node(right);
        let rl = self.drop_node(r.left);
        let new_left = self.new_node(left, value, rl.left);
        let new_right = self.new_node(rl.right, r.value, r.right);
        self.new_node(new_left, rl.value, new_right)
    }

    fn balance_r(&mut self, left: Key, value: T, right: Key) -> Key {
        if self.pair_is_balanced(right, left) {
            return self.new_node(left, value, right);
        }
        self.rotate_r(left, value, right)
    }

    fn rotate_r(&mut self, left: Key, value: T, right: Key) -> Key {
        let l = &self.sm[left];
        if self.is_single(l.right, l.left) {
            self.single_r(left, value, right)
        } else {
            self.double_r(left, value, right)
        }
    }

    fn single_r(&mut self, left: Key, value: T, right: Key) -> Key {
        let l = self.drop_node(left);
        let new_right = self.new_node(l.right, value, right);
        self.new_node(l.left, l.value, new_right)
    }

    fn double_r(&mut self, left: Key, value: T, right: Key) -> Key {
        let l = self.drop_node(left);
        let lr = self.drop_node(l.right);
        let new_right = self.new_node(lr.right, value, right);
        let new_left = self.new_node(l.left, l.value, lr.left);
        self.new_node(new_left, lr.value, new_right)
    }

    #[doc(hidden)]
    pub fn is_balanced(&self) -> bool {
        self.tree_is_balanced(self.root)
    }

    fn tree_is_balanced(&self, tree: Key) -> bool {
        if tree.is_null() {
            return true;
        }
        let n = &self.sm[tree];
        self.pair_is_balanced(n.left, n.right)
            && self.pair_is_balanced(n.right, n.left)
            && self.tree_is_balanced(n.left)
            && self.tree_is_balanced(n.right)
    }

    fn pair_is_balanced(&self, left: Key, right: Key) -> bool {
        let a = self.weight(left);
        let b = self.weight(right);
        DELTA * (a + 1) >= (b + 1) && DELTA * (b + 1) >= (a + 1)
    }

    fn is_single(&self, left: Key, right: Key) -> bool {
        let a = self.weight(left);
        let b = self.weight(right);
        a + 1 < GAMMA * (b + 1)
    }

    pub fn rank_lower(&self, bound: &T) -> Result<usize, usize> {
        self.rank_lower_inner(bound, self.root)
    }

    fn rank_lower_inner(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(1);
        }
        let n = &self.sm[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self.rank_lower_inner(value, n.left),
            Ordering::Equal => self
                .rank_lower_inner(value, n.left)
                .or(Ok(self.weight(n.left) + 1)),
            Ordering::Greater => self
                .rank_lower_inner(value, n.right)
                .map(|rank| self.weight(n.left) + 1 + rank)
                .map_err(|rank| self.weight(n.left) + 1 + rank),
        }
    }

    pub fn rank_upper(&self, bound: &T) -> Result<usize, usize> {
        self._rank_upper(bound, self.root)
    }

    fn _rank_upper(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(self.weight(tree) + 1);
        }
        let n = &self.sm[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self._rank_upper(value, n.left),
            Ordering::Equal => self
                ._rank_upper(value, n.right)
                .map(|rank| self.weight(n.left) + 1 + rank)
                .map_err(|rank| self.weight(n.left) + 1 + rank)
                .or(Ok(self.weight(n.left) + 1)),
            Ordering::Greater => self
                ._rank_upper(value, n.right)
                .map(|rank| self.weight(n.left) + 1 + rank)
                .map_err(|rank| self.weight(n.left) + 1 + rank),
        }
    }

    pub fn rank_unique(&self, value: &T) -> Result<usize, usize> {
        self.rank_unique_inner(value, self.root)
    }

    fn rank_unique_inner(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(1);
        }
        let n = &self.sm[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self.rank_unique_inner(value, n.left),
            Ordering::Equal if self._contains(value, n.left) => Ok(self.unique_weight(n.left)),
            Ordering::Equal => Ok(self.unique_weight(n.left) + 1),
            Ordering::Greater => self
                .rank_unique_inner(value, n.right)
                .map(|rank| self.unique_weight(tree) - self.unique_weight(n.right) + rank)
                .map_err(|rank| self.unique_weight(tree) - self.unique_weight(n.right) + rank),
        }
    }

    pub fn iter(&'a self) -> Iter<'a, T> {
        let capacity = usize::ilog2(self.len() + 1) as usize;
        let mut stack = Vec::with_capacity(capacity);
        stack.push(IterStackFrame {
            tree: self.root,
            state: IterNodeState::GoLeft,
        });
        Iter { tree: self, stack }
    }
}

#[derive(Debug, PartialEq)]
enum IterNodeState {
    GoLeft,
    VisitNode,
    GoRight,
}

#[derive(Debug)]
struct IterStackFrame {
    tree: Key,
    state: IterNodeState,
}

pub struct Iter<'a, T> {
    tree: &'a OrderStatisticTree<T>,
    stack: Vec<IterStackFrame>,
}

impl<'a, T: Debug> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        use IterNodeState::*;

        let mut sf = self.stack.pop()?;
        let mut n = self.tree.sm.get(sf.tree)?;
        while sf.state == GoRight && n.right.is_null() {
            sf = self.stack.pop()?;
            n = self.tree.sm.get(sf.tree)?;
        }
        if sf.state == GoRight {
            self.stack.push(IterStackFrame {
                tree: n.right,
                state: GoLeft,
            });
            return self.next();
        }
        if sf.state == GoLeft && n.left.is_null() {
            sf.state = VisitNode;
        }
        if sf.state == VisitNode {
            self.stack.push(IterStackFrame {
                state: GoRight,
                ..sf
            });
            return Some(&n.value);
        }
        if sf.state == GoLeft {
            self.stack.push(IterStackFrame {
                state: VisitNode,
                ..sf
            });
            self.stack.push(IterStackFrame {
                tree: n.left,
                state: GoLeft,
            });
            return self.next();
        }
        unreachable!()
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
            .run(&vec(0i32..100, 0..100), test_insert_inner)
            .unwrap()
    }

    fn test_insert_inner(items: Vec<i32>) -> Result<(), TestCaseError> {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in &items {
            ost.insert(*item);
            assert!(ost.is_balanced());
        }
        assert_eq!(ost.len(), items.len());
        let mut sorted_items = items.clone();
        sorted_items.sort();
        let collected_items: Vec<_> = ost.iter().cloned().collect();
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
                expected_rank_lower = Ok(items.iter().filter(|&x| *x < item).count() + 1);
                expected_rank_upper = Ok(items.iter().filter(|&x| *x <= item).count());
            } else {
                expected_rank_lower = Err(items.iter().filter(|&x| *x < item).count() + 1);
                expected_rank_upper = Err(items.iter().filter(|&x| *x <= item).count() + 1);
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
        assert_eq!(ost.unique_count(), items.len());
        for item in 0..50 {
            let unique_rank = ost.rank_unique(&item);
            let expected_unique_rank = if items.contains(&item) {
                Ok(items.iter().filter(|&x| *x < item).count() + 1)
            } else {
                Err(items.iter().filter(|&x| *x < item).count() + 1)
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
        assert_eq!(ost.unique_count(), 0);
        assert!(ost.is_balanced());
        assert!(!ost.contains(&1));
        assert_eq!(ost.rank_lower(&1), Err(1));
        assert_eq!(ost.rank_upper(&1), Err(1));
        assert_eq!(ost.rank_unique(&1), Err(1));
    }

    #[test]
    fn test_iter() {
        let mut runner = TestRunner::default();
        runner.run(&vec(0i32..50, 0..100), test_iter_inner).unwrap();
    }

    fn test_iter_inner(items: Vec<i32>) -> Result<(), TestCaseError> {
        let mut ost = OrderStatisticTree::new(i32::cmp);
        for item in &items {
            ost.insert(*item);
        }
        let mut sorted_items = items.clone();
        sorted_items.sort();
        let collected_items: Vec<_> = ost.iter().cloned().collect();
        assert_eq!(ost.len(), items.len());
        assert_eq!(&collected_items, &sorted_items);
        Ok(())
    }
}
