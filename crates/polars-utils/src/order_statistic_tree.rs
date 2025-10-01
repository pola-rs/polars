//! This module implements an order statistic multiset, which is implemented
//! as a weight-balanced tree (WBT).
//! It is based on the weight-balanced tree as described in:
//!
//! > Yoichi Hirai, Kazuhiko Yamamoto, "Balancing weight-balanced trees",
//! > JFP 21(3): 287–307, 2011, Cambridge University Press,
//! > https://doi.org/10.1017/S0956796811000104

use std::cmp::Ordering;
use std::f64;
use std::fmt::Debug;
use std::ops::Bound;

use slotmap::{Key as SlotMapKey, SlotMap, new_key_type};

const DELTA: f64 = 1.0 + f64::consts::SQRT_2;
const GAMMA: f64 = f64::consts::SQRT_2;

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

pub struct OrderStatisticTree<'a, T, CMP>
where
    CMP: Fn(&T, &T) -> Ordering,
{
    arena: SlotMap<Key, Node<T>>,
    root: Key,
    compare: &'a CMP,
}

impl<'a, T, CMP> std::fmt::Debug for OrderStatisticTree<'a, T, CMP>
where
    T: std::fmt::Debug,
    CMP: Fn(&T, &T) -> Ordering,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BinarySearchTree {{")?;
        writeln!(f, "  root: {:?}", self.root)?;
        for (k, v) in self.arena.iter() {
            writeln!(f, "  key: {:?}, value: {:?}", k, v)?;
        }
        Ok(())
    }
}

impl<'a, T: Debug + Default, CMP> OrderStatisticTree<'a, T, CMP>
where
    CMP: Fn(&T, &T) -> Ordering,
{
    pub fn new(compare: &'a CMP) -> Self {
        OrderStatisticTree {
            arena: SlotMap::<Key, Node<T>>::with_key(),
            root: Key::null(),
            compare,
        }
    }

    pub fn with_capacity(capacity: usize, compare: &'a CMP) -> Self {
        OrderStatisticTree {
            arena: SlotMap::<Key, Node<T>>::with_capacity_and_key(capacity),
            root: Key::null(),
            compare,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    pub fn count(&self) -> usize {
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
        self.arena[tree].weight
    }

    fn unique_weight(&self, tree: Key) -> usize {
        if tree.is_null() {
            return 0;
        }
        self.arena[tree].unique_weight
    }

    fn node(&mut self, left: Key, value: T, right: Key) -> Key {
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
        self.arena.insert(n)
    }

    fn leaf(&mut self, value: T) -> Key {
        self.node(Key::null(), value, Key::null())
    }

    fn minimum(&self, tree: Key) -> Option<&T> {
        if tree.is_null() {
            return None;
        }
        let n = &self.arena[tree];
        if n.left.is_null() {
            return Some(&n.value);
        }
        self.minimum(n.left)
    }

    fn maximum(&self, tree: Key) -> Option<&T> {
        if tree.is_null() {
            return None;
        }
        let n = &self.arena[tree];
        if n.right.is_null() {
            return Some(&n.value);
        }
        self.maximum(n.right)
    }

    pub fn insert(&mut self, value: T) {
        self.root = self._insert(value, self.root);
    }

    fn _insert(&mut self, value: T, tree: Key) -> Key {
        if tree.is_null() {
            return self.leaf(value);
        }

        let n = self.arena.remove(tree).unwrap();
        match (self.compare)(&value, &n.value) {
            Ordering::Less => {
                let left = self._insert(value, n.left);
                self.balance_r(left, n.value, n.right)
            },
            _ => {
                let right = self._insert(value, n.right);
                self.balance_l(n.left, n.value, right)
            },
        }
    }

    pub fn remove(&mut self, value: &T) -> Option<T> {
        let deleted;
        (deleted, self.root) = self._remove(value, self.root);
        deleted
    }

    fn _remove(&mut self, value: &T, tree: Key) -> (Option<T>, Key) {
        if tree.is_null() {
            return (None, tree);
        }
        let n = self.arena.remove(tree).unwrap();
        match (self.compare)(&value, &n.value) {
            Ordering::Less => {
                let (deleted, left) = self._remove(value, n.left);
                (deleted, self.balance_l(left, n.value, n.right))
            },
            Ordering::Greater => {
                let (deleted, right) = self._remove(value, n.right);
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
        let n = self.arena.remove(tree).unwrap();
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
        let n = self.arena.remove(tree).unwrap();
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
        let n = &self.arena[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self._contains(value, n.left),
            Ordering::Equal => true,
            Ordering::Greater => self._contains(value, n.right),
        }
    }

    fn balance_l(&mut self, left: Key, value: T, right: Key) -> Key {
        if self.pair_is_balanced(left, right) {
            return self.node(left, value, right);
        }
        self.rotate_l(left, value, right)
    }

    fn rotate_l(&mut self, left: Key, value: T, right: Key) -> Key {
        let r = &self.arena[right];
        if self.is_single(r.left, r.right) {
            self.single_l(left, value, right)
        } else {
            self.double_l(left, value, right)
        }
    }

    fn single_l(&mut self, left: Key, value: T, right: Key) -> Key {
        let r = self.arena.remove(right).unwrap();
        let new_left = self.node(left, value, r.left);
        self.node(new_left, r.value, r.right)
    }

    fn double_l(&mut self, left: Key, value: T, right: Key) -> Key {
        let r = self.arena.remove(right).unwrap();
        let rl = self.arena.remove(r.left).unwrap();
        let new_left = self.node(left, value, rl.left);
        let new_right = self.node(rl.right, r.value, r.right);
        self.node(new_left, rl.value, new_right)
    }

    fn balance_r(&mut self, left: Key, value: T, right: Key) -> Key {
        if self.pair_is_balanced(right, left) {
            return self.node(left, value, right);
        }
        self.rotate_r(left, value, right)
    }

    fn rotate_r(&mut self, left: Key, value: T, right: Key) -> Key {
        let l = &self.arena[left];
        if self.is_single(l.right, l.left) {
            self.single_r(left, value, right)
        } else {
            self.double_r(left, value, right)
        }
    }

    fn single_r(&mut self, left: Key, value: T, right: Key) -> Key {
        let l = self.arena.remove(left).unwrap();
        let new_right = self.node(l.right, value, right);
        self.node(l.left, l.value, new_right)
    }

    fn double_r(&mut self, left: Key, value: T, right: Key) -> Key {
        let l = self.arena.remove(left).unwrap();
        let lr = self.arena.remove(l.right).unwrap();
        let new_right = self.node(lr.right, value, right);
        let new_left = self.node(l.left, l.value, lr.left);
        self.node(new_left, lr.value, new_right)
    }

    #[doc(hidden)]
    pub fn is_balanced(&self) -> bool {
        self.tree_is_balanced(self.root)
    }

    fn tree_is_balanced(&self, tree: Key) -> bool {
        if tree.is_null() {
            return true;
        }
        let n = &self.arena[tree];
        self.pair_is_balanced(n.left, n.right)
            && self.pair_is_balanced(n.right, n.left)
            && self.tree_is_balanced(n.left)
            && self.tree_is_balanced(n.right)
    }

    fn pair_is_balanced(&self, left: Key, right: Key) -> bool {
        let a = self.weight(left) as f64;
        let b = self.weight(right) as f64;
        f64::abs((DELTA * (a + 1.0)) - (b + 1.0)) >= 0.0
    }

    fn is_single(&self, left: Key, right: Key) -> bool {
        let a = self.weight(left) as f64;
        let b = self.weight(right) as f64;
        f64::abs((GAMMA * (b + 1.0)) - (a + 1.0)) > 0.0
    }

    pub fn rank_lower(&self, bound: Bound<&T>) -> Result<usize, usize> {
        let bound_inner = match &bound {
            Bound::Included(b) => b,
            Bound::Excluded(b) => b,
            Bound::Unbounded => return Err(1),
        };
        let rank = self._rank_lower(bound_inner, self.root);
        match bound {
            Bound::Excluded(_) => rank.map(|x| x - 1),
            Bound::Included(_) => rank,
            Bound::Unbounded => unreachable!(),
        }
    }

    fn _rank_lower(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(1);
        }
        let n = &self.arena[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self._rank_lower(value, n.left),
            Ordering::Equal => self
                ._rank_lower(value, n.left)
                .or(Ok(self.weight(n.left) + 1)),
            Ordering::Greater => self
                ._rank_lower(value, n.right)
                .map(|rank| self.weight(n.left) + 1 + rank)
                .map_err(|rank| self.weight(n.left) + 1 + rank),
        }
    }

    pub fn rank_upper(&self, bound: Bound<&T>) -> Result<usize, usize> {
        let bound_inner = match &bound {
            Bound::Included(b) => b,
            Bound::Excluded(b) => b,
            Bound::Unbounded => return Err(self.count() + 1),
        };
        let rank = self._rank_upper(bound_inner, self.root);
        match bound {
            Bound::Excluded(_) => rank.map(|x| x + 1),
            Bound::Included(_) => rank,
            Bound::Unbounded => unreachable!(),
        }
    }

    fn _rank_upper(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(self.weight(tree) + 1);
        }
        let n = &self.arena[tree];
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

    pub fn unique_rank(&self, value: &T) -> Result<usize, usize> {
        self._unique_rank(value, self.root)
    }

    fn _unique_rank(&self, value: &T, tree: Key) -> Result<usize, usize> {
        if tree.is_null() {
            return Err(1);
        }
        let n = &self.arena[tree];
        match (self.compare)(value, &n.value) {
            Ordering::Less => self._unique_rank(value, n.left),
            Ordering::Equal if self._contains(value, n.left) => Ok(self.unique_weight(n.left)),
            Ordering::Equal => Ok(self.unique_weight(n.left) + 1),
            Ordering::Greater => self
                ._unique_rank(value, n.right)
                .map(|rank| self.unique_weight(tree) - self.unique_weight(n.right) + rank)
                .map_err(|rank| self.unique_weight(tree) - self.unique_weight(n.right) + rank),
        }
    }

    pub fn iter(&'a self) -> Iter<'a, T, CMP> {
        let capacity = usize::ilog2(self.count() + 1) as usize;
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

pub struct Iter<'a, T, CMP: Fn(&T, &T) -> Ordering> {
    tree: &'a OrderStatisticTree<'a, T, CMP>,
    stack: Vec<IterStackFrame>,
}

impl<'a, T: Debug, CMP: Fn(&T, &T) -> Ordering> Iterator for Iter<'a, T, CMP> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        use IterNodeState::*;

        let mut sf = self.stack.pop()?;
        let mut n = self.tree.arena.get(sf.tree)?;
        while sf.state == GoRight && n.right.is_null() {
            sf = self.stack.pop()?;
            n = self.tree.arena.get(sf.tree)?;
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
        let mut ost = OrderStatisticTree::new(&i32::cmp);
        for item in &items {
            ost.insert(*item);
            assert!(ost.is_balanced());
        }
        assert_eq!(ost.count(), items.len());
        let mut sorted_items = items.clone();
        sorted_items.sort();
        let collected_items: Vec<_> = ost.iter().cloned().collect();
        assert_eq!(ost.count(), items.len());
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
        let mut ost = OrderStatisticTree::new(&i32::cmp);
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
            assert_eq!(ost.count(), items.len());
        }
        assert_eq!(ost.count(), items.len());
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
        let mut ost = OrderStatisticTree::new(&i32::cmp);
        for item in &items {
            ost.insert(*item);
        }
        items.sort();
        for item in 0..100 {
            let rank_lower_inc = ost.rank_lower(Bound::Included(&item));
            let rank_lower_exc = ost.rank_lower(Bound::Excluded(&item));
            let rank_upper_inc = ost.rank_upper(Bound::Included(&item));
            let rank_upper_exc = ost.rank_upper(Bound::Excluded(&item));

            let expected_rank_lower_inc;
            let expected_rank_lower_exc;
            let expected_rank_upper_inc;
            let expected_rank_upper_exc;
            if items.contains(&item) {
                expected_rank_lower_inc = Ok(items.iter().filter(|&x| *x < item).count() + 1);
                expected_rank_lower_exc = Ok(items.iter().filter(|&x| *x < item).count());
                expected_rank_upper_inc = Ok(items.iter().filter(|&x| *x <= item).count());
                expected_rank_upper_exc = Ok(items.iter().filter(|&x| *x <= item).count() + 1);
            } else {
                expected_rank_lower_inc = Err(items.iter().filter(|&x| *x < item).count() + 1);
                expected_rank_lower_exc = expected_rank_lower_inc;
                expected_rank_upper_inc = Err(items.iter().filter(|&x| *x <= item).count() + 1);
                expected_rank_upper_exc = expected_rank_upper_inc;
            }

            assert_eq!(rank_lower_inc, expected_rank_lower_inc);
            assert_eq!(rank_lower_exc, expected_rank_lower_exc);
            assert_eq!(rank_upper_inc, expected_rank_upper_inc);
            assert_eq!(rank_upper_exc, expected_rank_upper_exc);
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
        let mut ost = OrderStatisticTree::new(&i32::cmp);
        for item in &items {
            ost.insert(*item);
        }
        assert_eq!(ost.count(), items.len());
        items.sort();
        items.dedup();
        assert_eq!(ost.unique_count(), items.len());
        for item in 0..50 {
            let unique_rank = ost.unique_rank(&item);
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
        let ost = OrderStatisticTree::<i32, _>::new(&i32::cmp);
        assert!(ost.is_empty());
        assert_eq!(ost.count(), 0);
        assert_eq!(ost.unique_count(), 0);
        assert!(ost.is_balanced());
        assert!(!ost.contains(&1));
        assert_eq!(ost.rank_lower(Bound::Included(&1)), Err(1));
        assert_eq!(ost.rank_lower(Bound::Excluded(&1)), Err(1));
        assert_eq!(ost.rank_upper(Bound::Included(&1)), Err(1));
        assert_eq!(ost.rank_upper(Bound::Excluded(&1)), Err(1));
        assert_eq!(ost.unique_rank(&1), Err(1));
    }

    #[test]
    fn test_iter() {
        let mut runner = TestRunner::default();
        runner.run(&vec(0i32..50, 0..100), test_iter_inner).unwrap();
    }

    fn test_iter_inner(items: Vec<i32>) -> Result<(), TestCaseError> {
        let mut ost = OrderStatisticTree::new(&i32::cmp);
        for item in &items {
            ost.insert(*item);
        }
        let mut sorted_items = items.clone();
        sorted_items.sort();
        let collected_items: Vec<_> = ost.iter().cloned().collect();
        assert_eq!(ost.count(), items.len());
        assert_eq!(&collected_items, &sorted_items);
        Ok(())
    }
}
