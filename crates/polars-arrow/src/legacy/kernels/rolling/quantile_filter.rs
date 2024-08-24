// Combine dancing links with sort merge.
// https://arxiv.org/abs/1406.1717
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Div, Mul, Sub};

use num_traits::NumCast;
use polars_utils::index::{Bounded, Indexable, NullCount};
use polars_utils::nulls::IsNull;
use polars_utils::slice::{GetSaferUnchecked, SliceAble};
use polars_utils::sort::arg_sort_ascending;
use polars_utils::total_ord::TotalOrd;

use crate::legacy::prelude::QuantileInterpolOptions;
use crate::pushable::Pushable;
use crate::types::NativeType;

struct Block<'a, A> {
    k: usize,
    tail: usize,
    n_element: usize,
    // Values buffer
    alpha: A,
    // Permutation
    pi: &'a mut [u32],
    prev: &'a mut Vec<u32>,
    next: &'a mut Vec<u32>,
    // permutation index in alpha
    m: usize,
    // index in the list
    current_index: usize,
    nulls_in_window: usize,
}

impl<'a, A> Debug for Block<'a, A>
where
    A: Indexable,
    A::Item: Debug + Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.n_element == 0 {
            return writeln!(f, "empty block");
        }
        writeln!(f, "elements in list: {}", self.n_element)?;
        writeln!(f, "m: {}", self.m)?;
        if self.current_index != self.n_element {
            writeln!(f, "m_index: {}", self.current_index)?;
            writeln!(f, "α[m]: {:?}", self.alpha.get(self.m))?;
        } else {
            // Index is at tail, so OOB.
            writeln!(f, "m_index: tail")?;
            writeln!(f, "α[m]: tail")?;
        }

        let mut p = self.m as u32;

        // Find start.
        loop {
            p = self.prev[p as usize];
            if p as usize == self.tail {
                p = self.next[p as usize];
                break;
            }
        }

        // Find all elements from start.
        let mut current = Vec::with_capacity(self.n_element);
        for _ in 0..self.n_element {
            current.push(self.alpha.get(p as usize));
            p = self.next[p as usize];
        }

        write!(f, "current buffer sorted: [")?;
        for (i, v) in current.iter().enumerate() {
            if i == self.current_index {
                write!(f, "[{v:?}], ")?;
            } else {
                let chars = if i == self.n_element - 1 { "" } else { ", " };
                write!(f, "{v:?}{chars}")?;
            }
        }
        write!(f, "]")
    }
}

impl<'a, A> Block<'a, A>
where
    A: Indexable + Bounded + NullCount + Clone,
    <A as Indexable>::Item: TotalOrd + Copy + IsNull + Debug + 'a,
{
    fn new(
        alpha: A,
        scratch: &'a mut Vec<u8>,
        prev: &'a mut Vec<u32>,
        next: &'a mut Vec<u32>,
    ) -> Self {
        debug_assert!(!alpha.is_empty());
        let k = alpha.len();
        let pi = arg_sort_ascending((0..alpha.len()).map(|i| alpha.get(i)), scratch, alpha.len());

        let nulls_in_window = alpha.null_count();
        let m_index = k / 2;
        let m = pi[m_index] as usize;

        prev.resize(k + 1, 0);
        next.resize(k + 1, 0);
        let mut b = Self {
            k,
            pi,
            prev,
            next,
            m,
            current_index: m_index,
            n_element: k,
            tail: k,
            alpha,
            nulls_in_window,
        };
        b.init_links();
        b
    }

    fn capacity(&self) -> usize {
        self.alpha.len()
    }

    fn init_links(&mut self) {
        let mut p = self.tail;

        for &q in self.pi.iter() {
            // SAFETY: bounded by pi
            unsafe {
                *self.next.get_unchecked_release_mut(p) = q;
                *self.prev.get_unchecked_release_mut(q as usize) = p as u32;
            }

            p = q as usize;
        }
        unsafe {
            *self.next.get_unchecked_release_mut(p) = self.tail as u32;
            *self.prev.get_unchecked_release_mut(self.tail) = p as u32;
        }
    }

    unsafe fn delete_link(&mut self, i: usize) {
        if <A as Indexable>::Item::HAS_NULLS && self.alpha.get_unchecked(i).is_null() {
            self.nulls_in_window -= 1
        }

        *self
            .next
            .get_unchecked_release_mut(*self.prev.get_unchecked_release(i) as usize) =
            *self.next.get_unchecked_release(i);
        *self
            .prev
            .get_unchecked_release_mut(*self.next.get_unchecked_release(i) as usize) =
            *self.prev.get_unchecked_release(i);
    }

    unsafe fn undelete_link(&mut self, i: usize) {
        if <A as Indexable>::Item::HAS_NULLS && self.alpha.get_unchecked(i).is_null() {
            self.nulls_in_window += 1
        }

        *self
            .next
            .get_unchecked_release_mut(*self.prev.get_unchecked_release(i) as usize) = i as u32;
        *self
            .prev
            .get_unchecked_release_mut(*self.next.get_unchecked_release(i) as usize) = i as u32;
    }

    fn unwind(&mut self) {
        for i in (0..self.k).rev() {
            // SAFETY: k is upper bound
            unsafe { self.delete_link(i) }
        }
        self.m = self.tail;
        self.n_element = 0;
    }

    #[cfg(test)]
    fn set_median(&mut self) {
        // median index position
        let new_index = self.n_element / 2;
        // SAFETY: only used in tests.
        unsafe { self.traverse_to_index(new_index) }
    }

    unsafe fn traverse_to_index(&mut self, i: usize) {
        match i as i64 - self.current_index as i64 {
            0 => {
                // pass
            },
            -1 => {
                self.current_index -= 1;
                self.m = *self.prev.get_unchecked_release(self.m) as usize;
            },
            1 => self.advance(),
            i64::MIN..=0 => {
                for _ in i..self.current_index {
                    self.m = *self.prev.get_unchecked_release(self.m) as usize;
                }
                self.current_index = i;
            },
            _ => {
                for _ in self.current_index..i {
                    self.m = *self.next.get_unchecked_release(self.m) as usize;
                }
                self.current_index = i;
            },
        }
    }

    fn reverse(&mut self) {
        if self.current_index > 0 {
            self.current_index -= 1;
            self.m = unsafe { *self.prev.get_unchecked_release(self.m) as usize };
        }
    }

    fn advance(&mut self) {
        if self.current_index < self.n_element {
            self.current_index += 1;
            self.m = unsafe { *self.next.get_unchecked_release(self.m) as usize };
        }
    }

    #[cfg(test)]
    fn reset(&mut self) {
        self.current_index = 0;
        self.m = self.next[self.tail] as usize;
    }

    unsafe fn delete(&mut self, i: usize) {
        if self.at_end() {
            self.reverse()
        }
        let delete = self.get_pair(i);

        let current = self.get_pair(self.m);

        // delete from links
        self.delete_link(i);

        self.n_element -= 1;

        match delete.tot_cmp(&current) {
            Ordering::Less => {
                // 1, 2, [3], 4, 5
                //    2, [3], 4, 5
                // the del changes index
                self.current_index -= 1
            },
            Ordering::Greater => {
                // 1, 2, [3], 4, 5
                // 1, 2, [3], 4
                // index position remains unaffected
            },
            Ordering::Equal => {
                // 1, 2, [3], 4, 5
                // 1, 2, [4], 5
                // go to next position because the link was deleted
                if self.n_element >= self.current_index {
                    let next_m = *self.next.get_unchecked_release(self.m) as usize;

                    if next_m == self.tail && self.n_element > 0 {
                        // The index points to tail,  set the index in the array again.
                        self.current_index -= 1;
                        self.m = *self.prev.get_unchecked_release(self.m) as usize
                    } else {
                        self.m = *self.next.get_unchecked_release(self.m) as usize;
                    }
                } else {
                    // move to previous position because the link was deleted
                    // 1, [2],
                    // [1]
                    self.m = *self.prev.get_unchecked_release(self.m) as usize
                }
            },
        };
    }

    unsafe fn undelete(&mut self, i: usize) {
        if !self.is_empty() && self.at_end() {
            self.reverse()
        }
        // undelete from links
        self.undelete_link(i);

        if self.is_empty() {
            self.m = self.prev[self.m] as usize;
            self.n_element = 1;
            self.current_index = 0;
            return;
        }
        let added = self.get_pair(i);
        let current = self.get_pair(self.m);

        self.n_element += 1;

        match added.tot_cmp(&current) {
            Ordering::Less => {
                //    2, [3], 4, 5
                // 1, 2, [3], 4, 5
                // the addition changes index
                self.current_index += 1
            },
            Ordering::Greater => {
                // 1, 2, [3], 4
                // 1, 2, [3], 4, 5
                // index position remains unaffected
            },
            Ordering::Equal => {
                // 1, 2,      4, 5
                // 1, 2, [3], 4, 5
                // go to prev position because the link was added
                // self.m = self.prev[self.m as usize] as usize;
            },
        };
    }

    #[cfg(test)]
    fn delete_set_median(&mut self, i: usize) {
        // SAFETY: only used in testing
        unsafe { self.delete(i) };
        self.set_median()
    }

    #[cfg(test)]
    fn undelete_set_median(&mut self, i: usize) {
        // SAFETY: only used in testing
        unsafe { self.undelete(i) };
        self.set_median()
    }

    fn at_end(&self) -> bool {
        self.m == self.tail
    }

    fn is_empty(&self) -> bool {
        self.n_element == 0
    }

    fn peek(&self) -> Option<<A as Indexable>::Item> {
        if self.at_end() {
            None
        } else {
            Some(self.alpha.get(self.m))
        }
    }

    fn peek_previous(&self) -> Option<<A as Indexable>::Item> {
        let m = self.prev[self.m];
        if m == self.tail as u32 {
            None
        } else {
            Some(self.alpha.get(m as usize))
        }
    }

    fn get_pair(&self, i: usize) -> (<A as Indexable>::Item, u32) {
        unsafe { (self.alpha.get_unchecked(i), i as u32) }
    }
}

trait LenGet {
    type Item;
    fn len(&self) -> usize;

    fn get(&mut self, i: usize) -> Self::Item;

    fn null_count(&self) -> usize;
}

impl<'a, A> LenGet for &mut Block<'a, A>
where
    A: Indexable + Bounded + NullCount + Clone,
    <A as Indexable>::Item: Copy + TotalOrd + Debug + 'a,
{
    type Item = <A as Indexable>::Item;

    fn len(&self) -> usize {
        self.n_element
    }

    fn get(&mut self, i: usize) -> Self::Item {
        // ONLY PRIVATE USE
        unsafe { self.traverse_to_index(i) };
        self.peek().unwrap()
    }

    fn null_count(&self) -> usize {
        self.nulls_in_window
    }
}

struct BlockUnion<'a, A: Indexable>
where
    A::Item: TotalOrd + Copy,
{
    block_left: &'a mut Block<'a, A>,
    block_right: &'a mut Block<'a, A>,
}

impl<'a, A> BlockUnion<'a, A>
where
    A: Indexable + Bounded + NullCount + Clone,
    <A as Indexable>::Item: TotalOrd + Copy + Debug,
{
    fn new(block_left: &'a mut Block<'a, A>, block_right: &'a mut Block<'a, A>) -> Self {
        Self {
            block_left,
            block_right,
        }
    }

    unsafe fn set_state(&mut self, i: usize) {
        self.block_left.delete(i);
        self.block_right.undelete(i);
    }

    fn reverse(&mut self) {
        let left = self.block_left.peek_previous();
        let right = self.block_right.peek_previous();
        match (left, right) {
            (Some(_), None) => {
                self.block_left.reverse();
            },
            (None, Some(_)) => {
                self.block_right.reverse();
            },
            (Some(left), Some(right)) => match left.tot_cmp(&right) {
                Ordering::Equal | Ordering::Less => {
                    self.block_right.reverse();
                },
                Ordering::Greater => {
                    self.block_left.reverse();
                },
            },
            (None, None) => {},
        }
    }
}

impl<'a, A> LenGet for BlockUnion<'a, A>
where
    A: Indexable + Bounded + NullCount + Clone,
    <A as Indexable>::Item: TotalOrd + Copy + Debug,
{
    type Item = <A as Indexable>::Item;

    fn len(&self) -> usize {
        self.block_left.n_element + self.block_right.n_element
    }

    fn get(&mut self, i: usize) -> Self::Item {
        debug_assert!(i < self.block_left.len() + self.block_right.len());
        // Simple case, all elements are left.
        if self.block_right.n_element == 0 {
            unsafe { self.block_left.traverse_to_index(i) };
            return self.block_left.peek().unwrap();
        } else if self.block_left.n_element == 0 {
            unsafe { self.block_right.traverse_to_index(i) };
            return self.block_right.peek().unwrap();
        }

        // Needed: one of the block can point too far depending on what was (un)deleted in the other
        // block.
        let mut peek_index = self.block_left.current_index + self.block_right.current_index + 1;
        while i <= peek_index {
            self.reverse();
            peek_index = self.block_left.current_index + self.block_right.current_index + 1;
            if peek_index <= 1 && i <= 1 {
                break;
            }
        }

        loop {
            // Current index position of merge sort
            let s = self.block_left.current_index + self.block_right.current_index;

            let left = self.block_left.peek();
            let right = self.block_right.peek();
            match (left, right) {
                (Some(left), None) => {
                    if s == i {
                        return left;
                    }
                    // Only advance on next iteration as the state can change when a new
                    // delete/undelete occurs. So next get call we might hit a different branch.
                    self.block_left.advance();
                },
                (None, Some(right)) => {
                    if s == i {
                        return right;
                    }
                    self.block_right.advance();
                },
                (Some(left), Some(right)) => {
                    match left.tot_cmp(&right) {
                        // On equality, take the left as that one was first.
                        Ordering::Equal | Ordering::Less => {
                            if s == i {
                                return left;
                            }
                            self.block_left.advance();
                        },
                        Ordering::Greater => {
                            if s == i {
                                return right;
                            }
                            self.block_right.advance();
                        },
                    }
                },
                (None, None) => {},
            }
        }
    }

    fn null_count(&self) -> usize {
        self.block_left.nulls_in_window + self.block_right.nulls_in_window
    }
}

pub(super) trait FinishLinear {
    fn finish(proportion: f64, lower: Self, upper: Self) -> Self;
    fn finish_midpoint(lower: Self, upper: Self) -> Self;
}

impl<
        T: NativeType
            + NumCast
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Mul<Output = T>
            + Debug,
    > FinishLinear for T
{
    fn finish(proportion: f64, lower: Self, upper: Self) -> Self {
        debug_assert!(proportion >= 0.0);
        debug_assert!(proportion <= 1.0);
        let proportion: T = NumCast::from(proportion).unwrap();
        proportion * (upper - lower) + lower
    }
    fn finish_midpoint(lower: Self, upper: Self) -> Self {
        (lower + upper) / NumCast::from(2).unwrap()
    }
}

impl<T: FinishLinear> FinishLinear for Option<T> {
    fn finish(proportion: f64, lower: Self, upper: Self) -> Self {
        match (lower, upper) {
            (Some(lower), Some(upper)) => Some(T::finish(proportion, lower, upper)),
            (Some(lower), _) => Some(lower),
            (None, Some(upper)) => Some(upper),
            _ => None,
        }
    }
    fn finish_midpoint(lower: Self, upper: Self) -> Self {
        match (lower, upper) {
            (Some(lower), Some(upper)) => Some(T::finish_midpoint(lower, upper)),
            (Some(lower), _) => Some(lower),
            (None, Some(upper)) => Some(upper),
            _ => None,
        }
    }
}

struct QuantileUpdate<M: LenGet> {
    inner: M,
    quantile: f64,
    min_periods: usize,
    interpol: QuantileInterpolOptions,
}

impl<M> QuantileUpdate<M>
where
    M: LenGet,
    <M as LenGet>::Item: Default + IsNull + Copy + FinishLinear + Debug,
{
    fn new(interpol: QuantileInterpolOptions, min_periods: usize, quantile: f64, inner: M) -> Self {
        Self {
            min_periods,
            quantile,
            inner,
            interpol,
        }
    }

    fn quantile(&mut self) -> M::Item {
        // nulls are ignored in median position.
        let null_count = self.inner.null_count();
        let valid_length = self.inner.len() - null_count;

        if M::Item::HAS_NULLS && valid_length < self.min_periods {
            // Default is None
            return M::Item::default();
        }

        let valid_length_f = valid_length as f64;

        use QuantileInterpolOptions::*;
        match self.interpol {
            Linear => {
                let float_idx_top = (valid_length_f - 1.0) * self.quantile;
                let idx = float_idx_top.floor() as usize;
                let top_idx = float_idx_top.ceil() as usize;

                if idx == top_idx {
                    self.inner.get(idx + null_count)
                } else {
                    let vi = self.inner.get(idx + null_count);
                    let vj = self.inner.get(top_idx + null_count);
                    let proportion = float_idx_top - idx as f64;
                    <<M as LenGet>::Item>::finish(proportion, vi, vj)
                }
            },
            Nearest => {
                let idx = (valid_length_f * self.quantile) as usize;
                let idx = std::cmp::min(idx, valid_length - 1);
                self.inner.get(idx + null_count)
            },
            Midpoint => {
                let idx = (valid_length_f * self.quantile) as usize;
                let idx = std::cmp::min(idx, valid_length - 1);

                let top_idx = ((valid_length_f - 1.0) * self.quantile).ceil() as usize;
                if top_idx == idx {
                    self.inner.get(idx + null_count)
                } else {
                    let mid = self.inner.get(idx + null_count);
                    let mid_1 = self.inner.get(top_idx + null_count);
                    <<M as LenGet>::Item>::finish_midpoint(mid, mid_1)
                }
            },
            Lower => {
                let idx = ((valid_length_f - 1.0) * self.quantile).floor() as usize;
                let idx = std::cmp::min(idx, valid_length - 1);
                self.inner.get(idx + null_count)
            },
            Higher => {
                let idx = ((valid_length_f - 1.0) * self.quantile).ceil() as usize;
                let idx = std::cmp::min(idx, valid_length - 1);
                self.inner.get(idx + null_count)
            },
        }
    }
}

pub(super) fn rolling_quantile<A, Out: Pushable<<A as Indexable>::Item>>(
    interpol: QuantileInterpolOptions,
    min_periods: usize,
    k: usize,
    values: A,
    quantile: f64,
) -> Out
where
    A: Indexable + SliceAble + Bounded + NullCount + Clone,
    <A as Indexable>::Item: Default + TotalOrd + Copy + FinishLinear + Debug,
{
    let mut scratch_left = vec![];
    let mut prev_left = vec![];
    let mut next_left = vec![];

    let mut scratch_right = vec![];
    let mut prev_right = vec![];
    let mut next_right = vec![];

    let k = std::cmp::min(k, values.len());
    let alpha = values.slice(0..k);

    let mut out = Out::with_capacity(values.len());

    let scratch_right_ptr = &mut scratch_right as *mut Vec<u8>;
    let scratch_left_ptr = &mut scratch_left as *mut Vec<u8>;
    let prev_right_ptr = &mut prev_right as *mut Vec<_>;
    let prev_left_ptr = &mut prev_left as *mut Vec<_>;
    let next_right_ptr = &mut next_right as *mut Vec<_>;
    let next_left_ptr = &mut next_left as *mut Vec<_>;

    let n_blocks = values.len() / k;

    let mut block_left = unsafe {
        Block::new(
            alpha,
            &mut *scratch_left_ptr,
            &mut *prev_left_ptr,
            &mut *next_left_ptr,
        )
    };
    let mut block_right = unsafe {
        Block::new(
            values.slice(0..1),
            &mut *scratch_right_ptr,
            &mut *prev_right_ptr,
            &mut *next_right_ptr,
        )
    };

    let ptr_left = &mut block_left as *mut Block<'_, _>;
    let ptr_right = &mut block_right as *mut Block<'_, _>;

    block_left.unwind();

    for i in 0..block_left.capacity() {
        // SAFETY: bounded by capacity
        unsafe { block_left.undelete(i) };

        let mut mu = QuantileUpdate::new(interpol, min_periods, quantile, &mut block_left);
        out.push(mu.quantile());
    }
    for i in 1..n_blocks + 1 {
        // Block left is now completely full as it is completely filled coming from the boundary effects.
        debug_assert!(block_left.n_element == k);

        // Windows state at this point.
        //
        //  - BLOCK_LEFT -- BLOCK_RIGHT -
        // |-------------||-------------|
        //   - WINDOW -
        // |--------------|
        let end = std::cmp::min((i + 1) * k, values.len());
        let alpha = unsafe { values.slice_unchecked(i * k..end) };

        if alpha.is_empty() {
            break;
        }

        // Find the scratch that belongs to the left window that has gone out of scope
        let (scratch, prev, next) = if i % 2 == 0 {
            (scratch_left_ptr, prev_left_ptr, next_left_ptr)
        } else {
            (scratch_right_ptr, prev_right_ptr, next_right_ptr)
        };

        block_right = unsafe { Block::new(alpha, &mut *scratch, &mut *prev, &mut *next) };

        // Time reverse the rhs so we can undelete in sorted order.
        block_right.unwind();

        // Here the window will move from BLOCK_LEFT into BLOCK_RIGHT
        for j in 0..block_right.capacity() {
            unsafe {
                let mut union = BlockUnion::new(&mut *ptr_left, &mut *ptr_right);
                union.set_state(j);
                let q: <A as Indexable>::Item =
                    QuantileUpdate::new(interpol, min_periods, quantile, union).quantile();
                out.push(q);
            }
        }

        std::mem::swap(&mut block_left, &mut block_right);
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_block_1() {
        //                    0, 1, 2, 3, 4, 5, 6, 7
        let values = [2, 8, 5, 9, 1, 3, 4, 10].as_ref();
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(values, &mut scratch, &mut prev, &mut next);

        // Unwind to get temporal window
        b.unwind();

        // Insert window in the right order
        b.undelete_set_median(0);
        // [[2]]
        assert_eq!(b.peek(), Some(2));
        b.undelete_set_median(1);
        // [2, [8]]
        assert_eq!(b.peek(), Some(8));
        b.undelete_set_median(2);
        // [2, [5], 8]
        assert_eq!(b.peek(), Some(5));
        b.undelete_set_median(3);
        // [2, 5, [8], 9]
        assert_eq!(b.peek(), Some(8));
        b.undelete_set_median(4);
        // [1, 2, [5], 8, 9]
        assert_eq!(b.peek(), Some(5));
        b.undelete_set_median(5);
        // [1, 2, 3, [5], 8, 9]
        assert_eq!(b.peek(), Some(5));
        b.undelete_set_median(6);
        // [1, 2, 3, [4], 5, 8, 9]
        assert_eq!(b.peek(), Some(4));
        b.undelete_set_median(7);
        // [1, 2, 3, 4, [5], 8, 9, 10]
        assert_eq!(b.peek(), Some(5));

        // Now we will delete as the block` will leave the window.
        b.delete_set_median(0);
        // [1, 3, 4, [5], 8, 9, 10]
        assert_eq!(b.peek(), Some(5));
        b.delete_set_median(1);
        // [1, 3, 4, [5], 9, 10]
        assert_eq!(b.peek(), Some(5));
        b.delete_set_median(2);
        // [1, 3, [4],  9, 10]
        assert_eq!(b.peek(), Some(4));
        b.delete_set_median(3);
        // [1, 3, [4], 10]
        assert_eq!(b.peek(), Some(4));
        b.delete_set_median(4);
        // [3, [4], 10]
        assert_eq!(b.peek(), Some(4));
        b.delete_set_median(5);
        // [4, [10]]
        assert_eq!(b.peek(), Some(10));
        b.delete_set_median(6);
        // [[10]]
        assert_eq!(b.peek(), Some(10));
    }

    #[test]
    fn test_block_2() {
        let values = [9, 1, 2].as_ref();
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(values, &mut scratch, &mut prev, &mut next);

        b.unwind();
        b.undelete_set_median(0);
        assert_eq!(b.peek(), Some(9));
        b.undelete_set_median(1);
        assert_eq!(b.peek(), Some(9));
        b.undelete_set_median(2);
        assert_eq!(b.peek(), Some(2));
    }

    #[test]
    fn test_block_union_1() {
        let alpha_a = [10, 4, 2];
        let alpha_b = [3, 4, 1];

        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut a = Block::new(alpha_a.as_ref(), &mut scratch, &mut prev, &mut next);

        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(alpha_b.as_ref(), &mut scratch, &mut prev, &mut next);

        b.unwind();
        let mut aub = BlockUnion::new(&mut a, &mut b);
        assert_eq!(aub.len(), 3);
        // STEP 0
        // block 1:
        // i:  10, 4, 2
        // s:  2, 4, 10
        // block 2: empty
        assert_eq!(aub.get(0), 2);
        assert_eq!(aub.get(1), 4);
        assert_eq!(aub.get(2), 10);

        unsafe {
            // STEP 1
            aub.block_left.reset();
            aub.set_state(0);
            assert_eq!(aub.len(), 3);
            // block 1:
            // i:  4, 2
            // s:  2, 4
            // block 2:
            // i:  3
            // s:  3
            // union s: [2, 3, 4]
            assert_eq!(aub.get(0), 2);
            assert_eq!(aub.get(1), 3);
            assert_eq!(aub.get(2), 4);

            // STEP 2
            // i:  2
            // s:  2
            // block 2:
            // i:  3, 4
            // s:  3, 4
            // union s: [2, 3, 4]
            aub.set_state(1);
            assert_eq!(aub.get(0), 2);
            assert_eq!(aub.get(1), 3);
            assert_eq!(aub.get(2), 4);
        }
    }

    #[test]
    fn test_block_union_2() {
        let alpha_a = [3, 4, 5, 7, 3, 9, 2, 6, 9, 8].as_ref();
        let alpha_b = [2, 2, 1, 7, 5, 3, 2, 6, 1, 7].as_ref();

        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut a = Block::new(alpha_a, &mut scratch, &mut prev, &mut next);

        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(alpha_b, &mut scratch, &mut prev, &mut next);

        b.unwind();
        let mut aub = BlockUnion::new(&mut a, &mut b);
        assert_eq!(aub.len(), 10);
        // STEP 0
        // block 1:
        // i:  3, 4, 5, 7, 3, 9, 2, 6, 9, 8
        // s:  2, 3, 3, 4, 5, 6, 7, 8, 9, 9
        // block 2: empty
        assert_eq!(aub.get(0), 2);
        assert_eq!(aub.get(1), 3);
        assert_eq!(aub.get(2), 3);
        // skip a step
        assert_eq!(aub.get(4), 5);
        // skip to end
        assert_eq!(aub.get(9), 9);

        // get median
        assert_eq!(aub.get(5), 6);

        unsafe {
            // STEP 1
            aub.set_state(0);
            assert_eq!(aub.len(), 10);
            // block 1:
            // i:  4, 5, 7, 3, 9, 2, 6, 9, 8
            // s:  2, 3, 4, 5, 6, 7, 8, 9, 9
            // block 2:
            // i:  2
            // s:  2
            // union s: 2, 2, 3, 4, 5, [6], 7, 8, 9, 9
            assert_eq!(aub.get(5), 6);
            assert_eq!(aub.get(7), 8);

            // STEP 2
            aub.set_state(1);

            // Back to index 4
            aub.block_left.reset();
            aub.block_right.reset();
            assert_eq!(aub.get(4), 5);
            // block 1:
            // i:  5, 7, 3, 9, 2, 6, 9, 8
            // s:  2, 3, 5, 6, 7, 8, 9, 9
            // block 2:
            // i:  2, 2
            // s:  2, 2
            // union s: 2, 2, 3, 4, 5, [6], 7, 8, 9, 9
            assert_eq!(aub.get(5), 6);

            // STEP 3
            aub.set_state(2);
            // block 1:
            // i:  7, 3, 9, 2, 6, 9, 8
            // s:  2, 3, 6, 7, 8, 9, 9
            // block 2:
            // i:  2, 2, 1
            // s:  1, 2, 2
            // union s: 1, 2, 2, 3, 4, [6], 7, 8, 9, 9
            assert_eq!(aub.get(5), 6);

            // STEP 4
            aub.set_state(3);
            // block 1:
            // i:  3, 9, 2, 6, 9, 8
            // s:  2, 3, 6, 8, 9, 9
            // block 2:
            // i:  2, 2, 1, 7
            // s:  1, 2, 2, 7
            // union s: 1, 2, 2, 3, 4, [6], 7, 8, 9, 9
            assert_eq!(aub.get(5), 6);

            // STEP 5
            aub.set_state(4);
            // block 1:
            // i:  9, 2, 6, 9, 8
            // s:  2, 6, 8, 9, 9
            // block 2:
            // i:  2, 2, 1, 7, 5
            // s:  1, 2, 2, 5, 7
            // union s: 1, 2, 2, 2, 5, [6], 7, 8, 9, 9
            assert_eq!(aub.get(5), 6);
            assert_eq!(aub.len(), 10);

            // STEP 6
            aub.set_state(5);
            // LEFT IS phasing out
            // block 1:
            // i:  2, 6, 9, 8
            // s:  2, 6, 8, 9
            // block 2:
            // i:  2, 2, 1, 7, 5, 3
            // s:  1, 2, 2, 3, 5, 7
            // union s: 1, 2, 2, 2, 4, [5], 6, 7, 8, 9
            assert_eq!(aub.len(), 10);
            assert_eq!(aub.get(5), 5);

            // STEP 7
            aub.set_state(6);
            // block 1:
            // i:  6, 9, 8
            // s:  6, 8, 9
            // block 2:
            // i:  2, 2, 1, 7, 5, 3, 2
            // s:  1, 2, 2, 2, 3, 5, 7
            // union s: 1, 2, 2, 2, 3, [5], 6, 7, 8, 9
            assert_eq!(aub.len(), 10);
            assert_eq!(aub.get(5), 5);

            // STEP 8
            aub.set_state(7);
            // block 1:
            // i:  9, 8
            // s:  8, 9
            // block 2:
            // i:  2, 2, 1, 7, 5, 3, 2, 6
            // s:  1, 2, 2, 2, 3, 5, 6, 7
            // union s: 1, 2, 2, 2, 3, [5], 6, 7, 8, 9
            assert_eq!(aub.len(), 10);
            assert_eq!(aub.get(5), 5);

            // STEP 9
            aub.set_state(8);
            // block 1:
            // i:  8
            // s:  8
            // block 2:
            // i:  2, 2, 1, 7, 5, 3, 2, 6, 1
            // s:  1, 1, 2, 2, 2, 3, 5, 6, 7
            // union s: 1, 1, 2, 2, 2, [3], 5, 6, 7, 8
            assert_eq!(aub.len(), 10);
            assert_eq!(aub.get(5), 3);

            // STEP 10
            aub.set_state(9);
            // block 1: empty
            // block 2:
            // i:  2, 2, 1, 7, 5, 3, 2, 6, 1, 7
            // s:  1, 1, 2, 2, 2, 3, 5, 6, 7
            // union s: 1, 1, 2, 2, 2, [3], 5, 6, 7, 7
            assert_eq!(aub.len(), 10);
            assert_eq!(aub.get(5), 3);
        }
    }

    #[test]
    fn test_median_1() {
        let values = [
            2.0, 8.0, 5.0, 9.0, 1.0, 2.0, 4.0, 2.0, 4.0, 8.1, -1.0, 2.9, 1.2, 23.0,
        ]
        .as_ref();
        let out: Vec<_> = rolling_quantile(QuantileInterpolOptions::Linear, 0, 3, values, 0.5);
        let expected = [
            2.0, 5.0, 5.0, 8.0, 5.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 2.9, 1.2, 2.9,
        ];
        assert_eq!(out, expected);
        let out: Vec<_> = rolling_quantile(QuantileInterpolOptions::Linear, 0, 5, values, 0.5);
        let expected = [
            2.0, 5.0, 5.0, 6.5, 5.0, 5.0, 4.0, 2.0, 2.0, 4.0, 4.0, 2.9, 2.9, 2.9,
        ];
        assert_eq!(out, expected);
        let out: Vec<_> = rolling_quantile(QuantileInterpolOptions::Linear, 0, 7, values, 0.5);
        let expected = [
            2.0, 5.0, 5.0, 6.5, 5.0, 3.5, 4.0, 4.0, 4.0, 4.0, 2.0, 2.9, 2.9, 2.9,
        ];
        assert_eq!(out, expected);
        let out: Vec<_> = rolling_quantile(QuantileInterpolOptions::Linear, 0, 4, values, 0.5);
        let expected = [
            2.0, 5.0, 5.0, 6.5, 6.5, 3.5, 3.0, 2.0, 3.0, 4.0, 3.0, 3.45, 2.05, 2.05,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_median_2() {
        let values = [10, 10, 15, 13, 9, 5, 3, 13, 19, 15, 19].as_ref();
        let out: Vec<_> = rolling_quantile(QuantileInterpolOptions::Linear, 0, 3, values, 0.5);
        let expected = [10, 10, 10, 13, 13, 9, 5, 5, 13, 15, 19];
        assert_eq!(out, expected);
    }
}
