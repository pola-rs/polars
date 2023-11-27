
// h: size of half window
// k: size of window
// alpha: slice of length k

use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use polars_utils::float::IsFloat;
use polars_utils::IdxSize;
use polars_utils::sort::arg_sort_ascending;
use crate::legacy::kernels::rolling::Idx;

fn get_h(k: usize) -> usize {
    (k - 1) / 2
}

struct Block<'a, T: Copy + IsFloat> {
    k: usize,
    tail: usize,
    n_element: usize,
    alpha: &'a [T],
    pi: &'a mut [u32],
    prev: &'a mut Vec<u32>,
    next: &'a mut Vec<u32>,
    // permutation index in alpha
    m: usize,
    // index in the list
    m_index: usize

}

impl<T: Copy + Debug + IsFloat> Debug for Block<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "elements in list: {}", self.n_element)?;
        writeln!(f, "m: {}", self.m)?;
        writeln!(f, "m_index: {}", self.m_index)?;
        writeln!(f, "median: {:?}", self.alpha[self.m])?;

        let mut p = self.m as u32;
        loop {
            p = self.prev[p as usize];
            if p as usize == self.tail {
                p = self.next[p as usize];
                break
            }
        }
        let mut current = Vec::with_capacity(self.n_element);
        let start = p;
        current.push(self.alpha[p as usize]);

        loop {
            p = self.next[p as usize];
            if p as usize == self.tail {
                break
            }
            current.push(self.alpha[p as usize]);
        }

        write!(f, "current buffer sorted: [")?;
        for (i, v) in current.iter().enumerate() {
            if i == self.m_index {
                write!(f, "[{v:?}], ")?;
            } else {
                let chars = if i == self.n_element - 1 {
                    ""
                } else {
                    ", "
                };
                write!(f, "{v:?}{chars}")?;
            }
        }
        write!(f, "]")
    }
}

impl<'a, T: Copy + IsFloat + PartialOrd + Debug> Block<'a, T> {

    fn new(
        alpha: &'a [T],
        scratch: &'a mut Vec<u8>,
           prev: &'a mut Vec<u32>,
           next: &'a mut Vec<u32>,
    ) -> Self {
        let k = alpha.len();
        let pi= arg_sort_ascending(alpha, scratch);

        let m_index = k / 2;
        let m = pi[m_index] as usize;

        prev.resize(k + 1, 0 as u32);
        next.resize(k + 1, 0 as u32);
        let mut b = Self {
            k,
            pi,
            prev,
            next,
            m,
            m_index,
            n_element: k,
            tail: k,
            alpha,
        };
        b.init_links();
        b
    }

    fn init_links(&mut self) {
        let mut p = self.tail;

        for &q in self.pi.iter() {
            self.next[p as usize] = q;
            self.prev[q as usize] = p as u32;

            p = q as usize;
        }
        self.next[p as usize] = self.tail as u32;
        self.prev[self.tail] = p as u32;
        dbg!(&self.pi);
        dbg!(&self.prev, &self.next);
    }

    fn unwind(&mut self) {
        for i in (0..self.k - 1).rev() {
            self.next[self.prev[i] as usize] = self.next[i];
            self.prev[self.next[i] as usize] = self.prev[i];
        }
        self.m = self.tail;
        self.n_element = 0;
    }

    fn delete(&mut self, i: usize) {
        let delete = self.get_pair(i);
        let current = self.get_pair(self.m);
        // delete from links
        self.next[self.prev[i] as usize] = self.next[i];
        self.prev[self.next[i] as usize] = self.prev[i];

        self.n_element -= 1;

        // median index position
        let new_index = self.n_element / 2;

        let mut current_index = match dbg!(delete.partial_cmp(&current).unwrap()) {
            Ordering::Less => {
                // 1, 2, [3], 4, 5
                //    2, [3], 4, 5
                // the del changes index
                self.m_index - 1
            },
            Ordering::Greater => {
                // 1, 2, [3], 4, 5
                // 1, 2, [3], 4
                // index position remains unaffected
                self.m_index
            },
            Ordering::Equal => {
                // 1, 2, [3], 4, 5
                // 1, 2, [4], 5
                // go to next
                self.m = self.next[self.m as usize] as usize;
                self.m_index
            }
        };

        if new_index < current_index {
            current_index -= 1;
            self.m = self.prev[self.m as usize] as usize;
        }
        if new_index > current_index {
            current_index += 1;
            self.m = self.next[self.m as usize] as usize;
        }
        self.m_index = current_index;
    }

    fn undelete(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = i as IdxSize;
        self.prev[self.next[i] as usize] = i as IdxSize;

        if self.is_small(i) {
            self.m = self.prev[self.m] as usize
        }
    }

    fn advance(&mut self) {
        self.m = self.next[self.m] as usize;
        self.n_element += 1;
    }

    fn at_end(&self) -> bool {
        self.m == self.tail
    }

    fn peek(&self) -> Option<T> {
        if self.at_end() {
            None
        } else {
            Some(self.alpha[self.m as usize])
        }
    }

    fn get_pair(&self, i: usize) -> (T, usize) {
        (self.alpha[i], i)
    }

    fn is_small(&self, i: usize) -> bool {
        self.at_end() || self.get_pair(i) < self.get_pair(self.m as usize)
    }

}

// fn sort_median<T>(k: usize, b: usize, slice: &[T])
// where T: Copy + IsFloat + PartialOrd + 'static
// {
//     let mut scratch_a = vec![];
//     let mut prev_a = vec![];
//     let mut next_a = vec![];
//
//     let mut scratch_b = vec![];
//     let mut prev_b = vec![];
//     let mut next_b = vec![];
//
//     let h= get_h(k);
//     let alpha = &slice[0..k];
//
//     let mut out = Vec::with_capacity(slice.len());
//     let mut block_a = Block::new(h, &alpha[..1], &mut scratch_a, &mut prev_a, &mut next_a);
//     let mut block_b = Block::new(h, alpha, &mut scratch_b, &mut prev_b, &mut next_b);
//     out.push(block_b.peek());
//
//     for j in 1..b {
//         block_a = block_b;
//
//         let alpha = &slice[j * k..(j + 1) *k];
//         block_b = if j % 2 == 0 {
//             Block::new(h, alpha, &mut scratch_b, &mut prev_b, &mut next_b)
//         } else {
//             Block::new(h, alpha, &mut scratch_a, &mut prev_a, &mut next_a)
//         };
//
//         block_b.unwind();
//         debug_assert_eq!(block_a.counter, h);
//         debug_assert_eq!(block_b.counter, h);
//
//         for i in 0..k {
//             block_a.delete(i);
//             block_b.undelete(i);
//             debug_assert!(block_a.counter + block_b.counter <= h);
//
//             if block_a.counter + block_b.counter <= h {
//                 if block_a.peek() <= block_b.peek() {
//                     block_a.advance()
//                 } else {
//                     block_b.advance()
//                 }
//             }
//             debug_assert_eq!(block_a.counter + block_b.counter, h)
//             out.push(std::cmp::min(block_a.peek(), block_b.peek()));
//         }
//         debug_assert_eq!(block_a.counter, 0);
//         debug_assert_eq!(block_b.counter, h);
//     }
//     dbg!(out);
//
// }

mod test {
    use super::*;

    #[test]
    fn test_block() {

        let values = [2, 8, 5, 9, 1, 3, 4, 10];
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(&values, &mut scratch, &mut prev, &mut next);
        b.delete(1);
        b.delete(0);
        b.delete(5);
        // b.delete(4);
        dbg!(b);

    }
}