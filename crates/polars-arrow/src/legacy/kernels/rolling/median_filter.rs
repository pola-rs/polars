
// h: size of half window
// k: size of window
// alpha: slice of length k

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
    m_elemnt: usize,
    counter: usize,
    alpha: &'a [T],
    pi: &'a mut [IdxSize],
    prev: &'a mut Vec<IdxSize>,
    next: &'a mut Vec<IdxSize>,
    m: usize,

}

impl<T: Copy + Debug + IsFloat> Debug for Block<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "m: {}", self.m)?;
        writeln!(f, "median: {:?}", self.alpha[self.m])?;
        writeln!(f, "elements in list: {}", self.counter)?;
        // dbg!(&self.prev);
        // dbg!(&self.next);
        // dbg!(&self.pi);
        // dbg!(&self.alpha);

        let mut p = self.m as u32;
        loop {
            p = self.prev[p as usize];
            if p as usize == self.tail {
                p = self.next[p as usize];
                break
            }
        }
        let mut current = Vec::with_capacity(self.counter);
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
        let median_point = self.pi[self.pi[self.m] as usize] as usize;
        for (i, v) in current.iter().enumerate() {
            if i == median_point {
                write!(f, "[{v:?}], ")?;
            } else {
                let chars = if i == self.counter - 1 {
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

impl<'a, T: Copy + IsFloat + PartialOrd> Block<'a, T> {

    fn new(h: usize, alpha: &'a [T],
           scratch: &'a mut Vec<u8>,
           prev: &'a mut Vec<u32>,
           next: &'a mut Vec<u32>,
    ) -> Self {
        let k = alpha.len();
        let pi= arg_sort_ascending(alpha, scratch);
        let m = pi[h] as usize;
        prev.resize(k + 1, 0 as u32);
        next.resize(k + 1, 0 as u32);
        let mut b = Self {
            k,
            pi,
            prev,
            next,
            m,
            counter: k,
            m_elemnt: h,
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
    }

    fn unwind(&mut self) {
        for i in (0..self.k - 1).rev() {
            self.next[self.prev[i] as usize] = self.next[i];
            self.prev[self.next[i] as usize] = self.prev[i];
        }
        self.m = self.tail;
        self.counter = 0;
    }

    fn delete(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = self.next[i];
        self.prev[self.next[i] as usize] = self.prev[i];


        if self.m == i {
            // Make sure that m is still well defined
            self.m = self.next[self.m] as usize;
        }
        if self.counter > 0 {
            if self.is_small(i) {
                self.m = self.prev[self.m] as usize;
            } else {
                self.m = self.next[self.m] as usize;
            }

            self.counter -= 1;
        }

        // if self.is_small(i) {
        //     self.counter -= 1
        // } else {
        //     if self.m == i {
        //         // Make sure that m is still well defined
        //         self.m = self.next[self.m] as usize;
        //     }
        //     if self.counter > 0 {
        //         self.m = self.prev[self.m] as usize;
        //         self.counter -= 1;
        //     }
        // }
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
        self.counter += 1;
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

        let values = [2, 4, 5, 9, 1, 2, 4, 9];
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(3, &values, &mut scratch, &mut prev, &mut next);
        b.delete(1);
        // b.delete(2);
        dbg!(b);

    }
}