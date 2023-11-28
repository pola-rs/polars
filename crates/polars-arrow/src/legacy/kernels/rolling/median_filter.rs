
// h: size of half window
// k: size of window
// alpha: slice of length k

use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use polars_utils::float::IsFloat;
use polars_utils::IdxSize;
use polars_utils::sort::arg_sort_ascending;
use crate::legacy::kernels::rolling::Idx;

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
    current_index: usize

}

impl<T: Copy + Debug + IsFloat> Debug for Block<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.n_element == 0 {
            return writeln!(f, "empty block")
        }
        writeln!(f, "elements in list: {}", self.n_element)?;
        writeln!(f, "m: {}", self.m)?;
        writeln!(f, "m_index: {}", self.current_index)?;
        writeln!(f, "α[m]: {:?}", self.alpha[self.m])?;

        let mut p = self.m as u32;

        // Find start.
        loop {
            p = self.prev[p as usize];
            if p as usize == self.tail {
                p = self.next[p as usize];
                break
            }
        }

        // Find all elements from start.
        let mut current = Vec::with_capacity(self.n_element);
        for _ in 0..self.n_element {
            current.push(self.alpha[p as usize]);
            p = self.next[p as usize];

        }

        write!(f, "current buffer sorted: [")?;
        for (i, v) in current.iter().enumerate() {
            if i == self.current_index {
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
            current_index: m_index,
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
    }

    fn delete_link(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = self.next[i];
        self.prev[self.next[i] as usize] = self.prev[i];
    }


    fn undelete_link(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = i as u32;
        self.prev[self.next[i] as usize] = i as u32;
    }

    fn unwind(&mut self) {
        for i in (0..self.k - 1).rev() {
            self.delete_link(i)
        }
        self.m = self.tail;
        self.n_element = 0;
    }

    fn set_median(&mut self) {
        // median index position
        let new_index = self.n_element / 2;
        dbg!(new_index);
        self.traverse_to_index(new_index)
    }

    fn traverse_to_index(&mut self, i: usize) {
        match i as i64 - self.current_index as i64 {
            0 => {
                // pass
            }
            -1 => {
                self.current_index -= 1;
                self.m = self.prev[self.m as usize] as usize;
            },
            1 => {
                self.current_index += 1;
                self.m = self.next[self.m as usize] as usize;
            },
            i64::MIN..=0 => {
                self.current_index -= i;
                for _ in i..0 {
                    self.m = self.prev[self.m as usize] as usize;
                }
            },
            _ => {
                self.current_index += i;
                for _ in 0..i {
                    self.m = self.next[self.m as usize] as usize;
                }
            }
        }
    }

    fn delete(&mut self, i: usize) {
        let delete = self.get_pair(i);
        let current = self.get_pair(self.m);

        // delete from links
        self.delete_link(i);

        self.n_element -= 1;

        match delete.partial_cmp(&current).unwrap() {
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
                // go to next position because hte link was deleted
                self.m = self.next[self.m as usize] as usize;
            }
        };
    }

    fn undelete(&mut self, i: usize) {
        // undelete from links
        self.undelete_link(i);

        if self.at_end() {
            self.m = i;
            self.n_element = 1;
            self.current_index = 0;
            return
        }
        let added = self.get_pair(i);
        let current = self.get_pair(self.m);

        self.n_element += 1;

        match added.partial_cmp(&current).unwrap() {
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
                // 1, 2, [4], 5
                // 1, 2, [3], 4, 5
                // go to prev position because hte link was added
                self.m = self.prev[self.m as usize] as usize;
            }
        };
    }

    fn undelete_set_median(&mut self, i: usize) {
        self.undelete(i);
        self.set_median()
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

    fn get_pair(&self, i: usize) -> (T, u32) {
        (self.alpha[i], i as u32)
    }

    fn is_small(&self, i: usize) -> bool {
        self.at_end() || self.get_pair(i) < self.get_pair(self.m as usize)
    }

}

fn rolling_median<T>(k: usize, slice: &[T])
where T: Copy + IsFloat + PartialOrd + Debug
{
    let mut scratch_a = vec![];
    let mut prev_a = vec![];
    let mut next_a = vec![];

    let mut scratch_b = vec![];
    let mut prev_b = vec![];
    let mut next_b = vec![];

    let k = std::cmp::min(k, slice.len());
    let alpha = &slice[..k];

    // let mut out = Vec::with_capacity(slice.len());
    let mut block_a = Block::new(alpha, &mut scratch_a, &mut prev_a, &mut next_a);
    let mut block_b = Block::new(&alpha[..1], &mut scratch_b, &mut prev_b, &mut next_b);

    let n_blocks = slice.len() / k;

    block_a.unwind();


    // let mut block_b = Block::new(h, alpha, &mut scratch_b, &mut prev_b, &mut next_b);
    // out.push(block_b.peek());
    //
    // for j in 1..b {
    //     block_a = block_b;
    //
    //     let alpha = &slice[j * k..(j + 1) *k];
    //     block_b = if j % 2 == 0 {
    //         Block::new(h, alpha, &mut scratch_b, &mut prev_b, &mut next_b)
    //     } else {
    //         Block::new(h, alpha, &mut scratch_a, &mut prev_a, &mut next_a)
    //     };
    //
    //     block_b.unwind();
    //     debug_assert_eq!(block_a.counter, h);
    //     debug_assert_eq!(block_b.counter, h);
    //
    //     for i in 0..k {
    //         block_a.delete(i);
    //         block_b.undelete(i);
    //         debug_assert!(block_a.counter + block_b.counter <= h);
    //
    //         if block_a.counter + block_b.counter <= h {
    //             if block_a.peek() <= block_b.peek() {
    //                 block_a.advance()
    //             } else {
    //                 block_b.advance()
    //             }
    //         }
    //         debug_assert_eq!(block_a.counter + block_b.counter, h)
    //         out.push(std::cmp::min(block_a.peek(), block_b.peek()));
    //     }
    //     debug_assert_eq!(block_a.counter, 0);
    //     debug_assert_eq!(block_b.counter, h);
    // }
    // dbg!(out);

}


mod test {
    use super::*;

    #[test]
    fn test_block() {

        let values = [2, 8, 5, 9, 1, 3, 4, 10];
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(&values, &mut scratch, &mut prev, &mut next);

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
        // TODO! write
        dbg!(b);

    }

    #[test]
    fn test_median() {
        let values = [2, 8, 5, 9, 1, 3, 4, 10];
        let k = 3;

        rolling_median(k, &values);


    }
}