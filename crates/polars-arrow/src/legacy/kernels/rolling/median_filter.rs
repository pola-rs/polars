
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
    current_index: usize

}

impl<T: Copy + Debug + IsFloat> Debug for Block<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "elements in list: {}", self.n_element)?;
        writeln!(f, "m: {}", self.m)?;
        writeln!(f, "m_index: {}", self.current_index)?;
        writeln!(f, "α[m]: {:?}", self.alpha[self.m])?;

        let mut p = self.m as u32;
        loop {
            p = self.prev[p as usize];
            if p as usize == self.tail {
                p = self.next[p as usize];
                break
            }
        }
        let mut current = Vec::with_capacity(self.n_element);
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

    fn unwind(&mut self) {
        for i in (0..self.k - 1).rev() {
            self.next[self.prev[i] as usize] = self.next[i];
            self.prev[self.next[i] as usize] = self.prev[i];
        }
        self.m = self.tail;
        self.n_element = 0;
    }

    fn set_median(&mut self) {
        // median index position
        let new_index = self.n_element / 2;
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
        self.next[self.prev[i] as usize] = self.next[i];
        self.prev[self.next[i] as usize] = self.prev[i];

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
        let added = self.get_pair(i);
        let current = self.get_pair(self.m);

        // undelete from links
        self.next[self.prev[i] as usize] = i as IdxSize;
        self.prev[self.next[i] as usize] = i as IdxSize;

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
        b.delete(4);
        b.undelete(4);
        b.undelete(5);
        b.undelete(0);

        b.set_median();
        dbg!(b);

    }
}