use std::cmp::Ordering;

use crate::IdxSize;

pub mod enumerate_idx;

/// Utility extension trait of iterator methods.
pub trait Itertools: Iterator {
    /// Equivalent to `.collect::<Vec<_>>()`.
    fn collect_vec(self) -> Vec<Self::Item>
    where
        Self: Sized,
    {
        self.collect()
    }

    /// Equivalent to `.collect::<Result<_, _>>()`.
    fn try_collect<T, U, E>(self) -> Result<U, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
        Result<U, E>: FromIterator<Result<T, E>>,
    {
        self.collect()
    }

    /// Equivalent to `.collect::<Result<Vec<_>, _>>()`.
    fn try_collect_vec<T, U, E>(self) -> Result<Vec<U>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
        Result<Vec<U>, E>: FromIterator<Result<T, E>>,
    {
        self.collect()
    }

    fn enumerate_idx(self) -> enumerate_idx::EnumerateIdx<Self, IdxSize>
    where
        Self: Sized,
    {
        enumerate_idx::EnumerateIdx::new(self)
    }

    fn enumerate_u32(self) -> enumerate_idx::EnumerateIdx<Self, u32>
    where
        Self: Sized,
    {
        enumerate_idx::EnumerateIdx::new(self)
    }

    fn all_equal(mut self) -> bool
    where
        Self: Sized,
        Self::Item: PartialEq,
    {
        match self.next() {
            None => true,
            Some(a) => self.all(|x| a == x),
        }
    }

    // Stable copy of the unstable eq_by from the stdlib.
    fn eq_by_<I, F>(mut self, other: I, mut eq: F) -> bool
    where
        Self: Sized,
        I: IntoIterator,
        F: FnMut(Self::Item, I::Item) -> bool,
    {
        let mut other = other.into_iter();
        loop {
            match (self.next(), other.next()) {
                (None, None) => return true,
                (None, Some(_)) => return false,
                (Some(_), None) => return false,
                (Some(l), Some(r)) => {
                    if eq(l, r) {
                        continue;
                    } else {
                        return false;
                    }
                },
            }
        }
    }

    // Stable copy of the unstable partial_cmp_by from the stdlib.
    fn partial_cmp_by_<I, F>(mut self, other: I, mut partial_cmp: F) -> Option<Ordering>
    where
        Self: Sized,
        I: IntoIterator,
        F: FnMut(Self::Item, I::Item) -> Option<Ordering>,
    {
        let mut other = other.into_iter();
        loop {
            match (self.next(), other.next()) {
                (None, None) => return Some(Ordering::Equal),
                (None, Some(_)) => return Some(Ordering::Less),
                (Some(_), None) => return Some(Ordering::Greater),
                (Some(l), Some(r)) => match partial_cmp(l, r) {
                    Some(Ordering::Equal) => continue,
                    ord => return ord,
                },
            }
        }
    }
}

impl<T: Iterator + ?Sized> Itertools for T {}
