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
}

impl<T: Iterator + ?Sized> Itertools for T {}
