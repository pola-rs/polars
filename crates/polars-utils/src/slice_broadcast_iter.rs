use core::iter::FusedIterator;
use core::mem::size_of;
use core::{fmt, slice};

/// An iterator over a slice that is either flat (one slot per element) or scalar (a single slot
/// every element shares).
pub struct SliceBroadcastIter<'a, T> {
    repr: Repr<'a, T>,
}

/// The mode the iterator turned out to be in, resolved once.
enum Repr<'a, T> {
    /// One slot per element, walked as the slice it is.
    Flat(slice::Iter<'a, T>),
    /// The single item every element shares, and how many are left to yield.
    Broadcast { item: &'a T, remaining: usize },
}

const _: () = {
    // One word more than the packed encoding this replaced, in exchange for a flat arm the
    // optimizer can widen.
    assert!(size_of::<SliceBroadcastIter<'static, u8>>() == 3 * size_of::<usize>());
    // The slice iterator's non-null pointer leaves a niche, so `Option` stays free.
    assert!(size_of::<Option<SliceBroadcastIter<'static, u8>>>() == 3 * size_of::<usize>());
};

impl<T> Clone for Repr<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Flat(iter) => Self::Flat(iter.clone()),
            Self::Broadcast { item, remaining } => Self::Broadcast {
                item,
                remaining: *remaining,
            },
        }
    }
}

impl<T> Clone for SliceBroadcastIter<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for SliceBroadcastIter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SliceBroadcastIter")
            .field("len", &self.len())
            .field("broadcast", &self.is_broadcast())
            .finish()
    }
}

impl<'a, T> SliceBroadcastIter<'a, T> {
    /// Broadcast `src` to length `n`. Returns `None` unless `src.len() == n` (normal mode) or
    /// `src.len() == 1` (broadcast mode).
    #[inline]
    pub fn new_broadcast(src: &'a [T], n: usize) -> Option<Self> {
        if src.len() == n {
            Some(Self::new(src))
        } else if let [item] = src {
            Some(Self::repeat(item, n))
        } else {
            None
        }
    }

    /// Normal mode: yields every element of `src`.
    #[inline]
    pub fn new(src: &'a [T]) -> Self {
        Self {
            repr: Repr::Flat(src.iter()),
        }
    }

    /// Broadcast mode: yields `item` exactly `n` times.
    #[inline]
    pub fn repeat(item: &'a T, n: usize) -> Self {
        Self {
            repr: Repr::Broadcast { item, remaining: n },
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Flat(iter) => iter.len(),
            Repr::Broadcast { remaining, .. } => *remaining,
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub const fn is_broadcast(&self) -> bool {
        matches!(self.repr, Repr::Broadcast { .. })
    }

    /// Random access into the elements left to yield.
    ///
    /// # Safety
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> &'a T {
        match &self.repr {
            // SAFETY: `i` is in bounds of the elements left to yield, which is what the slice the
            // iterator has left holds.
            Repr::Flat(iter) => unsafe { iter.as_slice().get_unchecked(i) },
            Repr::Broadcast { item, .. } => item,
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<&'a T> {
        (i < self.len()).then(|| unsafe { self.get_unchecked(i) })
    }

    /// Collapse the mode into a single branch so the caller can run a monomorphic, vectorizable
    /// loop: `Ok(slice)` in normal mode, `Err((item, count))` in broadcast mode.
    #[inline]
    pub fn split(self) -> Result<&'a [T], (&'a T, usize)> {
        match self.repr {
            Repr::Flat(iter) => Ok(iter.as_slice()),
            Repr::Broadcast { item, remaining } => Err((item, remaining)),
        }
    }
}

impl<'a, T> Iterator for SliceBroadcastIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match &mut self.repr {
            Repr::Flat(iter) => iter.next(),
            Repr::Broadcast { item, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(item)
            },
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(self) -> Option<&'a T> {
        match self.repr {
            Repr::Flat(iter) => iter.last(),
            Repr::Broadcast { item, remaining } => (remaining != 0).then_some(item),
        }
    }

    #[inline]
    fn nth(&mut self, k: usize) -> Option<&'a T> {
        match &mut self.repr {
            Repr::Flat(iter) => iter.nth(k),
            Repr::Broadcast { item, remaining } => {
                let Some(left) = remaining.checked_sub(k + 1) else {
                    *remaining = 0;
                    return None;
                };
                *remaining = left;
                Some(item)
            },
        }
    }

    /// Hoists the mode branch out of the loop. `for_each`, `sum`, `collect` and friends route
    /// through here.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a T) -> B,
    {
        match self.split() {
            Ok(s) => s.iter().fold(init, f),
            Err((x, n)) => {
                let mut acc = init;
                for _ in 0..n {
                    acc = f(acc, x);
                }
                acc
            },
        }
    }
}

impl<'a, T> DoubleEndedIterator for SliceBroadcastIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        match &mut self.repr {
            Repr::Flat(iter) => iter.next_back(),
            Repr::Broadcast { item, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(item)
            },
        }
    }

    #[inline]
    fn nth_back(&mut self, k: usize) -> Option<&'a T> {
        match &mut self.repr {
            Repr::Flat(iter) => iter.nth_back(k),
            // Every element is the same one, so walking in from either end is the same walk.
            Repr::Broadcast { .. } => self.nth(k),
        }
    }

    /// Hoists the mode branch out of the loop, the way [`Iterator::fold`] does. `rev().collect()`
    /// and friends route through here.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a T) -> B,
    {
        match self.split() {
            Ok(s) => s.iter().rfold(init, f),
            Err((x, n)) => {
                let mut acc = init;
                for _ in 0..n {
                    acc = f(acc, x);
                }
                acc
            },
        }
    }
}

impl<T> ExactSizeIterator for SliceBroadcastIter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        Self::len(self)
    }
}

impl<T> FusedIterator for SliceBroadcastIter<'_, T> {}

// ---------------------------------------------------------------------------
