use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr::NonNull;
use core::{fmt, slice};

pub struct SliceBroadcastIter<'a, T> {
    ptr: NonNull<T>,
    /// `(remaining << 1) | broadcast`
    state: usize,
    _lt: PhantomData<&'a [T]>,
}

const _: () = {
    assert!(size_of::<SliceBroadcastIter<'static, u8>>() == 2 * size_of::<usize>());
    // NonNull's niche keeps Option free.
    assert!(size_of::<Option<SliceBroadcastIter<'static, u8>>>() == 2 * size_of::<usize>());
};

// Same bounds as `slice::Iter`.
unsafe impl<T: Sync> Send for SliceBroadcastIter<'_, T> {}
unsafe impl<T: Sync> Sync for SliceBroadcastIter<'_, T> {}

impl<T> Clone for SliceBroadcastIter<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            state: self.state,
            _lt: PhantomData,
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
    /// Broadcast `src` to length `n`.
    ///
    /// Returns `None` unless `src.len() == n` (normal mode) or
    /// `src.len() == 1` (broadcast mode).
    #[inline]
    pub fn new_broadcast(src: &'a [T], n: usize) -> Option<Self> {
        let broadcast = if src.len() == n {
            false
        } else if src.len() == 1 {
            true
        } else {
            return None;
        };
        Some(Self::from_raw(src.as_ptr(), n, broadcast))
    }

    /// Normal mode: yields every element of `src`.
    #[inline]
    pub fn new(src: &'a [T]) -> Self {
        Self::from_raw(src.as_ptr(), src.len(), false)
    }

    /// Broadcast mode: yields `item` exactly `n` times.
    #[inline]
    pub fn repeat(item: &'a T, n: usize) -> Self {
        Self::from_raw(item, n, true)
    }

    #[inline]
    fn from_raw(ptr: *const T, n: usize, broadcast: bool) -> Self {
        // Non-ZST slices can never be this long, so this folds away.
        if size_of::<T>() == 0 {
            assert!(n <= usize::MAX >> 1, "broadcast length too large for ZST");
        }
        debug_assert!(n <= usize::MAX >> 1);
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr.cast_mut()) },
            state: (n << 1) | broadcast as usize,
            _lt: PhantomData,
        }
    }

    /// `0` when broadcasting, `usize::MAX` otherwise.
    #[inline(always)]
    const fn mask(&self) -> usize {
        (self.state & 1).wrapping_sub(1)
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.state >> 1
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.state < 2
    }

    #[inline(always)]
    pub const fn is_broadcast(&self) -> bool {
        self.state & 1 != 0
    }

    /// Branchless random access.
    ///
    /// # Safety
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> &'a T {
        unsafe { &*self.ptr.as_ptr().add(i & self.mask()) }
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<&'a T> {
        (i < self.len()).then(|| unsafe { self.get_unchecked(i) })
    }

    /// Collapse the mode into a single branch so the caller can run a
    /// monomorphic, vectorizable loop.
    ///
    /// `Ok(slice)` in normal mode; `Err((item, count))` in broadcast mode.
    #[inline]
    pub fn split(self) -> Result<&'a [T], (&'a T, usize)> {
        let n = self.len();
        if self.is_broadcast() {
            Err((unsafe { &*self.ptr.as_ptr() }, n))
        } else {
            Ok(unsafe { slice::from_raw_parts(self.ptr.as_ptr(), n) })
        }
    }
}

impl<'a, T> Iterator for SliceBroadcastIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.state < 2 {
            return None;
        }
        let p = self.ptr;
        let step = size_of::<T>() & self.mask();
        self.ptr = unsafe { NonNull::new_unchecked(p.as_ptr().byte_add(step)) };
        self.state -= 2;
        Some(unsafe { &*p.as_ptr() })
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
        let n = self.len();
        (n != 0).then(|| unsafe { self.get_unchecked(n - 1) })
    }

    #[inline]
    fn nth(&mut self, k: usize) -> Option<&'a T> {
        if k >= self.len() {
            self.state &= 1; // exhaust, keep mode
            return None;
        }
        let step = size_of::<T>().wrapping_mul(k) & self.mask();
        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().byte_add(step)) };
        self.state -= k << 1;
        self.next()
    }

    /// Hoists the mode branch out of the loop. `for_each`, `sum`, `collect`
    /// and friends route through here.
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
        if self.state < 2 {
            return None;
        }
        self.state -= 2;
        let off = size_of::<T>().wrapping_mul(self.state >> 1) & self.mask();
        Some(unsafe { &*self.ptr.as_ptr().byte_add(off) })
    }
}

impl<T> ExactSizeIterator for SliceBroadcastIter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.state >> 1
    }
}

impl<T> FusedIterator for SliceBroadcastIter<'_, T> {}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_mode() {
        let v = [1, 2, 3, 4];
        let it = SliceBroadcastIter::new_broadcast(&v, 4).unwrap();
        assert!(!it.is_broadcast());
        assert_eq!(it.len(), 4);
        assert_eq!(it.collect::<Vec<_>>(), vec![&1, &2, &3, &4]);
    }

    #[test]
    fn broadcast_mode() {
        let v = [7];
        let it = SliceBroadcastIter::new_broadcast(&v, 4).unwrap();
        assert!(it.is_broadcast());
        assert_eq!(it.collect::<Vec<_>>(), vec![&7, &7, &7, &7]);
    }

    #[test]
    fn mismatch_rejected() {
        let v = [1, 2];
        assert!(SliceBroadcastIter::new_broadcast(&v, 3).is_none());
    }

    #[test]
    fn empty_targets() {
        let v = [9];
        assert_eq!(SliceBroadcastIter::new_broadcast(&v, 0).unwrap().count(), 0);
        let e: [u8; 0] = [];
        assert_eq!(SliceBroadcastIter::new_broadcast(&e, 0).unwrap().count(), 0);
    }

    #[test]
    fn double_ended() {
        let v = [1, 2, 3];
        let got: Vec<_> = SliceBroadcastIter::new(&v).rev().copied().collect();
        assert_eq!(got, vec![3, 2, 1]);

        let one = [5];
        let got: Vec<_> = SliceBroadcastIter::new_broadcast(&one, 3)
            .unwrap()
            .rev()
            .copied()
            .collect();
        assert_eq!(got, vec![5, 5, 5]);

        let mut it = SliceBroadcastIter::new(&v);
        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next_back(), Some(&3));
        assert_eq!(it.next(), Some(&2));
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn nth_and_last() {
        let v = [10, 20, 30, 40];
        let mut it = SliceBroadcastIter::new(&v);
        assert_eq!(it.nth(2), Some(&30));
        assert_eq!(it.next(), Some(&40));
        assert_eq!(it.next(), None);

        let one = [8];
        let mut it = SliceBroadcastIter::new_broadcast(&one, 5).unwrap();
        assert_eq!(it.nth(3), Some(&8));
        assert_eq!(it.len(), 1);
        assert_eq!(
            SliceBroadcastIter::new_broadcast(&one, 5).unwrap().last(),
            Some(&8)
        );
        assert_eq!(SliceBroadcastIter::new(&v).last(), Some(&40));
    }

    #[test]
    fn fold_matches_next() {
        let v = [1, 2, 3];
        assert_eq!(SliceBroadcastIter::new(&v).sum::<i32>(), 6);
        let one = [4];
        assert_eq!(
            SliceBroadcastIter::new_broadcast(&one, 3)
                .unwrap()
                .sum::<i32>(),
            12
        );
    }

    #[test]
    fn indexed_access() {
        let one = [3u8];
        let it = SliceBroadcastIter::new_broadcast(&one, 100).unwrap();
        assert_eq!(it.get(99), Some(&3));
        assert_eq!(it.get(100), None);
    }

    #[test]
    fn zst() {
        let u = [()];
        assert_eq!(SliceBroadcastIter::new_broadcast(&u, 5).unwrap().count(), 5);
    }
}
