use core::iter::FusedIterator;
use core::mem::size_of;
use core::{fmt, slice};

/// An iterator over a slice that is either flat (one slot per element) or scalar (one slot).
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
    /// Broadcast `src` to length `n`.
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

    /// Collapse the mode into a single branch so the caller can run a monomorphic loop.
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

    /// Hoists the mode branch out of the loop.
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

    /// Hoists the mode branch out of the loop, the way [`Iterator::fold`] does.
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

#[cfg(test)]
mod test {
    use super::*;

    /// The two modes of the iterator, against the elements each of them stands for.
    fn modes(n: usize) -> Vec<(&'static str, SliceBroadcastIter<'static, usize>, Vec<usize>)> {
        // Leaked so both modes borrow for the same lifetime; the tests are the only owner.
        let flat: &'static [usize] = Vec::leak((0..n).collect::<Vec<_>>());
        let one: &'static usize = Box::leak(Box::new(7usize));

        vec![
            ("flat", SliceBroadcastIter::new(flat), flat.to_vec()),
            (
                "broadcast",
                SliceBroadcastIter::repeat(one, n),
                vec![*one; n],
            ),
        ]
    }

    #[test]
    fn walks_the_elements_it_stands_for() {
        for n in [0usize, 1, 2, 3, 8, 65] {
            for (mode, iter, elements) in modes(n) {
                assert_eq!(iter.len(), n, "{mode} of {n}");
                assert_eq!(iter.size_hint(), (n, Some(n)), "{mode} of {n}");
                assert_eq!(iter.is_empty(), n == 0, "{mode} of {n}");
                assert_eq!(iter.clone().count(), n, "{mode} of {n}");
                assert_eq!(
                    iter.clone().last().copied(),
                    elements.last().copied(),
                    "{mode} of {n}",
                );

                let front: Vec<usize> = iter.clone().copied().collect();
                assert_eq!(front, elements, "{mode} of {n}");
                let back: Vec<usize> = iter.clone().rev().copied().collect();
                assert_eq!(back, elements.iter().rev().copied().collect::<Vec<_>>());

                // `fold` and `rfold` hoist the mode out of the loop, so they are walks of
                // their own rather than `next`/`next_back` under another name.
                let folded = iter.clone().fold(Vec::new(), |mut acc, v| {
                    acc.push(*v);
                    acc
                });
                assert_eq!(folded, elements, "{mode} of {n}");
                let rfolded = iter.clone().rfold(Vec::new(), |mut acc, v| {
                    acc.push(*v);
                    acc
                });
                assert_eq!(
                    rfolded,
                    elements.iter().rev().copied().collect::<Vec<_>>(),
                    "{mode} of {n}",
                );
            }
        }
    }

    #[test]
    fn walking_from_both_ends_meets_in_the_middle() {
        for n in [0usize, 1, 2, 3, 8, 65] {
            for (mode, mut iter, elements) in modes(n) {
                let (mut front, mut back) = (Vec::new(), Vec::new());
                let mut take_front = true;
                while let Some(v) = match take_front {
                    true => iter.next(),
                    false => iter.next_back(),
                } {
                    match take_front {
                        true => front.push(*v),
                        false => back.push(*v),
                    }
                    take_front = !take_front;
                }

                back.reverse();
                front.extend(back);
                assert_eq!(front, elements, "{mode} of {n}");
                assert_eq!(iter.len(), 0, "{mode} of {n}");
                // The iterator is fused: it stays empty however it is asked again.
                assert_eq!(iter.next(), None, "{mode} of {n}");
                assert_eq!(iter.next_back(), None, "{mode} of {n}");
            }
        }
    }

    #[test]
    fn skipping_leaves_the_same_elements_the_reference_does() {
        for n in [0usize, 1, 2, 3, 8, 65] {
            for k in 0..n + 2 {
                for (mode, mut iter, elements) in modes(n) {
                    let mut reference = elements.iter().copied();
                    assert_eq!(
                        iter.nth(k).copied(),
                        reference.nth(k),
                        "{mode} of {n}, nth({k})",
                    );
                    assert_eq!(iter.len(), reference.len(), "{mode} of {n}, nth({k})");
                    assert_eq!(
                        iter.copied().collect::<Vec<_>>(),
                        reference.collect::<Vec<_>>(),
                        "{mode} of {n}, nth({k})",
                    );
                }

                for (mode, mut iter, elements) in modes(n) {
                    let mut reference = elements.iter().copied();
                    assert_eq!(
                        iter.nth_back(k).copied(),
                        reference.nth_back(k),
                        "{mode} of {n}, nth_back({k})",
                    );
                    assert_eq!(iter.len(), reference.len(), "{mode} of {n}, nth_back({k})");
                    assert_eq!(
                        iter.copied().collect::<Vec<_>>(),
                        reference.collect::<Vec<_>>(),
                        "{mode} of {n}, nth_back({k})",
                    );
                }
            }
        }
    }

    #[test]
    fn random_access_reads_what_is_left_to_yield() {
        for n in [1usize, 2, 3, 8, 65] {
            for consumed in 0..n {
                for (mode, mut iter, elements) in modes(n) {
                    for _ in 0..consumed {
                        iter.next();
                    }

                    let left = &elements[consumed..];
                    assert_eq!(iter.len(), left.len(), "{mode} of {n} less {consumed}");
                    for (i, expected) in left.iter().enumerate() {
                        assert_eq!(iter.get(i), Some(expected), "{mode} of {n}, get({i})");
                        // SAFETY: `i` is below the number of elements left to yield.
                        assert_eq!(unsafe { iter.get_unchecked(i) }, expected);
                    }
                    assert_eq!(iter.get(left.len()), None, "{mode} of {n} less {consumed}");
                }
            }
        }
    }

    #[test]
    fn split_hands_over_what_is_left_to_yield() {
        for n in [0usize, 1, 2, 8] {
            for (mode, mut iter, elements) in modes(n) {
                if n > 0 {
                    iter.next();
                }
                let left = match n {
                    0 => &elements[..],
                    _ => &elements[1..],
                };

                match iter.split() {
                    Ok(slice) => assert_eq!(slice, left, "{mode} of {n}"),
                    Err((item, remaining)) => {
                        assert_eq!(remaining, left.len(), "{mode} of {n}");
                        assert!(left.iter().all(|v| v == item), "{mode} of {n}");
                    },
                }
            }
        }
    }

    /// The collects that write into reserved room take a trusted iterator at its word, so the
    /// length it reports has to hold after every way of walking part of it.
    #[test]
    fn the_reported_length_is_the_number_of_elements_left() {
        for n in [0usize, 1, 2, 3, 8, 65] {
            for (mode, iter, _) in modes(n) {
                for step in [0usize, 1, 2, n / 2, n] {
                    let mut walked = iter.clone();
                    for _ in 0..step {
                        walked.next();
                    }
                    let reported = walked.size_hint();
                    let left = walked.count();
                    assert_eq!(
                        reported,
                        (left, Some(left)),
                        "{mode} of {n} after {step} steps",
                    );
                }
            }
        }
    }
}
