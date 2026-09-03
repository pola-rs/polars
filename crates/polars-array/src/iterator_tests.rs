//! What every iterator of this crate is expected to agree on.

use std::fmt::Debug;

/// Asserts that `iter` yields `expected`, however it is driven.
pub(crate) fn assert_iterates<I>(iter: I, expected: &[I::Item])
where
    I: DoubleEndedIterator + ExactSizeIterator + Clone,
    I::Item: PartialEq + Debug,
{
    let n = expected.len();

    assert_eq!(iter.len(), n, "length");
    assert_eq!(iter.clone().size_hint(), (n, Some(n)), "size hint");
    assert_eq!(iter.clone().count(), n, "count");

    // `next` in a loop, which is what a `for` loop and a trusted-length collect do.
    let mut walked = Vec::with_capacity(n);
    let mut it = iter.clone();
    for i in 0..n {
        assert_eq!(it.len(), n - i, "length after {i} elements");
        walked.push(it.next().expect("an element"));
    }
    assert!(it.next().is_none(), "past the end");
    assert!(it.next().is_none(), "past the end twice");
    assert_eq!(walked, expected, "walked with `next`");

    // `fold`, which `collect`, `for_each` and `sum` route through.
    let folded: Vec<_> = iter.clone().collect();
    assert_eq!(folded, expected, "folded");

    // The same, backwards.
    let mut walked = Vec::with_capacity(n);
    let mut it = iter.clone();
    while let Some(item) = it.next_back() {
        walked.push(item);
    }
    walked.reverse();
    assert_eq!(walked, expected, "walked with `next_back`");

    let mut folded: Vec<_> = iter.clone().rev().collect();
    folded.reverse();
    assert_eq!(folded, expected, "folded backwards");

    // `last`, and `nth` at every position, from either end.
    assert_eq!(iter.clone().last().as_ref(), expected.last(), "last");
    for i in 0..n {
        assert_eq!(iter.clone().nth(i).as_ref(), Some(&expected[i]), "nth({i})");
        assert_eq!(
            iter.clone().nth_back(i).as_ref(),
            Some(&expected[n - 1 - i]),
            "nth_back({i})",
        );
    }

    // Past the end, which leaves the iterator exhausted rather than out of step.
    let mut it = iter.clone();
    assert!(it.nth(n).is_none(), "nth past the end");
    assert_eq!(it.len(), 0, "length past the end");
    assert!(it.next().is_none(), "past the end after `nth`");

    let mut it = iter.clone();
    assert!(it.nth_back(n).is_none(), "nth_back past the end");
    assert_eq!(it.len(), 0, "length past the end backwards");

    // Both ends of the same iterator, which meet in the middle exactly once.
    let mut it = iter.clone();
    let mut front = Vec::with_capacity(n);
    let mut back = Vec::with_capacity(n);
    loop {
        match it.next() {
            Some(item) => front.push(item),
            None => break,
        }
        match it.next_back() {
            Some(item) => back.push(item),
            None => break,
        }
    }
    back.reverse();
    front.extend(back);
    assert_eq!(front, expected, "walked from both ends");
}
