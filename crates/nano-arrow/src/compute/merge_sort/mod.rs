//! Functions to perform merge-sorts.
//!
//! The goal of merge-sort is to merge two sorted arrays, `[a0, a1]`, `merge_sort(a0, a1)`,
//! so that the resulting array is sorted, i.e. the following invariant upholds:
//! `sort(merge_sort(a0, a1)) == merge_sort(a0, a1)` for any two sorted arrays `a0` and `a1`.
//!
//! Given that two sorted arrays are more likely to be partially sorted within each other,
//! and that the resulting array is built by taking elements from each array, it is
//! advantageous to `take` slices of items, not items, from each array.
//! As such, this module's main data representation is `(i: usize, start: usize, len: usize)`,
//! which represents a slice of array `i`.
//!
//! In this context, `merge_sort` is composed by two main operations:
//!
//! 1. compute the array of slices `v` that construct a new sorted array from `a0` and `a1`.
//! 2. `take_arrays` from `a0` and `a1`, creating the sorted array.
//!
//! In the extreme case where the two arrays are already sorted between then (e.g. `[0, 2]`, `[3, 4]`),
//! we need two slices, `v = vec![(0, 0, a0.len()), (1, 0, a1.len())]`. The higher the
//! inter-leave between the two arrays, the more slices will be needed, and
//! generally the more expensive the `take` operation will be.
//!
//! ## Merge-sort multiple arrays
//!
//! The main advantage of merge-sort over `sort` is that it can be parallelized.
//! For example, given a set of arrays `[a0, a1, a2, a3]` representing the same field,
//! e.g. over 4 batches of arrays, they can be sorted in parallel as follows (pseudo-code):
//!
//! ```rust,ignore
//! // in parallel
//! let a0 = sort(a0);
//! let a1 = sort(a1);
//! let a2 = sort(a2);
//! let a3 = sort(a3);
//!
//! // in parallel and recursively
//! let slices1 = merge_sort_slices(a0, a1);
//! let slices2 = merge_sort_slices(a2, a3);
//! let slices = merge_sort_slices(slices1, slices2);
//!
//! let array = take_arrays(&[a0, a1, a2, a3], slices, None);
//! ```
//!
//! A common operation in query engines is to merge multiple fields based on the
//! same sorting field (e.g. merge-sort multiple batches of arrays).
//! To perform this, use the same idea as above, but use `take_arrays` over
//! each independent field (which can again be parallelized):
//!
//! ```rust,ignore
//! // `slices` computed before-hand
//! // in parallel
//! let array1 = take_arrays(&[a0, a1, a2, a3], slices, None);
//! let array2 = take_arrays(&[b0, b1, b2, b3], slices, None);
//! ```
//!
//! To serialize slices, e.g. for checkpointing or transfer via Arrow's IPC, you can store
//! them as 3 non-null primitive arrays (e.g. `PrimitiveArray<i64>`).

use ahash::AHashMap;
use std::cmp::Ordering;
use std::iter::once;

use itertools::Itertools;

use crate::array::{
    growable::make_growable,
    ord::{build_compare, DynComparator},
    Array,
};
pub use crate::compute::sort::SortOptions;
use crate::error::Result;

/// A slice denoting `(array_index, start, len)` representing a slice from one of N arrays.
/// This is used to keep track of contiguous blocks of slots.
/// An array of MergeSlice, `[MergeSlice]`, represents inter-leaved array slices.
/// For example, `[(0, 0, 2), (1, 0, 1), (0, 2, 3)]` represents 2 arrays (a0 and a1) arranged as follows:
/// `[a0[0..2], a1[0..1], a0[2..3]]`
/// This representation is useful when building arrays in memory as it allows to memcopy slices of arrays.
/// This is particularly useful in merge-sort because sorted arrays (passed to the merge-sort) are more likely
/// to have contiguous blocks of sorted elements (than by random).
pub type MergeSlice = (usize, usize, usize);

/// Takes N arrays together through `slices` under the assumption that the slices have
/// a total coverage of the arrays.
/// I.e. they are such that all elements on all arrays are picked (which is the case in sorting).
/// # Panic
/// This function panics if:
/// * `max(slices[i].0) >= arrays.len()`, as it indicates that the slices point to an array out of bounds from `arrays`.
/// * the arrays do not have the same [`crate::datatypes::DataType`] (as it makes no sense to take together from them)
pub fn take_arrays<I: IntoIterator<Item = MergeSlice>>(
    arrays: &[&dyn Array],
    slices: I,
    limit: Option<usize>,
) -> Box<dyn Array> {
    let slices = slices.into_iter();
    let len = arrays.iter().map(|array| array.len()).sum();

    let limit = limit.unwrap_or(len);
    let limit = limit.min(len);
    let mut growable = make_growable(arrays, false, limit);

    if limit != len {
        let mut current_len = 0;
        for (index, start, len) in slices {
            if len + current_len >= limit {
                growable.extend(index, start, limit - current_len);
                break;
            } else {
                growable.extend(index, start, len);
                current_len += len;
            }
        }
    } else {
        for (index, start, len) in slices {
            growable.extend(index, start, len);
        }
    }

    growable.as_box()
}

/// Combines two sorted [Array]s of the same [`crate::datatypes::DataType`] into a single sorted array.
/// If the arrays are not sorted (which this function does not check), the result is wrong.
/// # Error
/// This function errors when:
/// * the arrays have a different [`crate::datatypes::DataType`]
/// * the arrays have a [`crate::datatypes::DataType`] that has no order relationship
/// # Example
/// ```rust
/// use arrow2::array::Int32Array;
/// use arrow2::compute::merge_sort::{merge_sort, SortOptions};
/// # use arrow2::error::Result;
/// # fn main() -> Result<()> {
/// let a = Int32Array::from_slice(&[2, 4, 6]);
/// let b = Int32Array::from_slice(&[0, 1, 3]);
/// let sorted = merge_sort(&a, &b, &SortOptions::default(), None)?;
/// let expected = Int32Array::from_slice(&[0, 1, 2, 3, 4, 6]);
/// assert_eq!(expected, sorted.as_ref());
/// # Ok(())
/// # }
/// ```
pub fn merge_sort(
    lhs: &dyn Array,
    rhs: &dyn Array,
    options: &SortOptions,
    limit: Option<usize>,
) -> Result<Box<dyn Array>> {
    let arrays = &[lhs, rhs];

    let pairs: &[(&[&dyn Array], &SortOptions)] = &[(arrays, options)];
    let comparator = build_comparator(pairs)?;

    let lhs = (0, 0, lhs.len());
    let rhs = (1, 0, rhs.len());
    let slices = merge_sort_slices(once(&lhs), once(&rhs), &comparator);
    Ok(take_arrays(arrays, slices, limit))
}

/// Returns a vector of slices from different sorted arrays that can be used to create sorted arrays.
/// `pairs` is an array representing multiple sorted array sets. The expected format is
///
/// pairs:  [([a00, a01], o1), ([a10, a11], o2), ...]
/// where aj0.len() == aj0.len()
///       aj1.len() == aj1.len()
///       ...
/// In other words, `pairs.i.0[j]` must be an array coming from a batch of equal len arrays.
/// # Example
/// ```rust
/// use arrow2::array::Int32Array;
/// use arrow2::compute::merge_sort::{slices, SortOptions};
/// # use arrow2::error::Result;
/// # fn main() -> Result<()> {
/// let a = Int32Array::from_slice(&[2, 4, 6]);
/// let b = Int32Array::from_slice(&[0, 1, 3]);
/// let slices = slices(&[(&[&a, &b], &SortOptions::default())])?;
/// assert_eq!(slices, vec![(1, 0, 2), (0, 0, 1), (1, 2, 1), (0, 1, 2)]);
///
/// # Ok(())
/// # }
/// ```
/// # Error
/// This function errors if the arrays `a0i` are not pairwise sortable. This happens when either
/// they have not the same [`crate::datatypes::DataType`] or when their [`crate::datatypes::DataType`]
/// does not correspond to a sortable type.
/// # Panic
/// This function panics if:
/// * `pairs` has no elements
/// * the length condition above is not fulfilled
pub fn slices(pairs: &[(&[&dyn Array], &SortOptions)]) -> Result<Vec<MergeSlice>> {
    assert!(!pairs.is_empty());
    let comparator = build_comparator(pairs)?;

    // pairs:  [([a00, a01], o1), ([a10, a11], o2), ...]
    // slices: [(0, 0, len), (1, 0, len)]

    let slices = pairs[0]
        .0
        .iter()
        .enumerate()
        .map(|(index, array)| vec![(index, 0, array.len())])
        .collect::<Vec<_>>();

    let slices = slices
        .iter()
        .map(|slice| slice.as_ref())
        .collect::<Vec<_>>();
    Ok(recursive_merge_sort(&slices, &comparator))
}

/// recursively sort-merges multiple `slices` representing slices of sorted arrays according
/// to a comparison function between those arrays.
/// Note that `slices` is an array of arrays, `slices[i][j]`. The index `i` represents
/// the set of arrays `i`, while the index `j` represents
/// the array `j` within that set.
/// Note that this does not split to the smallest element as arrays: the smallest unit is a `slice`
fn recursive_merge_sort(slices: &[&[MergeSlice]], comparator: &Comparator) -> Vec<MergeSlice> {
    let n = slices.len();
    let m = n / 2;

    if n == 1 {
        // slices are assumed sort arrays
        return slices[0].to_vec();
    }
    if n == 2 {
        return merge_sort_slices(slices[0].iter(), slices[1].iter(), comparator)
            .collect::<Vec<_>>();
    }

    // split in 2 and sort
    let lhs = recursive_merge_sort(&slices[0..m], comparator);
    let rhs = recursive_merge_sort(&slices[m..n], comparator);

    // merge-sort the splits
    merge_sort_slices(lhs.iter(), rhs.iter(), comparator).collect::<Vec<_>>()
}

/// An iterator adapter that merge-sorts two iterators of `MergeSlice` into a single `MergeSlice`
/// such that the resulting `MergeSlice`s are ordered according to `comparator`.
pub struct MergeSortSlices<'a, L, R>
where
    L: Iterator<Item = &'a MergeSlice>,
    R: Iterator<Item = &'a MergeSlice>,
{
    lhs: L,
    rhs: R,
    comparator: &'a Comparator<'a>,

    left: Option<(MergeSlice, usize)>, // current left pile and index
    right: Option<(MergeSlice, usize)>, // current right pile and index

    // track the current slice being constructed (from left or right)
    has_started: bool,
    current_start: usize,
    current_len: usize,
    current_is_left: bool,
}

impl<'a, L, R> MergeSortSlices<'a, L, R>
where
    L: Iterator<Item = &'a MergeSlice>,
    R: Iterator<Item = &'a MergeSlice>,
{
    fn new(lhs: L, rhs: R, comparator: &'a Comparator<'a>) -> Self {
        Self {
            lhs,
            rhs,
            comparator,
            left: None,
            right: None,
            has_started: false,
            current_start: 0,
            current_len: 0,
            current_is_left: true,
        }
    }

    fn next_left(&mut self) {
        match self.lhs.next() {
            Some(slice) => {
                self.left = Some((*slice, slice.1));
                self.current_start = slice.1;
            }
            None => self.left = None,
        }
    }

    fn next_right(&mut self) {
        match self.rhs.next() {
            Some(slice) => {
                self.right = Some((*slice, slice.1));
                self.current_start = slice.1;
            }
            None => self.right = None,
        }
    }

    /// Collect the MergeSortSlices to be a vec for reusing
    #[warn(dead_code)]
    pub fn to_vec(self, limit: Option<usize>) -> Vec<MergeSlice> {
        match limit {
            Some(limit) => {
                let mut v = Vec::with_capacity(limit);
                let mut current_len = 0;
                for (index, start, len) in self {
                    if len + current_len >= limit {
                        v.push((index, start, limit - current_len));
                        break;
                    } else {
                        v.push((index, start, len));
                    }
                    current_len += len;
                }

                v
            }
            None => self.into_iter().collect(),
        }
    }
}

impl<'a, L, R> Iterator for MergeSortSlices<'a, L, R>
where
    L: Iterator<Item = &'a MergeSlice>,
    R: Iterator<Item = &'a MergeSlice>,
{
    type Item = MergeSlice;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_started {
            // first call of `next`
            self.next_left();
            self.next_right();
        }

        match (self.left, self.right) {
            (None, None) => {
                // both ended
                None
            }
            (Some((left_slice, left_index)), None) => {
                // right ended => push left
                self.next_left();
                // pushing from left
                if left_index != left_slice.1 {
                    // we are in the middle of some slice: push the
                    // remaining of that slice
                    Some((
                        left_slice.0,
                        left_index,
                        left_slice.2 - (left_index - left_slice.1),
                    ))
                } else {
                    Some(left_slice)
                }
            }
            (None, Some((right_slice, right_index))) => {
                // left ended => push right
                self.next_right();
                if right_index != right_slice.1 {
                    // we are in the middle of some slice: push the
                    // remaining of that slice
                    Some((
                        right_slice.0,
                        right_index,
                        right_slice.2 - (right_index - right_slice.1),
                    ))
                } else {
                    Some(right_slice)
                }
            }
            // both sides have elements
            (Some((left_slice, mut left_index)), Some((right_slice, mut right_index))) => {
                if !self.has_started {
                    let ordering =
                        (self.comparator)(left_slice.0, left_index, right_slice.0, right_index);
                    if ordering == Ordering::Greater {
                        self.current_is_left = false;
                        self.current_start = right_index;
                    } else {
                        self.current_is_left = true;
                        self.current_start = left_index;
                    }
                    self.has_started = true;
                }

                // advance left_index or right_index until the next split
                while (left_index < left_slice.1 + left_slice.2)
                    && (right_index < right_slice.1 + right_slice.2)
                {
                    match (
                        (self.comparator)(left_slice.0, left_index, right_slice.0, right_index),
                        self.current_is_left,
                    ) {
                        (Ordering::Less, true) | (Ordering::Equal, true) => {
                            // on the left and take from the left
                            self.current_len += 1;
                            left_index += 1;
                        }
                        (Ordering::Greater, false) | (Ordering::Equal, false) => {
                            // on the right and take from the right
                            self.current_len += 1;
                            right_index += 1;
                        }
                        (Ordering::Less, false) => {
                            // switch from right side to left side => push new slice from the right
                            let start = self.current_start;
                            let len = self.current_len;
                            self.current_is_left = true;
                            self.current_len = 0;
                            self.current_start = left_index;
                            if len > 0 {
                                self.left = Some((left_slice, left_index));
                                self.right = Some((right_slice, right_index));
                                return Some((right_slice.0, start, len));
                            }
                        }
                        (Ordering::Greater, true) => {
                            // switch from left side to right side => push slice from the left
                            let start = self.current_start;
                            let len = self.current_len;
                            self.current_is_left = false;
                            self.current_len = 0;
                            self.current_start = right_index;
                            if len > 0 {
                                self.left = Some((left_slice, left_index));
                                self.right = Some((right_slice, right_index));
                                return Some((left_slice.0, start, len));
                            }
                        }
                    }
                }
                let start = self.current_start;
                let len = self.current_len;
                if left_index == left_slice.1 + left_slice.2 {
                    // reached end of left slice => push it
                    self.current_len = 0;
                    self.next_left();
                    Some((left_slice.0, start, len))
                } else {
                    debug_assert_eq!(right_index, right_slice.1 + right_slice.2);
                    // reached end of right slice => push it
                    self.current_len = 0;
                    self.next_right();
                    Some((right_slice.0, start, len))
                }
            }
        }
    }
}

/// Given two iterators of slices representing two sets of sorted [`Array`]s, and a `comparator` bound to those [`Array`]s,
/// returns a new iterator of slices denoting how to `take` slices from each of the arrays such that the resulting
/// array is sorted according to `comparator`
pub fn merge_sort_slices<
    'a,
    L: Iterator<Item = &'a MergeSlice>,
    R: Iterator<Item = &'a MergeSlice>,
>(
    lhs: L,
    rhs: R,
    comparator: &'a Comparator,
) -> MergeSortSlices<'a, L, R> {
    MergeSortSlices::new(lhs, rhs, comparator)
}

// (left index, left row), (right index, right row)
type Comparator<'a> = Box<dyn Fn(usize, usize, usize, usize) -> Ordering + 'a>;
type IsValid<'a> = Box<dyn Fn(usize) -> bool + 'a>;

/// returns a comparison function between any two arrays of each pair of arrays, according to `SortOptions`.
pub fn build_comparator<'a>(
    pairs: &'a [(&'a [&'a dyn Array], &SortOptions)],
) -> Result<Comparator<'a>> {
    build_comparator_impl(pairs, &build_compare)
}

/// returns a comparison function between any two arrays of each pair of arrays, according to `SortOptions`.
/// Implementing custom `build_compare_fn` for unsupportd data types.
pub fn build_comparator_impl<'a>(
    pairs: &'a [(&'a [&'a dyn Array], &SortOptions)],
    build_compare_fn: &dyn Fn(&dyn Array, &dyn Array) -> Result<DynComparator>,
) -> Result<Comparator<'a>> {
    // prepare the comparison function of _values_ between all pairs of arrays
    let indices_pairs = (0..pairs[0].0.len())
        .combinations(2)
        .map(|indices| (indices[0], indices[1]));

    let data = indices_pairs
        .map(|(lhs_index, rhs_index)| {
            let multi_column_comparator = pairs
                .iter()
                .map(move |(arrays, _)| {
                    Ok((
                        Box::new(move |row| arrays[lhs_index].is_valid(row)) as IsValid<'a>,
                        Box::new(move |row| arrays[rhs_index].is_valid(row)) as IsValid<'a>,
                        build_compare_fn(arrays[lhs_index], arrays[rhs_index])?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(((lhs_index, rhs_index), multi_column_comparator))
        })
        .collect::<Result<AHashMap<(usize, usize), Vec<(IsValid, IsValid, DynComparator)>>>>()?;

    // prepare a comparison function taking into account _nulls_ and sort options
    let cmp = move |left_index, left_row, right_index, right_row| {
        let data = data.get(&(left_index, right_index)).unwrap();
        //data.iter().zip(pairs.iter()).for_each()
        for c in 0..pairs.len() {
            let descending = pairs[c].1.descending;
            let null_first = pairs[c].1.nulls_first;
            let (l_is_valid, r_is_valid, value_comparator) = &data[c];
            let result = match ((l_is_valid)(left_row), (r_is_valid)(right_row)) {
                (true, true) => {
                    let result = (value_comparator)(left_row, right_row);
                    match descending {
                        true => result.reverse(),
                        false => result,
                    }
                }
                (false, true) => {
                    if null_first {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                (true, false) => {
                    if null_first {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                (false, false) => Ordering::Equal,
            };
            if result != Ordering::Equal {
                // we found a relevant comparison => short-circuit and return it
                return result;
            }
        }
        Ordering::Equal
    };
    Ok(Box::new(cmp))
}
