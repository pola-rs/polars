use crate::prelude::*;
use crate::utils::NoNull;
use itertools::Itertools;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

/// Sort with null values, to reverse, swap the arguments.
fn sort_with_nulls<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.partial_cmp(b).expect("could not compare"),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Reverse sorting when there are no nulls
fn order_reverse<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    b.partial_cmp(a).unwrap()
}

/// Default sorting when there are no nulls
fn order_default<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap()
}

/// Default sorting nulls
fn order_default_null<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    sort_with_nulls(a, b)
}

/// Default sorting nulls
fn order_reverse_null<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    sort_with_nulls(b, a)
}

fn sort_branch<T, Fd, Fr>(
    slice: &mut [T],
    sort_parallel: bool,
    reverse: bool,
    default_order_fn: Fd,
    reverse_order_fn: Fr,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
{
    match (sort_parallel, reverse) {
        (true, true) => slice.par_sort_unstable_by(reverse_order_fn),
        (true, false) => slice.par_sort_unstable_by(default_order_fn),
        (false, true) => slice.sort_unstable_by(reverse_order_fn),
        (false, false) => slice.sort_unstable_by(default_order_fn),
    }
}

fn argsort_branch<T, Fd, Fr>(
    slice: &mut [T],
    sort_parallel: bool,
    reverse: bool,
    default_order_fn: Fd,
    reverse_order_fn: Fr,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
{
    match (sort_parallel, reverse) {
        (true, true) => slice.par_sort_by(reverse_order_fn),
        (true, false) => slice.par_sort_by(default_order_fn),
        (false, true) => slice.sort_by(reverse_order_fn),
        (false, false) => slice.sort_by(default_order_fn),
    }
}

/// If the sort should be ran parallel or not.
fn sort_parallel<T>(ca: &ChunkedArray<T>) -> bool {
    ca.len()
        > std::env::var("POLARS_PAR_SORT_BOUND")
            .map(|v| v.parse::<usize>().expect("could not parse"))
            .unwrap_or(1000000)
}

macro_rules! argsort {
    ($self:expr, $reverse:expr) => {{
        let sort_parallel = sort_parallel($self);

        let ca: NoNull<UInt32Chunked> = if $self.null_count() == 0 {
            let mut count: u32 = 0;
            let mut vals: Vec<_> = $self
                .into_no_null_iter()
                .map(|v| {
                    let i = count;
                    count += 1;
                    (i, v)
                })
                .collect();

            argsort_branch(
                vals.as_mut_slice(),
                sort_parallel,
                $reverse,
                |(_, a), (_, b)| a.partial_cmp(b).unwrap(),
                |(_, a), (_, b)| b.partial_cmp(a).unwrap(),
            );

            vals.into_iter().map(|(idx, _v)| idx).collect()
        } else {
            let mut count: u32 = 0;
            let mut vals: Vec<_> = $self
                .into_iter()
                .map(|v| {
                    let i = count;
                    count += 1;
                    (i, v)
                })
                .collect();

            argsort_branch(
                vals.as_mut_slice(),
                sort_parallel,
                $reverse,
                |(_, a), (_, b)| order_default_null(a, b),
                |(_, a), (_, b)| order_reverse_null(a, b),
            );

            vals.into_iter().map(|(idx, _v)| idx).collect()
        };
        let mut ca = ca.into_inner();
        ca.rename($self.name());
        ca
    }};
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: std::cmp::PartialOrd,
{
    fn sort(&self, reverse: bool) -> ChunkedArray<T> {
        let sort_parallel = sort_parallel(self);

        if let Ok(vals) = self.cont_slice() {
            // Copy the values to a new aligned vec. This can be mutably sorted.
            let n = self.len();
            let vals_ptr = vals.as_ptr();
            // allocate aligned
            let mut new = AlignedVec::<T::Native>::with_capacity_aligned(n);
            let new_ptr = new.as_mut_ptr();

            // memcopy
            unsafe { std::ptr::copy_nonoverlapping(vals_ptr, new_ptr, n) };
            // set len to copied bytes
            unsafe { new.set_len(n) };

            sort_branch(
                new.as_mut_slice(),
                sort_parallel,
                reverse,
                order_default,
                order_reverse,
            );

            return ChunkedArray::new_from_aligned_vec(self.name(), new);
        }

        if self.null_count() == 0 {
            let mut av: AlignedVec<_> = self.into_no_null_iter().collect();
            sort_branch(
                av.as_mut_slice(),
                sort_parallel,
                reverse,
                order_default,
                order_reverse,
            );
            ChunkedArray::new_from_aligned_vec(self.name(), av)
        } else {
            let mut v = Vec::from_iter(self);
            sort_branch(
                v.as_mut_slice(),
                sort_parallel,
                reverse,
                order_default_null,
                order_reverse_null,
            );
            let mut ca: Self = v.into_iter().collect();
            ca.rename(self.name());
            ca
        }
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        argsort!(self, reverse)
    }
}

macro_rules! sort {
    ($self:ident, $reverse:ident) => {{
        if $reverse {
            $self.into_iter().sorted_by(|a, b| b.cmp(a)).collect()
        } else {
            $self.into_iter().sorted_by(|a, b| a.cmp(b)).collect()
        }
    }};
}

impl ChunkSort<Utf8Type> for Utf8Chunked {
    fn sort(&self, reverse: bool) -> Utf8Chunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        argsort!(self, reverse)
    }
}

impl ChunkSort<CategoricalType> for CategoricalChunked {
    fn sort(&self, reverse: bool) -> Self {
        self.as_ref().sort(reverse).cast().unwrap()
    }

    fn sort_in_place(&mut self, reverse: bool) {
        self.deref_mut().sort_in_place(reverse)
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        self.deref().argsort(reverse)
    }
}

impl ChunkSort<ListType> for ListChunked {
    fn sort(&self, _reverse: bool) -> Self {
        unimplemented!()
    }

    fn sort_in_place(&mut self, _reverse: bool) {
        unimplemented!()
    }

    fn argsort(&self, _reverse: bool) -> UInt32Chunked {
        unimplemented!()
    }
}

#[cfg(feature = "object")]
impl<T> ChunkSort<ObjectType<T>> for ObjectChunked<T> {
    fn sort(&self, _reverse: bool) -> Self {
        unimplemented!()
    }

    fn sort_in_place(&mut self, _reverse: bool) {
        unimplemented!()
    }

    fn argsort(&self, _reverse: bool) -> UInt32Chunked {
        unimplemented!()
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort(&self, reverse: bool) -> BooleanChunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        argsort!(self, reverse)
    }
}
