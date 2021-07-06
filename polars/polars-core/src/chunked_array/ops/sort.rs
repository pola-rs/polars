use crate::prelude::compare_inner::PartialOrdInner;
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use itertools::Itertools;
use polars_arrow::trusted_len::PushUnchecked;
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
    b.partial_cmp(a).unwrap_or_else(|| {
        // nan != nan
        #[allow(clippy::eq_op)]
        if a != a {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    })
}

/// Default sorting when there are no nulls
fn order_default<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // nan != nan
        // this is a simple way to check if it is nan
        // without convincing the compiler we deal with floats
        #[allow(clippy::eq_op)]
        if a != a {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    })
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

        let mut vals = Vec::with_capacity($self.len());
        let mut count: u32 = 0;
        $self.downcast_iter().for_each(|arr| {
            let iter = arr.iter().map(|v| {
                let i = count;
                count += 1;
                (i, v)
            });
            vals.extend_trusted_len(iter);
        });

        argsort_branch(
            vals.as_mut_slice(),
            sort_parallel,
            $reverse,
            |(_, a), (_, b)| a.partial_cmp(b).unwrap(),
            |(_, a), (_, b)| b.partial_cmp(a).unwrap(),
        );
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
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
            // rechunk and call again, then it will fall in the contiguous slice path.
            let ca = self.rechunk();
            ca.sort(reverse)
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
            ca.set_sorted(reverse);
            ca
        }
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        let sort_parallel = sort_parallel(self);

        let ca: NoNull<UInt32Chunked> = if self.null_count() == 0 {
            let mut vals = Vec::with_capacity(self.len());
            let mut count: u32 = 0;
            self.downcast_iter().for_each(|arr| {
                let values = arr.values();
                let iter = values.iter().map(|&v| {
                    let i = count;
                    count += 1;
                    (i, v)
                });
                vals.extend_trusted_len(iter);
            });

            argsort_branch(
                vals.as_mut_slice(),
                sort_parallel,
                reverse,
                |(_, a), (_, b)| a.partial_cmp(b).unwrap(),
                |(_, a), (_, b)| b.partial_cmp(a).unwrap(),
            );

            vals.into_iter().map(|(idx, _v)| idx).collect_trusted()
        } else {
            let mut vals = Vec::with_capacity(self.len());
            let mut count: u32 = 0;
            self.downcast_iter().for_each(|arr| {
                let iter = arr.iter().map(|v| {
                    let i = count;
                    count += 1;
                    (i, v)
                });
                vals.extend_trusted_len(iter);
            });

            argsort_branch(
                vals.as_mut_slice(),
                sort_parallel,
                reverse,
                |(_, a), (_, b)| a.partial_cmp(b).unwrap(),
                |(_, a), (_, b)| b.partial_cmp(a).unwrap(),
            );
            vals.into_iter().map(|(idx, _v)| idx).collect_trusted()
        };
        let mut ca = ca.into_inner();
        ca.rename(self.name());
        ca
    }

    #[cfg(feature = "sort_multiple")]
    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn argsort_multiple(&self, other: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
        for ca in other {
            assert_eq!(self.len(), ca.len());
        }
        if other.len() != (reverse.len() - 1) {
            return Err(PolarsError::ValueError(
                format!(
                    "The amount of ordering booleans: {} does not match that no. of Series: {}",
                    reverse.len(),
                    other.len() + 1
                )
                .into(),
            ));
        }

        assert_eq!(other.len(), reverse.len() - 1);

        let compare_inner: Vec<_> = other.iter().map(|s| s.into_partial_ord_inner()).collect();

        let mut count: u32 = 0;
        let mut vals: Vec<_> = self
            .into_iter()
            .map(|v| {
                let i = count;
                count += 1;
                (i, v)
            })
            .collect();

        vals.sort_by(
            |tpl_a, tpl_b| match (reverse[0], sort_with_nulls(&tpl_a.1, &tpl_b.1)) {
                // if ordering is equal, we check the other arrays until we find a non-equal ordering
                // if we have exhausted all arrays, we keep the equal ordering.
                (_, Ordering::Equal) => {
                    let idx_a = tpl_a.0 as usize;
                    let idx_b = tpl_b.0 as usize;
                    ordering_other_columns(&compare_inner, &reverse[1..], idx_a, idx_b)
                }
                (true, Ordering::Less) => Ordering::Greater,
                (true, Ordering::Greater) => Ordering::Less,
                (_, ord) => ord,
            },
        );
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect();
        let mut ca = ca.into_inner();
        ca.set_sorted(reverse[0]);
        Ok(ca)
    }
}

fn ordering_other_columns<'a>(
    compare_inner: &'a [Box<dyn PartialOrdInner + 'a>],
    reverse: &[bool],
    idx_a: usize,
    idx_b: usize,
) -> Ordering {
    for (cmp, reverse) in compare_inner.iter().zip(reverse) {
        // Safety:
        // indices are in bounds
        let ordering = unsafe { cmp.cmp_element_unchecked(idx_a, idx_b) };
        match (ordering, reverse) {
            (Ordering::Equal, _) => continue,
            (_, true) => return ordering.reverse(),
            _ => return ordering,
        }
    }
    // all arrays/columns exhausted, ordering equal it is.
    Ordering::Equal
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
        let sort_parallel = sort_parallel(self);

        let mut v = Vec::from_iter(self);
        sort_branch(
            v.as_mut_slice(),
            sort_parallel,
            reverse,
            order_default_null,
            order_reverse_null,
        );

        // We don't collect from an iterator because we know the total value size
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.get_values_size());
        v.into_iter().for_each(|opt_v| builder.append_option(opt_v));
        let mut ca = builder.finish();
        ca.set_sorted(reverse);
        ca
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        argsort!(self, reverse)
    }

    #[cfg(feature = "sort_multiple")]
    /// # Panics
    ///
    /// This function is very opinionated. On the implementation of `ChunkedArray<T>` for numeric types,
    /// we assume that all numeric `Series` are of the same type.
    ///
    /// In this case we assume that all numeric `Series` are `f64` types. The caller needs to
    /// uphold this contract. If not, it will panic.
    ///
    fn argsort_multiple(&self, other: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
        for ca in other {
            if self.len() != ca.len() {
                return Err(PolarsError::ShapeMisMatch(
                    "sort column should have equal length".into(),
                ));
            }
        }
        assert_eq!(other.len(), reverse.len() - 1);
        let mut count: u32 = 0;
        let mut vals: Vec<_> = self
            .into_iter()
            .map(|v| {
                let i = count;
                count += 1;
                (i, v)
            })
            .collect();
        let compare_inner: Vec<_> = other.iter().map(|s| s.into_partial_ord_inner()).collect();

        vals.sort_by(
            |tpl_a, tpl_b| match (reverse[0], sort_with_nulls(&tpl_a.1, &tpl_b.1)) {
                // if ordering is equal, we check the other arrays until we find a non-equal ordering
                // if we have exhausted all arrays, we keep the equal ordering.
                (_, Ordering::Equal) => {
                    let idx_a = tpl_a.0 as usize;
                    let idx_b = tpl_b.0 as usize;
                    ordering_other_columns(&compare_inner, &reverse[1..], idx_a, idx_b)
                }
                (true, Ordering::Less) => Ordering::Greater,
                (true, Ordering::Greater) => Ordering::Less,
                (_, ord) => ord,
            },
        );
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect();
        let mut ca = ca.into_inner();
        ca.set_sorted(reverse[0]);
        Ok(ca)
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

#[cfg(feature = "sort_multiple")]
pub(crate) fn prepare_argsort(
    columns: Vec<Series>,
    mut reverse: Vec<bool>,
) -> Result<(Series, Vec<Series>, Vec<bool>)> {
    let n_cols = columns.len();

    let mut columns = columns
        .iter()
        .map(|s| {
            use DataType::*;
            match s.dtype() {
                Float32 | Float64 | Int32 | Int64 | Utf8 | UInt32 | UInt64 => s.clone(),
                _ => s.cast::<Int32Type>().unwrap(),
            }
        })
        .collect::<Vec<_>>();

    let first = columns.remove(0);

    // broadcast ordering
    if n_cols > reverse.len() && reverse.len() == 1 {
        while n_cols != reverse.len() {
            reverse.push(reverse[0]);
        }
    }
    Ok((first, columns, reverse))
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    #[cfg(feature = "sort_multiple")]
    fn test_argsort_multiple() -> Result<()> {
        let a = Int32Chunked::new_from_slice("a", &[1, 2, 1, 1, 3, 4, 3, 3]);
        let b = Int64Chunked::new_from_slice("b", &[0, 1, 2, 3, 4, 5, 6, 1]);
        let c = Utf8Chunked::new_from_slice("c", &["a", "b", "c", "d", "e", "f", "g", "h"]);
        let df = DataFrame::new(vec![a.into_series(), b.into_series(), c.into_series()])?;

        let out = df.sort(&["a", "b", "c"], false)?;
        assert_eq!(
            Vec::from(out.column("b")?.i64()?),
            &[
                Some(0),
                Some(2),
                Some(3),
                Some(1),
                Some(1),
                Some(4),
                Some(6),
                Some(5)
            ]
        );

        // now let the first sort be a string
        let a = Utf8Chunked::new_from_slice("a", &["a", "b", "c", "a", "b", "c"]).into_series();
        let b = Int32Chunked::new_from_slice("b", &[5, 4, 2, 3, 4, 5]).into_series();
        let df = DataFrame::new(vec![a, b])?;

        let out = df.sort(&["a", "b"], false)?;
        let expected = df!(
            "a" => ["a", "a", "b", "b", "c", "c"],
            "b" => [3, 5, 4, 4, 2, 5]
        )?;
        assert!(out.frame_equal(&expected));

        let df = df!(
            "groups" => [1, 2, 3],
            "values" => ["a", "a", "b"]
        )?;

        let out = df.sort(&["groups", "values"], vec![true, false])?;
        let expected = df!(
            "groups" => [3, 2, 1],
            "values" => ["b", "a", "a"]
        )?;
        assert!(out.frame_equal(&expected));

        let out = df.sort(&["values", "groups"], vec![false, true])?;
        let expected = df!(
            "groups" => [2, 1, 3],
            "values" => ["a", "a", "b"]
        )?;
        assert!(out.frame_equal(&expected));

        Ok(())
    }
}
