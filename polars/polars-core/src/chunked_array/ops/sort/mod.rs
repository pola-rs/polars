mod arg_sort;
#[cfg(feature = "sort_multiple")]
mod arg_sort_multiple;
#[cfg(feature = "dtype-categorical")]
mod categorical;

use std::cmp::Ordering;
use std::hint::unreachable_unchecked;
use std::iter::FromIterator;

use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use num::Float;
use polars_arrow::array::default_arrays::FromDataUtf8;
use polars_arrow::kernels::rolling::compare_fn_nan_max;
use polars_arrow::prelude::{FromData, ValueSize};
use polars_arrow::trusted_len::PushUnchecked;
use rayon::prelude::*;

use crate::prelude::compare_inner::PartialOrdInner;
#[cfg(feature = "sort_multiple")]
use crate::prelude::sort::arg_sort_multiple::{arg_sort_multiple_impl, args_validate};
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{CustomIterTools, NoNull};

/// Reverse sorting when there are no nulls
fn order_reverse<T: Ord>(a: &T, b: &T) -> Ordering {
    b.cmp(a)
}

/// Default sorting when there are no nulls
fn order_default<T: Ord>(a: &T, b: &T) -> Ordering {
    a.cmp(b)
}

fn order_default_flt<T: Float>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // Safety: PlIsNan is only implemented for numbers
            _ => unsafe { unreachable_unchecked() },
        }
    })
}

fn order_reverse_flt<T: Float>(a: &T, b: &T) -> Ordering {
    order_default_flt(b, a)
}

fn sort_branch<T, Fd, Fr>(
    slice: &mut [T],
    reverse: bool,
    default_order_fn: Fd,
    reverse_order_fn: Fr,
    parallel: bool,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
{
    if parallel {
        match reverse {
            true => slice.par_sort_unstable_by(reverse_order_fn),
            false => slice.par_sort_unstable_by(default_order_fn),
        }
    } else {
        match reverse {
            true => slice.sort_unstable_by(reverse_order_fn),
            false => slice.sort_unstable_by(default_order_fn),
        }
    }
}

#[cfg(feature = "private")]
pub fn arg_sort_no_nulls<Idx, T>(slice: &mut [(Idx, T)], reverse: bool, parallel: bool)
where
    T: PartialOrd + Send + IsFloat,
    Idx: PartialOrd + Send,
{
    arg_sort_branch(
        slice,
        reverse,
        |(_, a), (_, b)| compare_fn_nan_max(a, b),
        |(_, a), (_, b)| compare_fn_nan_max(b, a),
        parallel,
    );
}

pub fn arg_sort_branch<T, Fd, Fr>(
    slice: &mut [T],
    reverse: bool,
    default_order_fn: Fd,
    reverse_order_fn: Fr,
    parallel: bool,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
{
    if parallel {
        match reverse {
            true => slice.par_sort_by(reverse_order_fn),
            false => slice.par_sort_by(default_order_fn),
        }
    } else {
        match reverse {
            true => slice.sort_by(reverse_order_fn),
            false => slice.sort_by(default_order_fn),
        }
    }
}

fn memcpy_values<T>(ca: &ChunkedArray<T>) -> Vec<T::Native>
where
    T: PolarsNumericType,
{
    let len = ca.len();
    let mut vals = Vec::with_capacity(len);

    ca.downcast_iter().for_each(|arr| {
        let values = arr.values().as_slice();
        vals.extend_from_slice(values);
    });
    vals
}

macro_rules! sort_with_fast_path {
    ($ca:ident, $options:expr) => {{
        if $ca.is_empty() {
            return $ca.clone();
        }

        // we can clone if we sort in same order
        if $options.descending && $ca.is_sorted_reverse_flag() || ($ca.is_sorted_flag() && !$options.descending) {
            // there are nulls
            if $ca.null_count() > 0 {
                // if the nulls are already last we can clone
                if $options.nulls_last && $ca.get($ca.len() - 1).is_none()  ||
                // if the nulls are already first we can clone
                $ca.get(0).is_none()
                {
                    return $ca.clone();
                }
                // nulls are not at the right place
                // continue w/ sorting
                // TODO: we can optimize here and just put the null at the correct place
            } else {
                return $ca.clone();
            }
        }
        // we can reverse if we sort in other order
        else if ($options.descending && $ca.is_sorted_flag() || $ca.is_sorted_reverse_flag()) && $ca.null_count() == 0 {
            return $ca.reverse()
        };


    }}
}

fn sort_with_numeric<T>(
    ca: &ChunkedArray<T>,
    options: SortOptions,
    order_default: fn(&T::Native, &T::Native) -> Ordering,
    order_reverse: fn(&T::Native, &T::Native) -> Ordering,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    sort_with_fast_path!(ca, options);
    if ca.null_count() == 0 {
        let mut vals = memcpy_values(ca);

        sort_branch(
            vals.as_mut_slice(),
            options.descending,
            order_default,
            order_reverse,
            options.multithreaded,
        );

        let mut ca = ChunkedArray::from_vec(ca.name(), vals);
        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        ca.set_sorted_flag(s);
        ca
    } else {
        let null_count = ca.null_count();
        let len = ca.len();

        let mut vals = Vec::with_capacity(ca.len());

        if !options.nulls_last {
            let iter = std::iter::repeat(T::Native::default()).take(null_count);
            vals.extend(iter);
        }

        ca.downcast_iter().for_each(|arr| {
            let iter = arr.iter().filter_map(|v| v.copied());
            vals.extend(iter);
        });
        let mut_slice = if options.nulls_last {
            &mut vals[..len - null_count]
        } else {
            &mut vals[null_count..]
        };

        sort_branch(
            mut_slice,
            options.descending,
            order_default,
            order_reverse,
            options.multithreaded,
        );

        let mut ca: ChunkedArray<T> = if options.nulls_last {
            vals.extend(std::iter::repeat(T::Native::default()).take(ca.null_count()));
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len - null_count, true);
            validity.extend_constant(null_count, false);

            (
                ca.name(),
                PrimitiveArray::new(
                    T::get_dtype().to_arrow(),
                    vals.into(),
                    Some(validity.into()),
                ),
            )
                .into()
        } else {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(null_count, false);
            validity.extend_constant(len - null_count, true);

            (
                ca.name(),
                PrimitiveArray::new(
                    T::get_dtype().to_arrow(),
                    vals.into(),
                    Some(validity.into()),
                ),
            )
                .into()
        };

        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        ca.set_sorted_flag(s);
        ca
    }
}

fn arg_sort_numeric<T>(ca: &ChunkedArray<T>, options: SortOptions) -> IdxCa
where
    T: PolarsNumericType,
{
    let reverse = options.descending;
    if ca.null_count() == 0 {
        let mut vals = Vec::with_capacity(ca.len());
        let mut count: IdxSize = 0;
        ca.downcast_iter().for_each(|arr| {
            let values = arr.values();
            let iter = values.iter().map(|&v| {
                let i = count;
                count += 1;
                (i, v)
            });
            vals.extend_trusted_len(iter);
        });

        arg_sort_no_nulls(vals.as_mut_slice(), reverse, options.multithreaded);

        let out: NoNull<IdxCa> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
        let mut out = out.into_inner();
        out.rename(ca.name());
        out
    } else {
        let iter = ca
            .downcast_iter()
            .map(|arr| arr.iter().map(|opt| opt.copied()));
        arg_sort::arg_sort(ca.name(), iter, options, ca.null_count(), ca.len())
    }
}

#[cfg(feature = "sort_multiple")]
fn arg_sort_multiple_numeric<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    other: &[Series],
    reverse: &[bool],
) -> PolarsResult<IdxCa> {
    args_validate(ca, other, reverse)?;
    let mut count: IdxSize = 0;
    let vals: Vec<_> = ca
        .into_iter()
        .map(|v| {
            let i = count;
            count += 1;
            (i, v)
        })
        .collect_trusted();

    arg_sort_multiple_impl(vals, other, reverse)
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Default + Ord,
{
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<T> {
        sort_with_numeric(self, options, order_default, order_reverse)
    }

    fn sort(&self, reverse: bool) -> ChunkedArray<T> {
        self.sort_with(SortOptions {
            descending: reverse,
            ..Default::default()
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort_numeric(self, options)
    }

    #[cfg(feature = "sort_multiple")]
    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn arg_sort_multiple(&self, other: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, other, reverse)
    }
}

impl ChunkSort<Float32Type> for Float32Chunked {
    fn sort_with(&self, options: SortOptions) -> Float32Chunked {
        sort_with_numeric(self, options, order_default_flt, order_reverse_flt)
    }

    fn sort(&self, reverse: bool) -> Float32Chunked {
        self.sort_with(SortOptions {
            descending: reverse,
            ..Default::default()
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort_numeric(self, options)
    }

    #[cfg(feature = "sort_multiple")]
    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn arg_sort_multiple(&self, other: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, other, reverse)
    }
}

impl ChunkSort<Float64Type> for Float64Chunked {
    fn sort_with(&self, options: SortOptions) -> Float64Chunked {
        sort_with_numeric(self, options, order_default_flt, order_reverse_flt)
    }

    fn sort(&self, reverse: bool) -> Float64Chunked {
        self.sort_with(SortOptions {
            descending: reverse,
            ..Default::default()
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort_numeric(self, options)
    }

    #[cfg(feature = "sort_multiple")]
    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn arg_sort_multiple(&self, other: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, other, reverse)
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

impl ChunkSort<Utf8Type> for Utf8Chunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<Utf8Type> {
        sort_with_fast_path!(self, options);
        let mut v: Vec<&str> = if self.null_count() > 0 {
            Vec::from_iter(self.into_iter().flatten())
        } else {
            Vec::from_iter(self.into_no_null_iter())
        };

        sort_branch(
            v.as_mut_slice(),
            options.descending,
            order_default,
            order_reverse,
            options.multithreaded,
        );

        let mut values = Vec::<u8>::with_capacity(self.get_values_size());
        let mut offsets = Vec::<i64>::with_capacity(self.len() + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let len = self.len();
        let null_count = self.null_count();
        let mut ca: Self = match (null_count, options.nulls_last) {
            (0, _) => {
                for val in v {
                    values.extend_from_slice(val.as_bytes());
                    length_so_far = values.len() as i64;
                    offsets.push(length_so_far);
                }
                // Safety:
                // we pass valid utf8
                let ar = unsafe {
                    Utf8Array::from_data_unchecked_default(offsets.into(), values.into(), None)
                };
                (self.name(), ar).into()
            }
            (_, true) => {
                for val in v {
                    values.extend_from_slice(val.as_bytes());
                    length_so_far = values.len() as i64;
                    offsets.push(length_so_far);
                }
                let mut validity = MutableBitmap::with_capacity(len);
                validity.extend_constant(len - null_count, true);
                validity.extend_constant(null_count, false);
                offsets.extend(std::iter::repeat(length_so_far).take(null_count));

                // Safety:
                // we pass valid utf8
                let ar = unsafe {
                    Utf8Array::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(validity.into()),
                    )
                };
                (self.name(), ar).into()
            }
            (_, false) => {
                let mut validity = MutableBitmap::with_capacity(len);
                validity.extend_constant(null_count, false);
                validity.extend_constant(len - null_count, true);
                offsets.extend(std::iter::repeat(length_so_far).take(null_count));

                for val in v {
                    values.extend_from_slice(val.as_bytes());
                    length_so_far = values.len() as i64;
                    offsets.push(length_so_far);
                }

                // Safety:
                // we pass valid utf8
                let ar = unsafe {
                    Utf8Array::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(validity.into()),
                    )
                };
                (self.name(), ar).into()
            }
        };

        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        ca.set_sorted_flag(s);
        ca
    }

    fn sort(&self, reverse: bool) -> Utf8Chunked {
        self.sort_with(SortOptions {
            descending: reverse,
            nulls_last: false,
            multithreaded: true,
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort::arg_sort(
            self.name(),
            self.downcast_iter().map(|arr| arr.iter()),
            options,
            self.null_count(),
            self.len(),
        )
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
    fn arg_sort_multiple(&self, other: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        args_validate(self, other, reverse)?;

        let mut count: IdxSize = 0;
        let vals: Vec<_> = self
            .into_iter()
            .map(|v| {
                let i = count;
                count += 1;
                (i, v)
            })
            .collect_trusted();
        arg_sort_multiple_impl(vals, other, reverse)
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkSort<BinaryType> for BinaryChunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<BinaryType> {
        sort_with_fast_path!(self, options);
        let mut v: Vec<&[u8]> = if self.null_count() > 0 {
            Vec::from_iter(self.into_iter().flatten())
        } else {
            Vec::from_iter(self.into_no_null_iter())
        };

        sort_branch(
            v.as_mut_slice(),
            options.descending,
            order_default,
            order_reverse,
            options.multithreaded,
        );

        let mut values = Vec::<u8>::with_capacity(self.get_values_size());
        let mut offsets = Vec::<i64>::with_capacity(self.len() + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let len = self.len();
        let null_count = self.null_count();
        let mut ca: Self = match (null_count, options.nulls_last) {
            (0, _) => {
                for val in v {
                    values.extend_from_slice(val);
                    length_so_far = values.len() as i64;
                    offsets.push(length_so_far);
                }
                // Safety:
                // we pass valid utf8
                let ar = unsafe {
                    BinaryArray::from_data_unchecked_default(offsets.into(), values.into(), None)
                };
                (self.name(), ar).into()
            }
            (_, true) => {
                for val in v {
                    values.extend_from_slice(val);
                    length_so_far = values.len() as i64;
                    offsets.push(length_so_far);
                }
                let mut validity = MutableBitmap::with_capacity(len);
                validity.extend_constant(len - null_count, true);
                validity.extend_constant(null_count, false);
                offsets.extend(std::iter::repeat(length_so_far).take(null_count));

                // Safety:
                // we pass valid utf8
                let ar = unsafe {
                    BinaryArray::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(validity.into()),
                    )
                };
                (self.name(), ar).into()
            }
            (_, false) => {
                let mut validity = MutableBitmap::with_capacity(len);
                validity.extend_constant(null_count, false);
                validity.extend_constant(len - null_count, true);
                offsets.extend(std::iter::repeat(length_so_far).take(null_count));

                for val in v {
                    values.extend_from_slice(val);
                    length_so_far = values.len() as i64;
                    offsets.push(length_so_far);
                }

                // Safety:
                // we pass valid utf8
                let ar = unsafe {
                    BinaryArray::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(validity.into()),
                    )
                };
                (self.name(), ar).into()
            }
        };

        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        ca.set_sorted_flag(s);
        ca
    }

    fn sort(&self, reverse: bool) -> BinaryChunked {
        self.sort_with(SortOptions {
            descending: reverse,
            nulls_last: false,
            multithreaded: true,
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort::arg_sort(
            self.name(),
            self.downcast_iter().map(|arr| arr.iter()),
            options,
            self.null_count(),
            self.len(),
        )
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
    fn arg_sort_multiple(&self, other: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        args_validate(self, other, reverse)?;

        let mut count: IdxSize = 0;
        let vals: Vec<_> = self
            .into_iter()
            .map(|v| {
                let i = count;
                count += 1;
                (i, v)
            })
            .collect_trusted();
        arg_sort_multiple_impl(vals, other, reverse)
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<BooleanType> {
        sort_with_fast_path!(self, options);
        assert!(
            !options.nulls_last,
            "null last not yet supported for bool dtype"
        );
        let mut vals = self.into_iter().collect::<Vec<_>>();

        if options.descending {
            vals.sort_by(|a, b| b.cmp(a))
        } else {
            vals.sort()
        }

        let mut ca: BooleanChunked = vals.into_iter().collect_trusted();
        ca.rename(self.name());
        ca
    }

    fn sort(&self, reverse: bool) -> BooleanChunked {
        self.sort_with(SortOptions {
            descending: reverse,
            nulls_last: false,
            multithreaded: true,
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort::arg_sort(
            self.name(),
            self.downcast_iter().map(|arr| arr.iter()),
            options,
            self.null_count(),
            self.len(),
        )
    }
}

#[cfg(feature = "sort_multiple")]
pub(crate) fn prepare_arg_sort(
    columns: Vec<Series>,
    mut reverse: Vec<bool>,
) -> PolarsResult<(Series, Vec<Series>, Vec<bool>)> {
    let n_cols = columns.len();

    let mut columns = columns
        .iter()
        .map(|s| {
            use DataType::*;
            match s.dtype() {
                Float32 | Float64 | Int32 | Int64 | Utf8 | UInt32 | UInt64 => s.clone(),
                #[cfg(feature = "dtype-categorical")]
                Categorical(_) => s.rechunk(),
                _ => {
                    // small integers i8, u8 etc are casted to reduce compiler bloat
                    // not that we don't expect any logical types at this point
                    if s.bit_repr_is_large() {
                        s.cast(&DataType::Int64).unwrap()
                    } else {
                        s.cast(&DataType::Int32).unwrap()
                    }
                }
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
    fn test_arg_sort() {
        let a = Int32Chunked::new(
            "a",
            &[
                Some(1), // 0
                Some(5), // 1
                None,    // 2
                Some(1), // 3
                None,    // 4
                Some(4), // 5
                Some(3), // 6
                Some(1), // 7
            ],
        );
        let idx = a.arg_sort(SortOptions {
            descending: false,
            ..Default::default()
        });
        let idx = idx.cont_slice().unwrap();

        let expected = [2, 4, 0, 3, 7, 6, 5, 1];
        assert_eq!(idx, expected);

        let idx = a.arg_sort(SortOptions {
            descending: true,
            ..Default::default()
        });
        let idx = idx.cont_slice().unwrap();
        // the duplicates are in reverse order of appearance, so we cannot reverse expected
        let expected = [1, 5, 6, 0, 3, 7, 4, 2];
        assert_eq!(idx, expected);
    }

    #[test]
    fn test_sort() {
        let a = Int32Chunked::new(
            "a",
            &[
                Some(1),
                Some(5),
                None,
                Some(1),
                None,
                Some(4),
                Some(3),
                Some(1),
            ],
        );
        let out = a.sort_with(SortOptions {
            descending: false,
            nulls_last: false,
            multithreaded: true,
        });
        assert_eq!(
            Vec::from(&out),
            &[
                None,
                None,
                Some(1),
                Some(1),
                Some(1),
                Some(3),
                Some(4),
                Some(5)
            ]
        );
        let out = a.sort_with(SortOptions {
            descending: false,
            nulls_last: true,
            multithreaded: true,
        });
        assert_eq!(
            Vec::from(&out),
            &[
                Some(1),
                Some(1),
                Some(1),
                Some(3),
                Some(4),
                Some(5),
                None,
                None
            ]
        );
    }

    #[test]
    #[cfg(feature = "sort_multiple")]
    #[cfg_attr(miri, ignore)]
    fn test_arg_sort_multiple() -> PolarsResult<()> {
        let a = Int32Chunked::new("a", &[1, 2, 1, 1, 3, 4, 3, 3]);
        let b = Int64Chunked::new("b", &[0, 1, 2, 3, 4, 5, 6, 1]);
        let c = Utf8Chunked::new("c", &["a", "b", "c", "d", "e", "f", "g", "h"]);
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
        let a = Utf8Chunked::new("a", &["a", "b", "c", "a", "b", "c"]).into_series();
        let b = Int32Chunked::new("b", &[5, 4, 2, 3, 4, 5]).into_series();
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

    #[test]
    fn test_sort_utf8() {
        let ca = Utf8Chunked::new("a", &[Some("a"), None, Some("c"), None, Some("b")]);
        let out = ca.sort_with(SortOptions {
            descending: false,
            nulls_last: false,
            multithreaded: true,
        });
        let expected = &[None, None, Some("a"), Some("b"), Some("c")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: true,
            nulls_last: false,
            multithreaded: true,
        });

        let expected = &[None, None, Some("c"), Some("b"), Some("a")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: false,
            nulls_last: true,
            multithreaded: true,
        });
        let expected = &[Some("a"), Some("b"), Some("c"), None, None];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: true,
            nulls_last: true,
            multithreaded: true,
        });
        let expected = &[Some("c"), Some("b"), Some("a"), None, None];
        assert_eq!(Vec::from(&out), expected);

        // no nulls
        let ca = Utf8Chunked::new("a", &[Some("a"), Some("c"), Some("b")]);
        let out = ca.sort(false);
        let expected = &[Some("a"), Some("b"), Some("c")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort(true);
        let expected = &[Some("c"), Some("b"), Some("a")];
        assert_eq!(Vec::from(&out), expected);
    }
}
