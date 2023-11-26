mod arg_sort;

pub mod arg_sort_multiple;
#[cfg(feature = "dtype-categorical")]
mod categorical;
mod slice;

use std::cmp::Ordering;
use std::hint::unreachable_unchecked;
use std::iter::FromIterator;

pub(crate) use arg_sort_multiple::argsort_multiple_row_fmt;
use arrow::array::ValueSize;
use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use arrow::legacy::prelude::FromData;
use arrow::legacy::trusted_len::TrustedLenPush;
use num_traits::Float;
use polars_utils::float::IsFloat;
use polars_utils::ord::compare_fn_nan_max;
use rayon::prelude::*;
pub use slice::*;

use crate::prelude::compare_inner::PartialOrdInner;
#[cfg(feature = "dtype-struct")]
use crate::prelude::sort::arg_sort_multiple::_get_rows_encoded_ca;
use crate::prelude::sort::arg_sort_multiple::{arg_sort_multiple_impl, args_validate};
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{CustomIterTools, NoNull};
use crate::POOL;

/// Reverse sorting when there are no nulls
#[inline]
fn order_descending<T: Ord>(a: &T, b: &T) -> Ordering {
    b.cmp(a)
}

/// Default sorting when there are no nulls
#[inline]
fn order_ascending<T: Ord>(a: &T, b: &T) -> Ordering {
    a.cmp(b)
}

#[inline]
fn order_ascending_flt<T: Float>(a: &T, b: &T) -> Ordering {
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

#[inline]
fn order_descending_flt<T: Float>(a: &T, b: &T) -> Ordering {
    order_ascending_flt(b, a)
}

#[inline]
fn sort_branch<T, Fd, Fr>(
    slice: &mut [T],
    descending: bool,
    ascending_order_fn: Fd,
    descending_order_fn: Fr,
    parallel: bool,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync + Send,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync + Send,
{
    if parallel {
        POOL.install(|| match descending {
            true => slice.par_sort_unstable_by(descending_order_fn),
            false => slice.par_sort_unstable_by(ascending_order_fn),
        })
    } else {
        match descending {
            true => slice.sort_unstable_by(descending_order_fn),
            false => slice.sort_unstable_by(ascending_order_fn),
        }
    }
}

pub fn arg_sort_no_nulls<Idx, T>(slice: &mut [(Idx, T)], descending: bool, parallel: bool)
where
    T: PartialOrd + Send + IsFloat,
    Idx: PartialOrd + Send,
{
    arg_sort_branch(
        slice,
        descending,
        |(_, a), (_, b)| compare_fn_nan_max(a, b),
        |(_, a), (_, b)| compare_fn_nan_max(b, a),
        parallel,
    );
}

pub(crate) fn arg_sort_branch<T, Fd, Fr>(
    slice: &mut [T],
    descending: bool,
    ascending_order_fn: Fd,
    descending_order_fn: Fr,
    parallel: bool,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync + Send,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync + Send,
{
    if parallel {
        POOL.install(|| match descending {
            true => slice.par_sort_by(descending_order_fn),
            false => slice.par_sort_by(ascending_order_fn),
        })
    } else {
        match descending {
            true => slice.sort_by(descending_order_fn),
            false => slice.sort_by(ascending_order_fn),
        }
    }
}

macro_rules! sort_with_fast_path {
    ($ca:ident, $options:expr) => {{
        if $ca.is_empty() {
            return $ca.clone();
        }

        // we can clone if we sort in same order
        if $options.descending && $ca.is_sorted_descending_flag() || ($ca.is_sorted_ascending_flag() && !$options.descending) {
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
        else if ($options.descending && $ca.is_sorted_ascending_flag() || $ca.is_sorted_descending_flag()) && $ca.null_count() == 0 {
            return $ca.reverse()
        };


    }}
}

fn sort_with_numeric<T>(
    ca: &ChunkedArray<T>,
    options: SortOptions,
    order_ascending: fn(&T::Native, &T::Native) -> Ordering,
    order_descending: fn(&T::Native, &T::Native) -> Ordering,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    sort_with_fast_path!(ca, options);
    if ca.null_count() == 0 {
        let mut vals = ca.to_vec_null_aware().left().unwrap();

        sort_branch(
            vals.as_mut_slice(),
            options.descending,
            order_ascending,
            order_descending,
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
            order_ascending,
            order_descending,
            options.multithreaded,
        );

        let mut validity = MutableBitmap::with_capacity(len);
        if options.nulls_last {
            vals.extend(std::iter::repeat(T::Native::default()).take(ca.null_count()));
            validity.extend_constant(len - null_count, true);
            validity.extend_constant(null_count, false);
        } else {
            validity.extend_constant(null_count, false);
            validity.extend_constant(len - null_count, true);
        };

        let arr = PrimitiveArray::new(
            T::get_dtype().to_arrow(),
            vals.into(),
            Some(validity.into()),
        );
        let mut new_ca = ChunkedArray::with_chunk(ca.name(), arr);
        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        new_ca.set_sorted_flag(s);
        new_ca
    }
}

fn arg_sort_numeric<T>(ca: &ChunkedArray<T>, options: SortOptions) -> IdxCa
where
    T: PolarsNumericType,
{
    let descending = options.descending;
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

        arg_sort_no_nulls(vals.as_mut_slice(), descending, options.multithreaded);

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

fn arg_sort_multiple_numeric<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    options: &SortMultipleOptions,
) -> PolarsResult<IdxCa> {
    args_validate(ca, &options.other, &options.descending)?;
    let mut count: IdxSize = 0;

    let no_nulls = ca.null_count() == 0;

    if no_nulls {
        let mut vals = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            vals.extend_trusted_len(arr.values().as_slice().iter().map(|v| {
                let i = count;
                count += 1;
                (i, *v)
            }))
        }
        arg_sort_multiple_impl(vals, options)
    } else {
        let mut vals = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            vals.extend_trusted_len(arr.into_iter().map(|v| {
                let i = count;
                count += 1;
                (i, v.copied())
            }));
        }
        arg_sort_multiple_impl(vals, options)
    }
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Default + Ord,
{
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<T> {
        sort_with_numeric(self, options, order_ascending, order_descending)
    }

    fn sort(&self, descending: bool) -> ChunkedArray<T> {
        self.sort_with(SortOptions {
            descending,
            ..Default::default()
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort_numeric(self, options)
    }

    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, options)
    }
}

impl ChunkSort<Float32Type> for Float32Chunked {
    fn sort_with(&self, options: SortOptions) -> Float32Chunked {
        sort_with_numeric(self, options, order_ascending_flt, order_descending_flt)
    }

    fn sort(&self, descending: bool) -> Float32Chunked {
        self.sort_with(SortOptions {
            descending,
            ..Default::default()
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort_numeric(self, options)
    }

    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, options)
    }
}

impl ChunkSort<Float64Type> for Float64Chunked {
    fn sort_with(&self, options: SortOptions) -> Float64Chunked {
        sort_with_numeric(self, options, order_ascending_flt, order_descending_flt)
    }

    fn sort(&self, descending: bool) -> Float64Chunked {
        self.sort_with(SortOptions {
            descending,
            ..Default::default()
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        arg_sort_numeric(self, options)
    }

    /// # Panics
    ///
    /// This function is very opinionated.
    /// We assume that all numeric `Series` are of the same type, if not it will panic
    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, options)
    }
}

fn ordering_other_columns<'a>(
    compare_inner: &'a [Box<dyn PartialOrdInner + 'a>],
    descending: &[bool],
    idx_a: usize,
    idx_b: usize,
) -> Ordering {
    for (cmp, descending) in compare_inner.iter().zip(descending) {
        // Safety:
        // indices are in bounds
        let ordering = unsafe { cmp.cmp_element_unchecked(idx_a, idx_b) };
        match (ordering, descending) {
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
        unsafe { self.as_binary().sort_with(options).to_utf8() }
    }

    fn sort(&self, descending: bool) -> Utf8Chunked {
        self.sort_with(SortOptions {
            descending,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.as_binary().arg_sort(options)
    }

    /// # Panics
    ///
    /// This function is very opinionated. On the implementation of `ChunkedArray<T>` for numeric types,
    /// we assume that all numeric `Series` are of the same type.
    ///
    /// In this case we assume that all numeric `Series` are `f64` types. The caller needs to
    /// uphold this contract. If not, it will panic.
    ///
    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        self.as_binary().arg_sort_multiple(options)
    }
}

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
            order_ascending,
            order_descending,
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
                // SAFETY: offsets are correctly created.
                let arr = unsafe {
                    BinaryArray::from_data_unchecked_default(offsets.into(), values.into(), None)
                };
                ChunkedArray::with_chunk(self.name(), arr)
            },
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

                // SAFETY: offsets are correctly created.
                let arr = unsafe {
                    BinaryArray::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(validity.into()),
                    )
                };
                ChunkedArray::with_chunk(self.name(), arr)
            },
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

                // SAFETY: we pass valid UTF-8.
                let arr = unsafe {
                    BinaryArray::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(validity.into()),
                    )
                };
                ChunkedArray::with_chunk(self.name(), arr)
            },
        };

        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        ca.set_sorted_flag(s);
        ca
    }

    fn sort(&self, descending: bool) -> BinaryChunked {
        self.sort_with(SortOptions {
            descending,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
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

    /// # Panics
    ///
    /// This function is very opinionated. On the implementation of `ChunkedArray<T>` for numeric types,
    /// we assume that all numeric `Series` are of the same type.
    ///
    /// In this case we assume that all numeric `Series` are `f64` types. The caller needs to
    /// uphold this contract. If not, it will panic.
    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        args_validate(self, &options.other, &options.descending)?;

        let mut count: IdxSize = 0;
        let vals: Vec<_> = self
            .into_iter()
            .map(|v| {
                let i = count;
                count += 1;
                (i, v)
            })
            .collect_trusted();
        arg_sort_multiple_impl(vals, options)
    }
}

#[cfg(feature = "dtype-struct")]
impl StructChunked {
    pub(crate) fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let bin = _get_rows_encoded_ca(
            self.name(),
            &[self.clone().into_series()],
            &[options.descending],
            options.nulls_last,
        )
        .unwrap();
        bin.arg_sort(Default::default())
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<BooleanType> {
        sort_with_fast_path!(self, options);
        assert!(
            !options.nulls_last,
            "null last not yet supported for bool dtype"
        );
        if self.null_count() == 0 {
            let len = self.len();
            let n_set = self.sum().unwrap() as usize;
            let mut bitmap = MutableBitmap::with_capacity(len);
            let (first, second) = if options.descending {
                (true, false)
            } else {
                (false, true)
            };
            bitmap.extend_constant(len - n_set, first);
            bitmap.extend_constant(n_set, second);
            let arr = BooleanArray::from_data_default(bitmap.into(), None);

            return unsafe { self.with_chunks(vec![Box::new(arr) as ArrayRef]) };
        }

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

    fn sort(&self, descending: bool) -> BooleanChunked {
        self.sort_with(SortOptions {
            descending,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
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
    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        let mut vals = Vec::with_capacity(self.len());
        let mut count: IdxSize = 0;
        for arr in self.downcast_iter() {
            vals.extend_trusted_len(arr.into_iter().map(|v| {
                let i = count;
                count += 1;
                (i, v.map(|v| v as u8))
            }));
        }
        arg_sort_multiple_impl(vals, options)
    }
}

pub(crate) fn convert_sort_column_multi_sort(s: &Series) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        Categorical(_) => s.rechunk(),
        Binary | Boolean => s.clone(),
        Utf8 => s.cast(&Binary).unwrap(),
        #[cfg(feature = "dtype-struct")]
        Struct(_) => {
            let ca = s.struct_().unwrap();
            let new_fields = ca
                .fields()
                .iter()
                .map(convert_sort_column_multi_sort)
                .collect::<PolarsResult<Vec<_>>>()?;
            return StructChunked::new(ca.name(), &new_fields).map(|ca| ca.into_series());
        },
        _ => {
            let phys = s.to_physical_repr().into_owned();
            polars_ensure!(
                phys.dtype().is_numeric(),
                ComputeError: "cannot sort column of dtype `{}`", s.dtype()
            );
            phys
        },
    };
    Ok(out)
}

pub fn _broadcast_descending(n_cols: usize, descending: &mut Vec<bool>) {
    if n_cols > descending.len() && descending.len() == 1 {
        while n_cols != descending.len() {
            descending.push(descending[0]);
        }
    }
}

pub(crate) fn prepare_arg_sort(
    columns: Vec<Series>,
    mut descending: Vec<bool>,
) -> PolarsResult<(Series, Vec<Series>, Vec<bool>)> {
    let n_cols = columns.len();

    let mut columns = columns
        .iter()
        .map(convert_sort_column_multi_sort)
        .collect::<PolarsResult<Vec<_>>>()?;

    let first = columns.remove(0);

    // broadcast ordering
    _broadcast_descending(n_cols, &mut descending);
    Ok((first, columns, descending))
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
        let expected = [4, 2, 1, 5, 6, 0, 3, 7];
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
            maintain_order: false,
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
            maintain_order: false,
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
    #[cfg_attr(miri, ignore)]
    fn test_arg_sort_multiple() -> PolarsResult<()> {
        let a = Int32Chunked::new("a", &[1, 2, 1, 1, 3, 4, 3, 3]);
        let b = Int64Chunked::new("b", &[0, 1, 2, 3, 4, 5, 6, 1]);
        let c = Utf8Chunked::new("c", &["a", "b", "c", "d", "e", "f", "g", "h"]);
        let df = DataFrame::new(vec![a.into_series(), b.into_series(), c.into_series()])?;

        let out = df.sort(["a", "b", "c"], false, false)?;
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

        let out = df.sort(["a", "b"], false, false)?;
        let expected = df!(
            "a" => ["a", "a", "b", "b", "c", "c"],
            "b" => [3, 5, 4, 4, 2, 5]
        )?;
        assert!(out.equals(&expected));

        let df = df!(
            "groups" => [1, 2, 3],
            "values" => ["a", "a", "b"]
        )?;

        let out = df.sort(["groups", "values"], vec![true, false], false)?;
        let expected = df!(
            "groups" => [3, 2, 1],
            "values" => ["b", "a", "a"]
        )?;
        assert!(out.equals(&expected));

        let out = df.sort(["values", "groups"], vec![false, true], false)?;
        let expected = df!(
            "groups" => [2, 1, 3],
            "values" => ["a", "a", "b"]
        )?;
        assert!(out.equals(&expected));

        Ok(())
    }

    #[test]
    fn test_sort_utf8() {
        let ca = Utf8Chunked::new("a", &[Some("a"), None, Some("c"), None, Some("b")]);
        let out = ca.sort_with(SortOptions {
            descending: false,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        });
        let expected = &[None, None, Some("a"), Some("b"), Some("c")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: true,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        });

        let expected = &[None, None, Some("c"), Some("b"), Some("a")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: false,
            nulls_last: true,
            multithreaded: true,
            maintain_order: false,
        });
        let expected = &[Some("a"), Some("b"), Some("c"), None, None];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: true,
            nulls_last: true,
            multithreaded: true,
            maintain_order: false,
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
