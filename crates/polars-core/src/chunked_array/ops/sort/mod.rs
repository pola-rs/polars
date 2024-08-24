mod arg_sort;

pub mod arg_sort_multiple;

pub mod arg_bottom_k;
pub mod options;

#[cfg(feature = "dtype-categorical")]
mod categorical;

use std::cmp::Ordering;

pub(crate) use arg_sort_multiple::argsort_multiple_row_fmt;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use arrow::legacy::trusted_len::TrustedLenPush;
use compare_inner::NonNull;
use rayon::prelude::*;
pub use slice::*;

use crate::prelude::compare_inner::TotalOrdInner;
use crate::prelude::sort::arg_sort_multiple::*;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::NoNull;
use crate::POOL;

fn partition_nulls<T: Copy>(
    values: &mut [T],
    mut validity: Option<Bitmap>,
    options: SortOptions,
) -> (&mut [T], Option<Bitmap>) {
    let partitioned = if let Some(bitmap) = &validity {
        // Partition null last first
        let mut out_len = 0;
        for idx in bitmap.true_idx_iter() {
            unsafe { *values.get_unchecked_mut(out_len) = *values.get_unchecked(idx) };
            out_len += 1;
        }
        let valid_count = out_len;
        let null_count = values.len() - valid_count;
        validity = Some(create_validity(
            bitmap.len(),
            bitmap.unset_bits(),
            options.nulls_last,
        ));

        // Views are correctly partitioned.
        if options.nulls_last {
            &mut values[..valid_count]
        }
        // We need to swap the ends.
        else {
            // swap nulls with end
            let mut end = values.len() - 1;

            for i in 0..null_count {
                unsafe { *values.get_unchecked_mut(end) = *values.get_unchecked(i) };
                end = end.saturating_sub(1);
            }
            &mut values[null_count..]
        }
    } else {
        values
    };
    (partitioned, validity)
}

pub(crate) fn sort_by_branch<T, C>(slice: &mut [T], descending: bool, cmp: C, parallel: bool)
where
    T: Send,
    C: Send + Sync + Fn(&T, &T) -> Ordering,
{
    if parallel {
        POOL.install(|| match descending {
            true => slice.par_sort_by(|a, b| cmp(b, a)),
            false => slice.par_sort_by(cmp),
        })
    } else {
        match descending {
            true => slice.sort_by(|a, b| cmp(b, a)),
            false => slice.sort_by(cmp),
        }
    }
}

fn sort_unstable_by_branch<T, C>(slice: &mut [T], options: SortOptions, cmp: C)
where
    T: Send,
    C: Send + Sync + Fn(&T, &T) -> Ordering,
{
    if options.multithreaded {
        POOL.install(|| match options.descending {
            true => slice.par_sort_unstable_by(|a, b| cmp(b, a)),
            false => slice.par_sort_unstable_by(cmp),
        })
    } else {
        match options.descending {
            true => slice.sort_unstable_by(|a, b| cmp(b, a)),
            false => slice.sort_unstable_by(cmp),
        }
    }
}

// Reduce monomorphisation.
fn sort_impl_unstable<T>(vals: &mut [T], options: SortOptions)
where
    T: TotalOrd + Send + Sync,
{
    sort_unstable_by_branch(vals, options, TotalOrd::tot_cmp);
}

fn create_validity(len: usize, null_count: usize, nulls_last: bool) -> Bitmap {
    let mut validity = MutableBitmap::with_capacity(len);
    if nulls_last {
        validity.extend_constant(len - null_count, true);
        validity.extend_constant(null_count, false);
    } else {
        validity.extend_constant(null_count, false);
        validity.extend_constant(len - null_count, true);
    }
    validity.into()
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

fn sort_with_numeric<T>(ca: &ChunkedArray<T>, options: SortOptions) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    sort_with_fast_path!(ca, options);
    if ca.null_count() == 0 {
        let mut vals = ca.to_vec_null_aware().left().unwrap();

        sort_impl_unstable(vals.as_mut_slice(), options);

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

        sort_impl_unstable(mut_slice, options);

        if options.nulls_last {
            vals.extend(std::iter::repeat(T::Native::default()).take(ca.null_count()));
        }

        let arr = PrimitiveArray::new(
            T::get_dtype().to_arrow(CompatLevel::newest()),
            vals.into(),
            Some(create_validity(len, null_count, options.nulls_last)),
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

fn arg_sort_numeric<T>(ca: &ChunkedArray<T>, mut options: SortOptions) -> IdxCa
where
    T: PolarsNumericType,
{
    options.multithreaded &= POOL.current_num_threads() > 1;
    if ca.null_count() == 0 {
        let iter = ca
            .downcast_iter()
            .map(|arr| arr.values().as_slice().iter().copied());
        arg_sort::arg_sort_no_nulls(ca.name(), iter, options, ca.len())
    } else {
        let iter = ca
            .downcast_iter()
            .map(|arr| arr.iter().map(|opt| opt.copied()));
        arg_sort::arg_sort(ca.name(), iter, options, ca.null_count(), ca.len())
    }
}

fn arg_sort_multiple_numeric<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    by: &[Series],
    options: &SortMultipleOptions,
) -> PolarsResult<IdxCa> {
    args_validate(ca, by, &options.descending, "descending")?;
    args_validate(ca, by, &options.nulls_last, "nulls_last")?;
    let mut count: IdxSize = 0;

    let no_nulls = ca.null_count() == 0;

    if no_nulls {
        let mut vals = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            vals.extend_trusted_len(arr.values().as_slice().iter().map(|v| {
                let i = count;
                count += 1;
                (i, NonNull(*v))
            }))
        }
        arg_sort_multiple_impl(vals, by, options)
    } else {
        let mut vals = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            vals.extend_trusted_len(arr.into_iter().map(|v| {
                let i = count;
                count += 1;
                (i, v.copied())
            }));
        }
        arg_sort_multiple_impl(vals, by, options)
    }
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn sort_with(&self, mut options: SortOptions) -> ChunkedArray<T> {
        options.multithreaded &= POOL.current_num_threads() > 1;
        sort_with_numeric(self, options)
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
    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        arg_sort_multiple_numeric(self, by, options)
    }
}

fn ordering_other_columns<'a>(
    compare_inner: &'a [Box<dyn TotalOrdInner + 'a>],
    descending: &[bool],
    nulls_last: &[bool],
    idx_a: usize,
    idx_b: usize,
) -> Ordering {
    for ((cmp, descending), null_last) in compare_inner.iter().zip(descending).zip(nulls_last) {
        // SAFETY: indices are in bounds
        let ordering = unsafe { cmp.cmp_element_unchecked(idx_a, idx_b, null_last ^ descending) };
        match (ordering, descending) {
            (Ordering::Equal, _) => continue,
            (_, true) => return ordering.reverse(),
            _ => return ordering,
        }
    }
    // all arrays/columns exhausted, ordering equal it is.
    Ordering::Equal
}

impl ChunkSort<StringType> for StringChunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<StringType> {
        unsafe { self.as_binary().sort_with(options).to_string_unchecked() }
    }

    fn sort(&self, descending: bool) -> StringChunked {
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
    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        self.as_binary().arg_sort_multiple(by, options)
    }
}

impl ChunkSort<BinaryType> for BinaryChunked {
    fn sort_with(&self, mut options: SortOptions) -> ChunkedArray<BinaryType> {
        options.multithreaded &= POOL.current_num_threads() > 1;
        sort_with_fast_path!(self, options);
        // We will sort by the views and reconstruct with sorted views. We leave the buffers as is.
        // We must rechunk to ensure that all views point into the proper buffers.
        let ca = self.rechunk();
        let arr = ca.downcast_into_array();

        let (views, buffers, validity, total_bytes_len, total_buffer_len) = arr.into_inner();
        let mut views = views.make_mut();

        let (partitioned_part, validity) = partition_nulls(&mut views, validity, options);

        sort_unstable_by_branch(partitioned_part, options, |a, b| unsafe {
            a.get_slice_unchecked(&buffers)
                .tot_cmp(&b.get_slice_unchecked(&buffers))
        });

        let array = unsafe {
            BinaryViewArray::new_unchecked(
                ArrowDataType::BinaryView,
                views.into(),
                buffers,
                validity,
                total_bytes_len,
                total_buffer_len,
            )
        };

        let mut out = Self::with_chunk_like(self, array);

        let s = if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        out.set_sorted_flag(s);
        out
    }

    fn sort(&self, descending: bool) -> ChunkedArray<BinaryType> {
        self.sort_with(SortOptions {
            descending,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        })
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        if self.null_count() == 0 {
            arg_sort::arg_sort_no_nulls(
                self.name(),
                self.downcast_iter().map(|arr| arr.values_iter()),
                options,
                self.len(),
            )
        } else {
            arg_sort::arg_sort(
                self.name(),
                self.downcast_iter().map(|arr| arr.iter()),
                options,
                self.null_count(),
                self.len(),
            )
        }
    }

    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        args_validate(self, by, &options.descending, "descending")?;
        args_validate(self, by, &options.nulls_last, "nulls_last")?;
        let mut count: IdxSize = 0;

        let mut vals = Vec::with_capacity(self.len());
        for arr in self.downcast_iter() {
            for v in arr {
                let i = count;
                count += 1;
                vals.push((i, v))
            }
        }

        arg_sort_multiple_impl(vals, by, options)
    }
}

impl ChunkSort<BinaryOffsetType> for BinaryOffsetChunked {
    fn sort_with(&self, mut options: SortOptions) -> BinaryOffsetChunked {
        options.multithreaded &= POOL.current_num_threads() > 1;
        sort_with_fast_path!(self, options);

        let mut v: Vec<&[u8]> = Vec::with_capacity(self.len());
        for arr in self.downcast_iter() {
            v.extend(arr.non_null_values_iter());
        }

        sort_impl_unstable(v.as_mut_slice(), options);

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
                offsets.extend(std::iter::repeat(length_so_far).take(null_count));

                // SAFETY: offsets are correctly created.
                let arr = unsafe {
                    BinaryArray::from_data_unchecked_default(
                        offsets.into(),
                        values.into(),
                        Some(create_validity(len, null_count, true)),
                    )
                };
                ChunkedArray::with_chunk(self.name(), arr)
            },
            (_, false) => {
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
                        Some(create_validity(len, null_count, false)),
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

    fn sort(&self, descending: bool) -> BinaryOffsetChunked {
        self.sort_with(SortOptions {
            descending,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        })
    }

    fn arg_sort(&self, mut options: SortOptions) -> IdxCa {
        options.multithreaded &= POOL.current_num_threads() > 1;
        let ca = self.rechunk();
        let arr = ca.downcast_into_array();
        let mut idx = (0..(arr.len() as IdxSize)).collect::<Vec<_>>();

        let argsort = |args| {
            sort_unstable_by_branch(args, options, |a, b| unsafe {
                let a = arr.value_unchecked(*a as usize);
                let b = arr.value_unchecked(*b as usize);
                a.tot_cmp(&b)
            });
        };

        if self.null_count() == 0 {
            argsort(&mut idx);
            IdxCa::from_vec(self.name(), idx)
        } else {
            // This branch (almost?) never gets called as the row-encoding also encodes nulls.
            let (partitioned_part, validity) =
                partition_nulls(&mut idx, arr.validity().cloned(), options);
            argsort(partitioned_part);
            IdxCa::with_chunk(self.name(), IdxArr::from_data_default(idx.into(), validity))
        }
    }

    /// # Panics
    ///
    /// This function is very opinionated. On the implementation of `ChunkedArray<T>` for numeric types,
    /// we assume that all numeric `Series` are of the same type.
    ///
    /// In this case we assume that all numeric `Series` are `f64` types. The caller needs to
    /// uphold this contract. If not, it will panic.
    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        args_validate(self, by, &options.descending, "descending")?;
        args_validate(self, by, &options.nulls_last, "nulls_last")?;
        let mut count: IdxSize = 0;

        let mut vals = Vec::with_capacity(self.len());
        for arr in self.downcast_iter() {
            for v in arr {
                let i = count;
                count += 1;
                vals.push((i, v))
            }
        }

        arg_sort_multiple_impl(vals, by, options)
    }
}

#[cfg(feature = "dtype-struct")]
impl StructChunked {
    pub(crate) fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let bin = _get_rows_encoded_ca(
            self.name(),
            &[self.clone().into_series()],
            &[options.descending],
            &[options.nulls_last],
        )
        .unwrap();
        bin.arg_sort(Default::default())
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkSort<StructType> for StructChunked {
    fn sort_with(&self, mut options: SortOptions) -> ChunkedArray<StructType> {
        options.multithreaded &= POOL.current_num_threads() > 1;
        let idx = self.arg_sort(options);
        unsafe { self.take_unchecked(&idx) }
    }

    fn sort(&self, descending: bool) -> ChunkedArray<StructType> {
        self.sort_with(SortOptions::new().with_order_descending(descending))
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let bin = self.get_row_encoded(options).unwrap();
        bin.arg_sort(Default::default())
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort_with(&self, mut options: SortOptions) -> ChunkedArray<BooleanType> {
        options.multithreaded &= POOL.current_num_threads() > 1;
        sort_with_fast_path!(self, options);
        assert!(
            !options.nulls_last,
            "null last not yet supported for bool dtype"
        );
        if self.null_count() == 0 {
            let len = self.len();
            let n_set = self.sum().unwrap() as usize;
            let mut bitmap = MutableBitmap::with_capacity(len);
            let (first, second, n_set) = if options.descending {
                (true, false, len - n_set)
            } else {
                (false, true, n_set)
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
        if self.null_count() == 0 {
            arg_sort::arg_sort_no_nulls(
                self.name(),
                self.downcast_iter().map(|arr| arr.values_iter()),
                options,
                self.len(),
            )
        } else {
            arg_sort::arg_sort(
                self.name(),
                self.downcast_iter().map(|arr| arr.iter()),
                options,
                self.null_count(),
                self.len(),
            )
        }
    }
    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        let mut vals = Vec::with_capacity(self.len());
        let mut count: IdxSize = 0;
        for arr in self.downcast_iter() {
            vals.extend_trusted_len(arr.into_iter().map(|v| {
                let i = count;
                count += 1;
                (i, v.map(|v| v as u8))
            }));
        }
        arg_sort_multiple_impl(vals, by, options)
    }
}

pub(crate) fn convert_sort_column_multi_sort(s: &Series) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        Categorical(_, _) | Enum(_, _) => s.rechunk(),
        Binary | Boolean => s.clone(),
        BinaryOffset => s.clone(),
        String => s.str().unwrap().as_binary().into_series(),
        #[cfg(feature = "dtype-struct")]
        Struct(_) => {
            let ca = s.struct_().unwrap();
            let new_fields = ca
                .fields_as_series()
                .iter()
                .map(convert_sort_column_multi_sort)
                .collect::<PolarsResult<Vec<_>>>()?;
            let mut out = StructChunked::from_series(ca.name(), &new_fields)?;
            out.zip_outer_validity(ca);
            out.into_series()
        },
        // we could fallback to default branch, but decimal is not numeric dtype for now, so explicit here
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => s.clone(),
        List(inner) if !inner.is_nested() => s.clone(),
        Null => s.clone(),
        _ => {
            let phys = s.to_physical_repr().into_owned();
            polars_ensure!(
                phys.dtype().is_numeric(),
                InvalidOperation: "cannot sort column of dtype `{}`", s.dtype()
            );
            phys
        },
    };
    Ok(out)
}

pub fn _broadcast_bools(n_cols: usize, values: &mut Vec<bool>) {
    if n_cols > values.len() && values.len() == 1 {
        while n_cols != values.len() {
            values.push(values[0]);
        }
    }
}

pub(crate) fn prepare_arg_sort(
    columns: Vec<Series>,
    sort_options: &mut SortMultipleOptions,
) -> PolarsResult<(Series, Vec<Series>)> {
    let n_cols = columns.len();

    let mut columns = columns
        .iter()
        .map(convert_sort_column_multi_sort)
        .collect::<PolarsResult<Vec<_>>>()?;

    _broadcast_bools(n_cols, &mut sort_options.descending);
    _broadcast_bools(n_cols, &mut sort_options.nulls_last);

    let first = columns.remove(0);
    Ok((first, columns))
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
        let expected = [2, 4, 1, 5, 6, 0, 3, 7];
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
        let b = BooleanChunked::new("b", &[Some(false), Some(true), Some(false)]);
        let out = b.sort_with(SortOptions::default().with_order_descending(true));
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = b.sort_with(SortOptions::default().with_order_descending(false));
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arg_sort_multiple() -> PolarsResult<()> {
        let a = Int32Chunked::new("a", &[1, 2, 1, 1, 3, 4, 3, 3]);
        let b = Int64Chunked::new("b", &[0, 1, 2, 3, 4, 5, 6, 1]);
        let c = StringChunked::new("c", &["a", "b", "c", "d", "e", "f", "g", "h"]);
        let df = DataFrame::new(vec![a.into_series(), b.into_series(), c.into_series()])?;

        let out = df.sort(["a", "b", "c"], SortMultipleOptions::default())?;
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
        let a = StringChunked::new("a", &["a", "b", "c", "a", "b", "c"]).into_series();
        let b = Int32Chunked::new("b", &[5, 4, 2, 3, 4, 5]).into_series();
        let df = DataFrame::new(vec![a, b])?;

        let out = df.sort(["a", "b"], SortMultipleOptions::default())?;
        let expected = df!(
            "a" => ["a", "a", "b", "b", "c", "c"],
            "b" => [3, 5, 4, 4, 2, 5]
        )?;
        assert!(out.equals(&expected));

        let df = df!(
            "groups" => [1, 2, 3],
            "values" => ["a", "a", "b"]
        )?;

        let out = df.sort(
            ["groups", "values"],
            SortMultipleOptions::default().with_order_descending_multi([true, false]),
        )?;
        let expected = df!(
            "groups" => [3, 2, 1],
            "values" => ["b", "a", "a"]
        )?;
        assert!(out.equals(&expected));

        let out = df.sort(
            ["values", "groups"],
            SortMultipleOptions::default().with_order_descending_multi([false, true]),
        )?;
        let expected = df!(
            "groups" => [2, 1, 3],
            "values" => ["a", "a", "b"]
        )?;
        assert!(out.equals(&expected));

        Ok(())
    }

    #[test]
    fn test_sort_string() {
        let ca = StringChunked::new("a", &[Some("a"), None, Some("c"), None, Some("b")]);
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
        let ca = StringChunked::new("a", &[Some("a"), Some("c"), Some("b")]);
        let out = ca.sort(false);
        let expected = &[Some("a"), Some("b"), Some("c")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort(true);
        let expected = &[Some("c"), Some("b"), Some("a")];
        assert_eq!(Vec::from(&out), expected);
    }
}
