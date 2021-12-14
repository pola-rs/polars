use crate::prelude::compare_inner::PartialOrdInner;
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use arrow::buffer::MutableBuffer;
use arrow::{bitmap::MutableBitmap, buffer::Buffer};
use itertools::Itertools;
use polars_arrow::array::default_arrays::FromDataUtf8;
use polars_arrow::prelude::ValueSize;
use polars_arrow::trusted_len::PushUnchecked;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::hint::unreachable_unchecked;
use std::iter::FromIterator;

/// # Safety
/// only may produce true, for f32/f64::NaN
pub unsafe trait PlIsNan {
    fn isnan(&self) -> bool {
        false
    }
}

unsafe impl PlIsNan for f32 {
    fn isnan(&self) -> bool {
        self.is_nan()
    }
}
unsafe impl PlIsNan for f64 {
    fn isnan(&self) -> bool {
        self.is_nan()
    }
}

unsafe impl PlIsNan for u8 {}
unsafe impl PlIsNan for u16 {}
unsafe impl PlIsNan for u32 {}
unsafe impl PlIsNan for u64 {}
unsafe impl PlIsNan for i8 {}
unsafe impl PlIsNan for i16 {}
unsafe impl PlIsNan for i32 {}
unsafe impl PlIsNan for i64 {}

/// Reverse sorting when there are no nulls
fn order_reverse<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    b.partial_cmp(a).unwrap()
}

/// Default sorting when there are no nulls
fn order_default<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap()
}

fn order_default_flt<T: PartialOrd + PlIsNan>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        match (a.isnan(), b.isnan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // Safety: PlIsNan is only implemented for numbers
            _ => unsafe { unreachable_unchecked() },
        }
    })
}

fn order_reverse_flt<T: PartialOrd + PlIsNan>(a: &T, b: &T) -> Ordering {
    order_default_flt(b, a)
}

/// Sort with null values, to reverse, swap the arguments.
fn sort_with_nulls<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => order_default(a, b),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
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
    reverse: bool,
    default_order_fn: Fd,
    reverse_order_fn: Fr,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
{
    match reverse {
        true => slice.par_sort_unstable_by(reverse_order_fn),
        false => slice.par_sort_unstable_by(default_order_fn),
    }
}

fn argsort_branch<T, Fd, Fr>(
    slice: &mut [T],
    reverse: bool,
    default_order_fn: Fd,
    reverse_order_fn: Fr,
) where
    T: PartialOrd + Send,
    Fd: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
    Fr: FnMut(&T, &T) -> Ordering + for<'r, 's> Fn(&'r T, &'s T) -> Ordering + Sync,
{
    match reverse {
        true => slice.par_sort_by(reverse_order_fn),
        false => slice.par_sort_by(default_order_fn),
    }
}

macro_rules! argsort {
    ($self:expr, $reverse:expr) => {{
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
            $reverse,
            |(_, a), (_, b)| order_default_null(a, b),
            |(_, a), (_, b)| order_reverse_null(a, b),
        );
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
        let mut ca = ca.into_inner();
        ca.rename($self.name());
        ca
    }};
}

fn memcpy_values<T>(ca: &ChunkedArray<T>) -> AlignedVec<T::Native>
where
    T: PolarsNumericType,
{
    let len = ca.len();
    let mut vals = AlignedVec::with_capacity(len);
    // Safety:
    // only primitives so no drop calls when writing
    unsafe { vals.set_len(len) }
    let vals_slice = vals.as_mut_slice();

    let mut offset = 0;
    ca.downcast_iter().for_each(|arr| {
        let values = arr.values().as_slice();
        let len = values.len();
        (vals_slice[offset..offset + len]).copy_from_slice(values);
        offset += len;
    });
    vals
}

macro_rules! sort_with_fast_path {
    ($ca:ident, $options:expr) => {{
        if $ca.is_empty() {
            return $ca.clone();
        }

        if $options.descending && $ca.is_sorted_reverse() || $ca.is_sorted() {
            // there are nulls
            if $ca.has_validity() {
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


    }}
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Default + PlIsNan,
{
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<T> {
        sort_with_fast_path!(self, options);
        if !self.has_validity() {
            let mut vals = memcpy_values(self);

            if matches!(self.dtype(), DataType::Float32 | DataType::Float64) {
                sort_branch(
                    vals.as_mut_slice(),
                    options.descending,
                    order_default_flt,
                    order_reverse_flt,
                );
            } else {
                sort_branch(
                    vals.as_mut_slice(),
                    options.descending,
                    order_default,
                    order_reverse,
                );
            }

            ChunkedArray::new_from_aligned_vec(self.name(), vals)
        } else {
            let null_count = self.null_count();
            let len = self.len();
            let mut vals = Vec::with_capacity(self.len());

            if !options.nulls_last {
                let iter = std::iter::repeat(T::Native::default()).take(null_count);
                vals.extend(iter);
            }

            self.downcast_iter().for_each(|arr| {
                let iter = arr
                    .iter()
                    .filter_map(|v| v.copied())
                    .trust_my_length(len - null_count);
                vals.extend_trusted_len(iter);
            });
            let mut_slice = if options.nulls_last {
                &mut vals[..len - null_count]
            } else {
                &mut vals[null_count..]
            };

            if matches!(self.dtype(), DataType::Float32 | DataType::Float64) {
                sort_branch(
                    mut_slice,
                    options.descending,
                    order_default_flt,
                    order_reverse_flt,
                );
            } else {
                sort_branch(mut_slice, options.descending, order_default, order_reverse);
            }

            let mut ca: Self = if options.nulls_last {
                vals.extend(std::iter::repeat(T::Native::default()).take(self.null_count()));
                let mut validity = MutableBitmap::with_capacity(len);
                validity.extend_constant(len - null_count, true);
                validity.extend_constant(null_count, false);

                (
                    self.name(),
                    PrimitiveArray::from_data(
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
                    self.name(),
                    PrimitiveArray::from_data(
                        T::get_dtype().to_arrow(),
                        vals.into(),
                        Some(validity.into()),
                    ),
                )
                    .into()
            };

            ca.set_sorted(options.descending);
            ca
        }
    }

    fn sort(&self, reverse: bool) -> ChunkedArray<T> {
        self.sort_with(SortOptions {
            descending: reverse,
            ..Default::default()
        })
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        if !self.has_validity() {
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
                reverse,
                |(_, a), (_, b)| order_default(a, b),
                |(_, a), (_, b)| order_reverse(a, b),
            );

            let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
            let mut ca = ca.into_inner();
            ca.rename(self.name());
            ca
        } else {
            let null_count = self.null_count();
            let len = self.len();
            let mut vals = Vec::with_capacity(len - null_count);

            // if we sort reverse, the nulls are last
            // and need to be extended to the indices in reverse order
            let null_cap = if reverse {
                null_count
            // if we sort normally, the nulls are first
            // and can be extended with the sorted indices
            } else {
                len
            };
            let mut nulls_idx = Vec::with_capacity(null_cap);
            let mut count: u32 = 0;
            self.downcast_iter().for_each(|arr| {
                let iter = arr.iter().filter_map(|v| {
                    let i = count;
                    count += 1;
                    match v {
                        Some(v) => Some((i, *v)),
                        None => {
                            // Safety:
                            // we allocated enough
                            unsafe { nulls_idx.push_unchecked(i) };
                            None
                        }
                    }
                });
                vals.extend(iter);
            });

            argsort_branch(
                vals.as_mut_slice(),
                reverse,
                |(_, a), (_, b)| a.partial_cmp(b).unwrap(),
                |(_, a), (_, b)| b.partial_cmp(a).unwrap(),
            );

            let iter = vals.into_iter().map(|(idx, _v)| idx);
            let idx = if reverse {
                let mut idx = Vec::with_capacity(len);
                idx.extend(iter);
                idx.extend(nulls_idx.into_iter().rev());
                idx
            } else {
                nulls_idx.extend(iter);
                nulls_idx
            };

            let arr = UInt32Array::from_data(ArrowDataType::UInt32, Buffer::from_vec(idx), None);
            UInt32Chunked::new_from_chunks(self.name(), vec![Arc::new(arr)])
        }
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

        let compare_inner: Vec<_> = other
            .iter()
            .map(|s| s.into_partial_ord_inner())
            .collect_trusted();

        let mut count: u32 = 0;
        let mut vals: Vec<_> = self
            .into_iter()
            .map(|v| {
                let i = count;
                count += 1;
                (i, v)
            })
            .collect_trusted();

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
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
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
    ($self:ident, $reverse:expr) => {{
        if $reverse {
            $self
                .into_iter()
                .sorted_by(|a, b| b.cmp(a))
                .collect_trusted()
        } else {
            $self
                .into_iter()
                .sorted_by(|a, b| a.cmp(b))
                .collect_trusted()
        }
    }};
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
        );

        let mut values = MutableBuffer::<u8>::with_capacity(self.get_values_size());
        let mut offsets = MutableBuffer::<i64>::with_capacity(self.len() + 1);
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
                offsets.extend_constant(null_count, length_so_far);

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
                offsets.extend_constant(null_count, length_so_far);

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

        ca.set_sorted(options.descending);
        ca
    }

    fn sort(&self, reverse: bool) -> Utf8Chunked {
        self.sort_with(SortOptions {
            descending: reverse,
            nulls_last: false,
        })
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
            .collect_trusted();
        let compare_inner: Vec<_> = other
            .iter()
            .map(|s| s.into_partial_ord_inner())
            .collect_trusted();

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
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
        let mut ca = ca.into_inner();
        ca.set_sorted(reverse[0]);
        Ok(ca)
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkSort<CategoricalType> for CategoricalChunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<CategoricalType> {
        assert!(
            !options.nulls_last,
            "null last not yet supported for categorical dtype"
        );
        let mut vals = self
            .into_iter()
            .zip(self.iter_str())
            .trust_my_length(self.len())
            .collect_trusted::<Vec<_>>();

        argsort_branch(
            vals.as_mut_slice(),
            options.descending,
            |(_, a), (_, b)| order_default_null(a, b),
            |(_, a), (_, b)| order_reverse_null(a, b),
        );
        let arr: UInt32Array = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
        let mut ca = self.clone();
        ca.chunks = vec![Arc::new(arr)];

        ca
    }

    fn sort(&self, reverse: bool) -> Self {
        self.sort_with(SortOptions {
            nulls_last: false,
            descending: reverse,
        })
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        let mut count: u32 = 0;
        let mut vals = self
            .iter_str()
            .map(|s| {
                let i = count;
                count += 1;
                (i, s)
            })
            .trust_my_length(self.len())
            .collect_trusted::<Vec<_>>();

        argsort_branch(
            vals.as_mut_slice(),
            reverse,
            |(_, a), (_, b)| order_default_null(a, b),
            |(_, a), (_, b)| order_reverse_null(a, b),
        );
        let ca: NoNull<UInt32Chunked> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
        let mut ca = ca.into_inner();
        ca.rename(self.name());
        ca
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<BooleanType> {
        sort_with_fast_path!(self, options);
        assert!(
            !options.nulls_last,
            "null last not yet supported for bool dtype"
        );
        sort!(self, options.descending)
    }

    fn sort(&self, reverse: bool) -> BooleanChunked {
        self.sort_with(SortOptions {
            descending: reverse,
            nulls_last: false,
        })
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
    fn test_argsort() {
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
        let idx = a.argsort(false);
        let idx = idx.cont_slice().unwrap();

        let expected = [2, 4, 0, 3, 7, 6, 5, 1];
        assert_eq!(idx, expected);

        let idx = a.argsort(true);
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
    fn test_argsort_multiple() -> Result<()> {
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
        });
        let expected = &[None, None, Some("a"), Some("b"), Some("c")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: true,
            nulls_last: false,
        });

        let expected = &[None, None, Some("c"), Some("b"), Some("a")];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: false,
            nulls_last: true,
        });
        let expected = &[Some("a"), Some("b"), Some("c"), None, None];
        assert_eq!(Vec::from(&out), expected);

        let out = ca.sort_with(SortOptions {
            descending: true,
            nulls_last: true,
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

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_sort_categorical() {
        let ca = Utf8Chunked::new("a", &[Some("a"), None, Some("c"), None, Some("b")]);
        let ca = ca.cast(&DataType::Categorical).unwrap();
        let ca = ca.categorical().unwrap();
        let out = ca.sort_with(SortOptions {
            descending: false,
            nulls_last: false,
        });
        let out = out.iter_str().collect::<Vec<_>>();
        let expected = &[None, None, Some("a"), Some("b"), Some("c")];
        assert_eq!(out, expected);
    }
}
