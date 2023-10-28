use std::convert::TryFrom;
use std::fmt::Write;

use arrow::legacy::kernels::list::sublist_get;
use arrow::legacy::prelude::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
#[cfg(feature = "list_take")]
use polars_core::export::num::ToPrimitive;
#[cfg(feature = "list_take")]
use polars_core::export::num::{NumCast, Signed, Zero};
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_core::utils::try_get_supertype;

use super::*;
#[cfg(feature = "list_any_all")]
use crate::chunked_array::list::any_all::*;
use crate::chunked_array::list::min_max::{list_max_function, list_min_function};
use crate::chunked_array::list::sum_mean::sum_with_nulls;
#[cfg(feature = "diff")]
use crate::prelude::diff;
use crate::prelude::list::sum_mean::{mean_list_numerical, sum_list_numerical};
use crate::series::ArgAgg;

pub(super) fn has_inner_nulls(ca: &ListChunked) -> bool {
    for arr in ca.downcast_iter() {
        if arr.values().null_count() > 0 {
            return true;
        }
    }
    false
}

fn cast_rhs(
    other: &mut [Series],
    inner_type: &DataType,
    dtype: &DataType,
    length: usize,
    allow_broadcast: bool,
) -> PolarsResult<()> {
    for s in other.iter_mut() {
        // make sure that inner types match before we coerce into list
        if !matches!(s.dtype(), DataType::List(_)) {
            *s = s.cast(inner_type)?
        }
        if !matches!(s.dtype(), DataType::List(_)) && s.dtype() == inner_type {
            // coerce to list JIT
            *s = s.reshape(&[-1, 1]).unwrap();
        }
        if s.dtype() != dtype {
            *s = s.cast(dtype).map_err(|e| {
                polars_err!(
                    SchemaMismatch:
                    "cannot concat `{}` into a list of `{}`: {}",
                    s.dtype(),
                    dtype,
                    e
                )
            })?;
        }

        if s.len() != length {
            polars_ensure!(
                s.len() == 1,
                ShapeMismatch: "series length {} does not match expected length of {}",
                s.len(), length
            );
            if allow_broadcast {
                // broadcast JIT
                *s = s.new_from_index(0, length)
            }
            // else do nothing
        }
    }
    Ok(())
}

pub trait ListNameSpaceImpl: AsList {
    /// In case the inner dtype [`DataType::Utf8`], the individual items will be joined into a
    /// single string separated by `separator`.
    fn lst_join(&self, separator: &Utf8Chunked) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_list();
        match ca.inner_dtype() {
            DataType::Utf8 => match separator.len() {
                1 => match separator.get(0) {
                    Some(separator) => self.join_literal(separator),
                    _ => Ok(Utf8Chunked::full_null(ca.name(), ca.len())),
                },
                _ => self.join_many(separator),
            },
            dt => polars_bail!(op = "`lst.join`", got = dt, expected = "Utf8"),
        }
    }

    fn join_literal(&self, separator: &str) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_list();
        // used to amortize heap allocs
        let mut buf = String::with_capacity(128);
        let mut builder = Utf8ChunkedBuilder::new(
            ca.name(),
            ca.len(),
            ca.get_values_size() + separator.len() * ca.len(),
        );

        ca.for_each_amortized(|opt_s| {
            let opt_val = opt_s.map(|s| {
                // make sure that we don't write values of previous iteration
                buf.clear();
                let ca = s.as_ref().utf8().unwrap();
                let iter = ca.into_iter().map(|opt_v| opt_v.unwrap_or("null"));

                for val in iter {
                    buf.write_str(val).unwrap();
                    buf.write_str(separator).unwrap();
                }
                // last value should not have a separator, so slice that off
                // saturating sub because there might have been nothing written.
                &buf[..buf.len().saturating_sub(separator.len())]
            });
            builder.append_option(opt_val)
        });
        Ok(builder.finish())
    }

    fn join_many(&self, separator: &Utf8Chunked) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_list();
        // used to amortize heap allocs
        let mut buf = String::with_capacity(128);
        let mut builder =
            Utf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size() + ca.len());
        // SAFETY: unstable series never lives longer than the iterator.
        unsafe {
            ca.amortized_iter()
                .zip(separator)
                .for_each(|(opt_s, opt_sep)| match opt_sep {
                    Some(separator) => {
                        let opt_val = opt_s.map(|s| {
                            // make sure that we don't write values of previous iteration
                            buf.clear();
                            let ca = s.as_ref().utf8().unwrap();
                            let iter = ca.into_iter().map(|opt_v| opt_v.unwrap_or("null"));

                            for val in iter {
                                buf.write_str(val).unwrap();
                                buf.write_str(separator).unwrap();
                            }
                            // last value should not have a separator, so slice that off
                            // saturating sub because there might have been nothing written.
                            &buf[..buf.len().saturating_sub(separator.len())]
                        });
                        builder.append_option(opt_val)
                    },
                    _ => builder.append_null(),
                })
        }
        Ok(builder.finish())
    }

    fn lst_max(&self) -> Series {
        list_max_function(self.as_list())
    }

    #[cfg(feature = "list_any_all")]
    fn lst_all(&self) -> PolarsResult<Series> {
        let ca = self.as_list();
        list_all(ca)
    }

    #[cfg(feature = "list_any_all")]
    fn lst_any(&self) -> PolarsResult<Series> {
        let ca = self.as_list();
        list_any(ca)
    }

    fn lst_min(&self) -> Series {
        list_min_function(self.as_list())
    }

    fn lst_sum(&self) -> Series {
        let ca = self.as_list();

        if has_inner_nulls(ca) {
            return sum_with_nulls(ca, &ca.inner_dtype());
        };

        match ca.inner_dtype() {
            DataType::Boolean => count_boolean_bits(ca).into_series(),
            dt if dt.is_numeric() => sum_list_numerical(ca, &dt),
            dt => sum_with_nulls(ca, &dt),
        }
    }

    fn lst_mean(&self) -> Series {
        let ca = self.as_list();

        if has_inner_nulls(ca) {
            return sum_mean::mean_with_nulls(ca);
        };

        match ca.inner_dtype() {
            dt if dt.is_numeric() => mean_list_numerical(ca, &dt),
            _ => sum_mean::mean_with_nulls(ca),
        }
    }

    fn same_type(&self, out: ListChunked) -> ListChunked {
        let ca = self.as_list();
        let dtype = ca.dtype();
        if out.dtype() != dtype {
            out.cast(ca.dtype()).unwrap().list().unwrap().clone()
        } else {
            out
        }
    }

    #[must_use]
    fn lst_sort(&self, options: SortOptions) -> ListChunked {
        let ca = self.as_list();
        let out = ca.apply_amortized(|s| s.as_ref().sort_with(options));
        self.same_type(out)
    }

    #[must_use]
    fn lst_reverse(&self) -> ListChunked {
        let ca = self.as_list();
        let out = ca.apply_amortized(|s| s.as_ref().reverse());
        self.same_type(out)
    }

    fn lst_unique(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let out = ca.try_apply_amortized(|s| s.as_ref().unique())?;
        Ok(self.same_type(out))
    }

    fn lst_unique_stable(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let out = ca.try_apply_amortized(|s| s.as_ref().unique_stable())?;
        Ok(self.same_type(out))
    }

    fn lst_arg_min(&self) -> IdxCa {
        let ca = self.as_list();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_min().map(|idx| idx as IdxSize))
        })
        .with_name(ca.name())
    }

    fn lst_arg_max(&self) -> IdxCa {
        let ca = self.as_list();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_max().map(|idx| idx as IdxSize))
        })
        .with_name(ca.name())
    }

    #[cfg(feature = "diff")]
    fn lst_diff(&self, n: i64, null_behavior: NullBehavior) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        ca.try_apply_amortized(|s| diff(s.as_ref(), n, null_behavior))
    }

    fn lst_shift(&self, periods: &Series) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let periods_s = periods.cast(&DataType::Int64)?;
        let periods = periods_s.i64()?;
        let out = match periods.len() {
            1 => {
                if let Some(periods) = periods.get(0) {
                    ca.apply_amortized(|s| s.as_ref().shift(periods))
                } else {
                    ListChunked::full_null_with_dtype(ca.name(), ca.len(), &ca.inner_dtype())
                }
            },
            _ => ca.zip_and_apply_amortized(periods, |opt_s, opt_periods| {
                match (opt_s, opt_periods) {
                    (Some(s), Some(periods)) => Some(s.as_ref().shift(periods)),
                    _ => None,
                }
            }),
        };
        Ok(self.same_type(out))
    }

    fn lst_slice(&self, offset: i64, length: usize) -> ListChunked {
        let ca = self.as_list();
        let out = ca.apply_amortized(|s| s.as_ref().slice(offset, length));
        self.same_type(out)
    }

    fn lst_lengths(&self) -> IdxCa {
        let ca = self.as_list();
        let mut lengths = Vec::with_capacity(ca.len());
        ca.downcast_iter().for_each(|arr| {
            let offsets = arr.offsets().as_slice();
            let mut last = offsets[0];
            for o in &offsets[1..] {
                lengths.push((*o - last) as IdxSize);
                last = *o;
            }
        });
        IdxCa::from_vec(ca.name(), lengths)
    }

    /// Get the value by index in the sublists.
    /// So index `0` would return the first item of every sublist
    /// and index `-1` would return the last item of every sublist
    /// if an index is out of bounds, it will return a `None`.
    fn lst_get(&self, idx: i64) -> PolarsResult<Series> {
        let ca = self.as_list();
        let chunks = ca
            .downcast_iter()
            .map(|arr| sublist_get(arr, idx))
            .collect::<Vec<_>>();
        Series::try_from((ca.name(), chunks))
            .unwrap()
            .cast(&ca.inner_dtype())
    }

    #[cfg(feature = "list_take")]
    fn lst_take(&self, idx: &Series, null_on_oob: bool) -> PolarsResult<Series> {
        let list_ca = self.as_list();

        let index_typed_index = |idx: &Series| {
            let idx = idx.cast(&IDX_DTYPE).unwrap();
            // SAFETY: unstable series never lives longer than the iterator.
            unsafe {
                list_ca
                    .amortized_iter()
                    .map(|s| {
                        s.map(|s| {
                            let s = s.as_ref();
                            take_series(s, idx.clone(), null_on_oob)
                        })
                        .transpose()
                    })
                    .collect::<PolarsResult<ListChunked>>()
                    .map(|mut ca| {
                        ca.rename(list_ca.name());
                        ca.into_series()
                    })
            }
        };

        use DataType::*;
        match idx.dtype() {
            List(_) => {
                let idx_ca = idx.list().unwrap();
                // SAFETY: unstable series never lives longer than the iterator.
                let mut out = unsafe {
                    list_ca
                        .amortized_iter()
                        .zip(idx_ca)
                        .map(|(opt_s, opt_idx)| {
                            {
                                match (opt_s, opt_idx) {
                                    (Some(s), Some(idx)) => {
                                        Some(take_series(s.as_ref(), idx, null_on_oob))
                                    },
                                    _ => None,
                                }
                            }
                            .transpose()
                        })
                        .collect::<PolarsResult<ListChunked>>()?
                };
                out.rename(list_ca.name());

                Ok(out.into_series())
            },
            UInt32 | UInt64 => index_typed_index(idx),
            dt if dt.is_signed() => {
                if let Some(min) = idx.min::<i64>() {
                    if min >= 0 {
                        index_typed_index(idx)
                    } else {
                        // SAFETY: unstable series never lives longer than the iterator.
                        let mut out = unsafe {
                            list_ca
                                .amortized_iter()
                                .map(|opt_s| {
                                    opt_s
                                        .map(|s| take_series(s.as_ref(), idx.clone(), null_on_oob))
                                        .transpose()
                                })
                                .collect::<PolarsResult<ListChunked>>()?
                        };
                        out.rename(list_ca.name());
                        Ok(out.into_series())
                    }
                } else {
                    polars_bail!(ComputeError: "all indices are null");
                }
            },
            dt => polars_bail!(ComputeError: "cannot use dtype `{}` as an index", dt),
        }
    }

    #[cfg(feature = "list_drop_nulls")]
    fn lst_drop_nulls(&self) -> ListChunked {
        let list_ca = self.as_list();

        list_ca.apply_amortized(|s| s.as_ref().drop_nulls())
    }

    #[cfg(feature = "list_sample")]
    fn lst_sample_n(
        &self,
        n: &Series,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<ListChunked> {
        let ca = self.as_list();

        let n_s = n.cast(&IDX_DTYPE)?;
        let n = n_s.idx()?;

        let out = match n.len() {
            1 => {
                if let Some(n) = n.get(0) {
                    ca.try_apply_amortized(|s| {
                        s.as_ref()
                            .sample_n(n as usize, with_replacement, shuffle, seed)
                    })
                } else {
                    Ok(ListChunked::full_null_with_dtype(
                        ca.name(),
                        ca.len(),
                        &ca.inner_dtype(),
                    ))
                }
            },
            _ => ca.try_zip_and_apply_amortized(n, |opt_s, opt_n| match (opt_s, opt_n) {
                (Some(s), Some(n)) => s
                    .as_ref()
                    .sample_n(n as usize, with_replacement, shuffle, seed)
                    .map(Some),
                _ => Ok(None),
            }),
        };
        out.map(|ok| self.same_type(ok))
    }

    #[cfg(feature = "list_sample")]
    fn lst_sample_fraction(
        &self,
        fraction: &Series,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<ListChunked> {
        let ca = self.as_list();

        let fraction_s = fraction.cast(&DataType::Float64)?;
        let fraction = fraction_s.f64()?;

        let out = match fraction.len() {
            1 => {
                if let Some(fraction) = fraction.get(0) {
                    ca.try_apply_amortized(|s| {
                        let n = (s.as_ref().len() as f64 * fraction) as usize;
                        s.as_ref().sample_n(n, with_replacement, shuffle, seed)
                    })
                } else {
                    Ok(ListChunked::full_null_with_dtype(
                        ca.name(),
                        ca.len(),
                        &ca.inner_dtype(),
                    ))
                }
            },
            _ => ca.try_zip_and_apply_amortized(fraction, |opt_s, opt_n| match (opt_s, opt_n) {
                (Some(s), Some(fraction)) => {
                    let n = (s.as_ref().len() as f64 * fraction) as usize;
                    s.as_ref()
                        .sample_n(n, with_replacement, shuffle, seed)
                        .map(Some)
                },
                _ => Ok(None),
            }),
        };
        out.map(|ok| self.same_type(ok))
    }

    fn lst_concat(&self, other: &[Series]) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let other_len = other.len();
        let length = ca.len();
        let mut other = other.to_vec();
        let mut inner_super_type = ca.inner_dtype();

        for s in &other {
            match s.dtype() {
                DataType::List(inner_type) => {
                    inner_super_type = try_get_supertype(&inner_super_type, inner_type)?;
                    #[cfg(feature = "dtype-categorical")]
                    if let DataType::Categorical(_) = &inner_super_type {
                        inner_super_type = merge_dtypes(&inner_super_type, inner_type)?;
                    }
                },
                dt => {
                    inner_super_type = try_get_supertype(&inner_super_type, dt)?;
                    #[cfg(feature = "dtype-categorical")]
                    if let DataType::Categorical(_) = &inner_super_type {
                        inner_super_type = merge_dtypes(&inner_super_type, dt)?;
                    }
                },
            }
        }

        // cast lhs
        let dtype = &DataType::List(Box::new(inner_super_type.clone()));
        let ca = ca.cast(dtype)?;
        let ca = ca.list().unwrap();

        // broadcasting path in case all unit length
        // this path will not expand the series, so saves memory
        let out = if other.iter().all(|s| s.len() == 1) && ca.len() != 1 {
            cast_rhs(&mut other, &inner_super_type, dtype, length, false)?;
            let to_append = other
                .iter()
                .flat_map(|s| {
                    let lst = s.list().unwrap();
                    lst.get_as_series(0)
                })
                .collect::<Vec<_>>();
            // there was a None, so all values will be None
            if to_append.len() != other_len {
                return Ok(ListChunked::full_null_with_dtype(
                    ca.name(),
                    length,
                    &inner_super_type,
                ));
            }

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();

            let mut builder = get_list_builder(
                &inner_super_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name(),
            )?;
            ca.into_iter().for_each(|opt_s| {
                let opt_s = opt_s.map(|mut s| {
                    for append in &to_append {
                        s.append(append).unwrap();
                    }
                    match inner_super_type {
                        // structs don't have chunks, so we must first rechunk the underlying series
                        #[cfg(feature = "dtype-struct")]
                        DataType::Struct(_) => s = s.rechunk(),
                        // nothing
                        _ => {},
                    }
                    s
                });
                builder.append_opt_series(opt_s.as_ref()).unwrap();
            });
            builder.finish()
        } else {
            // normal path which may contain same length list or unit length lists
            cast_rhs(&mut other, &inner_super_type, dtype, length, true)?;

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();
            let mut iters = Vec::with_capacity(other_len + 1);

            for s in other.iter_mut() {
                // SAFETY: unstable series never lives longer than the iterator.
                iters.push(unsafe { s.list()?.amortized_iter() })
            }
            let mut first_iter = ca.into_iter();
            let mut builder = get_list_builder(
                &inner_super_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name(),
            )?;

            for _ in 0..ca.len() {
                let mut acc = match first_iter.next().unwrap() {
                    Some(s) => s,
                    None => {
                        builder.append_null();
                        // make sure that the iterators advance before we continue
                        for it in &mut iters {
                            it.next().unwrap();
                        }
                        continue;
                    },
                };

                let mut has_nulls = false;
                for it in &mut iters {
                    match it.next().unwrap() {
                        Some(s) => {
                            if !has_nulls {
                                acc.append(s.as_ref())?;
                            }
                        },
                        None => {
                            has_nulls = true;
                        },
                    }
                }
                if has_nulls {
                    builder.append_null();
                    continue;
                }

                match inner_super_type {
                    // structs don't have chunks, so we must first rechunk the underlying series
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => acc = acc.rechunk(),
                    // nothing
                    _ => {},
                }
                builder.append_series(&acc).unwrap();
            }
            builder.finish()
        };
        Ok(out)
    }
}

impl ListNameSpaceImpl for ListChunked {}

#[cfg(feature = "list_take")]
fn take_series(s: &Series, idx: Series, null_on_oob: bool) -> PolarsResult<Series> {
    let len = s.len();
    let idx = cast_index(idx, len, null_on_oob)?;
    let idx = idx.idx().unwrap();
    s.take(idx)
}

#[cfg(feature = "list_take")]
fn cast_signed_index_ca<T: PolarsNumericType>(idx: &ChunkedArray<T>, len: usize) -> Series
where
    T::Native: Copy + PartialOrd + PartialEq + NumCast + Signed + Zero,
{
    idx.into_iter()
        .map(|opt_idx| opt_idx.and_then(|idx| idx.negative_to_usize(len).map(|idx| idx as IdxSize)))
        .collect::<IdxCa>()
        .into_series()
}

#[cfg(feature = "list_take")]
fn cast_unsigned_index_ca<T: PolarsNumericType>(idx: &ChunkedArray<T>, len: usize) -> Series
where
    T::Native: Copy + PartialOrd + ToPrimitive,
{
    idx.into_iter()
        .map(|opt_idx| {
            opt_idx.and_then(|idx| {
                let idx = idx.to_usize().unwrap();
                if idx >= len {
                    None
                } else {
                    Some(idx as IdxSize)
                }
            })
        })
        .collect::<IdxCa>()
        .into_series()
}

#[cfg(feature = "list_take")]
fn cast_index(idx: Series, len: usize, null_on_oob: bool) -> PolarsResult<Series> {
    let idx_null_count = idx.null_count();
    use DataType::*;
    let out = match idx.dtype() {
        #[cfg(feature = "big_idx")]
        UInt32 => {
            if null_on_oob {
                let a = idx.u32().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx.cast(&IDX_DTYPE).unwrap()
            }
        },
        #[cfg(feature = "big_idx")]
        UInt64 => {
            if null_on_oob {
                let a = idx.u64().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx
            }
        },
        #[cfg(not(feature = "big_idx"))]
        UInt64 => {
            if null_on_oob {
                let a = idx.u64().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx.cast(&IDX_DTYPE).unwrap()
            }
        },
        #[cfg(not(feature = "big_idx"))]
        UInt32 => {
            if null_on_oob {
                let a = idx.u32().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx
            }
        },
        dt if dt.is_unsigned() => idx.cast(&IDX_DTYPE).unwrap(),
        Int8 => {
            let a = idx.i8().unwrap();
            cast_signed_index_ca(a, len)
        },
        Int16 => {
            let a = idx.i16().unwrap();
            cast_signed_index_ca(a, len)
        },
        Int32 => {
            let a = idx.i32().unwrap();
            cast_signed_index_ca(a, len)
        },
        Int64 => {
            let a = idx.i64().unwrap();
            cast_signed_index_ca(a, len)
        },
        _ => {
            unreachable!()
        },
    };
    polars_ensure!(
        out.null_count() == idx_null_count || null_on_oob,
        ComputeError: "take indices are out of bounds"
    );
    Ok(out)
}

// TODO: implement the above for ArrayChunked as well?
