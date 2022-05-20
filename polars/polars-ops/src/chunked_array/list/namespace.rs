use super::*;
use polars_arrow::kernels::list::sublist_get;
use polars_arrow::prelude::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::series::ops::NullBehavior;
use polars_core::utils::CustomIterTools;
use std::convert::TryFrom;
use std::fmt::Write;

fn cast_rhs(
    other: &mut [Series],
    inner_type: &DataType,
    dtype: &DataType,
    length: usize,
    allow_broadcast: bool,
) -> Result<()> {
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
            match s.cast(dtype) {
                Ok(out) => {
                    *s = out;
                }
                Err(_) => {
                    return Err(PolarsError::SchemaMisMatch(
                        format!("cannot concat {:?} into a list of {:?}", s.dtype(), dtype).into(),
                    ));
                }
            }
        }

        if s.len() != length {
            if s.len() == 1 {
                if allow_broadcast {
                    // broadcast JIT
                    *s = s.expand_at_index(0, length)
                }
                // else do nothing
            } else {
                return Err(PolarsError::ShapeMisMatch(
                    format!("length {} does not match {}", s.len(), length).into(),
                ));
            }
        }
    }
    Ok(())
}

pub trait ListNameSpaceImpl: AsList {
    /// In case the inner dtype [`DataType::Utf8`], the individual items will be joined into a
    /// single string separated by `separator`.
    fn lst_join(&self, separator: &str) -> Result<Utf8Chunked> {
        let ca = self.as_list();
        match ca.inner_dtype() {
            DataType::Utf8 => {
                // used to amortize heap allocs
                let mut buf = String::with_capacity(128);

                let mut builder = Utf8ChunkedBuilder::new(
                    ca.name(),
                    ca.len(),
                    ca.get_values_size() + separator.len() * ca.len(),
                );

                ca.amortized_iter().for_each(|opt_s| {
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
            dt => Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot call lst.join on Series with dtype {:?}.\
                Inner type must be Utf8",
                    dt
                )
                .into(),
            )),
        }
    }

    fn lst_max(&self) -> Series {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().max_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    fn lst_min(&self) -> Series {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().min_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    fn lst_sum(&self) -> Series {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().sum_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    fn lst_mean(&self) -> Float64Chunked {
        let ca = self.as_list();
        ca.amortized_iter()
            .map(|s| s.and_then(|s| s.as_ref().mean()))
            .collect()
    }

    #[must_use]
    fn lst_sort(&self, reverse: bool) -> ListChunked {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().sort(reverse))
    }

    #[must_use]
    fn lst_reverse(&self) -> ListChunked {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().reverse())
    }

    fn lst_unique(&self) -> Result<ListChunked> {
        let ca = self.as_list();
        ca.try_apply_amortized(|s| s.as_ref().unique())
    }

    fn lst_arg_min(&self) -> IdxCa {
        let ca = self.as_list();
        let mut out: IdxCa = ca
            .amortized_iter()
            .map(|opt_s| opt_s.and_then(|s| s.as_ref().arg_min().map(|idx| idx as IdxSize)))
            .collect_trusted();
        out.rename(ca.name());
        out
    }

    fn lst_arg_max(&self) -> IdxCa {
        let ca = self.as_list();
        let mut out: IdxCa = ca
            .amortized_iter()
            .map(|opt_s| opt_s.and_then(|s| s.as_ref().arg_max().map(|idx| idx as IdxSize)))
            .collect_trusted();
        out.rename(ca.name());
        out
    }

    #[cfg(feature = "diff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    fn lst_diff(&self, n: usize, null_behavior: NullBehavior) -> ListChunked {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().diff(n, null_behavior))
    }

    fn lst_shift(&self, periods: i64) -> ListChunked {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().shift(periods))
    }

    fn lst_slice(&self, offset: i64, length: usize) -> ListChunked {
        let ca = self.as_list();
        ca.apply_amortized(|s| s.as_ref().slice(offset, length))
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
    fn lst_get(&self, idx: i64) -> Result<Series> {
        let ca = self.as_list();
        let chunks = ca
            .downcast_iter()
            .map(|arr| sublist_get(arr, idx))
            .collect::<Vec<_>>();
        Series::try_from((ca.name(), chunks))
    }

    fn lst_concat(&self, other: &[Series]) -> Result<ListChunked> {
        let ca = self.as_list();
        let other_len = other.len();
        let length = ca.len();
        let mut other = other.to_vec();
        let dtype = ca.dtype();
        let inner_type = ca.inner_dtype();

        // broadcasting path in case all unit length
        // this path will not expand the series, so saves memory
        if other.iter().all(|s| s.len() == 1) && ca.len() != 1 {
            cast_rhs(&mut other, &inner_type, dtype, length, false)?;
            let to_append = other
                .iter()
                .flat_map(|s| {
                    let lst = s.list().unwrap();
                    lst.get(0)
                })
                .collect::<Vec<_>>();
            // there was a None, so all values will be None
            if to_append.len() != other_len {
                return Ok(ListChunked::full_null_with_dtype(
                    ca.name(),
                    length,
                    &inner_type,
                ));
            }

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();

            let mut builder = get_list_builder(
                &inner_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name(),
            )?;
            ca.into_iter().for_each(|opt_s| {
                let opt_s = opt_s.map(|mut s| {
                    for append in &to_append {
                        s.append(append).unwrap();
                    }
                    match inner_type {
                        // structs don't have chunks, so we must first rechunk the underlying series
                        #[cfg(feature = "dtype-struct")]
                        DataType::Struct(_) => s = s.rechunk(),
                        // nothing
                        _ => {}
                    }
                    s
                });
                builder.append_opt_series(opt_s.as_ref())
            });
            Ok(builder.finish())
        } else {
            // normal path which may contain same length list or unit length lists
            cast_rhs(&mut other, &inner_type, dtype, length, true)?;

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();
            let mut iters = Vec::with_capacity(other_len + 1);

            for s in other.iter_mut() {
                iters.push(s.list()?.amortized_iter())
            }
            let mut first_iter = ca.into_iter();
            let mut builder = get_list_builder(
                &inner_type,
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
                    }
                };
                let mut already_null = false;
                for it in &mut iters {
                    match it.next().unwrap() {
                        Some(s) => {
                            acc.append(s.as_ref())?;
                        }
                        None => {
                            if !already_null {
                                builder.append_null();
                                already_null = true;
                            }

                            continue;
                        }
                    }
                }
                match inner_type {
                    // structs don't have chunks, so we must first rechunk the underlying series
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => acc = acc.rechunk(),
                    // nothing
                    _ => {}
                }
                builder.append_series(&acc);
            }
            Ok(builder.finish())
        }
    }
}

impl ListNameSpaceImpl for ListChunked {}
