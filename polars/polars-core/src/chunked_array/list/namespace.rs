use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::series::ops::NullBehavior;
use polars_arrow::kernels::list::sublist_get;
use polars_arrow::prelude::ValueSize;
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

impl ListChunked {
    /// In case the inner dtype [`DataType::Utf8`], the individual items will be joined into a
    /// single string separated by `separator`.
    pub fn lst_join(&self, separator: &str) -> Result<Utf8Chunked> {
        match self.inner_dtype() {
            DataType::Utf8 => {
                // used to amortize heap allocs
                let mut buf = String::with_capacity(128);

                let mut builder = Utf8ChunkedBuilder::new(
                    self.name(),
                    self.len(),
                    self.get_values_size() + separator.len() * self.len(),
                );

                self.amortized_iter().for_each(|opt_s| {
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

    pub fn lst_max(&self) -> Series {
        self.apply_amortized(|s| s.as_ref().max_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    pub fn lst_min(&self) -> Series {
        self.apply_amortized(|s| s.as_ref().min_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    pub fn lst_sum(&self) -> Series {
        self.apply_amortized(|s| s.as_ref().sum_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    pub fn lst_mean(&self) -> Float64Chunked {
        self.amortized_iter()
            .map(|s| s.and_then(|s| s.as_ref().mean()))
            .collect()
    }

    #[must_use]
    pub fn lst_sort(&self, reverse: bool) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().sort(reverse))
    }

    #[must_use]
    pub fn lst_reverse(&self) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().reverse())
    }

    pub fn lst_unique(&self) -> Result<ListChunked> {
        self.try_apply_amortized(|s| s.as_ref().unique())
    }

    pub fn lst_arg_min(&self) -> IdxCa {
        let mut out: IdxCa = self
            .amortized_iter()
            .map(|opt_s| opt_s.and_then(|s| s.as_ref().arg_min().map(|idx| idx as IdxSize)))
            .collect_trusted();
        out.rename(self.name());
        out
    }

    pub fn lst_arg_max(&self) -> IdxCa {
        let mut out: IdxCa = self
            .amortized_iter()
            .map(|opt_s| opt_s.and_then(|s| s.as_ref().arg_max().map(|idx| idx as IdxSize)))
            .collect_trusted();
        out.rename(self.name());
        out
    }

    #[cfg(feature = "diff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    pub fn lst_diff(&self, n: usize, null_behavior: NullBehavior) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().diff(n, null_behavior))
    }

    pub fn lst_shift(&self, periods: i64) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().shift(periods))
    }

    pub fn lst_slice(&self, offset: i64, length: usize) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().slice(offset, length))
    }

    pub fn lst_lengths(&self) -> UInt32Chunked {
        let mut lengths = Vec::with_capacity(self.len());
        self.downcast_iter().for_each(|arr| {
            let offsets = arr.offsets().as_slice();
            let mut last = offsets[0];
            for o in &offsets[1..] {
                lengths.push((*o - last) as u32);
                last = *o;
            }
        });
        UInt32Chunked::from_vec(self.name(), lengths)
    }

    /// Get the value by index in the sublists.
    /// So index `0` would return the first item of every sublist
    /// and index `-1` would return the last item of every sublist
    /// if an index is out of bounds, it will return a `None`.
    pub fn lst_get(&self, idx: i64) -> Result<Series> {
        let chunks = self
            .downcast_iter()
            .map(|arr| sublist_get(arr, idx))
            .collect::<Vec<_>>();
        Series::try_from((self.name(), chunks))
    }

    pub fn lst_concat(&self, other: &[Series]) -> Result<ListChunked> {
        let other_len = other.len();
        let length = self.len();
        let mut other = other.to_vec();
        let dtype = self.dtype();
        let inner_type = self.inner_dtype();

        // broadcasting path in case all unit length
        // this path will not expand the series, so saves memory
        if other.iter().all(|s| s.len() == 1) && self.len() != 1 {
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
                return Ok(Self::full_null_with_dtype(self.name(), length, &inner_type));
            }

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();

            let mut builder = get_list_builder(
                &inner_type,
                self.get_values_size() + vals_size_other + 1,
                length,
                self.name(),
            );
            self.into_iter().for_each(|opt_s| {
                let opt_s = opt_s.map(|mut s| {
                    for append in &to_append {
                        s.append(append).unwrap();
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
            let mut first_iter = self.into_iter();
            let mut builder = get_list_builder(
                &inner_type,
                self.get_values_size() + vals_size_other + 1,
                length,
                self.name(),
            );

            for _ in 0..self.len() {
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
                builder.append_series(&acc);
            }
            Ok(builder.finish())
        }
    }
}
