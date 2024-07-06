use std::borrow::Cow;

use either::Either;

use super::*;

impl DataFrame {
    pub(crate) fn transpose_from_dtype(
        &self,
        dtype: &DataType,
        keep_names_as: Option<&str>,
        names_out: &[String],
    ) -> PolarsResult<DataFrame> {
        let new_width = self.height();
        let new_height = self.width();
        // Allocate space for the transposed columns, putting the "row names" first if needed
        let mut cols_t = match keep_names_as {
            None => Vec::<Series>::with_capacity(new_width),
            Some(name) => {
                let mut tmp = Vec::<Series>::with_capacity(new_width + 1);
                tmp.push(StringChunked::new(name, self.get_column_names()).into());
                tmp
            },
        };

        let cols = &self.columns;
        match dtype {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => numeric_transpose::<Int8Type>(cols, names_out, &mut cols_t),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => numeric_transpose::<Int16Type>(cols, names_out, &mut cols_t),
            DataType::Int32 => numeric_transpose::<Int32Type>(cols, names_out, &mut cols_t),
            DataType::Int64 => numeric_transpose::<Int64Type>(cols, names_out, &mut cols_t),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => numeric_transpose::<UInt8Type>(cols, names_out, &mut cols_t),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => numeric_transpose::<UInt16Type>(cols, names_out, &mut cols_t),
            DataType::UInt32 => numeric_transpose::<UInt32Type>(cols, names_out, &mut cols_t),
            DataType::UInt64 => numeric_transpose::<UInt64Type>(cols, names_out, &mut cols_t),
            DataType::Float32 => numeric_transpose::<Float32Type>(cols, names_out, &mut cols_t),
            DataType::Float64 => numeric_transpose::<Float64Type>(cols, names_out, &mut cols_t),
            #[cfg(feature = "object")]
            DataType::Object(_, _) => {
                // this requires to support `Object` in Series::iter which we don't yet
                polars_bail!(InvalidOperation: "Object dtype not supported in 'transpose'")
            },
            _ => {
                let phys_dtype = dtype.to_physical();
                let mut buffers = (0..new_width)
                    .map(|_| {
                        let buf: AnyValueBufferTrusted = (&phys_dtype, new_height).into();
                        buf
                    })
                    .collect::<Vec<_>>();

                let columns = self
                    .columns
                    .iter()
                    // first cast to supertype before casting to physical to ensure units are correct
                    .map(|s| s.cast(dtype).unwrap().cast(&phys_dtype).unwrap())
                    .collect::<Vec<_>>();

                // this is very expensive. A lot of cache misses here.
                // This is the part that is performance critical.
                for s in columns {
                    polars_ensure!(s.dtype() == &phys_dtype, ComputeError: "cannot transpose with supertype: {}", dtype);
                    s.iter().zip(buffers.iter_mut()).for_each(|(av, buf)| {
                        // SAFETY: we checked the type and we borrow
                        unsafe {
                            buf.add_unchecked_borrowed_physical(&av);
                        }
                    });
                }
                cols_t.extend(buffers.into_iter().zip(names_out).map(|(buf, name)| {
                    // SAFETY: we are casting back to the supertype
                    let mut s = unsafe { buf.into_series().cast_unchecked(dtype).unwrap() };
                    s.rename(name);
                    s
                }));
            },
        };
        Ok(unsafe { DataFrame::new_no_checks(cols_t) })
    }

    /// Transpose a DataFrame. This is a very expensive operation.
    pub fn transpose(
        &mut self,
        keep_names_as: Option<&str>,
        new_col_names: Option<Either<String, Vec<String>>>,
    ) -> PolarsResult<DataFrame> {
        // We must iterate columns as [`AnyValue`], so we must be contiguous.
        self.as_single_chunk_par();

        let mut df = Cow::Borrowed(self); // Can't use self because we might drop a name column
        let names_out = match new_col_names {
            None => (0..self.height()).map(|i| format!("column_{i}")).collect(),
            Some(cn) => match cn {
                Either::Left(name) => {
                    let new_names = self.column(&name).and_then(|x| x.str())?;
                    polars_ensure!(new_names.null_count() == 0, ComputeError: "Column with new names can't have null values");
                    df = Cow::Owned(self.drop(&name)?);
                    new_names
                        .into_no_null_iter()
                        .map(|s| s.to_owned())
                        .collect()
                },
                Either::Right(names) => {
                    polars_ensure!(names.len() == self.height(), ShapeMismatch: "Length of new column names must be the same as the row count");
                    names
                },
            },
        };
        if let Some(cn) = keep_names_as {
            // Check that the column name we're using for the original column names is unique before
            // wasting time transposing
            polars_ensure!(names_out.iter().all(|a| a.as_str() != cn), Duplicate: "{} is already in output column names", cn)
        }
        polars_ensure!(
            df.height() != 0 && df.width() != 0,
            NoData: "unable to transpose an empty DataFrame"
        );
        let dtype = df.get_supertype().unwrap()?;
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                let mut valid = true;
                let mut rev_map: Option<&Arc<RevMapping>> = None;
                for s in self.columns.iter() {
                    if let DataType::Categorical(Some(col_rev_map), _)
                    | DataType::Enum(Some(col_rev_map), _) = &s.dtype()
                    {
                        match rev_map {
                            Some(rev_map) => valid = valid && rev_map.same_src(col_rev_map),
                            None => {
                                rev_map = Some(col_rev_map);
                            },
                        }
                    }
                }
                polars_ensure!(valid, string_cache_mismatch);
            },
            _ => {},
        }
        df.transpose_from_dtype(&dtype, keep_names_as, &names_out)
    }
}

#[inline]
unsafe fn add_value<T: NumericNative>(
    values_buf_ptr: usize,
    col_idx: usize,
    row_idx: usize,
    value: T,
) {
    let column = (*(values_buf_ptr as *mut Vec<Vec<T>>)).get_unchecked_mut(col_idx);
    let el_ptr = column.as_mut_ptr();
    *el_ptr.add(row_idx) = value;
}

// This just fills a pre-allocated mutable series vector, which may have a name column.
// Nothing is returned and the actual DataFrame is constructed above.
pub(super) fn numeric_transpose<T>(cols: &[Series], names_out: &[String], cols_t: &mut Vec<Series>)
where
    T: PolarsNumericType,
    //S: AsRef<str>,
    ChunkedArray<T>: IntoSeries,
{
    let new_width = cols[0].len();
    let new_height = cols.len();

    let has_nulls = cols.iter().any(|s| s.null_count() > 0);

    let mut values_buf: Vec<Vec<T::Native>> = (0..new_width)
        .map(|_| Vec::with_capacity(new_height))
        .collect();
    let mut validity_buf: Vec<_> = if has_nulls {
        // we first use bools instead of bits, because we can access these in parallel without aliasing
        (0..new_width).map(|_| vec![true; new_height]).collect()
    } else {
        (0..new_width).map(|_| vec![]).collect()
    };

    // work with *mut pointers because we it is UB write to &refs.
    let values_buf_ptr = &mut values_buf as *mut Vec<Vec<T::Native>> as usize;
    let validity_buf_ptr = &mut validity_buf as *mut Vec<Vec<bool>> as usize;

    POOL.install(|| {
        cols.iter().enumerate().for_each(|(row_idx, s)| {
            let s = s.cast(&T::get_dtype()).unwrap();
            let ca = s.unpack::<T>().unwrap();

            // SAFETY:
            // we access in parallel, but every access is unique, so we don't break aliasing rules
            // we also ensured we allocated enough memory, so we never reallocate and thus
            // the pointers remain valid.
            if has_nulls {
                for (col_idx, opt_v) in ca.iter().enumerate() {
                    match opt_v {
                        None => unsafe {
                            let column = (*(validity_buf_ptr as *mut Vec<Vec<bool>>))
                                .get_unchecked_mut(col_idx);
                            let el_ptr = column.as_mut_ptr();
                            *el_ptr.add(row_idx) = false;
                            // we must initialize this memory otherwise downstream code
                            // might access uninitialized memory when the masked out values
                            // are changed.
                            add_value(values_buf_ptr, col_idx, row_idx, T::Native::default());
                        },
                        Some(v) => unsafe {
                            add_value(values_buf_ptr, col_idx, row_idx, v);
                        },
                    }
                }
            } else {
                for (col_idx, v) in ca.into_no_null_iter().enumerate() {
                    unsafe {
                        let column = (*(values_buf_ptr as *mut Vec<Vec<T::Native>>))
                            .get_unchecked_mut(col_idx);
                        let el_ptr = column.as_mut_ptr();
                        *el_ptr.add(row_idx) = v;
                    }
                }
            }
        })
    });

    let par_iter = values_buf
        .into_par_iter()
        .zip(validity_buf)
        .zip(names_out)
        .map(|((mut values, validity), name)| {
            // SAFETY:
            // all values are written we can now set len
            unsafe {
                values.set_len(new_height);
            }

            let validity = if has_nulls {
                let validity = Bitmap::from_trusted_len_iter(validity.iter().copied());
                if validity.unset_bits() > 0 {
                    Some(validity)
                } else {
                    None
                }
            } else {
                None
            };

            let arr = PrimitiveArray::<T::Native>::new(
                T::get_dtype().to_arrow(CompatLevel::newest()),
                values.into(),
                validity,
            );
            ChunkedArray::with_chunk(name.as_str(), arr).into_series()
        });
    POOL.install(|| cols_t.par_extend(par_iter));
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() -> PolarsResult<()> {
        let mut df = df![
            "a" => [1, 2, 3],
            "b" => [10, 20, 30],
        ]?;

        let out = df.transpose(None, None)?;
        let expected = df![
            "column_0" => [1, 10],
            "column_1" => [2, 20],
            "column_2" => [3, 30],

        ]?;
        assert!(out.equals_missing(&expected));

        let mut df = df![
            "a" => [Some(1), None, Some(3)],
            "b" => [Some(10), Some(20), None],
        ]?;
        let out = df.transpose(None, None)?;
        let expected = df![
            "column_0" => [1, 10],
            "column_1" => [None, Some(20)],
            "column_2" => [Some(3), None],

        ]?;
        assert!(out.equals_missing(&expected));

        let mut df = df![
            "a" => ["a", "b", "c"],
            "b" => [Some(10), Some(20), None],
        ]?;
        let out = df.transpose(None, None)?;
        let expected = df![
            "column_0" => ["a", "10"],
            "column_1" => ["b", "20"],
            "column_2" => [Some("c"), None],

        ]?;
        assert!(out.equals_missing(&expected));
        Ok(())
    }
}
