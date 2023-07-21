use either::Either;

use super::*;

impl DataFrame {
    pub(crate) fn transpose_from_dtype(
        &self,
        dtype: &DataType,
        keep_names_as: Option<&str>,
        colnames: &Vec<String>,
    ) -> PolarsResult<DataFrame> {
        let new_width = self.height();
        let new_height = self.width();

        let mut out = match dtype {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => numeric_transpose::<Int8Type>(&self.columns),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => numeric_transpose::<Int16Type>(&self.columns),
            DataType::Int32 => numeric_transpose::<Int32Type>(&self.columns),
            DataType::Int64 => numeric_transpose::<Int64Type>(&self.columns),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => numeric_transpose::<UInt8Type>(&self.columns),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => numeric_transpose::<UInt16Type>(&self.columns),
            DataType::UInt32 => numeric_transpose::<UInt32Type>(&self.columns),
            DataType::UInt64 => numeric_transpose::<UInt64Type>(&self.columns),
            DataType::Float32 => numeric_transpose::<Float32Type>(&self.columns),
            DataType::Float64 => numeric_transpose::<Float64Type>(&self.columns),
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                // this requires to support `Object` in Series::iter which we don't yet
                polars_bail!(InvalidOperation: "Object dtype not supported in 'transpose'")
            }
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
                        // safety: we checked the type and we borrow
                        unsafe {
                            buf.add_unchecked_borrowed_physical(&av);
                        }
                    });
                }
                let cols = buffers
                    .into_iter()
                    .map(|buf| buf.into_series().cast(dtype).unwrap())
                    .collect::<Vec<_>>();
                Ok(DataFrame::new_no_checks(cols))
            }
        }?;
        out.set_column_names(colnames)?;
        match keep_names_as {
            Some(cn) => {
                let namecol = Utf8Chunked::new(&cn, self.get_column_names());
                out.insert_at_idx_no_name_check(0, namecol.into()).cloned()
            }
            None => Ok(out),
        }
    }

    /// Transpose a DataFrame. This is a very expensive operation.
    pub fn transpose(
        &self,
        keep_names_as: Option<&str>,
        column_names: Option<Either<String, Vec<String>>>,
    ) -> PolarsResult<DataFrame> {
        let mut df = self.clone(); // Must be owned so we get the same type if dropping a column.
        let colnames_t = match column_names {
            None => (0..self.height()).map(|i| format!("column_{i}")).collect(),
            Some(cn) => match cn {
                Either::Left(cname) => {
                    let new_names = self.column(&cname).and_then(|x| x.utf8())?;
                    polars_ensure!(!new_names.has_validity(), ComputeError: "Column with new names can't have null values");
                    df = self.drop(&cname)?;
                    new_names
                        .into_no_null_iter()
                        .map(|s| s.to_owned())
                        .collect()
                }
                Either::Right(names) => {
                    polars_ensure!(names.len() == self.height(), ShapeMismatch: "Length of new column names must be the same as the row count");
                    names
                }
            },
        };
        match keep_names_as {
            // Check that the column name we're using for the original column names is unique before
            // wasting time transposing
            Some(cn) => {
                polars_ensure!(!colnames_t.contains(&cn.to_owned()), Duplicate: "{} is already in output column names", cn)
            }
            None => {}
        }
        polars_ensure!(
            df.height() != 0 && df.width() != 0,
            NoData: "unable to transpose an empty dataframe"
        );
        let dtype = df.get_supertype().unwrap()?;
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let mut valid = true;
                let mut cache_id = None;
                for s in self.columns.iter() {
                    if let DataType::Categorical(Some(rev_map)) = &s.dtype() {
                        match &**rev_map {
                            RevMapping::Local(_) => valid = false,
                            RevMapping::Global(_, _, id) => {
                                if let Some(cache_id) = cache_id {
                                    if cache_id != *id {
                                        valid = false;
                                    }
                                }
                                cache_id = Some(*id);
                            }
                        }
                    }
                }
                polars_ensure!(valid, ComputeError: "'transpose' of categorical can only be done if all are from the same global string cache")
            }
            _ => {}
        }
        df.transpose_from_dtype(&dtype, keep_names_as, &colnames_t)
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

pub(super) fn numeric_transpose<T>(cols: &[Series]) -> PolarsResult<DataFrame>
where
    T: PolarsNumericType,
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

            // Safety
            // we access in parallel, but every access is unique, so we don't break aliasing rules
            // we also ensured we allocated enough memory, so we never reallocate and thus
            // the pointers remain valid.
            if has_nulls {
                for (col_idx, opt_v) in ca.into_iter().enumerate() {
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

    let series = POOL.install(|| {
        values_buf
            .into_par_iter()
            .zip(validity_buf)
            .enumerate()
            .map(|(i, (mut values, validity))| {
                // Safety:
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
                    T::get_dtype().to_arrow(),
                    values.into(),
                    validity,
                );
                let name = format!("column_{i}");
                unsafe {
                    ChunkedArray::<T>::from_chunks(&name, vec![Box::new(arr) as ArrayRef])
                        .into_series()
                }
            })
            .collect()
    });

    Ok(DataFrame::new_no_checks(series))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() -> PolarsResult<()> {
        let df = df![
            "a" => [1, 2, 3],
            "b" => [10, 20, 30],
        ]?;

        let out = df.transpose(None, None)?;
        let expected = df![
            "column_0" => [1, 10],
            "column_1" => [2, 20],
            "column_2" => [3, 30],

        ]?;
        assert!(out.frame_equal_missing(&expected));

        let df = df![
            "a" => [Some(1), None, Some(3)],
            "b" => [Some(10), Some(20), None],
        ]?;
        let out = df.transpose(None, None)?;
        let expected = df![
            "column_0" => [1, 10],
            "column_1" => [None, Some(20)],
            "column_2" => [Some(3), None],

        ]?;
        assert!(out.frame_equal_missing(&expected));

        let df = df![
            "a" => ["a", "b", "c"],
            "b" => [Some(10), Some(20), None],
        ]?;
        let out = df.transpose(None, None)?;
        let expected = df![
            "column_0" => ["a", "10"],
            "column_1" => ["b", "20"],
            "column_2" => [Some("c"), None],

        ]?;
        assert!(out.frame_equal_missing(&expected));
        Ok(())
    }
}
