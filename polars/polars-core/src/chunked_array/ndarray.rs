use ndarray::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;
use crate::POOL;

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// If data is aligned in a single chunk and has no Null values a zero copy view is returned
    /// as an `ndarray`
    pub fn to_ndarray(&self) -> PolarsResult<ArrayView1<T::Native>> {
        let slice = self.cont_slice()?;
        Ok(aview1(slice))
    }
}

impl ListChunked {
    /// If all nested `Series` have the same length, a 2 dimensional `ndarray::Array` is returned.
    pub fn to_ndarray<N>(&self) -> PolarsResult<Array2<N::Native>>
    where
        N: PolarsNumericType,
    {
        if self.null_count() != 0 {
            Err(PolarsError::ComputeError(
                "Creation of ndarray with null values is not supported.".into(),
            ))
        } else {
            let mut iter = self.into_no_null_iter();

            let mut ndarray;
            let width;

            // first iteration determine the size
            if let Some(series) = iter.next() {
                width = series.len();

                let mut row_idx = 0;
                ndarray = ndarray::Array::uninit((self.len(), width));

                let series = series.cast(&N::get_dtype())?;
                let ca = series.unpack::<N>()?;
                let a = ca.to_ndarray()?;
                let mut row = ndarray.slice_mut(s![row_idx, ..]);
                a.assign_to(&mut row);
                row_idx += 1;

                for series in iter {
                    if series.len() != width {
                        return Err(PolarsError::ShapeMisMatch(
                            "Could not create a 2D array. Series have different lengths".into(),
                        ));
                    }
                    let series = series.cast(&N::get_dtype())?;
                    let ca = series.unpack::<N>()?;
                    let a = ca.to_ndarray()?;
                    let mut row = ndarray.slice_mut(s![row_idx, ..]);
                    a.assign_to(&mut row);
                    row_idx += 1;
                }

                debug_assert_eq!(row_idx, self.len());
                // Safety:
                // We have assigned to every row and element of the array
                unsafe { Ok(ndarray.assume_init()) }
            } else {
                Err(PolarsError::NoData(
                    "cannot create ndarray of empty ListChunked".into(),
                ))
            }
        }
    }
}

impl DataFrame {
    /// Create a 2D `ndarray::Array` from this `DataFrame`. This requires all columns in the
    /// `DataFrame` to be non-null and numeric. They will be casted to the same data type
    /// (if they aren't already).
    ///
    /// For floating point data we implicitly convert `None` to `NaN` without failure.
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// let a = UInt32Chunked::new("a", &[1, 2, 3]).into_series();
    /// let b = Float64Chunked::new("b", &[10., 8., 6.]).into_series();
    ///
    /// let df = DataFrame::new(vec![a, b]).unwrap();
    /// let ndarray = df.to_ndarray::<Float64Type>().unwrap();
    /// println!("{:?}", ndarray);
    /// ```
    /// Outputs:
    /// ```text
    /// [[1.0, 10.0],
    ///  [2.0, 8.0],
    ///  [3.0, 6.0]], shape=[3, 2], strides=[2, 1], layout=C (0x1), const ndim=2/
    /// ```
    pub fn to_ndarray<N>(&self) -> PolarsResult<Array2<N::Native>>
    where
        N: PolarsNumericType,
    {
        let columns = POOL.install(|| {
            self.get_columns()
                .par_iter()
                .map(|s| {
                    let s = s.cast(&N::get_dtype())?;
                    let s = match s.dtype() {
                        DataType::Float32 => {
                            let ca = s.f32().unwrap();
                            ca.none_to_nan().into_series()
                        }
                        DataType::Float64 => {
                            let ca = s.f64().unwrap();
                            ca.none_to_nan().into_series()
                        }
                        _ => s,
                    };
                    Ok(s.rechunk())
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        let shape = self.shape();
        let height = self.height();
        let mut membuf = Vec::with_capacity(shape.0 * shape.1);
        let ptr = membuf.as_ptr() as usize;

        POOL.install(||{columns.par_iter().enumerate().map(|(col_idx, s)| {
            if s.null_count() != 0 {
                return Err(PolarsError::ComputeError(
                    "Creation of ndarray with null values is not supported. Consider using floats and NaNs".into(),
                ));
            }

            // this is an Arc clone if already of type N
            let s = s.cast(&N::get_dtype())?;
            let ca = s.unpack::<N>()?;
            let vals = ca.cont_slice().unwrap();

            // Safety:
            // we get parallel access to the vector
            // but we make sure that we don't get aliased access by offsetting the column indices + length
            unsafe {
                let offset_ptr = (ptr as *mut N::Native).add(col_idx * height) ;
                // Safety:
                // this is uninitialized memory, so we must never read from this data
                // copy_from_slice does not read
                let buf = std::slice::from_raw_parts_mut(offset_ptr, height);
                buf.copy_from_slice(vals)
            }

            Ok(())
        }).collect::<PolarsResult<Vec<_>>>()})?;

        // Safety:
        // we have written all data, so we can now safely set length
        unsafe {
            membuf.set_len(shape.0 * shape.1);
        }
        let ndarr = Array2::from_shape_vec((shape.1, shape.0), membuf).unwrap();
        Ok(ndarr.reversed_axes())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ndarray_from_ca() -> PolarsResult<()> {
        let ca = Float64Chunked::new("", &[1.0, 2.0, 3.0]);
        let ndarr = ca.to_ndarray()?;
        assert_eq!(ndarr, ArrayView1::from(&[1.0, 2.0, 3.0]));

        let mut builder =
            ListPrimitiveChunkedBuilder::<Float64Type>::new("", 10, 10, DataType::Float64);
        builder.append_opt_slice(Some(&[1.0, 2.0, 3.0]));
        builder.append_opt_slice(Some(&[2.0, 4.0, 5.0]));
        builder.append_opt_slice(Some(&[6.0, 7.0, 8.0]));
        let list = builder.finish();

        let ndarr = list.to_ndarray::<Float64Type>()?;
        let expected = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        assert_eq!(ndarr, expected);

        // test list array that is not square
        let mut builder =
            ListPrimitiveChunkedBuilder::<Float64Type>::new("", 10, 10, DataType::Float64);
        builder.append_opt_slice(Some(&[1.0, 2.0, 3.0]));
        builder.append_opt_slice(Some(&[2.0]));
        builder.append_opt_slice(Some(&[6.0, 7.0, 8.0]));
        let list = builder.finish();
        assert!(list.to_ndarray::<Float64Type>().is_err());
        Ok(())
    }

    #[test]
    fn test_ndarray_from_df() -> PolarsResult<()> {
        let df = df!["a"=> [1.0, 2.0, 3.0],
            "b" => [2.0, 3.0, 4.0]
        ]?;

        let ndarr = df.to_ndarray::<Float64Type>()?;
        let expected = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        assert_eq!(ndarr, expected);

        Ok(())
    }
}
