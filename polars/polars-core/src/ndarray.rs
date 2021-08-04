use crate::prelude::*;
use ndarray::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// If data is aligned in a single chunk and has no Null values a zero copy view is returned
    /// as an `ndarray`
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    pub fn to_ndarray(&self) -> Result<ArrayView1<T::Native>> {
        let slice = self.cont_slice()?;
        Ok(aview1(slice))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'a, T> From<&'a ChunkedArray<T>> for ArrayView1<'a, T::Native>
    where
        T: PolarsNumericType
{
    /// Convert from a ChunkedArray into a 1-dimensional ArrayView
    fn from(a: &'a ChunkedArray<T>) -> Self {
        a.to_ndarray().unwrap()
    }
}

impl ListChunked {
    /// If all nested `Series` have the same length, a 2 dimensional `ndarray::Array` is returned.
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    // allow deprecated for now. Fix when removed
    #[allow(deprecated)]
    pub fn to_ndarray<N>(&self) -> Result<Array2<N::Native>>
    where
        N: PolarsNumericType,
    {
        if self.null_count() != 0 {
            Err(PolarsError::HasNullValues(
                "Creation of ndarray with null values is not supported.".into(),
            ))
        } else {
            let mut iter = self.into_no_null_iter();

            let mut ndarray;
            let width;

            // first iteration determine the size
            if let Some(series) = iter.next() {
                width = series.len();

                ndarray = unsafe { Array::uninitialized((self.len(), series.len())) };

                let series = series.cast::<N>()?;
                let ca = series.unpack::<N>()?;
                let a = ca.to_ndarray()?;
                let mut row = ndarray.slice_mut(s![0, ..]);
                row.assign(&a);

                for series in iter {
                    if series.len() != width {
                        return Err(PolarsError::ShapeMisMatch(
                            "Could not create a 2D array. Series have different lengths".into(),
                        ));
                    }
                    let series = series.cast::<N>()?;
                    let ca = series.unpack::<N>()?;
                    let a = ca.to_ndarray()?;
                    let mut row = ndarray.slice_mut(s![0, ..]);
                    row.assign(&a)
                }
                Ok(ndarray)
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
    /// ```rust
    /// use polars_core::prelude::*;
    /// let a = UInt32Chunked::new_from_slice("a", &[1, 2, 3]).into_series();
    /// let b = Float64Chunked::new_from_slice("b", &[10., 8., 6.]).into_series();
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
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    pub fn to_ndarray<N>(&self) -> Result<Array2<N::Native>>
    where
        N: PolarsNumericType,
        N::Native: num::Zero + Copy,
    {
        let mut ndarr = Array2::zeros(self.shape());
        for (col_idx, series) in self.get_columns().iter().enumerate() {
            if series.null_count() != 0 {
                return Err(PolarsError::HasNullValues(
                    "Creation of ndarray with null values is not supported.".into(),
                ));
            }
            // this is an Arc clone if already of type N
            let series = series.cast::<N>()?;
            let ca = series.unpack::<N>()?;

            ca.into_no_null_iter()
                .enumerate()
                .for_each(|(row_idx, val)| {
                    ndarr[[row_idx, col_idx]] = val;
                })
        }
        Ok(ndarr)
    }
}

macro_rules! make_ndarray_type_conversion {
    ($polars:ty, $native:ty) => {
        #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
        impl From<DataFrame> for Array2<$native> {
            fn from(d: DataFrame) -> Self {
                d.to_ndarray::<$polars>().unwrap()
            }
        }

        #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
        impl From<ListChunked> for Array2<$native> {
            fn from(l: ListChunked) -> Self {
                l.to_ndarray::<$polars>().unwrap()
            }
        }
    };
}

// Todo add try from
make_ndarray_type_conversion!(Int8Type, i8);
make_ndarray_type_conversion!(Int16Type, i16);
make_ndarray_type_conversion!(Int32Type, i32);
make_ndarray_type_conversion!(Int64Type, i64);
make_ndarray_type_conversion!(UInt8Type, u8);
make_ndarray_type_conversion!(UInt16Type, u16);
make_ndarray_type_conversion!(UInt32Type, u32);
make_ndarray_type_conversion!(UInt64Type, u64);
make_ndarray_type_conversion!(Float32Type, f32);
make_ndarray_type_conversion!(Float64Type, f64);

// #[cfg(feature = "ndarray")]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_df_to_ndarray(){
        let df = df![
            "a" => [1, 2, 3],
            "b" => [0.1, 0.2, 0.3]
        ].unwrap();
        let arr: Array2<f64> = df.into();
        assert!(!arr.is_empty());
    }
}

// make_df_to_type!(UInt8Type);
// make_df_to_type!(UInt16Type);
// make_df_to_type!(UInt32Type);
// make_df_to_type!(UInt64Type);
// make_df_to_type!(Int8Type);
// make_df_to_type!(Int16Type);
// make_df_to_type!(Int32Type);
// make_df_to_type!(Int64Type);
// make_df_to_type!(Float32Type);
// make_df_to_type!(Float64Type);
// make_df_to_type!(Date32Type);
// make_df_to_type!(Date64Type);
// make_df_to_type!(Time64NanosecondType);
// make_df_to_type!(DurationNanosecondType);
// make_df_to_type!(DurationMillisecondType);

// #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
// impl From<DataFrame> for Array2<<Float64Type as ArrowPrimitiveType>::Native>
// {
//     fn from(d: DataFrame) -> Self {
//         d.to_ndarray::<Float64Type>().unwrap()
//     }
//     // fn from(d: DataFrame) -> Array2<N::Native> {
//     //     d.to_ndarray::<N>().unwrap()
//     // }
// }
