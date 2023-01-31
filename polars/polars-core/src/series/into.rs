#[cfg(any(
    feature = "dtype-datetime",
    feature = "dtype-date",
    feature = "dtype-duration",
    feature = "dtype-time"
))]
use polars_arrow::compute::cast::cast;

use crate::prelude::*;

impl Series {
    /// Returns a reference to the Arrow ArrayRef
    #[inline]
    pub fn array_ref(&self, chunk_idx: usize) -> &ArrayRef {
        &self.chunks()[chunk_idx] as &ArrayRef
    }

    /// Convert a chunk in the Series to the correct Arrow type.
    /// This conversion is needed because polars doesn't use a
    /// 1 on 1 mapping for logical/ categoricals, etc.
    pub fn to_arrow(&self, chunk_idx: usize) -> ArrayRef {
        match self.dtype() {
            // special list branch to
            // make sure that we recursively apply all logical types.
            DataType::List(inner) => {
                let ca = self.list().unwrap();
                let arr = ca.chunks[chunk_idx].clone();
                let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();

                let new_values = if let DataType::Null = &**inner {
                    arr.values().clone()
                } else {
                    let s = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            "",
                            vec![arr.values().clone()],
                            inner,
                        )
                    };
                    s.to_arrow(0)
                };

                let data_type = ListArray::<i64>::default_datatype(inner.to_arrow());
                let arr = ListArray::<i64>::new(
                    data_type,
                    arr.offsets().clone(),
                    new_values,
                    arr.validity().cloned(),
                );
                Box::new(arr)
            }
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let ca = self.categorical().unwrap();
                let arr = ca.logical().chunks()[chunk_idx].clone();
                let cats = unsafe { UInt32Chunked::from_chunks("", vec![arr]) };

                // safety:
                // we only take a single chunk and change nothing about the index/rev_map mapping
                let new = unsafe {
                    CategoricalChunked::from_cats_and_rev_map_unchecked(
                        cats,
                        ca.get_rev_map().clone(),
                    )
                };

                let arr: DictionaryArray<u32> = (&new).into();
                Box::new(arr) as ArrayRef
            }
            #[cfg(feature = "dtype-date")]
            DataType::Date => cast(&*self.chunks()[chunk_idx], &DataType::Date.to_arrow()).unwrap(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                cast(&*self.chunks()[chunk_idx], &self.dtype().to_arrow()).unwrap()
            }
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                cast(&*self.chunks()[chunk_idx], &self.dtype().to_arrow()).unwrap()
            }
            #[cfg(feature = "dtype-time")]
            DataType::Time => cast(&*self.chunks()[chunk_idx], &DataType::Time.to_arrow()).unwrap(),
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                use crate::chunked_array::object::builder::object_series_to_arrow_array;
                if self.chunks().len() == 1 && chunk_idx == 0 {
                    object_series_to_arrow_array(self)
                } else {
                    // we slice the series to only that chunk
                    let offset = self.chunks()[..chunk_idx]
                        .iter()
                        .map(|arr| arr.len())
                        .sum::<usize>() as i64;
                    let len = self.chunks()[chunk_idx].len();
                    let s = self.slice(offset, len);
                    object_series_to_arrow_array(&s)
                }
            }
            _ => self.array_ref(chunk_idx).clone(),
        }
    }
}
