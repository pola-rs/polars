use crate::prelude::*;
#[cfg(any(
    feature = "dtype-datetime",
    feature = "dtype-date",
    feature = "dtype-duration",
    feature = "dtype-time"
))]
use polars_arrow::compute::cast::cast;

impl Series {
    /// Convert a chunk in the Series to the correct Arrow type.
    /// This conversion is needed because polars doesn't use a
    /// 1 on 1 mapping for logical/ categoricals, etc.
    pub fn to_arrow(&self, chunk_idx: usize) -> ArrayRef {
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let ca = self.categorical().unwrap();
                let arr = ca.logical().chunks()[chunk_idx].clone();
                let cats = UInt32Chunked::from_chunks("", vec![arr]);

                let new = CategoricalChunked::from_cats_and_rev_map(cats, ca.get_rev_map().clone());

                let arr: DictionaryArray<u32> = (&new).into();
                Arc::new(arr) as ArrayRef
            }
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                let arr = cast(&*self.chunks()[chunk_idx], &DataType::Date.to_arrow()).unwrap();
                Arc::from(arr)
            }
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                let arr = cast(&*self.chunks()[chunk_idx], &self.dtype().to_arrow()).unwrap();
                Arc::from(arr)
            }
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                let arr = cast(&*self.chunks()[chunk_idx], &self.dtype().to_arrow()).unwrap();
                Arc::from(arr)
            }
            #[cfg(feature = "dtype-time")]
            DataType::Time => {
                let arr = cast(&*self.chunks()[chunk_idx], &DataType::Time.to_arrow()).unwrap();
                Arc::from(arr)
            }
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => {
                let ca = self.struct_().unwrap();
                ca.arrow_array().clone()
            }
            _ => self.chunks()[chunk_idx].clone(),
        }
    }
}
