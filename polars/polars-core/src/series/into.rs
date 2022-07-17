use crate::prelude::*;
#[cfg(any(
    feature = "dtype-datetime",
    feature = "dtype-date",
    feature = "dtype-duration",
    feature = "dtype-time"
))]
use polars_arrow::compute::cast::cast;

impl Series {
    /// Returns a reference to the Arrow ArrayRef
    pub fn array_ref(&self, chunk_idx: usize) -> &ArrayRef {
        match self.dtype() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => {
                let ca = self.struct_().unwrap();
                ca.arrow_array()
            }
            _ => &self.chunks()[chunk_idx] as &ArrayRef,
        }
    }

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
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => self.array_ref(chunk_idx).clone(),
            _ => self.array_ref(chunk_idx).clone(),
        }
    }
}
