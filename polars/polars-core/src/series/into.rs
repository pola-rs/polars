use crate::prelude::*;
use arrow::compute::cast::cast;

impl Series {
    /// Convert a chunk in the Series to the correct Arrow type.
    /// This conversion is needed because polars doesn't use a
    /// 1 on 1 mapping for logical/ categoricals, etc.
    pub(crate) fn to_arrow(&self, chunk_idx: usize) -> ArrayRef {
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical => {
                let ca = self.categorical().unwrap();
                let mut new = CategoricalChunked::new_from_chunks(
                    ca.name(),
                    vec![ca.chunks()[chunk_idx].clone()],
                );
                new.set_categorical_map(ca.get_categorical_map().cloned().unwrap());

                let arr: DictionaryArray<u32> = (&new).into();
                Arc::new(arr) as ArrayRef
            }
            DataType::Date => {
                let arr = cast(&*self.chunks()[chunk_idx], &DataType::Date.to_arrow()).unwrap();
                Arc::from(arr)
            }
            DataType::Datetime => {
                let arr = cast(&*self.chunks()[chunk_idx], &DataType::Datetime.to_arrow()).unwrap();
                Arc::from(arr)
            }
            _ => self.chunks()[chunk_idx].clone(),
        }
    }
}
