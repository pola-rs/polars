use polars_utils::float::IsFloat;

use crate::prelude::*;

impl<'a, T> ChunkSearch<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn index_of(&'a self, value: T::Native) -> Option<usize> {
        if value.is_nan() {
            let mut index = 0;
            for opt_val in self.iter() {
                if opt_val.map(|v| v.is_nan()) == Some(true) {
                    return Some(index);
                };
                index += 1;
            }
            return None;
        }
        let mut index = 0;
        for opt_val in self.iter() {//self.downcast_iter().map(|arr| arr.into_iter()).flatten() {
            if Some(value) == opt_val {
                return Some(index);
            };
            index += 1;
        }
        None
    }
}
