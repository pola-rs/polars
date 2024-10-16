use polars_utils::float::IsFloat;

use crate::prelude::*;

impl<'a, T> ChunkSearch<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn index_of(&'a self, value: T::Native) -> Option<usize> {
        if value.is_nan() {
            return self
                .iter()
                .position(|opt_val| opt_val.map(|v| v.is_nan()) == Some(true));
        }
        return self.iter().position(|opt_val| opt_val == Some(value));
    }
}
