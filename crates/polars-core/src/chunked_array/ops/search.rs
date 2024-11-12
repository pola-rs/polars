use polars_utils::float::IsFloat;

use crate::prelude::*;

impl<'a, T> ChunkSearch<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn index_of(&'a self, value: Option<T::Native>) -> Option<usize> {
        // A NaN is never equal to anything, including itself. But we still want
        // to be able to search for NaNs, so we handle them specially.
        if value.map(|v| v.is_nan()) == Some(true) {
            return self
                .iter()
                .position(|opt_val| opt_val.map(|v| v.is_nan()) == Some(true));
        }

        self.iter().position(|opt_val| opt_val == value)
    }
}
