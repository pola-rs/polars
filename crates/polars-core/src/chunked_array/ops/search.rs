use crate::prelude::*;

impl<'a, T> ChunkSearch<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn index_of(&'a self, value: T::Native) -> Option<usize> {
        // TODO handle NaN
        let mut index = 0;
        for opt_val in self.downcast_iter().map(|arr| arr.into_iter()).flatten() {
            println!("{value} {opt_val:?}, {index}");
            if Some(&value) == opt_val {
                return Some(index);
            };
            index += 1;
        }
        None
    }
}
