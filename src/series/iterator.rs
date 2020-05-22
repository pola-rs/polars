use crate::series::chunked_array::ChunkedArray;
use arrow::array::{Array, PrimitiveArray, PrimitiveArrayOps, StringArray};
use arrow::datatypes::ArrowPrimitiveType;

pub struct ChunkIter<'a, T>
where
    T: ArrowPrimitiveType,
{
    arrays: Vec<&'a PrimitiveArray<T>>,
    chunk_i: usize,
    array_i: usize,
    out_of_bounds: bool,
}

impl<T> ChunkIter<'_, T>
where
    T: ArrowPrimitiveType,
{
    fn set_indexes(&mut self, arr: &PrimitiveArray<T>) {
        self.array_i += 1;
        if self.array_i >= arr.len() {
            // go to next array in the chunks
            self.array_i = 0;
            self.chunk_i += 1;
        }
        if self.chunk_i >= self.arrays.len() {
            self.out_of_bounds = true;
        }
    }
}

impl<T> Iterator for ChunkIter<'_, T>
where
    T: ArrowPrimitiveType,
{
    // nullable, therefore an option
    type Item = Option<T::Native>;

    /// Because arrow types are nullable an option is returned. This is wrapped in another option
    /// to indicate if the iterator returns Some or None.
    fn next(&mut self) -> Option<Self::Item> {
        if self.out_of_bounds {
            return None;
        }

        let arr = unsafe { self.arrays.get_unchecked(self.chunk_i) };
        let data = arr.data();
        let ret;
        if data.is_null(self.array_i) {
            ret = Some(None)
        } else {
            let v = arr.value(self.array_i);
            ret = Some(Some(v))
        }
        self.set_indexes(arr);
        ret
    }
}

impl<T> ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    pub fn iter(&self) -> ChunkIter<T> {
        let arrays = self
            .chunks
            .iter()
            .map(|a| {
                a.as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .expect("could not downcast")
            })
            .collect::<Vec<_>>();

        ChunkIter {
            arrays,
            chunk_i: 0,
            array_i: 0,
            out_of_bounds: false,
        }
    }
}
