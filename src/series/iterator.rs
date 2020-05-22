use crate::{datatypes, series::chunked_array::ChunkedArray};
use arrow::array::{Array, PrimitiveArray, PrimitiveArrayOps, StringArray};
use arrow::datatypes::ArrowPrimitiveType;
use std::marker::PhantomData;

/// K is a phantom type to make specialization work.
/// K is set to u8 for all ArrowPrimitiveTypes
/// K is set to String for datatypes::Utf8DataType
pub struct ChunkIterPrimitive<'a, T, K>
where
    T: ArrowPrimitiveType,
{
    arrays: Vec<&'a PrimitiveArray<T>>,
    stringarrays: Vec<&'a StringArray>,
    chunk_i: usize,
    array_i: usize,
    out_of_bounds: bool,
    phantom: PhantomData<K>,
}

impl<T, S> ChunkIterPrimitive<'_, T, S>
where
    T: ArrowPrimitiveType,
{
    fn set_indexes(&mut self, arr: &dyn Array) {
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

impl<T> Iterator for ChunkIterPrimitive<'_, T, u8>
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

        let arr = unsafe { *self.arrays.get_unchecked(self.chunk_i) };
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

impl<'a, T> Iterator for ChunkIterPrimitive<'a, T, String>
where
    T: ArrowPrimitiveType,
{
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.out_of_bounds {
            return None;
        }
        let arr = unsafe { *self.stringarrays.get_unchecked(self.chunk_i) };
        let data = arr.data();
        let ret;
        if data.is_null(self.array_i) {
            ret = None
        } else {
            let v = arr.value(self.array_i);
            ret = Some(v)
        }
        self.set_indexes(arr);
        ret
    }
}

pub trait ChunkIter<T, S>
where
    T: ArrowPrimitiveType,
{
    fn iter(&self) -> ChunkIterPrimitive<T, S>;
}

/// ChunkIter for all the ArrowPrimitiveTypes
impl<T> ChunkIter<T, u8> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    default fn iter(&self) -> ChunkIterPrimitive<T, u8> {
        let arrays = self
            .chunks
            .iter()
            .map(|a| {
                a.as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .expect("could not downcast")
            })
            .collect::<Vec<_>>();

        ChunkIterPrimitive {
            arrays,
            stringarrays: vec![],
            chunk_i: 0,
            array_i: 0,
            out_of_bounds: false,
            phantom: PhantomData,
        }
    }
}

/// ChunkIter for the Utf8Type Chose by unstable specialization
impl ChunkIter<datatypes::Int32Type, String> for ChunkedArray<datatypes::Utf8Type> {
    fn iter(&self) -> ChunkIterPrimitive<datatypes::Int32Type, String> {
        let stringarrays = self
            .chunks
            .iter()
            .map(|a| {
                a.as_any()
                    .downcast_ref::<StringArray>()
                    .expect("could not downcast")
            })
            .collect::<Vec<_>>();

        ChunkIterPrimitive {
            arrays: vec![],
            stringarrays,
            chunk_i: 0,
            array_i: 0,
            out_of_bounds: false,
            phantom: PhantomData,
        }
    }
}
