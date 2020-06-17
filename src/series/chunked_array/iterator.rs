use crate::datatypes::Utf8Chunked;
use crate::series::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::{
    datatypes,
    series::chunked_array::{ChunkedArray, SeriesOps},
};
use arrow::array::{Array, PrimitiveArray, PrimitiveArrayOps, StringArray};
use arrow::datatypes::ArrowPrimitiveType;
use std::iter::FromIterator;
use std::marker::PhantomData;

// This module implements an iter method for both ArrowPrimitiveType and Utf8 type.
// As both expose a different api in arrow, this required some hacking.
// A solution was found by returning an auxiliary struct ChunkIterState from the .iter methods.
// This ChunkIterState has a generic type T: ArrowPrimitiveType to work well with the arrow api.
// It also has a phantomtype K that is only used as distinctive type to be able to implement a trait
// method twice. Sort of a specialization hack.

enum Either<L, R> {
    Left(L),
    Right(R),
}

// K is a phantom type to make specialization work.
// K is set to u8 for all ArrowPrimitiveTypes
// K is set to String for datatypes::Utf8DataType
pub struct ChunkIterState<'a, T, K>
where
    T: ArrowPrimitiveType,
{
    arrays: Either<Vec<&'a PrimitiveArray<T>>, Vec<&'a StringArray>>,
    chunk_i: usize,
    array_i: usize,
    out_of_bounds: bool,
    length: usize,
    phantom: PhantomData<K>,
}

impl<T, S> ChunkIterState<'_, T, S>
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

        if self.chunk_i >= self.length {
            self.out_of_bounds = true;
        }
    }
}

impl<T> Iterator for ChunkIterState<'_, T, u8>
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

        let arrays;
        if let Either::Left(arr) = &self.arrays {
            arrays = arr;
        } else {
            panic!("implementation error")
        }

        let arr = unsafe { *arrays.get_unchecked(self.chunk_i) };
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

impl<'a, T> Iterator for ChunkIterState<'a, T, String>
where
    T: ArrowPrimitiveType,
{
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.out_of_bounds {
            return None;
        }

        let arrays;
        if let Either::Right(arr) = &self.arrays {
            arrays = arr;
        } else {
            panic!("implementation error")
        }

        let arr = unsafe { *arrays.get_unchecked(self.chunk_i) };

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

pub trait ChunkIterator<T, S>
where
    T: ArrowPrimitiveType,
{
    fn iter(&self) -> ChunkIterState<T, S>;
}

/// ChunkIter for all the ArrowPrimitiveTypes
/// Note that u8 is only a phantom type to be able to have stable specialization.
impl<T> ChunkIterator<T, u8> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn iter(&self) -> ChunkIterState<T, u8> {
        let arrays = self
            .chunks
            .iter()
            .map(|a| {
                a.as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .expect("could not downcast")
            })
            .collect::<Vec<_>>();

        ChunkIterState {
            arrays: Either::Left(arrays),
            chunk_i: 0,
            array_i: 0,
            out_of_bounds: false,
            length: self.len(),
            phantom: PhantomData,
        }
    }
}

/// ChunkIter for the Utf8Type
/// Note that datatypes::Int32Type is just a random ArrowPrimitveType for the Left variant
/// of the Either enum. We don't need it.
impl ChunkIterator<datatypes::Int32Type, String> for ChunkedArray<datatypes::Utf8Type> {
    fn iter(&self) -> ChunkIterState<datatypes::Int32Type, String> {
        let arrays = self
            .chunks
            .iter()
            .map(|a| {
                a.as_any()
                    .downcast_ref::<StringArray>()
                    .expect("could not downcast")
            })
            .collect::<Vec<_>>();

        ChunkIterState {
            arrays: Either::Right(arrays),
            chunk_i: 0,
            array_i: 0,
            out_of_bounds: false,
            length: self.len(),
            phantom: PhantomData,
        }
    }
}

impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let mut builder = PrimitiveChunkedBuilder::new("", 1024);

        for opt_val in iter {
            builder.append_option(opt_val).expect("could not append");
        }

        builder.finish()
    }
}

impl<'a> FromIterator<&'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let mut builder = Utf8ChunkedBuilder::new("", 1024);

        for val in iter {
            builder.append_value(val).expect("could not append");
        }
        builder.finish()
    }
}
