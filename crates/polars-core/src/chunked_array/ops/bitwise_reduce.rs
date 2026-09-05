use polars_array::{PlBooleanArray, PlPrimitiveArray};
use polars_compute::bitwise::BitwiseKernel;

use super::{BooleanType, ChunkBitwiseReduce, ChunkedArray, PolarsNumericType};

/// The kernels read a chunk in whichever representation it is in: a scalar one reduces to the
/// single value it repeats without being walked. See `polars_compute::bitwise`.
impl<T> ChunkBitwiseReduce for ChunkedArray<T>
where
    T: PolarsNumericType,
    PlPrimitiveArray<T::Native>: BitwiseKernel<Scalar = T::Native>,
{
    type Physical = T::Native;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_and)
            .reduce(<PlPrimitiveArray<T::Native> as BitwiseKernel>::bit_and)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_or)
            .reduce(<PlPrimitiveArray<T::Native> as BitwiseKernel>::bit_or)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_xor)
            .reduce(<PlPrimitiveArray<T::Native> as BitwiseKernel>::bit_xor)
    }
}

impl ChunkBitwiseReduce for ChunkedArray<BooleanType> {
    type Physical = bool;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(<PlBooleanArray as BitwiseKernel>::reduce_and)
            .reduce(|a, b| a & b)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(<PlBooleanArray as BitwiseKernel>::reduce_or)
            .reduce(|a, b| a | b)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(<PlBooleanArray as BitwiseKernel>::reduce_xor)
            .reduce(|a, b| a ^ b)
    }
}
