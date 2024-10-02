use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use polars_compute::bitwise::BitwiseKernel;

use super::{BooleanType, ChunkBitwiseReduce, ChunkedArray, PolarsNumericType};

impl<T> ChunkBitwiseReduce for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NativeType,
    PrimitiveArray<T::Native>: BitwiseKernel<Scalar = T::Native>,
{
    type Physical = T::Native;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .map(|arr| BitwiseKernel::reduce_and(arr))
            .flatten()
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_and)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .map(|arr| BitwiseKernel::reduce_or(arr))
            .flatten()
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_or)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .map(|arr| BitwiseKernel::reduce_xor(arr))
            .flatten()
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_xor)
    }
}

impl ChunkBitwiseReduce for ChunkedArray<BooleanType> {
    type Physical = bool;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .map(|arr| BitwiseKernel::reduce_and(arr))
            .flatten()
            .reduce(|a, b| a & b)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .map(|arr| BitwiseKernel::reduce_or(arr))
            .flatten()
            .reduce(|a, b| a | b)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .map(|arr| BitwiseKernel::reduce_xor(arr))
            .flatten()
            .reduce(|a, b| a ^ b)
    }
}
