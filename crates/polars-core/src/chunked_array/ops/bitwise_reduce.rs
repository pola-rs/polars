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
            .filter_map(BitwiseKernel::reduce_and)
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_and)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_or)
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_or)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_xor)
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_xor)
    }
}

impl ChunkBitwiseReduce for ChunkedArray<BooleanType> {
    type Physical = bool;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_and)
            .reduce(|a, b| a & b)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_or)
            .reduce(|a, b| a | b)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(BitwiseKernel::reduce_xor)
            .reduce(|a, b| a ^ b)
    }
}
