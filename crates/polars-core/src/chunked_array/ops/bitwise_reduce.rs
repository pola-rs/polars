use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use polars_array::arrow::bridge::chunk_to_arrow;
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
            // TODO(polars-array-scalar): the kernel is the Arrow one, so each chunk crosses over;
            // the reduction of a scalar chunk follows from its single value and its length.
            .filter_map(|arr| BitwiseKernel::reduce_and(&chunk_to_arrow(arr)))
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_and)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            // TODO(polars-array-scalar): the kernel is the Arrow one, so each chunk crosses over;
            // the reduction of a scalar chunk follows from its single value and its length.
            .filter_map(|arr| BitwiseKernel::reduce_or(&chunk_to_arrow(arr)))
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_or)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            // TODO(polars-array-scalar): the kernel is the Arrow one, so each chunk crosses over;
            // the reduction of a scalar chunk follows from its single value and its length.
            .filter_map(|arr| BitwiseKernel::reduce_xor(&chunk_to_arrow(arr)))
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_xor)
    }
}

impl ChunkBitwiseReduce for ChunkedArray<BooleanType> {
    type Physical = bool;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            // TODO(polars-array-scalar): the kernel is the Arrow one, so each chunk crosses over;
            // the reduction of a scalar chunk follows from its single value and its length.
            .filter_map(|arr| BitwiseKernel::reduce_and(&chunk_to_arrow(arr)))
            .reduce(|a, b| a & b)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            // TODO(polars-array-scalar): the kernel is the Arrow one, so each chunk crosses over;
            // the reduction of a scalar chunk follows from its single value and its length.
            .filter_map(|arr| BitwiseKernel::reduce_or(&chunk_to_arrow(arr)))
            .reduce(|a, b| a | b)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            // TODO(polars-array-scalar): the kernel is the Arrow one, so each chunk crosses over;
            // the reduction of a scalar chunk follows from its single value and its length.
            .filter_map(|arr| BitwiseKernel::reduce_xor(&chunk_to_arrow(arr)))
            .reduce(|a, b| a ^ b)
    }
}
