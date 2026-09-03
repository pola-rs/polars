use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use polars_array::arrow::bridge::chunk_to_arrow;
use polars_compute::bitwise::BitwiseKernel;

use super::{BooleanType, ChunkBitwiseReduce, ChunkedArray, PolarsNumericType};

/// The one value a chunk repeats and the number of its elements that are not null, if its values
/// buffer holds the single slot every element reads.
///
/// `None` says the chunk has to be walked: its values are laid out one per element, or every
/// element is null and there is no value to reduce. The macro serves both the primitive and the
/// boolean array, which share the shape of these methods but not a trait.
macro_rules! repeated_value {
    ($arr:expr) => {{
        let arr = $arr;
        let count = arr.len() - arr.null_count();
        arr.scalar_values()
            .filter(|_| count > 0)
            .map(|v| (v, count))
    }};
}

/// `and` and `or` are idempotent, so a chunk that repeats one value reduces to that value; `xor`
/// cancels in pairs, so an even number of copies leaves nothing and an odd one leaves a single
/// value. A chunk that has to be walked crosses over to Arrow for the kernel.
impl<T> ChunkBitwiseReduce for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NativeType,
    PrimitiveArray<T::Native>: BitwiseKernel<Scalar = T::Native>,
{
    type Physical = T::Native;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(|arr| match repeated_value!(arr) {
                Some((value, _)) => Some(value),
                None => BitwiseKernel::reduce_and(&chunk_to_arrow(arr)),
            })
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_and)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(|arr| match repeated_value!(arr) {
                Some((value, _)) => Some(value),
                None => BitwiseKernel::reduce_or(&chunk_to_arrow(arr)),
            })
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_or)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(|arr| match repeated_value!(arr) {
                Some((value, count)) if count % 2 == 1 => Some(value),
                // What is left of a value cancelled against itself, whatever `xor` means for it.
                Some((value, _)) => Some(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_xor(
                    value, value,
                )),
                None => BitwiseKernel::reduce_xor(&chunk_to_arrow(arr)),
            })
            .reduce(<PrimitiveArray<T::Native> as BitwiseKernel>::bit_xor)
    }
}

/// See the primitive impl above: the reductions read a scalar chunk the same way.
impl ChunkBitwiseReduce for ChunkedArray<BooleanType> {
    type Physical = bool;

    fn and_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(|arr| match repeated_value!(arr) {
                Some((value, _)) => Some(value),
                None => BitwiseKernel::reduce_and(&chunk_to_arrow(arr)),
            })
            .reduce(|a, b| a & b)
    }

    fn or_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(|arr| match repeated_value!(arr) {
                Some((value, _)) => Some(value),
                None => BitwiseKernel::reduce_or(&chunk_to_arrow(arr)),
            })
            .reduce(|a, b| a | b)
    }

    fn xor_reduce(&self) -> Option<Self::Physical> {
        self.downcast_iter()
            .filter_map(|arr| match repeated_value!(arr) {
                Some((value, count)) => Some(value && count % 2 == 1),
                None => BitwiseKernel::reduce_xor(&chunk_to_arrow(arr)),
            })
            .reduce(|a, b| a ^ b)
    }
}
