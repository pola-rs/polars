use crate::prelude::*;
use arrow::array::Array;
use arrow::buffer::Buffer;

impl<T> ToBitRepr for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn bit_repr_is_large() -> bool {
        std::mem::size_of::<T::Native>() == 8
    }

    fn bit_repr_large(&self) -> UInt64Chunked {
        if std::mem::size_of::<T::Native>() == 8 {
            let chunks = self
                .downcast_iter()
                .map(|array| {
                    let buf = array.values().clone();
                    // Safety:
                    // we just check the size of T::Native to be 64 bits
                    let buf = unsafe { std::mem::transmute::<_, Buffer<u64>>(buf) };
                    Arc::new(PrimitiveArray::from_data(
                        ArrowDataType::UInt64,
                        buf,
                        array.validity().clone(),
                    )) as Arc<dyn Array>
                })
                .collect::<Vec<_>>();
            UInt64Chunked::new_from_chunks(self.name(), chunks)
        } else {
            unreachable!()
        }
    }

    fn bit_repr_small(&self) -> UInt32Chunked {
        if std::mem::size_of::<T::Native>() == 4 {
            let chunks = self
                .downcast_iter()
                .map(|array| {
                    let buf = array.values().clone();
                    // Safety:
                    // we just check the size of T::Native to be 32 bits
                    let buf = unsafe { std::mem::transmute::<_, Buffer<u32>>(buf) };
                    Arc::new(PrimitiveArray::from_data(
                        ArrowDataType::UInt32,
                        buf,
                        array.validity().clone(),
                    )) as Arc<dyn Array>
                })
                .collect::<Vec<_>>();
            UInt32Chunked::new_from_chunks(self.name(), chunks)
        } else {
            unreachable!()
        }
    }
}

impl ToBitRepr for CategoricalChunked {
    fn bit_repr_is_large() -> bool {
        // u32
        false
    }

    fn bit_repr_large(&self) -> UInt64Chunked {
        unimplemented!()
    }

    fn bit_repr_small(&self) -> UInt32Chunked {
        self.cast::<UInt32Type>().unwrap()
    }
}
