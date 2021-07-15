use crate::prelude::*;
use arrow::array::{make_array, Array, ArrayData};

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
                    let data = array.data();
                    let buffers = data.buffers().to_vec();
                    let null_buf = data.null_buffer().cloned();

                    let mut builder = ArrayData::builder(ArrowDataType::UInt64)
                        .buffers(buffers)
                        .len(array.len())
                        .offset(array.offset());
                    if let Some(null_buf) = null_buf {
                        builder = builder.null_bit_buffer(null_buf);
                    }
                    make_array(builder.build())
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
                    let data = array.data();
                    let buffers = data.buffers().to_vec();
                    let null_buf = data.null_buffer().cloned();

                    let mut builder = ArrayData::builder(ArrowDataType::UInt32)
                        .buffers(buffers)
                        .len(array.len())
                        .offset(array.offset());
                    if let Some(null_buf) = null_buf {
                        builder = builder.null_bit_buffer(null_buf);
                    }
                    make_array(builder.build())
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

#[cfg(feature = "reinterpret")]
impl Reinterpret for UInt64Chunked {
    fn reinterpret_signed(&self) -> Series {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let data = array.data();
                let buffers = data.buffers().to_vec();
                let null_buf = data.null_buffer().cloned();

                let mut builder = ArrayData::builder(ArrowDataType::Int64)
                    .buffers(buffers)
                    .len(array.len())
                    .offset(array.offset());
                if let Some(null_buf) = null_buf {
                    builder = builder.null_bit_buffer(null_buf);
                }
                make_array(builder.build())
            })
            .collect::<Vec<_>>();
        Int64Chunked::new_from_chunks(self.name(), chunks).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}
#[cfg(feature = "reinterpret")]
impl Reinterpret for Int64Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.bit_repr_large().into_series()
    }
}
