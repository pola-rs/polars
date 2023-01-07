use arrow::buffer::Buffer;

use crate::prelude::*;

#[cfg(feature = "performant")]
impl Int16Chunked {
    pub(crate) fn reinterpret_unsigned(&self) -> UInt16Chunked {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let buf = array.values().clone();
                // Safety
                // same bit length i16 <-> u16
                // The fields can still be reordered between generic types
                // so we do some extra assertions
                let len = buf.len();
                let offset = buf.offset();
                let ptr = buf.as_slice().as_ptr() as usize;
                #[allow(clippy::transmute_undefined_repr)]
                let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<u16>>(buf) };
                debug_assert_eq!(reinterpreted_buf.len(), len);
                debug_assert_eq!(reinterpreted_buf.offset(), offset);
                debug_assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                Box::new(PrimitiveArray::new(
                    ArrowDataType::UInt16,
                    reinterpreted_buf,
                    array.validity().cloned(),
                )) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { UInt16Chunked::from_chunks(self.name(), chunks) }
    }
}

#[cfg(feature = "performant")]
impl Int8Chunked {
    pub(crate) fn reinterpret_unsigned(&self) -> UInt8Chunked {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let buf = array.values().clone();
                // Safety
                // same bit length i8 <-> u8
                // The fields can still be reordered between generic types
                // so we do some extra assertions
                let len = buf.len();
                let offset = buf.offset();
                let ptr = buf.as_slice().as_ptr() as usize;
                #[allow(clippy::transmute_undefined_repr)]
                let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<u8>>(buf) };
                debug_assert_eq!(reinterpreted_buf.len(), len);
                debug_assert_eq!(reinterpreted_buf.offset(), offset);
                debug_assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                Box::new(PrimitiveArray::new(
                    ArrowDataType::UInt8,
                    reinterpreted_buf,
                    array.validity().cloned(),
                )) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { UInt8Chunked::from_chunks(self.name(), chunks) }
    }
}

impl<T> ToBitRepr for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn bit_repr_is_large() -> bool {
        std::mem::size_of::<T::Native>() == 8
    }

    fn bit_repr_large(&self) -> UInt64Chunked {
        if std::mem::size_of::<T::Native>() == 8 {
            if matches!(self.dtype(), DataType::UInt64) {
                let ca = self.clone();
                // convince the compiler we are this type. This keeps flags
                return unsafe { std::mem::transmute(ca) };
            }
            let chunks = self
                .downcast_iter()
                .map(|array| {
                    let buf = array.values().clone();
                    // Safety:
                    // we just check the size of T::Native to be 64 bits
                    // The fields can still be reordered between generic types
                    // so we do some extra assertions
                    let len = buf.len();
                    let offset = buf.offset();
                    let ptr = buf.as_slice().as_ptr() as usize;
                    #[allow(clippy::transmute_undefined_repr)]
                    let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<u64>>(buf) };
                    assert_eq!(reinterpreted_buf.len(), len);
                    assert_eq!(reinterpreted_buf.offset(), offset);
                    assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                    Box::new(PrimitiveArray::new(
                        ArrowDataType::UInt64,
                        reinterpreted_buf,
                        array.validity().cloned(),
                    )) as ArrayRef
                })
                .collect::<Vec<_>>();
            unsafe { UInt64Chunked::from_chunks(self.name(), chunks) }
        } else {
            unreachable!()
        }
    }

    fn bit_repr_small(&self) -> UInt32Chunked {
        if std::mem::size_of::<T::Native>() == 4 {
            if matches!(self.dtype(), DataType::UInt32) {
                let ca = self.clone();
                // convince the compiler we are this type. This keeps flags
                return unsafe { std::mem::transmute(ca) };
            }
            let chunks = self
                .downcast_iter()
                .map(|array| {
                    let buf = array.values().clone();
                    // Safety:
                    // we just check the size of T::Native to be 32 bits
                    // The fields can still be reordered between generic types
                    // so we do some extra assertions
                    let len = buf.len();
                    let offset = buf.offset();
                    let ptr = buf.as_slice().as_ptr() as usize;
                    #[allow(clippy::transmute_undefined_repr)]
                    let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<u32>>(buf) };
                    assert_eq!(reinterpreted_buf.len(), len);
                    assert_eq!(reinterpreted_buf.offset(), offset);
                    assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                    Box::new(PrimitiveArray::new(
                        ArrowDataType::UInt32,
                        reinterpreted_buf,
                        array.validity().cloned(),
                    )) as ArrayRef
                })
                .collect::<Vec<_>>();
            unsafe { UInt32Chunked::from_chunks(self.name(), chunks) }
        } else {
            self.cast_unchecked(&DataType::UInt32)
                .unwrap()
                .u32()
                .unwrap()
                .clone()
        }
    }
}

#[cfg(feature = "reinterpret")]
impl Reinterpret for UInt64Chunked {
    fn reinterpret_signed(&self) -> Series {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let buf = array.values().clone();
                // Safety
                // same bit length u64 <-> i64
                // The fields can still be reordered between generic types
                // so we do some extra assertions
                let len = buf.len();
                let offset = buf.offset();
                let ptr = buf.as_slice().as_ptr() as usize;
                #[allow(clippy::transmute_undefined_repr)]
                let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<i64>>(buf) };
                assert_eq!(reinterpreted_buf.len(), len);
                assert_eq!(reinterpreted_buf.offset(), offset);
                assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                Box::new(PrimitiveArray::new(
                    ArrowDataType::Int64,
                    reinterpreted_buf,
                    array.validity().cloned(),
                )) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { Int64Chunked::from_chunks(self.name(), chunks).into_series() }
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

#[cfg(feature = "reinterpret")]
impl Reinterpret for UInt32Chunked {
    fn reinterpret_signed(&self) -> Series {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let buf = array.values().clone();
                // Safety
                // same bit length u32 <-> i32
                // The fields can still be reordered between generic types
                // so we do some extra assertions
                let len = buf.len();
                let offset = buf.offset();
                let ptr = buf.as_slice().as_ptr() as usize;
                #[allow(clippy::transmute_undefined_repr)]
                let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<i32>>(buf) };
                assert_eq!(reinterpreted_buf.len(), len);
                assert_eq!(reinterpreted_buf.offset(), offset);
                assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                Box::new(PrimitiveArray::new(
                    ArrowDataType::Int32,
                    reinterpreted_buf,
                    array.validity().cloned(),
                )) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { Int32Chunked::from_chunks(self.name(), chunks).into_series() }
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}

#[cfg(feature = "reinterpret")]
impl Reinterpret for Int32Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.bit_repr_large().into_series()
    }
}

impl UInt64Chunked {
    #[doc(hidden)]
    pub fn _reinterpret_float(&self) -> Float64Chunked {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let buf = array.values().clone();
                // Safety
                // same bit length u64 <-> f64
                // The fields can still be reordered between generic types
                // so we do some extra assertions
                let len = buf.len();
                let offset = buf.offset();
                let ptr = buf.as_slice().as_ptr() as usize;
                #[allow(clippy::transmute_undefined_repr)]
                let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<f64>>(buf) };
                assert_eq!(reinterpreted_buf.len(), len);
                assert_eq!(reinterpreted_buf.offset(), offset);
                assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                Box::new(PrimitiveArray::new(
                    ArrowDataType::Float64,
                    reinterpreted_buf,
                    array.validity().cloned(),
                )) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { Float64Chunked::from_chunks(self.name(), chunks) }
    }
}
impl UInt32Chunked {
    #[doc(hidden)]
    pub fn _reinterpret_float(&self) -> Float32Chunked {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let buf = array.values().clone();
                // Safety
                // same bit length u32 <-> f32
                // The fields can still be reordered between generic types
                // so we do some extra assertions
                let len = buf.len();
                let offset = buf.offset();
                let ptr = buf.as_slice().as_ptr() as usize;
                #[allow(clippy::transmute_undefined_repr)]
                let reinterpreted_buf = unsafe { std::mem::transmute::<_, Buffer<f32>>(buf) };
                assert_eq!(reinterpreted_buf.len(), len);
                assert_eq!(reinterpreted_buf.offset(), offset);
                assert_eq!(reinterpreted_buf.as_slice().as_ptr() as usize, ptr);
                Box::new(PrimitiveArray::new(
                    ArrowDataType::Float32,
                    reinterpreted_buf,
                    array.validity().cloned(),
                )) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { Float32Chunked::from_chunks(self.name(), chunks) }
    }
}

/// Used to save compilation paths. Use carefully. Although this is safe,
/// if misused it can lead to incorrect results.
impl Float32Chunked {
    pub(crate) fn apply_as_ints<F>(&self, f: F) -> Series
    where
        F: Fn(&Series) -> Series,
    {
        let s = self.bit_repr_small().into_series();
        let out = f(&s);
        let out = out.u32().unwrap();
        out._reinterpret_float().into()
    }
}
impl Float64Chunked {
    pub(crate) fn apply_as_ints<F>(&self, f: F) -> Series
    where
        F: Fn(&Series) -> Series,
    {
        let s = self.bit_repr_large().into_series();
        let out = f(&s);
        let out = out.u64().unwrap();
        out._reinterpret_float().into()
    }
}
