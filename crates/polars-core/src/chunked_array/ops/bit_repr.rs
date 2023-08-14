use arrow::buffer::Buffer;

use crate::prelude::*;

/// Reinterprets the type of a ChunkedArray. T and U must have the same size
/// and alignment.
fn reinterpret_chunked_array<T: PolarsNumericType, U: PolarsNumericType>(
    ca: &ChunkedArray<T>,
) -> ChunkedArray<U> {
    assert!(std::mem::size_of::<T::Native>() == std::mem::size_of::<U::Native>());
    assert!(std::mem::align_of::<T::Native>() == std::mem::align_of::<U::Native>());

    let chunks = ca.downcast_iter().map(|array| {
        let buf = array.values().clone();
        // SAFETY: we checked that the size and alignment matches.
        #[allow(clippy::transmute_undefined_repr)]
        let reinterpreted_buf =
            unsafe { std::mem::transmute::<Buffer<T::Native>, Buffer<U::Native>>(buf) };
        PrimitiveArray::from_data_default(reinterpreted_buf, array.validity().cloned())
    });

    ChunkedArray::from_chunk_iter(ca.name(), chunks)
}

#[cfg(feature = "performant")]
impl Int16Chunked {
    pub(crate) fn reinterpret_unsigned(&self) -> UInt16Chunked {
        reinterpret_chunked_array(self)
    }
}

#[cfg(feature = "performant")]
impl Int8Chunked {
    pub(crate) fn reinterpret_unsigned(&self) -> UInt8Chunked {
        reinterpret_chunked_array(self)
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
                // Convince the compiler we are this type. This keeps flags.
                return unsafe { std::mem::transmute(ca) };
            }
            reinterpret_chunked_array(self)
        } else {
            unreachable!()
        }
    }

    fn bit_repr_small(&self) -> UInt32Chunked {
        if std::mem::size_of::<T::Native>() == 4 {
            if matches!(self.dtype(), DataType::UInt32) {
                let ca = self.clone();
                // Convince the compiler we are this type. This preserves flags.
                return unsafe { std::mem::transmute(ca) };
            }
            reinterpret_chunked_array(self)
        } else {
            // SAFETY: an unchecked cast to uint32 (which has no invariants) is
            // always sound.
            unsafe {
                self.cast_unchecked(&DataType::UInt32)
                    .unwrap()
                    .u32()
                    .unwrap()
                    .clone()
            }
        }
    }
}

#[cfg(feature = "reinterpret")]
impl Reinterpret for UInt64Chunked {
    fn reinterpret_signed(&self) -> Series {
        let signed: Int64Chunked = reinterpret_chunked_array(self);
        signed.into_series()
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
        let signed: Int32Chunked = reinterpret_chunked_array(self);
        signed.into_series()
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
        reinterpret_chunked_array(self)
    }
}
impl UInt32Chunked {
    #[doc(hidden)]
    pub fn _reinterpret_float(&self) -> Float32Chunked {
        reinterpret_chunked_array(self)
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
