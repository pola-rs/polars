use arrow::buffer::Buffer;
use polars_error::feature_gated;

use crate::prelude::*;
use crate::series::BitRepr;

/// Reinterprets the type of a [`ChunkedArray`]. T and U must have the same size
/// and alignment.
fn reinterpret_chunked_array<T: PolarsNumericType, U: PolarsNumericType>(
    ca: &ChunkedArray<T>,
) -> ChunkedArray<U> {
    assert!(size_of::<T::Native>() == size_of::<U::Native>());
    assert!(align_of::<T::Native>() == align_of::<U::Native>());

    let chunks = ca.downcast_iter().map(|array| {
        let buf = array.values().clone();
        let reinterpreted_buf = Buffer::try_transmute::<U::Native>(buf).unwrap();
        PrimitiveArray::from_data_default(reinterpreted_buf, array.validity().cloned())
    });

    ChunkedArray::from_chunk_iter(ca.name().clone(), chunks)
}

/// Reinterprets the type of a [`ListChunked`]. T and U must have the same size
/// and alignment.
fn reinterpret_list_chunked<T: PolarsNumericType, U: PolarsNumericType>(
    ca: &ListChunked,
) -> ListChunked {
    assert!(size_of::<T::Native>() == size_of::<U::Native>());
    assert!(align_of::<T::Native>() == align_of::<U::Native>());

    let chunks = ca.downcast_iter().map(|array| {
        let inner_arr = array
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<T::Native>>()
            .unwrap();
        let reinterpreted_buf =
            Buffer::try_transmute::<U::Native>(inner_arr.values().clone()).unwrap();
        let pa =
            PrimitiveArray::from_data_default(reinterpreted_buf, inner_arr.validity().cloned());
        LargeListArray::new(
            DataType::List(Box::new(U::get_static_dtype())).to_arrow(CompatLevel::newest()),
            array.offsets().clone(),
            pa.to_boxed(),
            array.validity().cloned(),
        )
    });

    ListChunked::from_chunk_iter(ca.name().clone(), chunks)
}

#[cfg(all(feature = "dtype-i16", feature = "dtype-u16"))]
impl Reinterpret for Int16Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt16Type>(self).into_series()
    }
}

#[cfg(all(feature = "dtype-u16", feature = "dtype-i16"))]
impl Reinterpret for UInt16Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int16Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}

#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
impl Reinterpret for Int8Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt8Type>(self).into_series()
    }
}

#[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
impl Reinterpret for UInt8Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int8Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}

impl<T> ToBitRepr for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn to_bit_repr(&self) -> BitRepr {
        match size_of::<T::Native>() {
            16 => {
                feature_gated!("dtype-i128", {
                    if matches!(self.dtype(), DataType::Int128) {
                        let ca: &Int128Chunked = self.as_any().downcast_ref().unwrap();
                        return BitRepr::I128(ca.clone());
                    }

                    BitRepr::I128(reinterpret_chunked_array(self))
                })
            },

            8 => {
                if matches!(self.dtype(), DataType::UInt64) {
                    let ca: &UInt64Chunked = self.as_any().downcast_ref().unwrap();
                    return BitRepr::U64(ca.clone());
                }

                BitRepr::U64(reinterpret_chunked_array(self))
            },

            4 => {
                if matches!(self.dtype(), DataType::UInt32) {
                    let ca: &UInt32Chunked = self.as_any().downcast_ref().unwrap();
                    return BitRepr::U32(ca.clone());
                }

                BitRepr::U32(reinterpret_chunked_array(self))
            },

            2 => {
                if matches!(self.dtype(), DataType::UInt16) {
                    let ca: &UInt16Chunked = self.as_any().downcast_ref().unwrap();
                    return BitRepr::U16(ca.clone());
                }

                BitRepr::U16(reinterpret_chunked_array(self))
            },

            1 => {
                if matches!(self.dtype(), DataType::UInt8) {
                    let ca: &UInt8Chunked = self.as_any().downcast_ref().unwrap();
                    return BitRepr::U8(ca.clone());
                }

                BitRepr::U8(reinterpret_chunked_array(self))
            },

            _ => unreachable!(),
        }
    }
}

impl Reinterpret for UInt64Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int64Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}

impl Reinterpret for Int64Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt64Type>(self).into_series()
    }
}

impl Reinterpret for UInt32Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int32Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}

impl Reinterpret for Int32Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt32Type>(self).into_series()
    }
}

impl Reinterpret for Float32Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int32Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt32Type>(self).into_series()
    }
}

impl Reinterpret for ListChunked {
    fn reinterpret_signed(&self) -> Series {
        match self.inner_dtype() {
            DataType::Float32 => reinterpret_list_chunked::<Float32Type, Int32Type>(self),
            DataType::Float64 => reinterpret_list_chunked::<Float64Type, Int64Type>(self),
            _ => unimplemented!(),
        }
        .into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        match self.inner_dtype() {
            DataType::Float32 => reinterpret_list_chunked::<Float32Type, UInt32Type>(self),
            DataType::Float64 => reinterpret_list_chunked::<Float64Type, UInt64Type>(self),
            _ => unimplemented!(),
        }
        .into_series()
    }
}

impl Reinterpret for Float64Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int64Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt64Type>(self).into_series()
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
    pub fn apply_as_ints<F>(&self, f: F) -> Series
    where
        F: Fn(&Series) -> Series,
    {
        let BitRepr::U32(s) = self.to_bit_repr() else {
            unreachable!()
        };
        let s = s.into_series();
        let out = f(&s);
        let out = out.u32().unwrap();
        out._reinterpret_float().into()
    }
}
impl Float64Chunked {
    pub fn apply_as_ints<F>(&self, f: F) -> Series
    where
        F: Fn(&Series) -> Series,
    {
        let BitRepr::U64(s) = self.to_bit_repr() else {
            unreachable!()
        };
        let s = s.into_series();
        let out = f(&s);
        let out = out.u64().unwrap();
        out._reinterpret_float().into()
    }
}
