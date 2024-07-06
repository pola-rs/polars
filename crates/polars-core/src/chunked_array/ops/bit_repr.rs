use arrow::buffer::Buffer;

use crate::prelude::*;
use crate::series::BitRepr;

/// Reinterprets the type of a [`ChunkedArray`]. T and U must have the same size
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

/// Reinterprets the type of a [`ListChunked`]. T and U must have the same size
/// and alignment.
#[cfg(feature = "reinterpret")]
fn reinterpret_list_chunked<T: PolarsNumericType, U: PolarsNumericType>(
    ca: &ListChunked,
) -> ListChunked {
    assert!(std::mem::size_of::<T::Native>() == std::mem::size_of::<U::Native>());
    assert!(std::mem::align_of::<T::Native>() == std::mem::align_of::<U::Native>());

    let chunks = ca.downcast_iter().map(|array| {
        let inner_arr = array
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<T::Native>>()
            .unwrap();
        // SAFETY: we checked that the size and alignment matches.
        #[allow(clippy::transmute_undefined_repr)]
        let reinterpreted_buf = unsafe {
            std::mem::transmute::<Buffer<T::Native>, Buffer<U::Native>>(inner_arr.values().clone())
        };
        let pa =
            PrimitiveArray::from_data_default(reinterpreted_buf, inner_arr.validity().cloned());
        LargeListArray::new(
            DataType::List(Box::new(U::get_dtype())).to_arrow(CompatLevel::newest()),
            array.offsets().clone(),
            pa.to_boxed(),
            array.validity().cloned(),
        )
    });

    ListChunked::from_chunk_iter(ca.name(), chunks)
}

#[cfg(all(feature = "reinterpret", feature = "dtype-i16", feature = "dtype-u16"))]
impl Reinterpret for Int16Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt16Type>(self).into_series()
    }
}

#[cfg(all(feature = "reinterpret", feature = "dtype-u16", feature = "dtype-i16"))]
impl Reinterpret for UInt16Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int16Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        self.clone().into_series()
    }
}

#[cfg(all(feature = "reinterpret", feature = "dtype-i8", feature = "dtype-u8"))]
impl Reinterpret for Int8Chunked {
    fn reinterpret_signed(&self) -> Series {
        self.clone().into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt8Type>(self).into_series()
    }
}

#[cfg(all(feature = "reinterpret", feature = "dtype-u8", feature = "dtype-i8"))]
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
        let is_large = std::mem::size_of::<T::Native>() == 8;

        if is_large {
            if matches!(self.dtype(), DataType::UInt64) {
                let ca = self.clone();
                // Convince the compiler we are this type. This keeps flags.
                return BitRepr::Large(unsafe {
                    std::mem::transmute::<ChunkedArray<T>, UInt64Chunked>(ca)
                });
            }

            BitRepr::Large(reinterpret_chunked_array(self))
        } else {
            BitRepr::Small(if std::mem::size_of::<T::Native>() == 4 {
                if matches!(self.dtype(), DataType::UInt32) {
                    let ca = self.clone();
                    // Convince the compiler we are this type. This preserves flags.
                    return BitRepr::Small(unsafe {
                        std::mem::transmute::<ChunkedArray<T>, UInt32Chunked>(ca)
                    });
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
            })
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
        let BitRepr::Large(b) = self.to_bit_repr() else {
            unreachable!()
        };
        b.into_series()
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
        let BitRepr::Small(b) = self.to_bit_repr() else {
            unreachable!()
        };
        b.into_series()
    }
}

#[cfg(feature = "reinterpret")]
impl Reinterpret for Float32Chunked {
    fn reinterpret_signed(&self) -> Series {
        reinterpret_chunked_array::<_, Int32Type>(self).into_series()
    }

    fn reinterpret_unsigned(&self) -> Series {
        reinterpret_chunked_array::<_, UInt32Type>(self).into_series()
    }
}

#[cfg(feature = "reinterpret")]
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

#[cfg(feature = "reinterpret")]
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
        let BitRepr::Small(s) = self.to_bit_repr() else {
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
        let BitRepr::Large(s) = self.to_bit_repr() else {
            unreachable!()
        };
        let s = s.into_series();
        let out = f(&s);
        let out = out.u64().unwrap();
        out._reinterpret_float().into()
    }
}
