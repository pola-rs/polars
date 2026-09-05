use polars_array::ArrayRepr;
use polars_buffer::Buffer;
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
        let length = array.len();
        // The values are handed over as they are, so a scalar chunk stays one value.
        let out = match array.values_repr() {
            ArrayRepr::Flat(buf) => PlPrimitiveArray::new(
                Buffer::try_transmute::<U::Native>(buf.clone()).unwrap(),
                length,
                None,
            ),
            ArrayRepr::Scalar(value) => PlPrimitiveArray::new_broadcast(
                Buffer::try_transmute::<U::Native>(Buffer::from(vec![value])).unwrap(),
                length,
                None,
            ),
        };
        out.with_validity(array.validity().map(PlBitmap::from))
    });

    ChunkedArray::from_chunk_iter(ca.name().clone(), chunks)
}

impl<T> ToBitRepr for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn to_bit_repr(&self) -> BitRepr {
        match size_of::<T::Native>() {
            16 => {
                feature_gated!("dtype-u128", {
                    if matches!(self.dtype(), DataType::UInt128) {
                        let ca: &UInt128Chunked = self.as_any().downcast_ref().unwrap();
                        return BitRepr::U128(ca.clone());
                    }

                    BitRepr::U128(reinterpret_chunked_array(self))
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

pub fn reinterpret(s: &Series, dtype: &DataType) -> PolarsResult<Series> {
    if s.dtype().is_numeric() && s.dtype() == dtype {
        return Ok(s.clone());
    }

    Ok(match (s.dtype(), dtype) {
        #[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
        (DataType::UInt8, DataType::Int8) => {
            reinterpret_chunked_array::<_, Int8Type>(s.u8().unwrap()).into_series()
        },
        #[cfg(all(feature = "dtype-u16", feature = "dtype-i16"))]
        (DataType::UInt16, DataType::Int16) => {
            reinterpret_chunked_array::<_, Int16Type>(s.u16().unwrap()).into_series()
        },
        (DataType::UInt32, DataType::Int32) => {
            reinterpret_chunked_array::<_, Int32Type>(s.u32().unwrap()).into_series()
        },
        (DataType::UInt64, DataType::Int64) => {
            reinterpret_chunked_array::<_, Int64Type>(s.u64().unwrap()).into_series()
        },
        #[cfg(all(feature = "dtype-u128", feature = "dtype-i128"))]
        (DataType::UInt128, DataType::Int128) => {
            reinterpret_chunked_array::<_, Int128Type>(s.u128().unwrap()).into_series()
        },

        #[cfg(all(feature = "dtype-u16", feature = "dtype-f16"))]
        (DataType::UInt16, DataType::Float16) => {
            reinterpret_chunked_array::<_, Float16Type>(s.u16().unwrap()).into_series()
        },
        (DataType::UInt32, DataType::Float32) => {
            reinterpret_chunked_array::<_, Float32Type>(s.u32().unwrap()).into_series()
        },
        (DataType::UInt64, DataType::Float64) => {
            reinterpret_chunked_array::<_, Float64Type>(s.u64().unwrap()).into_series()
        },

        #[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
        (DataType::Int8, DataType::UInt8) => {
            reinterpret_chunked_array::<_, UInt8Type>(s.i8().unwrap()).into_series()
        },
        #[cfg(all(feature = "dtype-i16", feature = "dtype-u16"))]
        (DataType::Int16, DataType::UInt16) => {
            reinterpret_chunked_array::<_, UInt16Type>(s.i16().unwrap()).into_series()
        },
        (DataType::Int32, DataType::UInt32) => {
            reinterpret_chunked_array::<_, UInt32Type>(s.i32().unwrap()).into_series()
        },
        (DataType::Int64, DataType::UInt64) => {
            reinterpret_chunked_array::<_, UInt64Type>(s.i64().unwrap()).into_series()
        },
        #[cfg(all(feature = "dtype-i128", feature = "dtype-u128"))]
        (DataType::Int128, DataType::UInt128) => {
            reinterpret_chunked_array::<_, UInt128Type>(s.i128().unwrap()).into_series()
        },

        #[cfg(all(feature = "dtype-i16", feature = "dtype-f16"))]
        (DataType::Int16, DataType::Float16) => {
            reinterpret_chunked_array::<_, Float16Type>(s.i16().unwrap()).into_series()
        },
        (DataType::Int32, DataType::Float32) => {
            reinterpret_chunked_array::<_, Float32Type>(s.i32().unwrap()).into_series()
        },
        (DataType::Int64, DataType::Float64) => {
            reinterpret_chunked_array::<_, Float64Type>(s.i64().unwrap()).into_series()
        },

        #[cfg(all(feature = "dtype-f16", feature = "dtype-u16"))]
        (DataType::Float16, DataType::UInt16) => {
            reinterpret_chunked_array::<_, UInt16Type>(s.f16().unwrap()).into_series()
        },
        (DataType::Float32, DataType::UInt32) => {
            reinterpret_chunked_array::<_, UInt32Type>(s.f32().unwrap()).into_series()
        },
        (DataType::Float64, DataType::UInt64) => {
            reinterpret_chunked_array::<_, UInt64Type>(s.f64().unwrap()).into_series()
        },

        #[cfg(all(feature = "dtype-f16", feature = "dtype-i16"))]
        (DataType::Float16, DataType::Int16) => {
            reinterpret_chunked_array::<_, Int16Type>(s.f16().unwrap()).into_series()
        },
        (DataType::Float32, DataType::Int32) => {
            reinterpret_chunked_array::<_, Int32Type>(s.f32().unwrap()).into_series()
        },
        (DataType::Float64, DataType::Int64) => {
            reinterpret_chunked_array::<_, Int64Type>(s.f64().unwrap()).into_series()
        },

        (source_dtype, target_dtype) => polars_bail!(
            ComputeError:
            "cannot reinterpret from {source_dtype:?} to {target_dtype:?}"
        ),
    })
}

#[cfg(feature = "dtype-f16")]
impl UInt16Chunked {
    #[doc(hidden)]
    pub fn _reinterpret_float(&self) -> Float16Chunked {
        reinterpret_chunked_array(self)
    }
}

impl UInt32Chunked {
    #[doc(hidden)]
    pub fn _reinterpret_float(&self) -> Float32Chunked {
        reinterpret_chunked_array(self)
    }
}

impl UInt64Chunked {
    #[doc(hidden)]
    pub fn _reinterpret_float(&self) -> Float64Chunked {
        reinterpret_chunked_array(self)
    }
}

/// Used to save compilation paths. Use carefully. Although this is safe,
/// if misused it can lead to incorrect results.
#[cfg(feature = "dtype-f16")]
impl Float16Chunked {
    pub fn apply_as_ints<F>(&self, f: F) -> Series
    where
        F: Fn(&Series) -> Series,
    {
        let BitRepr::U16(s) = self.to_bit_repr() else {
            unreachable!()
        };
        let s = s.into_series();
        let out = f(&s);
        let out = out.u16().unwrap();
        out._reinterpret_float().into()
    }
}

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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::series::BitRepr;

    /// An empty chunk is flat over an empty buffer, and a scalar chunk over a single slot: both
    /// are reinterpreted through the buffer they hold, with no third case to fall through to.
    #[test]
    fn reinterpreting_empty_and_scalar_chunks() {
        let empty = Int64Chunked::from_slice(PlSmallStr::from_static("a"), &[]);
        let BitRepr::U64(out) = empty.to_bit_repr() else {
            panic!("an i64 reinterprets as a u64")
        };
        assert_eq!(out.len(), 0);

        let scalar = Int64Chunked::full(PlSmallStr::from_static("a"), -1, 5);
        let BitRepr::U64(out) = scalar.to_bit_repr() else {
            panic!("an i64 reinterprets as a u64")
        };
        assert_eq!(out.len(), 5);
        assert_eq!(out.get(0), Some(u64::MAX));
        assert!(out.downcast_iter().all(|arr| arr.values_are_scalar()));
    }
}
