use serde::de::Deserializer;
use serde::{Serialize, Serializer};

use crate::prelude::*;

// We don't use this internally (we call Series::serialize instead), but Rust users might need it.
impl<T> Serialize for ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
{
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.clone().into_series().serialize(serializer)
    }
}

macro_rules! impl_chunked_array_deserialize {
    ($chunked_array_type:ty, $series_downcast_func:expr) => {
        impl<'de> serde::de::Deserialize<'de> for $chunked_array_type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                $series_downcast_func(&Series::deserialize(deserializer)?)
                    .cloned()
                    .map_err(|e| {
                        e.wrap_msg(|e| {
                            format!(
                                "error deserializing into {}: {}",
                                stringify!($chunked_array_type),
                                e
                            )
                        })
                    })
                    .map_err(serde::de::Error::custom)
            }
        }
    };
}

#[cfg(feature = "dtype-array")]
impl_chunked_array_deserialize!(ArrayChunked, Series::array);
impl_chunked_array_deserialize!(ListChunked, Series::list);
impl_chunked_array_deserialize!(BooleanChunked, Series::bool);
impl_chunked_array_deserialize!(UInt8Chunked, Series::u8);
impl_chunked_array_deserialize!(UInt16Chunked, Series::u16);
impl_chunked_array_deserialize!(UInt32Chunked, Series::u32);
impl_chunked_array_deserialize!(UInt64Chunked, Series::u64);
impl_chunked_array_deserialize!(Int8Chunked, Series::i8);
impl_chunked_array_deserialize!(Int16Chunked, Series::i16);
impl_chunked_array_deserialize!(Int32Chunked, Series::i32);
impl_chunked_array_deserialize!(Int64Chunked, Series::i64);
#[cfg(feature = "dtype-i128")]
impl_chunked_array_deserialize!(Int128Chunked, Series::i128);
impl_chunked_array_deserialize!(Float32Chunked, Series::f32);
impl_chunked_array_deserialize!(Float64Chunked, Series::f64);
impl_chunked_array_deserialize!(StringChunked, Series::str);
impl_chunked_array_deserialize!(BinaryChunked, Series::binary);
impl_chunked_array_deserialize!(BinaryOffsetChunked, Series::binary_offset);
