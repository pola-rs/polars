mod binary;
mod boolean;
mod fixed_len_binary;
mod primitive;

pub use binary::BinaryStatistics;
pub use boolean::BooleanStatistics;
pub use fixed_len_binary::FixedLenStatistics;
pub use primitive::PrimitiveStatistics;

use crate::parquet::error::ParquetResult;
use crate::parquet::schema::types::{PhysicalType, PrimitiveType};
pub use crate::parquet::thrift_format::Statistics as ParquetStatistics;

#[derive(Debug, PartialEq)]
pub enum Statistics {
    Binary(BinaryStatistics),
    Boolean(BooleanStatistics),
    FixedLen(FixedLenStatistics),
    Int32(PrimitiveStatistics<i32>),
    Int64(PrimitiveStatistics<i64>),
    Int96(PrimitiveStatistics<[u32; 3]>),
    Float(PrimitiveStatistics<f32>),
    Double(PrimitiveStatistics<f64>),
}

impl Statistics {
    #[inline]
    pub const fn physical_type(&self) -> &PhysicalType {
        use Statistics as S;

        match self {
            S::Binary(_) => &PhysicalType::ByteArray,
            S::Boolean(_) => &PhysicalType::Boolean,
            S::FixedLen(s) => &s.primitive_type.physical_type,
            S::Int32(_) => &PhysicalType::Int32,
            S::Int64(_) => &PhysicalType::Int64,
            S::Int96(_) => &PhysicalType::Int96,
            S::Float(_) => &PhysicalType::Float,
            S::Double(_) => &PhysicalType::Double,
        }
    }

    /// Deserializes a raw parquet statistics into [`Statistics`].
    /// # Error
    /// This function errors if it is not possible to read the statistics to the
    /// corresponding `physical_type`.
    #[inline]
    pub fn deserialize(
        statistics: &ParquetStatistics,
        primitive_type: PrimitiveType,
    ) -> ParquetResult<Self> {
        use {PhysicalType as T, PrimitiveStatistics as PrimStat};
        Ok(match primitive_type.physical_type {
            T::ByteArray => BinaryStatistics::deserialize(statistics, primitive_type)?.into(),
            T::Boolean => BooleanStatistics::deserialize(statistics)?.into(),
            T::Int32 => PrimStat::<i32>::deserialize(statistics, primitive_type)?.into(),
            T::Int64 => PrimStat::<i64>::deserialize(statistics, primitive_type)?.into(),
            T::Int96 => PrimStat::<[u32; 3]>::deserialize(statistics, primitive_type)?.into(),
            T::Float => PrimStat::<f32>::deserialize(statistics, primitive_type)?.into(),
            T::Double => PrimStat::<f64>::deserialize(statistics, primitive_type)?.into(),
            T::FixedLenByteArray(size) => {
                FixedLenStatistics::deserialize(statistics, size, primitive_type)?.into()
            },
        })
    }
}

macro_rules! statistics_from_as {
    ($($variant:ident($struct:ty) => ($as_ident:ident, $into_ident:ident, $expect_ident:ident, $owned_expect_ident:ident),)+) => {
        $(
            impl From<$struct> for Statistics {
                #[inline]
                fn from(stats: $struct) -> Self {
                    Self::$variant(stats)
                }
            }
        )+

        impl Statistics {
            #[inline]
            pub const fn null_count(&self) -> Option<i64> {
                match self {
                    $(Self::$variant(s) => s.null_count,)+
                }
            }

            /// Serializes [`Statistics`] into a raw parquet statistics.
            #[inline]
            pub fn serialize(&self) -> ParquetStatistics {
                match self {
                    $(Self::$variant(s) => s.serialize(),)+
                }
            }

            const fn variant_str(&self) -> &'static str {
                match self {
                    $(Self::$variant(_) => stringify!($struct),)+
                }
            }

            $(
                #[doc = concat!("Try to take [`Statistics`] as [`", stringify!($struct), "`]")]
                #[inline]
                pub fn $as_ident(&self) -> Option<&$struct> {
                    match self {
                        Self::$variant(s) => Some(s),
                        _ => None,
                    }
                }

                #[doc = concat!("Try to take [`Statistics`] as [`", stringify!($struct), "`]")]
                #[inline]
                pub fn $into_ident(self) -> Option<$struct> {
                    match self {
                        Self::$variant(s) => Some(s),
                        _ => None,
                    }
                }

                #[doc = concat!("Interpret [`Statistics`] to be [`", stringify!($struct), "`]")]
                ///
                /// Panics if it is not the correct variant.
                #[track_caller]
                #[inline]
                pub fn $expect_ident(&self) -> &$struct {
                    let Self::$variant(s) = self else {
                        panic!("Expected Statistics to be {}, found {} instead", stringify!($struct), self.variant_str());
                    };

                    s
                }

                #[doc = concat!("Interpret [`Statistics`] to be [`", stringify!($struct), "`]")]
                ///
                /// Panics if it is not the correct variant.
                #[track_caller]
                #[inline]
                pub fn $owned_expect_ident(self) -> $struct {
                    let Self::$variant(s) = self else {
                        panic!("Expected Statistics to be {}, found {} instead", stringify!($struct), self.variant_str());
                    };

                    s
                }
            )+

        }
    };
}

statistics_from_as! {
    Binary    (BinaryStatistics             ) => (as_binary,   into_binary,   expect_as_binary,   expect_binary  ),
    Boolean   (BooleanStatistics            ) => (as_boolean,  into_boolean,  expect_as_boolean,  expect_boolean ),
    FixedLen  (FixedLenStatistics           ) => (as_fixedlen, into_fixedlen, expect_as_fixedlen, expect_fixedlen),
    Int32     (PrimitiveStatistics<i32>     ) => (as_int32,    into_int32,    expect_as_int32,    expect_int32   ),
    Int64     (PrimitiveStatistics<i64>     ) => (as_int64,    into_int64,    expect_as_int64,    expect_int64   ),
    Int96     (PrimitiveStatistics<[u32; 3]>) => (as_int96,    into_int96,    expect_as_int96,    expect_int96   ),
    Float     (PrimitiveStatistics<f32>     ) => (as_float,    into_float,    expect_as_float,    expect_float   ),
    Double    (PrimitiveStatistics<f64>     ) => (as_double,   into_double,   expect_as_double,   expect_double  ),
}
