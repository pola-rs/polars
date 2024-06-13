use parquet_format_safe::Type;
#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use crate::parquet::error::ParquetError;

/// The set of all physical types representable in Parquet
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum PhysicalType {
    Boolean,
    Int32,
    Int64,
    Int96,
    Float,
    Double,
    ByteArray,
    FixedLenByteArray(usize),
}

impl TryFrom<(Type, Option<i32>)> for PhysicalType {
    type Error = ParquetError;

    fn try_from((type_, length): (Type, Option<i32>)) -> Result<Self, Self::Error> {
        Ok(match type_ {
            Type::BOOLEAN => PhysicalType::Boolean,
            Type::INT32 => PhysicalType::Int32,
            Type::INT64 => PhysicalType::Int64,
            Type::INT96 => PhysicalType::Int96,
            Type::FLOAT => PhysicalType::Float,
            Type::DOUBLE => PhysicalType::Double,
            Type::BYTE_ARRAY => PhysicalType::ByteArray,
            Type::FIXED_LEN_BYTE_ARRAY => {
                let length = length.ok_or_else(|| {
                    ParquetError::oos("Length must be defined for FixedLenByteArray")
                })?;
                PhysicalType::FixedLenByteArray(length.try_into()?)
            },
            _ => return Err(ParquetError::oos("Unknown type")),
        })
    }
}

impl From<PhysicalType> for (Type, Option<i32>) {
    fn from(physical_type: PhysicalType) -> Self {
        match physical_type {
            PhysicalType::Boolean => (Type::BOOLEAN, None),
            PhysicalType::Int32 => (Type::INT32, None),
            PhysicalType::Int64 => (Type::INT64, None),
            PhysicalType::Int96 => (Type::INT96, None),
            PhysicalType::Float => (Type::FLOAT, None),
            PhysicalType::Double => (Type::DOUBLE, None),
            PhysicalType::ByteArray => (Type::BYTE_ARRAY, None),
            PhysicalType::FixedLenByteArray(length) => {
                (Type::FIXED_LEN_BYTE_ARRAY, Some(length as i32))
            },
        }
    }
}
