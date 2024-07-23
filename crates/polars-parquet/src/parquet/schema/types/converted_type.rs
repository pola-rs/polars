use parquet_format_safe::ConvertedType;
#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use crate::parquet::error::ParquetError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum PrimitiveConvertedType {
    Utf8,
    /// an enum is converted into a binary field
    Enum,
    /// A decimal value.
    ///
    /// This may be used to annotate binary or fixed primitive types. The underlying byte array
    /// stores the unscaled value encoded as two's complement using big-endian byte order (the most
    /// significant byte is the zeroth element). The value of the decimal is the value *
    /// 10^{-scale}.
    ///
    /// This must be accompanied by a (maximum) precision and a scale in the SchemaElement. The
    /// precision specifies the number of digits in the decimal and the scale stores the location
    /// of the decimal point. For example 1.23 would have precision 3 (3 total digits) and scale 2
    /// (the decimal point is 2 digits over).
    // (precision, scale)
    Decimal(usize, usize),
    /// A Date
    ///
    /// Stored as days since Unix epoch, encoded as the INT32 physical type.
    ///
    Date,
    /// A time
    ///
    /// The total number of milliseconds since midnight.  The value is stored as an INT32 physical
    /// type.
    TimeMillis,
    /// A time.
    ///
    /// The total number of microseconds since midnight.  The value is stored as an INT64 physical
    /// type.
    TimeMicros,
    /// A date/time combination
    ///
    /// Date and time recorded as milliseconds since the Unix epoch.  Recorded as a physical type
    /// of INT64.
    TimestampMillis,
    /// A date/time combination
    ///
    /// Date and time recorded as microseconds since the Unix epoch.  The value is stored as an
    /// INT64 physical type.
    TimestampMicros,
    /// An unsigned integer value.
    ///
    /// The number describes the maximum number of meaningful data bits in the stored value. 8, 16
    /// and 32 bit values are stored using the INT32 physical type.  64 bit values are stored using
    /// the INT64 physical type.
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    /// A signed integer value.
    ///
    /// The number describes the maximum number of meainful data bits in the stored value. 8, 16
    /// and 32 bit values are stored using the INT32 physical type.  64 bit values are stored using
    /// the INT64 physical type.
    ///
    Int8,
    Int16,
    Int32,
    Int64,
    /// An embedded JSON document
    ///
    /// A JSON document embedded within a single UTF8 column.
    Json,
    /// An embedded BSON document
    ///
    /// A BSON document embedded within a single BINARY column.
    Bson,
    /// An interval of time
    ///
    /// This type annotates data stored as a FIXED_LEN_BYTE_ARRAY of length 12 This data is
    /// composed of three separate little endian unsigned integers.  Each stores a component of a
    /// duration of time.  The first integer identifies the number of months associated with the
    /// duration, the second identifies the number of days associated with the duration and the
    /// third identifies the number of milliseconds associated with the provided duration.  This
    /// duration of time is independent of any particular timezone or date.
    Interval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum GroupConvertedType {
    /// a map is converted as an optional field containing a repeated key/value pair
    Map,
    /// a key/value pair is converted into a group of two fields
    MapKeyValue,
    /// a list is converted into an optional field containing a repeated field for its values
    List,
}

impl TryFrom<(ConvertedType, Option<(i32, i32)>)> for PrimitiveConvertedType {
    type Error = ParquetError;

    fn try_from(
        (ty, maybe_decimal): (ConvertedType, Option<(i32, i32)>),
    ) -> Result<Self, Self::Error> {
        use PrimitiveConvertedType::*;
        Ok(match ty {
            ConvertedType::UTF8 => Utf8,
            ConvertedType::ENUM => Enum,
            ConvertedType::DECIMAL => {
                if let Some((precision, scale)) = maybe_decimal {
                    Decimal(precision.try_into()?, scale.try_into()?)
                } else {
                    return Err(ParquetError::oos("Decimal requires a precision and scale"));
                }
            },
            ConvertedType::DATE => Date,
            ConvertedType::TIME_MILLIS => TimeMillis,
            ConvertedType::TIME_MICROS => TimeMicros,
            ConvertedType::TIMESTAMP_MILLIS => TimestampMillis,
            ConvertedType::TIMESTAMP_MICROS => TimestampMicros,
            ConvertedType::UINT_8 => Uint8,
            ConvertedType::UINT_16 => Uint16,
            ConvertedType::UINT_32 => Uint32,
            ConvertedType::UINT_64 => Uint64,
            ConvertedType::INT_8 => Int8,
            ConvertedType::INT_16 => Int16,
            ConvertedType::INT_32 => Int32,
            ConvertedType::INT_64 => Int64,
            ConvertedType::JSON => Json,
            ConvertedType::BSON => Bson,
            ConvertedType::INTERVAL => Interval,
            _ => {
                return Err(ParquetError::oos(format!(
                    "Converted type \"{:?}\" cannot be applied to a primitive type",
                    ty
                )))
            },
        })
    }
}

impl TryFrom<ConvertedType> for GroupConvertedType {
    type Error = ParquetError;

    fn try_from(type_: ConvertedType) -> Result<Self, Self::Error> {
        Ok(match type_ {
            ConvertedType::LIST => GroupConvertedType::List,
            ConvertedType::MAP => GroupConvertedType::Map,
            ConvertedType::MAP_KEY_VALUE => GroupConvertedType::MapKeyValue,
            _ => return Err(ParquetError::oos("LogicalType value out of range")),
        })
    }
}

impl From<GroupConvertedType> for ConvertedType {
    fn from(type_: GroupConvertedType) -> Self {
        match type_ {
            GroupConvertedType::Map => ConvertedType::MAP,
            GroupConvertedType::List => ConvertedType::LIST,
            GroupConvertedType::MapKeyValue => ConvertedType::MAP_KEY_VALUE,
        }
    }
}

impl From<PrimitiveConvertedType> for (ConvertedType, Option<(i32, i32)>) {
    fn from(ty: PrimitiveConvertedType) -> Self {
        use PrimitiveConvertedType::*;
        match ty {
            Utf8 => (ConvertedType::UTF8, None),
            Enum => (ConvertedType::ENUM, None),
            Decimal(precision, scale) => (
                ConvertedType::DECIMAL,
                Some((precision as i32, scale as i32)),
            ),
            Date => (ConvertedType::DATE, None),
            TimeMillis => (ConvertedType::TIME_MILLIS, None),
            TimeMicros => (ConvertedType::TIME_MICROS, None),
            TimestampMillis => (ConvertedType::TIMESTAMP_MILLIS, None),
            TimestampMicros => (ConvertedType::TIMESTAMP_MICROS, None),
            Uint8 => (ConvertedType::UINT_8, None),
            Uint16 => (ConvertedType::UINT_16, None),
            Uint32 => (ConvertedType::UINT_32, None),
            Uint64 => (ConvertedType::UINT_64, None),
            Int8 => (ConvertedType::INT_8, None),
            Int16 => (ConvertedType::INT_16, None),
            Int32 => (ConvertedType::INT_32, None),
            Int64 => (ConvertedType::INT_64, None),
            Json => (ConvertedType::JSON, None),
            Bson => (ConvertedType::BSON, None),
            Interval => (ConvertedType::INTERVAL, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() -> Result<(), ParquetError> {
        use PrimitiveConvertedType::*;
        let a = vec![
            Utf8,
            Enum,
            Decimal(3, 1),
            Date,
            TimeMillis,
            TimeMicros,
            TimestampMillis,
            TimestampMicros,
            Uint8,
            Uint16,
            Uint32,
            Uint64,
            Int8,
            Int16,
            Int32,
            Int64,
            Json,
            Bson,
            Interval,
        ];
        for a in a {
            let (c, d): (ConvertedType, Option<(i32, i32)>) = a.into();
            let e: PrimitiveConvertedType = (c, d).try_into()?;
            assert_eq!(e, a);
        }
        Ok(())
    }
}
