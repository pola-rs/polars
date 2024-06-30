// see https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
use super::{IntegerType, PhysicalType, PrimitiveConvertedType, PrimitiveLogicalType, TimeUnit};
use crate::parquet::error::{ParquetError, ParquetResult};

fn check_decimal_invariants(
    physical_type: &PhysicalType,
    precision: usize,
    scale: usize,
) -> ParquetResult<()> {
    if precision < 1 {
        return Err(ParquetError::oos(format!(
            "DECIMAL precision must be larger than 0; It is {}",
            precision,
        )));
    }
    if scale > precision {
        return Err(ParquetError::oos(format!(
            "Invalid DECIMAL: scale ({}) cannot be greater than precision \
            ({})",
            scale, precision
        )));
    }

    match physical_type {
        PhysicalType::Int32 => {
            if !(1..=9).contains(&precision) {
                return Err(ParquetError::oos(format!(
                    "Cannot represent INT32 as DECIMAL with precision {}",
                    precision
                )));
            }
        },
        PhysicalType::Int64 => {
            if !(1..=18).contains(&precision) {
                return Err(ParquetError::oos(format!(
                    "Cannot represent INT64 as DECIMAL with precision {}",
                    precision
                )));
            }
        },
        PhysicalType::FixedLenByteArray(length) => {
            let oos_error =
                || ParquetError::oos(format!("Byte Array length {} out of spec", length));
            let max_precision = (2f64.powi(
                (*length as i32)
                    .checked_mul(8)
                    .ok_or_else(oos_error)?
                    .checked_sub(1)
                    .ok_or_else(oos_error)?,
            ) - 1f64)
                .log10()
                .floor() as usize;

            if precision > max_precision {
                return Err(ParquetError::oos(format!(
                    "Cannot represent FIXED_LEN_BYTE_ARRAY as DECIMAL with length {} and \
                    precision {}. The max precision can only be {}",
                    length, precision, max_precision
                )));
            }
        },
        PhysicalType::ByteArray => {},
        _ => {
            return Err(ParquetError::oos(
                "DECIMAL can only annotate INT32, INT64, BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY",
            ))
        },
    };
    Ok(())
}

pub fn check_converted_invariants(
    physical_type: &PhysicalType,
    converted_type: &Option<PrimitiveConvertedType>,
) -> ParquetResult<()> {
    if converted_type.is_none() {
        return Ok(());
    };
    let converted_type = converted_type.as_ref().unwrap();

    use PrimitiveConvertedType::*;
    match converted_type {
        Utf8 | Bson | Json => {
            if physical_type != &PhysicalType::ByteArray {
                return Err(ParquetError::oos(format!(
                    "{:?} can only annotate BYTE_ARRAY fields",
                    converted_type
                )));
            }
        },
        Decimal(precision, scale) => {
            check_decimal_invariants(physical_type, *precision, *scale)?;
        },
        Date | TimeMillis | Uint8 | Uint16 | Uint32 | Int8 | Int16 | Int32 => {
            if physical_type != &PhysicalType::Int32 {
                return Err(ParquetError::oos(format!(
                    "{:?} can only annotate INT32",
                    converted_type
                )));
            }
        },
        TimeMicros | TimestampMillis | TimestampMicros | Uint64 | Int64 => {
            if physical_type != &PhysicalType::Int64 {
                return Err(ParquetError::oos(format!(
                    "{:?} can only annotate INT64",
                    converted_type
                )));
            }
        },
        Interval => {
            if physical_type != &PhysicalType::FixedLenByteArray(12) {
                return Err(ParquetError::oos(
                    "INTERVAL can only annotate FIXED_LEN_BYTE_ARRAY(12)",
                ));
            }
        },
        Enum => {
            if physical_type != &PhysicalType::ByteArray {
                return Err(ParquetError::oos(
                    "ENUM can only annotate BYTE_ARRAY fields",
                ));
            }
        },
    };
    Ok(())
}

pub fn check_logical_invariants(
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
) -> ParquetResult<()> {
    if logical_type.is_none() {
        return Ok(());
    };
    let logical_type = logical_type.unwrap();

    // Check that logical type and physical type are compatible
    use PrimitiveLogicalType::*;
    match (logical_type, physical_type) {
        (Enum, PhysicalType::ByteArray) => {},
        (Decimal(precision, scale), _) => {
            check_decimal_invariants(physical_type, precision, scale)?;
        },
        (Date, PhysicalType::Int32) => {},
        (
            Time {
                unit: TimeUnit::Milliseconds,
                ..
            },
            PhysicalType::Int32,
        ) => {},
        (Time { unit, .. }, PhysicalType::Int64) => {
            if unit == TimeUnit::Milliseconds {
                return Err(ParquetError::oos(
                    "Cannot use millisecond unit on INT64 type",
                ));
            }
        },
        (Timestamp { .. }, PhysicalType::Int64) => {},
        (Integer(IntegerType::Int8), PhysicalType::Int32) => {},
        (Integer(IntegerType::Int16), PhysicalType::Int32) => {},
        (Integer(IntegerType::Int32), PhysicalType::Int32) => {},
        (Integer(IntegerType::UInt8), PhysicalType::Int32) => {},
        (Integer(IntegerType::UInt16), PhysicalType::Int32) => {},
        (Integer(IntegerType::UInt32), PhysicalType::Int32) => {},
        (Integer(IntegerType::UInt64), PhysicalType::Int64) => {},
        (Integer(IntegerType::Int64), PhysicalType::Int64) => {},
        // Null type
        (Unknown, PhysicalType::Int32) => {},
        (String | Json | Bson, PhysicalType::ByteArray) => {},
        // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#uuid
        (Uuid, PhysicalType::FixedLenByteArray(16)) => {},
        (a, b) => {
            return Err(ParquetError::oos(format!(
                "Cannot annotate {:?} from {:?} fields",
                a, b
            )))
        },
    };
    Ok(())
}
