//! Decoding of Iceberg values into Arrow arrays: column bounds (single-value binary
//! serialization), partition tuple values, and `initial-default` values (JSON single-value
//! serialization).
//!
//! The rules follow the Python resolver (`polars.io.iceberg._utils`), so that both paths produce
//! the same statistics and defaults.
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars_arrow::array::{
    Array, BinaryViewArray, BooleanArray, PrimitiveArray, Utf8ViewArray, new_null_array,
};
use serde_json::Value as JsonValue;

use crate::iceberg::arrow_types::value_dtype;
use crate::iceberg::avro::Datum;
use crate::iceberg::error::{IcebergResult, err_invalid_data, err_not_implemented};
use crate::iceberg::spec::{PrimitiveType, Type};

const MICROS_TO_NANOS: i64 = 1000;
const UNIX_EPOCH_DAYS_FROM_CE: i32 = 719_163;

/// Whether bounds of a field can be loaded, given all the types it had across schema versions.
/// Mirrors `LoadFromBytesImpl.init_for_field_type`.
pub fn bounds_supported(current: &Type, all_types: &[&Type]) -> bool {
    use PrimitiveType as P;
    let Some(current) = current.as_primitive() else {
        return false;
    };
    let same_kind = |a: &PrimitiveType, b: &PrimitiveType| match (a, b) {
        (P::Decimal { .. }, P::Decimal { .. }) | (P::Fixed(_), P::Fixed(_)) => true,
        (a, b) => a == b,
    };
    let allowed = |t: &PrimitiveType| match current {
        P::Long => matches!(t, P::Long | P::Int),
        c => same_kind(c, t),
    };
    let supported = matches!(
        current,
        P::Boolean
            | P::Date
            | P::Time
            | P::Timestamp
            | P::Timestamptz
            | P::Int
            | P::Long
            | P::String
            | P::Binary
            | P::Decimal { .. }
            | P::Fixed(_)
    );
    supported
        && all_types
            .iter()
            .all(|t| t.as_primitive().is_some_and(allowed))
}

/// Decode bounds of a field whose bounds are supported (see [`bounds_supported`]).
pub fn bounds_array(ty: &Type, values: &[Option<&[u8]>]) -> IcebergResult<Box<dyn Array>> {
    use PrimitiveType as P;
    let dtype = value_dtype(ty);
    let Some(p) = ty.as_primitive() else {
        return Ok(new_null_array(dtype, values.len()));
    };

    let le_i32 = |b: &[u8]| <[u8; 4]>::try_from(b).ok().map(i32::from_le_bytes);
    let le_i64 = |b: &[u8]| <[u8; 8]>::try_from(b).ok().map(i64::from_le_bytes);
    let map = |f: &dyn Fn(&[u8]) -> Option<i64>| -> Vec<Option<i64>> {
        values.iter().map(|v| v.and_then(f)).collect()
    };

    Ok(match p {
        P::Boolean => BooleanArray::from_iter(
            values
                .iter()
                .map(|v| v.and_then(|b| (b.len() == 1).then(|| b[0] != 0))),
        )
        .boxed(),
        P::Int | P::Date => {
            PrimitiveArray::<i32>::from_iter(values.iter().map(|v| v.and_then(le_i32)))
                .to(dtype)
                .boxed()
        },
        // A long field may have been promoted from int, so 4-byte bounds are read as int.
        P::Long => PrimitiveArray::<i64>::from(map(&|b| le_i64(b).or(le_i32(b).map(i64::from))))
            .to(dtype)
            .boxed(),
        P::Time => PrimitiveArray::<i64>::from(map(&|b| le_i64(b).map(|us| us * MICROS_TO_NANOS)))
            .to(dtype)
            .boxed(),
        P::Timestamp | P::Timestamptz => {
            PrimitiveArray::<i64>::from(map(&le_i64)).to(dtype).boxed()
        },
        P::String => Utf8ViewArray::from_slice(
            values
                .iter()
                .map(|v| v.and_then(|b| std::str::from_utf8(b).ok()))
                .collect::<Vec<_>>(),
        )
        .boxed(),
        P::Binary | P::Fixed(_) => BinaryViewArray::from_slice(values).boxed(),
        P::Decimal { precision, .. } => PrimitiveArray::<i128>::from(
            values
                .iter()
                .map(|v| v.map(|b| decimal_from_be_bytes(b, *precision)).transpose())
                .collect::<IcebergResult<Vec<_>>>()?,
        )
        .to(dtype)
        .boxed(),
        _ => new_null_array(dtype, values.len()),
    })
}

/// Iceberg decimal: two's-complement big-endian unscaled value with the minimum number of bytes.
pub fn decimal_from_be_bytes(be: &[u8], precision: u32) -> IcebergResult<i128> {
    if be.len() > 16 {
        return Err(err_invalid_data(format!(
            "binary data for decimal exceeded 16 bytes: {}",
            be.len()
        )));
    }
    let negative = be.first().is_some_and(|b| b & 0x80 != 0);
    let mut le = if negative { [0xFF; 16] } else { [0; 16] };
    for (i, byte) in be.iter().rev().enumerate() {
        le[i] = *byte;
    }
    let value = i128::from_le_bytes(le);
    let max = 10_i128.pow(precision) - 1;
    if value.unsigned_abs() > max.unsigned_abs() {
        return Err(err_invalid_data(format!(
            "decoded value for decimal exceeded precision: value: {value}, precision: {precision}"
        )));
    }
    Ok(value)
}

/// Whether a field's type change across schemas still allows identity partition values to be
/// used. Mirrors `IdentityTransformedPartitionValuesBuilder`.
pub fn partition_type_change_allowed(projected: &Type, other: &Type) -> bool {
    use PrimitiveType as P;
    projected == other
        || matches!(
            (projected, other),
            (Type::Primitive(P::Long), Type::Primitive(P::Int))
                | (
                    Type::Primitive(P::Double | P::Float),
                    Type::Primitive(P::Double | P::Float)
                )
        )
}

/// Build an array from identity partition values of a primitive field.
pub fn partition_values_array(
    ty: &Type,
    values: &[Option<&Datum>],
) -> Result<Box<dyn Array>, String> {
    use PrimitiveType as P;
    let dtype = value_dtype(ty);
    let Some(p) = ty.as_primitive() else {
        return Err(format!("non-primitive type: {ty:?}"));
    };
    let mismatch = |d: &Datum| format!("unexpected partition value {d:?} for type {p:?}");

    let ints = |f: &dyn Fn(&Datum) -> Option<i64>| -> Result<Vec<Option<i64>>, String> {
        values
            .iter()
            .map(|v| v.map(|d| f(d).ok_or_else(|| mismatch(d))).transpose())
            .collect()
    };
    let as_i64 = |d: &Datum| match d {
        Datum::Int(v) => Some(i64::from(*v)),
        Datum::Long(v) => Some(*v),
        _ => None,
    };

    Ok(match p {
        P::Boolean => BooleanArray::from_iter(
            values
                .iter()
                .map(|v| {
                    v.map(|d| match d {
                        Datum::Bool(b) => Ok(*b),
                        d => Err(mismatch(d)),
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .boxed(),
        P::Int | P::Date => PrimitiveArray::<i32>::from(
            ints(&as_i64)?
                .into_iter()
                .map(|v| v.map(|v| v as i32))
                .collect::<Vec<_>>(),
        )
        .to(dtype)
        .boxed(),
        P::Long | P::Timestamp | P::Timestamptz | P::TimestampNs | P::TimestamptzNs => {
            PrimitiveArray::<i64>::from(ints(&as_i64)?)
                .to(dtype)
                .boxed()
        },
        P::Time => PrimitiveArray::<i64>::from(
            ints(&as_i64)?
                .into_iter()
                .map(|v| v.map(|us| us * MICROS_TO_NANOS))
                .collect::<Vec<_>>(),
        )
        .to(dtype)
        .boxed(),
        P::Float | P::Double => {
            let vals = values
                .iter()
                .map(|v| {
                    v.map(|d| match d {
                        Datum::Float(f) => Ok(f64::from(*f)),
                        Datum::Double(f) => Ok(*f),
                        d => Err(mismatch(d)),
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(p, P::Float) {
                PrimitiveArray::<f32>::from(
                    vals.into_iter()
                        .map(|v| v.map(|v| v as f32))
                        .collect::<Vec<_>>(),
                )
                .boxed()
            } else {
                PrimitiveArray::<f64>::from(vals).boxed()
            }
        },
        P::String => Utf8ViewArray::from_slice(
            values
                .iter()
                .map(|v| {
                    v.map(|d| match d {
                        Datum::String(s) => Ok(s.as_str()),
                        d => Err(mismatch(d)),
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .boxed(),
        P::Binary | P::Fixed(_) | P::Uuid => BinaryViewArray::from_slice(
            values
                .iter()
                .map(|v| {
                    v.map(|d| match d {
                        Datum::Bytes(b) => Ok(b.as_slice()),
                        Datum::String(s) => Ok(s.as_bytes()),
                        d => Err(mismatch(d)),
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .boxed(),
        P::Decimal { precision, .. } => PrimitiveArray::<i128>::from(
            values
                .iter()
                .map(|v| {
                    v.map(|d| match d {
                        Datum::Bytes(b) => {
                            decimal_from_be_bytes(b, *precision).map_err(|e| e.message())
                        },
                        Datum::Int(i) => Ok(i128::from(*i)),
                        Datum::Long(i) => Ok(i128::from(*i)),
                        d => Err(mismatch(d)),
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .to(dtype)
        .boxed(),
        P::Unknown => new_null_array(dtype, values.len()),
    })
}

/// A length-1 array holding a field's `initial-default` (JSON single-value serialization).
pub fn initial_default_array(ty: &Type, json: &JsonValue) -> IcebergResult<Box<dyn Array>> {
    use PrimitiveType as P;
    let dtype = value_dtype(ty);
    let Some(p) = ty.as_primitive() else {
        return Err(err_not_implemented(format!(
            "initial-default for nested type {ty:?}"
        )));
    };
    let invalid = || err_invalid_data(format!("invalid initial-default {json} for type {p:?}"));
    let as_str = || json.as_str().ok_or_else(invalid);

    Ok(match p {
        P::Boolean => BooleanArray::from_slice([json.as_bool().ok_or_else(invalid)?]).boxed(),
        P::Int => {
            PrimitiveArray::<i32>::from_slice([json.as_i64().ok_or_else(invalid)? as i32]).boxed()
        },
        P::Long => PrimitiveArray::<i64>::from_slice([json.as_i64().ok_or_else(invalid)?]).boxed(),
        P::Float => {
            PrimitiveArray::<f32>::from_slice([json.as_f64().ok_or_else(invalid)? as f32]).boxed()
        },
        P::Double => {
            PrimitiveArray::<f64>::from_slice([json.as_f64().ok_or_else(invalid)?]).boxed()
        },
        P::Date => {
            let d = NaiveDate::parse_from_str(as_str()?, "%Y-%m-%d").map_err(|_| invalid())?;
            PrimitiveArray::<i32>::from_slice([days_since_epoch(d)])
                .to(dtype)
                .boxed()
        },
        P::Time => {
            let t = NaiveTime::parse_from_str(as_str()?, "%H:%M:%S%.f").map_err(|_| invalid())?;
            let ns = i64::from(t.num_seconds_from_midnight()) * 1_000_000_000
                + i64::from(t.nanosecond());
            PrimitiveArray::<i64>::from_slice([ns]).to(dtype).boxed()
        },
        P::Timestamp | P::TimestampNs => {
            let ts = NaiveDateTime::parse_from_str(as_str()?, "%Y-%m-%dT%H:%M:%S%.f")
                .map_err(|_| invalid())?
                .and_utc();
            let v = if matches!(p, P::Timestamp) {
                ts.timestamp_micros()
            } else {
                ts.timestamp_nanos_opt().ok_or_else(invalid)?
            };
            PrimitiveArray::<i64>::from_slice([v]).to(dtype).boxed()
        },
        P::Timestamptz | P::TimestamptzNs => {
            let s = as_str()?;
            let ts = DateTime::parse_from_rfc3339(s)
                .or_else(|_| DateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f%:z"))
                .map_err(|_| invalid())?;
            let v = if matches!(p, P::Timestamptz) {
                ts.timestamp_micros()
            } else {
                ts.timestamp_nanos_opt().ok_or_else(invalid)?
            };
            PrimitiveArray::<i64>::from_slice([v]).to(dtype).boxed()
        },
        P::String => Utf8ViewArray::from_slice_values([as_str()?]).boxed(),
        P::Uuid => {
            let hex_str: String = as_str()?.chars().filter(|c| *c != '-').collect();
            let bytes = hex::decode(hex_str).map_err(|_| invalid())?;
            if bytes.len() != 16 {
                return Err(invalid());
            }
            BinaryViewArray::from_slice_values([bytes]).boxed()
        },
        P::Binary | P::Fixed(_) => {
            let bytes = hex::decode(as_str()?).map_err(|_| invalid())?;
            BinaryViewArray::from_slice_values([bytes]).boxed()
        },
        P::Decimal { scale, .. } => {
            PrimitiveArray::<i128>::from_slice([
                parse_decimal(as_str()?, *scale).ok_or_else(invalid)?
            ])
            .to(dtype)
            .boxed()
        },
        P::Unknown => new_null_array(dtype, 1),
    })
}

fn days_since_epoch(d: NaiveDate) -> i32 {
    use chrono::Datelike;
    d.num_days_from_ce() - UNIX_EPOCH_DAYS_FROM_CE
}

/// Parse a decimal string such as `-1.50` into an unscaled integer with the given scale.
fn parse_decimal(s: &str, scale: u32) -> Option<i128> {
    let (negative, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
    if frac_part.len() > scale as usize
        || !int_part
            .chars()
            .chain(frac_part.chars())
            .all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let mut value: i128 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().ok()?
    };
    for i in 0..scale as usize {
        let digit = frac_part
            .as_bytes()
            .get(i)
            .map_or(0, |b| i128::from(b - b'0'));
        value = value.checked_mul(10)?.checked_add(digit)?;
    }
    Some(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal() {
        assert_eq!(decimal_from_be_bytes(&[0xFF], 10).unwrap(), -1);
        assert_eq!(decimal_from_be_bytes(&[0x01, 0x00], 10).unwrap(), 256);
        assert_eq!(parse_decimal("1.00", 2), Some(100));
        assert_eq!(parse_decimal("-1.5", 2), Some(-150));
        assert_eq!(parse_decimal("3", 0), Some(3));
        assert_eq!(parse_decimal("1.234", 2), None);
    }

    #[test]
    fn test_days_since_epoch() {
        assert_eq!(
            days_since_epoch(NaiveDate::from_ymd_opt(1970, 1, 2).unwrap()),
            1
        );
    }
}
