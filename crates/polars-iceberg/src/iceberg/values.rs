//! Decoding of Iceberg values into Arrow arrays: column bounds (single-value binary
//! serialization), partition tuple values, and `initial-default` values (JSON single-value
//! serialization).
//!
//! The rules follow the Python resolver (`polars.io.iceberg._utils`), so that both paths produce
//! the same statistics and defaults.
use std::borrow::Cow;

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
/// used. Mirrors `IdentityTransformedPartitionValuesBuilder`: promotions in either direction (a
/// scan of an older snapshot projects the type from before later promotions), and decimal
/// precision changes (unscaled values are unchanged).
pub fn partition_type_change_allowed(projected: &Type, other: &Type) -> bool {
    use PrimitiveType as P;
    projected == other
        || matches!(
            (projected, other),
            (
                Type::Primitive(P::Int | P::Long),
                Type::Primitive(P::Int | P::Long)
            ) | (
                Type::Primitive(P::Double | P::Float),
                Type::Primitive(P::Double | P::Float)
            )
        )
        || matches!(
            (projected, other),
            (
                Type::Primitive(P::Decimal { scale: s1, .. }),
                Type::Primitive(P::Decimal { scale: s2, .. }),
            ) if s1 == s2
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
                .map(|v| {
                    v.map(|v| i32::try_from(v).map_err(|_| format!("{v} out of range for {p:?}")))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?,
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
                        Datum::Bytes(b) => Ok(Cow::Borrowed(b.as_slice())),
                        // A UUID in an Avro `string` is in its text form.
                        Datum::String(s) if matches!(p, P::Uuid) => {
                            parse_uuid(s).map(Cow::Owned).ok_or_else(|| mismatch(d))
                        },
                        Datum::String(s) => Ok(Cow::Borrowed(s.as_bytes())),
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
            let v = json.as_i64().ok_or_else(invalid)?;
            PrimitiveArray::<i32>::from_slice([i32::try_from(v).map_err(|_| invalid())?]).boxed()
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
            // ISO-8601 local time, whose seconds are optional (as in Java's `LocalTime.parse`).
            let s = as_str()?;
            let t = NaiveTime::parse_from_str(s, "%H:%M:%S%.f")
                .or_else(|_| NaiveTime::parse_from_str(s, "%H:%M"))
                .map_err(|_| invalid())?;
            let ns = i64::from(t.num_seconds_from_midnight()) * 1_000_000_000
                + i64::from(t.nanosecond());
            PrimitiveArray::<i64>::from_slice([ns]).to(dtype).boxed()
        },
        P::Timestamp | P::TimestampNs => {
            let s = as_str()?;
            let ts = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M"))
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
                .or_else(|_| DateTime::parse_from_str(s, "%Y-%m-%dT%H:%M%:z"))
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
            BinaryViewArray::from_slice_values([parse_uuid(as_str()?).ok_or_else(invalid)?]).boxed()
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

/// The 16 bytes of a UUID in its text form (hyphens optional).
pub fn parse_uuid(s: &str) -> Option<Vec<u8>> {
    let hex_str: String = s.chars().filter(|c| *c != '-').collect();
    let bytes = hex::decode(hex_str).ok()?;
    (bytes.len() == 16).then_some(bytes)
}

fn days_since_epoch(d: NaiveDate) -> i32 {
    use chrono::Datelike;
    d.num_days_from_ce() - UNIX_EPOCH_DAYS_FROM_CE
}

/// Parse a decimal string such as `-1.50` or `1E-7` (as written by Python's `str(Decimal)`) into
/// an unscaled integer with the given scale; `None` if it is not exactly representable at `scale`.
pub fn parse_decimal(s: &str, scale: u32) -> Option<i128> {
    let (mantissa, exponent) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i32>().ok()?),
        None => (s, 0),
    };
    let (negative, unsigned) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
    };
    let (int_part, frac_part) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if int_part.is_empty() && frac_part.is_empty()
        || !int_part
            .chars()
            .chain(frac_part.chars())
            .all(|c| c.is_ascii_digit())
    {
        return None;
    }

    // value = digits * 10^shift, at the target scale.
    let digits = format!("{int_part}{frac_part}");
    let shift = i64::from(scale) + i64::from(exponent) - frac_part.len() as i64;
    let digits = digits.trim_start_matches('0');
    let digits = if shift < 0 {
        // Dropped digits must be zeros.
        let drop = usize::try_from(-shift).ok()?;
        let keep = digits.len().saturating_sub(drop);
        if !digits[keep..].bytes().all(|b| b == b'0') {
            return None;
        }
        &digits[..keep]
    } else {
        digits
    };
    if digits.is_empty() {
        return Some(0);
    }
    let value: i128 = digits.parse().ok()?;
    let value = value.checked_mul(10_i128.checked_pow(u32::try_from(shift.max(0)).ok()?)?)?;
    Some(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_encoded_uuid_partition_values_are_decoded() {
        let bytes: Vec<u8> = (0..16).collect();
        let text = Datum::String("00010203-0405-0607-0809-0a0b0c0d0e0f".into());
        let raw = Datum::Bytes(bytes.clone());
        let array = partition_values_array(
            &Type::Primitive(PrimitiveType::Uuid),
            &[Some(&text), Some(&raw), None],
        )
        .unwrap();
        let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
        assert_eq!(array.value(0), bytes.as_slice());
        assert_eq!(array.value(1), bytes.as_slice());
        assert!(array.is_null(2));

        let invalid = Datum::String("not-a-uuid".into());
        assert!(
            partition_values_array(&Type::Primitive(PrimitiveType::Uuid), &[Some(&invalid)])
                .is_err()
        );
    }

    #[test]
    fn out_of_range_int_initial_default_is_rejected() {
        let int = Type::Primitive(PrimitiveType::Int);
        assert!(initial_default_array(&int, &serde_json::json!(i32::MAX)).is_ok());
        assert!(initial_default_array(&int, &serde_json::json!(1_i64 << 32)).is_err());
    }

    #[test]
    fn temporal_initial_defaults_without_seconds() {
        let value = |p: PrimitiveType, s: &str| {
            let array = initial_default_array(&Type::Primitive(p), &serde_json::json!(s)).unwrap();
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap()
                .value(0)
        };

        let ten_hours_us = 10 * 3600 * 1_000_000;
        assert_eq!(value(PrimitiveType::Time, "10:00"), ten_hours_us * 1000);
        assert_eq!(value(PrimitiveType::Time, "10:00:00"), ten_hours_us * 1000);
        assert_eq!(
            value(PrimitiveType::Timestamp, "1970-01-01T10:00"),
            ten_hours_us
        );
        assert_eq!(
            value(PrimitiveType::Timestamptz, "1970-01-01T11:00+01:00"),
            ten_hours_us
        );
    }

    #[test]
    fn parse_decimal_strings() {
        assert_eq!(parse_decimal("-1.50", 2), Some(-150));
        assert_eq!(parse_decimal("1.500", 2), Some(150));
        assert_eq!(parse_decimal("+3", 1), Some(30));
        assert_eq!(parse_decimal(".5", 1), Some(5));
        assert_eq!(parse_decimal("1.555", 2), None);
        assert_eq!(parse_decimal("", 2), None);
        assert_eq!(parse_decimal("1.x", 2), None);

        // Exponent notation, as written by Python's `str(Decimal)`.
        assert_eq!(parse_decimal("1E-7", 7), Some(1));
        assert_eq!(parse_decimal("0E-7", 7), Some(0));
        assert_eq!(parse_decimal("-2.5E-6", 7), Some(-25));
        assert_eq!(parse_decimal("1.5e+2", 0), Some(150));
        assert_eq!(parse_decimal("1E+2", 2), Some(10000));
        assert_eq!(parse_decimal("1E-8", 7), None);
        assert_eq!(parse_decimal("1E", 7), None);
        assert_eq!(parse_decimal("1E99", 2), None);
    }
}
