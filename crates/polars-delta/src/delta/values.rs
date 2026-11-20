//! Typed scalar values for statistics and partition columns, and the builders that turn them
//! into Arrow arrays. Numbers keep their exact JSON text until the target type is known.
use std::borrow::Cow;

use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType as D, Field, TimeUnit};

use super::error::{Result, invalid};
use super::schema::Column;

/// A statistics or partition value. Borrows from the log text where it can.
#[derive(Debug, Clone, PartialEq)]
pub enum Sv<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    /// Unscaled value and scale (checkpoint decimals).
    Decimal(i128, usize),
    /// A JSON number as written: integers, fractions and exponents alike.
    Num(Cow<'a, str>),
    Str(Cow<'a, str>),
    Obj(Vec<(Cow<'a, str>, Sv<'a>)>),
}
static NULL: Sv<'static> = Sv::Null;
impl<'a> Sv<'a> {
    pub fn is_null(&self) -> bool {
        matches!(self, Sv::Null)
    }
    /// A child of an object; `Null` when absent or not an object.
    pub fn get(&self, key: &str) -> &Sv<'a> {
        match self {
            Sv::Obj(entries) => entries
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map_or(&NULL, |(_, v)| v),
            _ => &NULL,
        }
    }
    /// The value as text; numbers and booleans as written.
    fn text(&self) -> Option<Cow<'_, str>> {
        Some(match self {
            Sv::Null => return None,
            Sv::Bool(b) => Cow::Borrowed(if *b { "true" } else { "false" }),
            Sv::Int(v) => Cow::Owned(v.to_string()),
            Sv::UInt(v) => Cow::Owned(v.to_string()),
            Sv::Float(v) => Cow::Owned(v.to_string()),
            Sv::Decimal(v, scale) => Cow::Owned(format_decimal(*v, *scale)),
            Sv::Num(s) | Sv::Str(s) => Cow::Borrowed(s),
            Sv::Obj(_) => Cow::Borrowed("{}"),
        })
    }
    /// The value as an integer: integers directly, others by parsing their text.
    fn as_i64(&self) -> Result<Option<i64>> {
        match self {
            Sv::Null => Ok(None),
            Sv::Int(v) => Ok(Some(*v)),
            Sv::UInt(v) => i64::try_from(*v).map(Some).map_err(invalid),
            _ => parse_text(self),
        }
    }
    fn as_u64(&self) -> Result<Option<u64>> {
        match self {
            Sv::Null => Ok(None),
            Sv::UInt(v) => Ok(Some(*v)),
            Sv::Int(v) => u64::try_from(*v).map(Some).map_err(invalid),
            _ => parse_text(self),
        }
    }
}
fn parse_text<T: std::str::FromStr>(v: &Sv) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    v.text()
        .map(|s| s.parse::<T>().map_err(invalid))
        .transpose()
}
pub fn format_decimal(v: i128, scale: usize) -> String {
    if scale == 0 {
        return v.to_string();
    }
    let digits = format!("{:0>width$}", v.unsigned_abs(), width = scale + 1);
    let split = digits.len() - scale;
    format!(
        "{}{}.{}",
        if v < 0 { "-" } else { "" },
        &digits[..split],
        &digits[split..]
    )
}
pub fn decimal(s: &str, scale: usize, precision: usize) -> Result<i128> {
    let (s, exponent) = match s.split_once(['e', 'E']) {
        Some((s, e)) => (s, e.parse::<i32>().map_err(invalid)?),
        None => (s, 0),
    };
    let negative = s.starts_with('-');
    let s = s.trim_start_matches(['-', '+']);
    let (whole, frac) = s.split_once('.').unwrap_or((s, ""));
    if !whole
        .bytes()
        .chain(frac.bytes())
        .all(|b| b.is_ascii_digit())
    {
        return Err(invalid("invalid decimal digits"));
    }
    let digits = format!("{whole}{frac}");
    let mut v = digits.parse::<i128>().map_err(invalid)?;
    let adjustment = (scale as i64) + (exponent as i64) - (frac.len() as i64);
    if adjustment >= 0 {
        let power = u32::try_from(adjustment).map_err(invalid)?;
        v = v
            .checked_mul(
                10i128
                    .checked_pow(power)
                    .ok_or_else(|| invalid("decimal scale overflow"))?,
            )
            .ok_or_else(|| invalid("decimal overflow"))?;
    } else {
        let power = u32::try_from(-adjustment).map_err(invalid)?;
        let divisor = 10i128
            .checked_pow(power)
            .ok_or_else(|| invalid("decimal scale overflow"))?;
        if v % divisor != 0 {
            return Err(invalid("inexact decimal value"));
        }
        v /= divisor;
    }
    if v.unsigned_abs() >= 10u128.pow(precision as u32) {
        return Err(invalid("decimal precision exceeded"));
    }
    Ok(if negative { -v } else { v })
}
pub fn timestamp(s: &str, utc: bool) -> Result<i64> {
    if utc {
        if let Ok(v) = chrono::DateTime::parse_from_rfc3339(s) {
            return Ok(v.timestamp_micros());
        }
    }
    let v = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f"))
        .map_err(invalid)?;
    Ok(v.and_utc().timestamp_micros())
}
/// `value` (unscaled, at `from` scale) as an unscaled decimal of `scale` and `precision`.
fn rescale(value: i128, from: usize, scale: usize, precision: usize) -> Result<i128> {
    let v = if scale >= from {
        let power = u32::try_from(scale - from).map_err(invalid)?;
        value
            .checked_mul(
                10i128
                    .checked_pow(power)
                    .ok_or_else(|| invalid("decimal scale overflow"))?,
            )
            .ok_or_else(|| invalid("decimal overflow"))?
    } else {
        let power = u32::try_from(from - scale).map_err(invalid)?;
        let divisor = 10i128
            .checked_pow(power)
            .ok_or_else(|| invalid("decimal scale overflow"))?;
        if value % divisor != 0 {
            return Err(invalid("inexact decimal value"));
        }
        value / divisor
    };
    if v.unsigned_abs() >= 10u128.pow(precision as u32) {
        return Err(invalid("decimal precision exceeded"));
    }
    Ok(v)
}
/// Accumulates values of one Arrow type. A value that cannot be converted is an error and is
/// not pushed; [`Builder::push_or_null`] is for optional data.
pub struct Builder {
    dtype: D,
    inner: Inner,
    len: usize,
}
enum Inner {
    Null,
    I8(Vec<Option<i8>>),
    I16(Vec<Option<i16>>),
    I32(Vec<Option<i32>>),
    I64(Vec<Option<i64>>),
    U64(Vec<Option<u64>>),
    F32(Vec<Option<f32>>),
    F64(Vec<Option<f64>>),
    Bool(MutableBooleanArray),
    Utf8(MutableBinaryViewArray<str>),
    Binary(MutableBinaryViewArray<[u8]>),
    Decimal(Vec<Option<i128>>, usize, usize),
    Date(Vec<Option<i32>>),
    Timestamp(Vec<Option<i64>>, bool),
    Struct(Vec<Builder>, Vec<bool>),
}
impl Builder {
    pub fn new(dtype: &D, capacity: usize) -> Self {
        fn v<T>(capacity: usize) -> Vec<Option<T>> {
            Vec::with_capacity(capacity)
        }
        let inner = match dtype {
            D::Int8 => Inner::I8(v(capacity)),
            D::Int16 => Inner::I16(v(capacity)),
            D::Int32 => Inner::I32(v(capacity)),
            D::Int64 => Inner::I64(v(capacity)),
            D::UInt64 => Inner::U64(v(capacity)),
            D::Float32 => Inner::F32(v(capacity)),
            D::Float64 => Inner::F64(v(capacity)),
            D::Boolean => Inner::Bool(MutableBooleanArray::with_capacity(capacity)),
            D::Utf8View => Inner::Utf8(MutableBinaryViewArray::with_capacity(capacity)),
            D::BinaryView => Inner::Binary(MutableBinaryViewArray::with_capacity(capacity)),
            D::Decimal(p, s) => Inner::Decimal(v(capacity), *s, *p),
            D::Date32 => Inner::Date(v(capacity)),
            D::Timestamp(TimeUnit::Microsecond, tz) => Inner::Timestamp(v(capacity), tz.is_some()),
            D::Struct(fields) => Inner::Struct(
                fields
                    .iter()
                    .map(|f| Builder::new(&f.dtype, capacity))
                    .collect(),
                Vec::with_capacity(capacity),
            ),
            // Bounds on lists, maps, variants and other types are unavailable.
            _ => Inner::Null,
        };
        Self {
            dtype: dtype.clone(),
            inner,
            len: 0,
        }
    }
    pub fn push_null(&mut self) {
        match &mut self.inner {
            Inner::Null => {},
            Inner::I8(v) => v.push(None),
            Inner::I16(v) => v.push(None),
            Inner::I32(v) | Inner::Date(v) => v.push(None),
            Inner::I64(v) | Inner::Timestamp(v, _) => v.push(None),
            Inner::U64(v) => v.push(None),
            Inner::F32(v) => v.push(None),
            Inner::F64(v) => v.push(None),
            Inner::Bool(b) => b.push_null(),
            Inner::Utf8(b) => b.push_null(),
            Inner::Binary(b) => b.push_null(),
            Inner::Decimal(v, ..) => v.push(None),
            Inner::Struct(children, validity) => {
                children.iter_mut().for_each(Builder::push_null);
                validity.push(false);
            },
        }
        self.len += 1;
    }
    /// Push `value`, or null if it cannot be converted.
    pub fn push_or_null(&mut self, value: &Sv) {
        if self.push(value).is_err() {
            self.push_null();
        }
    }
    pub fn push(&mut self, value: &Sv) -> Result<()> {
        if value.is_null() {
            self.push_null();
            return Ok(());
        }
        // Convert before pushing, so that an error leaves the builder unchanged.
        match &mut self.inner {
            Inner::Null => {},
            Inner::I8(v) => v.push(
                value
                    .as_i64()?
                    .map(i8::try_from)
                    .transpose()
                    .map_err(invalid)?,
            ),
            Inner::I16(v) => v.push(
                value
                    .as_i64()?
                    .map(i16::try_from)
                    .transpose()
                    .map_err(invalid)?,
            ),
            Inner::I32(v) => v.push(
                value
                    .as_i64()?
                    .map(i32::try_from)
                    .transpose()
                    .map_err(invalid)?,
            ),
            Inner::I64(v) => v.push(value.as_i64()?),
            Inner::U64(v) => v.push(value.as_u64()?),
            Inner::F32(v) => v.push(match value {
                Sv::Float(f) => Some(*f as f32),
                Sv::Int(i) => Some(*i as f32),
                _ => parse_text::<f32>(value)?,
            }),
            Inner::F64(v) => v.push(match value {
                Sv::Float(f) => Some(*f),
                Sv::Int(i) => Some(*i as f64),
                _ => parse_text::<f64>(value)?,
            }),
            Inner::Bool(b) => b.push(match value {
                Sv::Bool(v) => Some(*v),
                _ => parse_text::<bool>(value)?,
            }),
            Inner::Utf8(b) => b.push(value.text().as_deref()),
            Inner::Binary(b) => {
                let bytes = value
                    .text()
                    .map(|s| {
                        s.chars()
                            .map(|c| u8::try_from(c as u32).map_err(invalid))
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()?;
                b.push(bytes.as_deref())
            },
            Inner::Decimal(v, scale, precision) => v.push(match value {
                Sv::Decimal(d, from) => Some(rescale(*d, *from, *scale, *precision)?),
                Sv::Int(i) => Some(rescale(*i as i128, 0, *scale, *precision)?),
                _ => value
                    .text()
                    .map(|s| decimal(&s, *scale, *precision))
                    .transpose()?,
            }),
            Inner::Date(v) => v.push(match value {
                Sv::Int(i) => Some(i32::try_from(*i).map_err(invalid)?),
                Sv::Str(s) => {
                    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let d = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(invalid)?;
                    Some(
                        i32::try_from(d.signed_duration_since(epoch).num_days())
                            .map_err(invalid)?,
                    )
                },
                Sv::Num(_) | Sv::UInt(_) => value
                    .as_i64()?
                    .map(|i| i32::try_from(i).map_err(invalid))
                    .transpose()?,
                _ => return Err(invalid("invalid date")),
            }),
            Inner::Timestamp(v, utc) => v.push(match value {
                Sv::Int(i) => Some(*i),
                Sv::Str(s) => Some(timestamp(s, *utc)?),
                Sv::Num(_) | Sv::UInt(_) => value.as_i64()?,
                _ => return Err(invalid("invalid timestamp")),
            }),
            Inner::Struct(children, validity) => {
                let Sv::Obj(_) = value else {
                    return Err(invalid("statistics value is not a struct"));
                };
                let D::Struct(fields) = &self.dtype else {
                    unreachable!()
                };
                // Children of one row either all convert or the row is null.
                for (child, field) in children.iter_mut().zip(fields) {
                    child.push_or_null(value.get(field.name.as_str()));
                }
                validity.push(true);
            },
        }
        self.len += 1;
        Ok(())
    }
    pub fn finish(self) -> Box<dyn Array> {
        let dtype = self.dtype;
        match self.inner {
            Inner::Null => new_null_array(dtype, self.len),
            Inner::I8(v) => PrimitiveArray::from(v).boxed(),
            Inner::I16(v) => PrimitiveArray::from(v).boxed(),
            Inner::I32(v) => PrimitiveArray::from(v).boxed(),
            Inner::I64(v) => PrimitiveArray::from(v).boxed(),
            Inner::U64(v) => PrimitiveArray::from(v).boxed(),
            Inner::F32(v) => PrimitiveArray::from(v).boxed(),
            Inner::F64(v) => PrimitiveArray::from(v).boxed(),
            Inner::Bool(b) => BooleanArray::from(b).boxed(),
            Inner::Utf8(b) => b.freeze().boxed(),
            Inner::Binary(b) => b.freeze().boxed(),
            Inner::Decimal(v, ..) => PrimitiveArray::from(v).to(dtype).boxed(),
            Inner::Date(v) => PrimitiveArray::from(v).to(dtype).boxed(),
            Inner::Timestamp(v, _) => PrimitiveArray::from(v).to(dtype).boxed(),
            Inner::Struct(children, validity) => {
                let arrays = children.into_iter().map(Builder::finish).collect();
                let all_valid = validity.iter().all(|v| *v);
                StructArray::try_new(
                    dtype,
                    self.len,
                    arrays,
                    (!all_valid).then(|| Bitmap::from_iter(validity)),
                )
                .expect("struct builder children have the builder's length")
                .boxed()
            },
        }
    }
}
/// An array of `dtype` from `values`; a value that does not convert is an error.
#[cfg(test)]
pub fn array<'a>(dtype: &D, values: impl IntoIterator<Item = Sv<'a>>) -> Result<Box<dyn Array>> {
    let values = values.into_iter();
    let mut builder = Builder::new(dtype, values.size_hint().0);
    for v in values {
        builder.push(&v)?;
    }
    Ok(builder.finish())
}
pub fn stat_dtype(column: &Column, null_count: bool) -> D {
    if !column.children.is_empty() && matches!(column.field.dtype, D::Struct(_)) {
        D::Struct(
            column
                .children
                .iter()
                .map(|c| Field::new(c.field.name.clone(), stat_dtype(c, null_count), true))
                .collect(),
        )
    } else if null_count {
        D::UInt64
    } else {
        column.field.dtype.clone()
    }
}
/// The statistic of `column` given its own value in the file's statistics (by physical name), as
/// a value for [`stat_dtype`]: struct columns are rebuilt under their logical child names.
pub fn stat_value<'a>(column: &Column, value: &Sv<'a>, null_count: bool, max: bool) -> Sv<'a> {
    if !column.children.is_empty() && matches!(column.field.dtype, D::Struct(_)) {
        return Sv::Obj(
            column
                .children
                .iter()
                .map(|c| {
                    (
                        Cow::Owned(c.field.name.to_string()),
                        stat_value(c, value.get(&c.physical), null_count, max),
                    )
                })
                .collect(),
        );
    }
    if max && !null_count && matches!(column.field.dtype, D::Timestamp(..)) {
        // Delta truncates timestamp maxima to milliseconds.
        let micros = match value {
            Sv::Int(v) => Some(*v),
            Sv::Num(_) | Sv::UInt(_) => value.as_i64().ok().flatten(),
            Sv::Str(s) => {
                timestamp(s, matches!(&column.field.dtype, D::Timestamp(_, Some(_)))).ok()
            },
            _ => None,
        };
        return micros
            .and_then(|v| v.checked_add(1000))
            .map_or(Sv::Null, Sv::Int);
    }
    value.clone()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimals() {
        assert_eq!(decimal("-1.230", 2, 5).unwrap(), -123);
        assert_eq!(decimal("1.2e2", 2, 5).unwrap(), 12000);
        assert!(decimal("1.001", 2, 5).is_err());
        assert!(decimal("1000", 0, 3).is_err())
    }
    #[test]
    fn typed_values_convert_without_text() {
        let dtype = D::Decimal(10, 2);
        let a = array(&dtype, [Sv::Decimal(12345, 3), Sv::Int(7), Sv::Null]);
        // 12.345 does not fit scale 2.
        assert!(a.is_err());
        let a = array(&dtype, [Sv::Decimal(12340, 3), Sv::Int(7), Sv::Null]).unwrap();
        let a = a.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();
        assert_eq!(
            a.iter().collect::<Vec<_>>(),
            [Some(&1234), Some(&700), None]
        );
        let a = array(
            &D::Int32,
            [Sv::Int(5), Sv::Num("6".into()), Sv::Str("7".into())],
        )
        .unwrap();
        let a = a.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();
        assert_eq!(a.iter().collect::<Vec<_>>(), [Some(&5), Some(&6), Some(&7)]);
        assert!(array(&D::Int8, [Sv::Int(300)]).is_err());
    }
    #[test]
    fn push_or_null_keeps_builder_aligned() {
        let mut b = Builder::new(&D::Int64, 3);
        b.push_or_null(&Sv::Int(1));
        b.push_or_null(&Sv::Str("x".into()));
        b.push_or_null(&Sv::Num("3".into()));
        let a = b.finish();
        let a = a.as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
        assert_eq!(a.iter().collect::<Vec<_>>(), [Some(&1), None, Some(&3)]);
    }
}
