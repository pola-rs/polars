use std::hash::{Hash, Hasher};

use arrow::types::NativeType;
use num_traits::NumCast;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::prelude::string::Pattern;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::prelude::string::infer::{DatetimeInfer, TryFromWithUnit, infer_pattern_single};
use polars_utils::format_pl_smallstr;
use simd_json::{BorrowedValue as Value, KnownKey, StaticNode};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BufferKey<'a>(pub(crate) KnownKey<'a>);
impl Eq for BufferKey<'_> {}

impl Hash for BufferKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.key().hash(state)
    }
}

pub(crate) struct Buffer<'a> {
    name: &'a str,
    ignore_errors: bool,
    buf: AnyValueBuffer<'a>,
}

impl Buffer<'_> {
    pub fn into_series(self) -> PolarsResult<Series> {
        let mut buf = self.buf;
        let mut s = buf.reset(0, !self.ignore_errors)?;
        s.rename(PlSmallStr::from_str(self.name));
        Ok(s)
    }

    #[inline]
    pub(crate) fn add(&mut self, value: &Value) -> PolarsResult<()> {
        use AnyValueBuffer::*;
        match &mut self.buf {
            Boolean(buf) => {
                match value {
                    Value::Static(StaticNode::Bool(b)) => buf.append_value(*b),
                    Value::Static(StaticNode::Null) => buf.append_null(),
                    _ if self.ignore_errors => buf.append_null(),
                    v => polars_bail!(ComputeError: "cannot parse '{}' as Boolean", v),
                }
                Ok(())
            },
            Int32(buf) => {
                let n = deserialize_number::<i32>(value, self.ignore_errors)?;
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            },
            Int64(buf) => {
                let n = deserialize_number::<i64>(value, self.ignore_errors)?;
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            },
            UInt64(buf) => {
                let n = deserialize_number::<u64>(value, self.ignore_errors)?;
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            },
            UInt32(buf) => {
                let n = deserialize_number::<u32>(value, self.ignore_errors)?;
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            },
            Float32(buf) => {
                let n = deserialize_number::<f32>(value, self.ignore_errors)?;
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            },
            Float64(buf) => {
                let n = deserialize_number::<f64>(value, self.ignore_errors)?;
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            },

            String(buf) => {
                match value {
                    Value::String(v) => buf.append_value(v),
                    Value::Static(StaticNode::Null) => buf.append_null(),
                    // Forcibly convert to String using the Display impl.
                    v => buf.append_value(format_pl_smallstr!("{}", ValueDisplay(v))),
                }
                Ok(())
            },
            #[cfg(feature = "dtype-datetime")]
            Datetime(buf, tu, _) => {
                let v =
                    deserialize_datetime::<Int64Type>(value, "Datetime", self.ignore_errors, *tu)?;
                buf.append_option(v);
                Ok(())
            },
            #[cfg(feature = "dtype-date")]
            Date(buf) => {
                let v = deserialize_datetime::<Int32Type>(
                    value,
                    "Date",
                    self.ignore_errors,
                    TimeUnit::Microseconds, // ignored
                )?;
                buf.append_option(v);
                Ok(())
            },
            All(dtype, buf) => {
                let av = deserialize_all(value, dtype, self.ignore_errors)?;
                buf.push(av);
                Ok(())
            },
            Null(builder) => {
                if !(matches!(value, Value::Static(StaticNode::Null)) || self.ignore_errors) {
                    polars_bail!(ComputeError: "got non-null value for NULL-typed column: {}", value)
                };

                builder.append_null();
                Ok(())
            },
            _ => panic!("unexpected dtype when deserializing ndjson"),
        }
    }

    pub fn add_null(&mut self) {
        self.buf.add(AnyValue::Null).expect("should not fail");
    }
}
pub(crate) fn init_buffers(
    schema: &Schema,
    capacity: usize,
    ignore_errors: bool,
) -> PolarsResult<PlIndexMap<BufferKey, Buffer>> {
    schema
        .iter()
        .map(|(name, dtype)| {
            let av_buf = (dtype, capacity).into();
            let key = KnownKey::from(name.as_str());
            Ok((
                BufferKey(key),
                Buffer {
                    name,
                    buf: av_buf,
                    ignore_errors,
                },
            ))
        })
        .collect()
}

fn deserialize_number<T: NativeType + NumCast>(
    value: &Value,
    ignore_errors: bool,
) -> PolarsResult<Option<T>> {
    let to_result = |x: Option<T>| {
        let out = if ignore_errors {
            x
        } else {
            Some(x.ok_or_else(|| {
                polars_err!(ComputeError: "cannot parse '{}' as {:?}", value, T::PRIMITIVE
                )
            })?)
        };

        Ok(out)
    };

    match value {
        Value::Static(StaticNode::F64(f)) => to_result(num_traits::cast(*f)),
        Value::Static(StaticNode::I64(i)) => to_result(num_traits::cast(*i)),
        Value::Static(StaticNode::U64(u)) => to_result(num_traits::cast(*u)),
        Value::Static(StaticNode::Bool(b)) => to_result(num_traits::cast(*b as i32)),
        Value::Static(StaticNode::Null) => Ok(None),
        _ => to_result(None),
    }
}

#[cfg(feature = "dtype-datetime")]
fn deserialize_datetime<T>(
    value: &Value,
    type_name: &str,
    ignore_errors: bool,
    tu: TimeUnit,
) -> PolarsResult<Option<T::Native>>
where
    T: PolarsNumericType,
    DatetimeInfer<T>: TryFromWithUnit<Pattern>,
{
    match value {
        Value::String(val) => {
            if let Some(pattern) = infer_pattern_single(val) {
                if let Ok(mut infer) = DatetimeInfer::try_from_with_unit(pattern, Some(tu)) {
                    if let Some(v) = infer.parse(val) {
                        return Ok(Some(v));
                    }
                }
            }
        },
        Value::Static(StaticNode::Null) => return Ok(None),
        _ => {},
    };

    if ignore_errors {
        return Ok(None);
    }

    polars_bail!(ComputeError: "cannot parse '{}' as {}", value, type_name)
}

fn deserialize_all<'a>(
    json: &Value,
    dtype: &DataType,
    ignore_errors: bool,
) -> PolarsResult<AnyValue<'a>> {
    if let Value::Static(StaticNode::Null) = json {
        return Ok(AnyValue::Null);
    }
    match dtype {
        #[cfg(feature = "dtype-datetime")]
        DataType::Date => {
            let value = deserialize_datetime::<Int32Type>(
                json,
                "Date",
                ignore_errors,
                TimeUnit::Microseconds, // ignored
            )?;
            return Ok(if let Some(value) = value {
                AnyValue::Date(value)
            } else {
                AnyValue::Null
            });
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, tz) => {
            let value = deserialize_datetime::<Int64Type>(json, "Datetime", ignore_errors, *tu)?;
            return Ok(if let Some(value) = value {
                AnyValue::DatetimeOwned(value, *tu, tz.as_ref().map(|s| Arc::from(s.clone())))
            } else {
                AnyValue::Null
            });
        },
        DataType::Float32 => {
            return Ok(
                if let Some(v) = deserialize_number::<f32>(json, ignore_errors)? {
                    AnyValue::Float32(v)
                } else {
                    AnyValue::Null
                },
            );
        },
        DataType::Float64 => {
            return Ok(
                if let Some(v) = deserialize_number::<f64>(json, ignore_errors)? {
                    AnyValue::Float64(v)
                } else {
                    AnyValue::Null
                },
            );
        },
        DataType::String => {
            return Ok(match json {
                Value::String(s) => AnyValue::StringOwned(s.as_ref().into()),
                v => AnyValue::StringOwned(format_pl_smallstr!("{}", ValueDisplay(v))),
            });
        },
        dt if dt.is_primitive_numeric() => {
            return Ok(
                if let Some(v) = deserialize_number::<i128>(json, ignore_errors)? {
                    AnyValue::Int128(v).cast(dt).into_static()
                } else {
                    AnyValue::Null
                },
            );
        },
        _ => {},
    }

    let out = match json {
        Value::Static(StaticNode::Bool(b)) => AnyValue::Boolean(*b),
        Value::Static(StaticNode::I64(i)) => AnyValue::Int64(*i),
        Value::Static(StaticNode::U64(u)) => AnyValue::UInt64(*u),
        Value::Static(StaticNode::F64(f)) => AnyValue::Float64(*f),
        Value::String(s) => AnyValue::StringOwned(s.as_ref().into()),
        Value::Array(arr) => {
            let Some(inner_dtype) = dtype.inner_dtype() else {
                if ignore_errors {
                    return Ok(AnyValue::Null);
                }
                polars_bail!(ComputeError: "expected dtype '{}' in JSON value, got dtype: Array\n\nEncountered value: {}", dtype, json);
            };
            let vals: Vec<AnyValue> = arr
                .iter()
                .map(|val| deserialize_all(val, inner_dtype, ignore_errors))
                .collect::<PolarsResult<_>>()?;
            let strict = !ignore_errors;
            let s =
                Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &vals, inner_dtype, strict)?;
            AnyValue::List(s)
        },
        #[cfg(feature = "dtype-struct")]
        Value::Object(doc) => {
            if let DataType::Struct(fields) = dtype {
                let document = &**doc;

                let vals = fields
                    .iter()
                    .map(|field| {
                        if let Some(value) = document.get(field.name.as_str()) {
                            deserialize_all(value, &field.dtype, ignore_errors)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                AnyValue::StructOwned(Box::new((vals, fields.clone())))
            } else {
                if ignore_errors {
                    return Ok(AnyValue::Null);
                }
                polars_bail!(
                    ComputeError: "expected {} in json value, got object", dtype,
                );
            }
        },
        val => AnyValue::StringOwned(format!("{:#?}", val).into()),
    };
    Ok(out)
}

/// Wrapper for serde_json's `Value` with a human-friendly Display impl for nested types:
///
/// * Default: `{"x": Static(U64(1))}`
/// * ValueDisplay: `{x: 1}`
///
/// This intended for reading in arbitrary `Value` types into a String type. Note that the output
/// is not guaranteed to be valid JSON as we don't do any escaping of e.g. quote/newline values.
struct ValueDisplay<'a>(&'a Value<'a>);

impl std::fmt::Display for ValueDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Value::*;

        match self.0 {
            Static(s) => write!(f, "{s}"),
            String(s) => write!(f, r#""{s}""#),
            Array(a) => {
                write!(f, "[")?;

                let mut iter = a.iter();

                for v in (&mut iter).take(1) {
                    write!(f, "{}", ValueDisplay(v))?;
                }

                for v in iter {
                    write!(f, ", {}", ValueDisplay(v))?;
                }

                write!(f, "]")
            },
            Object(o) => {
                write!(f, "{{")?;

                let mut iter = o.iter();

                for (k, v) in (&mut iter).take(1) {
                    write!(f, r#""{}": {}"#, k, ValueDisplay(v))?;
                }

                for (k, v) in iter {
                    write!(f, r#", "{}": {}"#, k, ValueDisplay(v))?;
                }

                write!(f, "}}")
            },
        }
    }
}
