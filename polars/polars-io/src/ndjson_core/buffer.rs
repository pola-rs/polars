use arrow::types::NativeType;
use num::traits::NumCast;
use polars_core::prelude::*;
use polars_time::prelude::utf8::infer::infer_pattern_single;
use polars_time::prelude::utf8::infer::DatetimeInfer;
use polars_time::prelude::utf8::Pattern;
use serde_json::Value;
pub(crate) fn init_buffers(schema: &Schema, capacity: usize) -> Result<PlIndexMap<String, Buffer>> {
    schema
        .iter()
        .map(|(name, dtype)| {
            let builder = match dtype {
                &DataType::Boolean => Buffer::Boolean(BooleanChunkedBuilder::new(name, capacity)),
                &DataType::Int32 => Buffer::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Int64 => Buffer::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt32 => Buffer::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt64 => Buffer::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Float32 => Buffer::Float32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Float64 => Buffer::Float64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Utf8 => {
                    Buffer::Utf8(Utf8ChunkedBuilder::new(name, capacity, capacity * 25))
                }
                #[cfg(feature = "dtype-datetime")]
                &DataType::Datetime(_, _) => {
                    Buffer::Datetime(PrimitiveChunkedBuilder::new(name, capacity))
                }
                #[cfg(feature = "dtype-date")]
                &DataType::Date => Buffer::Date(PrimitiveChunkedBuilder::new(name, capacity)),
                _ => Buffer::All((Vec::with_capacity(capacity), name)),
            };
            Ok((name.clone(), builder))
        })
        .collect()
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Buffer<'a> {
    Boolean(BooleanChunkedBuilder),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    Utf8(Utf8ChunkedBuilder),
    #[cfg(feature = "dtype-datetime")]
    Datetime(PrimitiveChunkedBuilder<Int64Type>),
    #[cfg(feature = "dtype-date")]
    Date(PrimitiveChunkedBuilder<Int32Type>),
    All((Vec<AnyValue<'a>>, &'a str)),
}

impl<'a> Buffer<'a> {
    pub(crate) fn into_series(self) -> Result<Series> {
        let s = match self {
            Buffer::Boolean(v) => v.finish().into_series(),
            Buffer::Int32(v) => v.finish().into_series(),
            Buffer::Int64(v) => v.finish().into_series(),
            Buffer::UInt32(v) => v.finish().into_series(),
            Buffer::UInt64(v) => v.finish().into_series(),
            Buffer::Float32(v) => v.finish().into_series(),
            Buffer::Float64(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-datetime")]
            Buffer::Datetime(v) => v
                .finish()
                .into_series()
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap(),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(v) => v.finish().into_series().cast(&DataType::Date).unwrap(),
            Buffer::Utf8(v) => v.finish().into_series(),
            Buffer::All((vals, name)) => Series::new(name, vals),
        };
        Ok(s)
    }

    pub(crate) fn add_null(&mut self) {
        match self {
            Buffer::Boolean(v) => v.append_null(),
            Buffer::Int32(v) => v.append_null(),
            Buffer::Int64(v) => v.append_null(),
            Buffer::UInt32(v) => v.append_null(),
            Buffer::UInt64(v) => v.append_null(),
            Buffer::Float32(v) => v.append_null(),
            Buffer::Float64(v) => v.append_null(),
            Buffer::Utf8(v) => v.append_null(),
            #[cfg(feature = "dtype-datetime")]
            Buffer::Datetime(v) => v.append_null(),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(v) => v.append_null(),
            Buffer::All((v, _)) => v.push(AnyValue::Null),
        };
    }

    #[inline]
    pub(crate) fn add(&mut self, value: &Value) -> Result<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => {
                match value {
                    Value::Bool(v) => buf.append_value(*v),
                    _ => buf.append_null(),
                }
                Ok(())
            }
            Int32(buf) => {
                let n = deserialize_number::<i32>(value);
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Int64(buf) => {
                let n = deserialize_number::<i64>(value);
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            }
            UInt64(buf) => {
                let n = deserialize_number::<u64>(value);
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            }
            UInt32(buf) => {
                let n = deserialize_number::<u32>(value);
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Float32(buf) => {
                let n = deserialize_number::<f32>(value);
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Float64(buf) => {
                let n = deserialize_float::<f64>(value);
                match n {
                    Some(v) => buf.append_value(v),
                    None => buf.append_null(),
                }
                Ok(())
            }

            Utf8(buf) => {
                match value {
                    Value::String(v) => buf.append_value(v),
                    _ => buf.append_null(),
                }
                Ok(())
            }
            #[cfg(feature = "dtype-datetime")]
            Datetime(buf) => {
                let v = deserialize_datetime::<Int64Type>(value);
                buf.append_option(v);
                Ok(())
            }
            #[cfg(feature = "dtype-date")]
            Date(buf) => {
                let v = deserialize_datetime::<Int32Type>(value);
                buf.append_option(v);
                Ok(())
            }
            All((buf, _)) => {
                let av = deserialize_all(value);
                buf.push(av);
                Ok(())
            }
        }
    }
}

fn deserialize_float<T: NativeType + NumCast>(value: &Value) -> Option<T> {
    match value {
        Value::Number(number) => number.as_f64().and_then(num::traits::cast::<f64, T>),
        Value::Bool(number) => num::traits::cast::<i32, T>(*number as i32),
        _ => None,
    }
}

fn deserialize_number<T: NativeType + NumCast>(value: &Value) -> Option<T> {
    match value {
        Value::Number(v) => v.as_i64().and_then(num::traits::cast::<i64, T>),
        Value::Bool(number) => num::traits::cast::<i32, T>(*number as i32),
        _ => None,
    }
}

fn deserialize_datetime<T>(value: &Value) -> Option<T::Native>
where
    T: PolarsNumericType,
    DatetimeInfer<T::Native>: TryFrom<Pattern>,
{
    let val = match value {
        Value::String(s) => s,
        _ => return None,
    };
    match infer_pattern_single(val) {
        None => None,
        Some(pattern) => match DatetimeInfer::<T::Native>::try_from(pattern) {
            Ok(mut infer) => infer.parse(val),
            Err(_) => None,
        },
    }
}

#[cfg(feature = "dtype-struct")]
fn value_to_dtype(val: &Value) -> DataType {
    match val {
        Value::Null => DataType::Null,
        Value::Bool(_) => DataType::Boolean,
        Value::Number(n) => {
            if n.is_i64() {
                DataType::Int64
            } else if n.is_u64() {
                DataType::UInt64
            } else {
                DataType::Float64
            }
        }
        Value::Array(arr) => {
            let dtype = value_to_dtype(&arr[0]);

            DataType::List(Box::new(dtype))
        }
        #[cfg(feature = "dtype-struct")]
        Value::Object(doc) => {
            let fields = doc.iter().map(|(key, value)| {
                let dtype = value_to_dtype(value);
                Field::new(key, dtype)
            });
            DataType::Struct(fields.collect())
        }
        _ => DataType::Utf8,
    }
}

fn deserialize_all<'a, 'b>(json: &'b Value) -> AnyValue<'a> {
    match json {
        Value::Bool(b) => AnyValue::Boolean(*b),
        Value::Number(n) => {
            if n.is_i64() {
                AnyValue::Int64(n.as_i64().unwrap())
            } else if n.is_u64() {
                AnyValue::UInt64(n.as_u64().unwrap())
            } else {
                AnyValue::Float64(n.as_f64().unwrap())
            }
        }
        Value::Array(arr) => {
            let vals: Vec<AnyValue> = arr.iter().map(deserialize_all).collect();

            let s = Series::new("", vals);
            AnyValue::List(s)
        }
        Value::Null => AnyValue::Null,
        #[cfg(feature = "dtype-struct")]
        Value::Object(doc) => {
            let vals: (Vec<AnyValue>, Vec<Field>) = doc
                .into_iter()
                .map(|(key, value)| {
                    let dt = value_to_dtype(value);
                    let fld = Field::new(key, dt);
                    let av: AnyValue<'a> = deserialize_all(value);
                    (av, fld)
                })
                .unzip();
            AnyValue::StructOwned(Box::new(vals))
        }
        val => AnyValue::Utf8Owned(format!("{:#?}", val)),
    }
}
