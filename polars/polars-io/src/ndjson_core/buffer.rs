use std::hash::{Hash, Hasher};

use arrow::types::NativeType;
use num::traits::NumCast;
use polars_core::prelude::*;
use polars_time::prelude::utf8::infer::{infer_pattern_single, DatetimeInfer};
use polars_time::prelude::utf8::Pattern;
use simd_json::{BorrowedValue as Value, KnownKey, StaticNode};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BufferKey<'a>(pub(crate) KnownKey<'a>);
impl<'a> Eq for BufferKey<'a> {}

impl<'a> Hash for BufferKey<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.key().hash(state)
    }
}

pub(crate) fn init_buffers(
    schema: &Schema,
    capacity: usize,
) -> PolarsResult<PlIndexMap<BufferKey, Buffer>> {
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
            let key = KnownKey::from(name);

            Ok((BufferKey(key), builder))
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
    pub(crate) fn into_series(self) -> PolarsResult<Series> {
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
    pub(crate) fn add(&mut self, value: &Value) -> PolarsResult<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => {
                match value {
                    Value::Static(StaticNode::Bool(b)) => buf.append_value(*b),
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
                let n = deserialize_number::<f64>(value);
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

fn deserialize_number<T: NativeType + NumCast>(value: &Value) -> Option<T> {
    match value {
        Value::Static(StaticNode::F64(f)) => num::traits::cast::<f64, T>(*f),
        Value::Static(StaticNode::I64(i)) => num::traits::cast::<i64, T>(*i),
        Value::Static(StaticNode::U64(u)) => num::traits::cast::<u64, T>(*u),
        Value::Static(StaticNode::Bool(b)) => num::traits::cast::<i32, T>(*b as i32),
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
        Value::Static(StaticNode::Bool(_)) => DataType::Boolean,
        Value::Static(StaticNode::I64(_)) => DataType::Int64,
        Value::Static(StaticNode::U64(_)) => DataType::UInt64,
        Value::Static(StaticNode::F64(_)) => DataType::Float64,
        Value::Static(StaticNode::Null) => DataType::Null,
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
        Value::Static(StaticNode::Bool(b)) => AnyValue::Boolean(*b),
        Value::Static(StaticNode::I64(i)) => AnyValue::Int64(*i),
        Value::Static(StaticNode::U64(u)) => AnyValue::UInt64(*u),
        Value::Static(StaticNode::F64(f)) => AnyValue::Float64(*f),
        Value::Static(StaticNode::Null) => AnyValue::Null,
        Value::String(s) => AnyValue::Utf8Owned(s.to_string()),
        Value::Array(arr) => {
            let vals: Vec<AnyValue> = arr.iter().map(deserialize_all).collect();

            let s = Series::new("", vals);
            AnyValue::List(s)
        }
        #[cfg(feature = "dtype-struct")]
        Value::Object(doc) => {
            let vals: (Vec<AnyValue>, Vec<Field>) = doc
                .iter()
                .map(|(key, value)| {
                    let dt = value_to_dtype(value);
                    let fld = Field::new(key, dt);
                    let av: AnyValue<'a> = deserialize_all(value);
                    (av, fld)
                })
                .unzip();
            AnyValue::StructOwned(Box::new(vals))
        }
        #[cfg(not(feature = "dtype-struct"))]
        val => AnyValue::Utf8Owned(format!("{:#?}", val)),
    }
}
