use std::hash::{Hash, Hasher};

use arrow::types::NativeType;
use num::traits::NumCast;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::prelude::utf8::infer::{infer_pattern_single, DatetimeInfer};
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
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

pub(crate) struct Buffer<'a>(&'a str, AnyValueBuffer<'a>);

impl Buffer<'_> {
    pub fn into_series(self) -> Series {
        let mut s = self.1.into_series();
        s.rename(self.0);
        s
    }

    #[inline]
    pub(crate) fn add(&mut self, value: &Value) -> PolarsResult<()> {
        use AnyValueBuffer::*;
        match &mut self.1 {
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
            Datetime(buf, _, _) => {
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
            All(dtype, buf) => {
                let av = deserialize_all(value, dtype)?;
                buf.push(av);
                Ok(())
            }
            _ => panic!("unexpected dtype when deserializing ndjson"),
        }
    }
    pub fn add_null(&mut self) {
        self.1.add(AnyValue::Null).expect("should not fail");
    }
}
pub(crate) fn init_buffers(
    schema: &Schema,
    capacity: usize,
) -> PolarsResult<PlIndexMap<BufferKey, Buffer>> {
    schema
        .iter()
        .map(|(name, dtype)| {
            let av_buf = (dtype, capacity).into();
            let key = KnownKey::from(name);
            Ok((BufferKey(key), Buffer(name, av_buf)))
        })
        .collect()
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

#[cfg(feature = "dtype-datetime")]
fn deserialize_datetime<T>(value: &Value) -> Option<T::Native>
where
    T: PolarsNumericType,
    DatetimeInfer<T::Native>: TryFrom<Pattern>,
{
    let val = match value {
        Value::String(s) => s,
        _ => return None,
    };
    infer_pattern_single(val).and_then(|pattern| {
        match DatetimeInfer::<T::Native>::try_from(pattern) {
            Ok(mut infer) => infer.parse(val),
            Err(_) => None,
        }
    })
}

fn deserialize_all<'a>(json: &Value, dtype: &DataType) -> PolarsResult<AnyValue<'a>> {
    let out = match json {
        Value::Static(StaticNode::Bool(b)) => AnyValue::Boolean(*b),
        Value::Static(StaticNode::I64(i)) => AnyValue::Int64(*i),
        Value::Static(StaticNode::U64(u)) => AnyValue::UInt64(*u),
        Value::Static(StaticNode::F64(f)) => AnyValue::Float64(*f),
        Value::Static(StaticNode::Null) => AnyValue::Null,
        Value::String(s) => AnyValue::Utf8Owned(s.as_ref().into()),
        Value::Array(arr) => {
            let inner_dtype = dtype.inner_dtype().ok_or_else(|| {
                PolarsError::ComputeError(
                    format!("Expected List/Array in json value got: {dtype}").into(),
                )
            })?;
            let vals: Vec<AnyValue> = arr
                .iter()
                .map(|val| deserialize_all(val, inner_dtype))
                .collect::<PolarsResult<_>>()?;
            let s = Series::from_any_values_and_dtype("", &vals, inner_dtype)?;
            AnyValue::List(s)
        }
        #[cfg(feature = "dtype-struct")]
        Value::Object(doc) => {
            if let DataType::Struct(fields) = dtype {
                let document = &**doc;

                let vals = fields
                    .iter()
                    .map(|field| {
                        if let Some(value) = document.get(field.name.as_str()) {
                            deserialize_all(value, &field.dtype)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                AnyValue::StructOwned(Box::new((vals, fields.clone())))
            } else {
                return Err(PolarsError::ComputeError(
                    format!("Expected: {dtype} but got object instead.").into(),
                ));
            }
        }
        #[cfg(not(feature = "dtype-struct"))]
        val => AnyValue::Utf8Owned(format!("{:#?}", val).into()),
    };
    Ok(out)
}
