use num::traits::NumCast;
use polars_core::prelude::*;
use serde_json::Value;

use arrow::types::NativeType;
pub(crate) fn init_buffers(schema: &Schema, capacity: usize) -> Result<PlHashMap<String, Buffer>> {
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
                // #[cfg(feature = "dtype-datetime")]
                // &DataType::Datetime(_, _) => Buffer::Datetime(DatetimeField::new(name, capacity)),
                // #[cfg(feature = "dtype-date")]
                // &DataType::Date => Buffer::Date(DatetimeField::new(name, capacity)),
                other => {
                    return Err(PolarsError::ComputeError(
                        format!("Unsupported data type {:?} when reading a csv", other).into(),
                    ))
                }
            };
            Ok((name.clone(), builder))
        })
        .collect()
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Buffer {
    Boolean(BooleanChunkedBuilder),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    /// Stores the Utf8 fields and the total string length seen for that column
    Utf8(Utf8ChunkedBuilder),
    // #[cfg(feature = "dtype-datetime")]
    // Datetime(DatetimeField<Int64Type>),
    // #[cfg(feature = "dtype-date")]
    // Date(DatetimeField<Int32Type>),
}

impl Buffer {
    pub(crate) fn into_series(self) -> Result<Series> {
        let s = match self {
            Buffer::Boolean(v) => v.finish().into_series(),
            Buffer::Int32(v) => v.finish().into_series(),
            Buffer::Int64(v) => v.finish().into_series(),
            Buffer::UInt32(v) => v.finish().into_series(),
            Buffer::UInt64(v) => v.finish().into_series(),
            Buffer::Float32(v) => v.finish().into_series(),
            Buffer::Float64(v) => v.finish().into_series(),
            // #[cfg(feature = "dtype-datetime")]
            // Buffer::Datetime(v) => v
            //   .builder
            //   .finish()
            //   .into_series()
            //   .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            //   .unwrap(),
            // #[cfg(feature = "dtype-date")]
            // Buffer::Date(v) => v
            //   .builder
            //   .finish()
            //   .into_series()
            //   .cast(&DataType::Date)
            //   .unwrap(),
            // Safety:
            // We already checked utf8 validity during parsing
            Buffer::Utf8(v) => v.finish().into_series(),
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
            // #[cfg(feature = "dtype-datetime")]
            // Buffer::Datetime(v) => v.builder.append_null(),
            // #[cfg(feature = "dtype-date")]
            // Buffer::Date(v) => v.builder.append_null(),
        };
    }

    pub(crate) fn dtype(&self) -> DataType {
        match self {
            Buffer::Boolean(_) => DataType::Boolean,
            Buffer::Int32(_) => DataType::Int32,
            Buffer::Int64(_) => DataType::Int64,
            Buffer::UInt32(_) => DataType::UInt32,
            Buffer::UInt64(_) => DataType::UInt64,
            Buffer::Float32(_) => DataType::Float32,
            Buffer::Float64(_) => DataType::Float64,
            Buffer::Utf8(_) => DataType::Utf8,
            // #[cfg(feature = "dtype-datetime")]
            // Buffer::Datetime(_) => DataType::Datetime(TimeUnit::Microseconds, None),
            // #[cfg(feature = "dtype-date")]
            // Buffer::Date(_) => DataType::Date,
        }
    }

    #[inline]
    pub(crate) fn add(&mut self, value: &Value) -> Result<()> {
        use Buffer::*;
        match self {
      Boolean(buf) => {
        match value {
          Value::Bool(v) => buf.append_value(*v),
          _ => buf.append_null()
        }
        Ok(())
      },
      Int32(buf) => {
        let n = deserialize_number::<i32>(value);
        match n {
          Some(v) => buf.append_value(v),
          None => buf.append_null()
        }
        Ok(())

      }
      Int64(buf) => {
        let n = deserialize_number::<i64>(value);
        match n {
          Some(v) => buf.append_value(v),
          None => buf.append_null()
        }
        Ok(())

      },
      UInt64(buf) => {
        let n = deserialize_number::<u64>(value);
        match n {
          Some(v) => buf.append_value(v),
          None => buf.append_null()
        }
        Ok(())
      },
      UInt32(buf) => {
        let n = deserialize_number::<u32>(value);
        match n {
          Some(v) => buf.append_value(v),
          None => buf.append_null()
        }
        Ok(())
      },
      Float32(buf) => {
        let n = deserialize_number::<f32>(value);
        match n {
          Some(v) => buf.append_value(v),
          None => buf.append_null()
        }
        Ok(())
      },
      Float64(buf) => {
        let n = deserialize_float::<f64>(value);
        match n {
          Some(v) => buf.append_value(v),
          None => buf.append_null()
        }
        Ok(())
      },

      Utf8(buf) => {
        match value {
          Value::String(v) => buf.append_value(v),
          _ => buf.append_null()
        }
        Ok(())
      },
      _ => todo!()

      // #[cfg(feature = "dtype-datetime")]
      // Datetime(buf) => <DatetimeField<Int64Type> as ParsedBuffer>::parse_bytes(
      //   buf,
      //   bytes,
      //   ignore_errors,
      //   needs_escaping,
      // ),
      // #[cfg(feature = "dtype-date")]
      // Date(buf) => <DatetimeField<Int32Type> as ParsedBuffer>::parse_bytes(
      //   buf,
      //   bytes,
      //   ignore_errors,
      //   needs_escaping,
      // ),
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
