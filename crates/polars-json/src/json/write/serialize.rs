use std::io::Write;

use arrow::array::*;
use arrow::bitmap::utils::ZipValidity;
#[cfg(feature = "dtype-decimal")]
use arrow::compute::decimal::{format_decimal, get_trim_decimal_zeros};
use arrow::datatypes::{ArrowDataType, IntegerType, TimeUnit};
use arrow::io::iterator::BufStreamingIterator;
use arrow::offset::Offset;
#[cfg(feature = "timezones")]
use arrow::temporal_conversions::parse_offset_tz;
use arrow::temporal_conversions::{
    date32_to_date, duration_ms_to_duration, duration_ns_to_duration, duration_s_to_duration,
    duration_us_to_duration, parse_offset, timestamp_ms_to_datetime, timestamp_ns_to_datetime,
    timestamp_s_to_datetime, timestamp_to_datetime, timestamp_us_to_datetime,
};
use arrow::types::NativeType;
use chrono::{Duration, NaiveDate, NaiveDateTime};
use streaming_iterator::StreamingIterator;

use super::utf8;

fn write_integer<I: itoa::Integer>(buf: &mut Vec<u8>, val: I) {
    let mut buffer = itoa::Buffer::new();
    let value = buffer.format(val);
    buf.extend_from_slice(value.as_bytes())
}

fn write_float<I: ryu::Float>(f: &mut Vec<u8>, val: I) {
    let mut buffer = ryu::Buffer::new();
    let value = buffer.format(val);
    f.extend_from_slice(value.as_bytes())
}

fn materialize_serializer<'a, I, F, T>(
    f: F,
    iterator: I,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync>
where
    T: 'a,
    I: Iterator<Item = T> + Send + Sync + 'a,
    F: FnMut(T, &mut Vec<u8>) + Send + Sync + 'a,
{
    if offset > 0 || take < usize::MAX {
        Box::new(BufStreamingIterator::new(
            iterator.skip(offset).take(take),
            f,
            vec![],
        ))
    } else {
        Box::new(BufStreamingIterator::new(iterator, f, vec![]))
    }
}

fn boolean_serializer<'a>(
    array: &'a BooleanArray,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let f = |x: Option<bool>, buf: &mut Vec<u8>| match x {
        Some(true) => buf.extend_from_slice(b"true"),
        Some(false) => buf.extend_from_slice(b"false"),
        None => buf.extend_from_slice(b"null"),
    };
    materialize_serializer(f, array.iter(), offset, take)
}

fn null_serializer(
    len: usize,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + Send + Sync> {
    let f = |_x: (), buf: &mut Vec<u8>| buf.extend_from_slice(b"null");
    materialize_serializer(f, std::iter::repeat(()).take(len), offset, take)
}

fn primitive_serializer<'a, T: NativeType + itoa::Integer>(
    array: &'a PrimitiveArray<T>,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let f = |x: Option<&T>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            write_integer(buf, *x)
        } else {
            buf.extend(b"null")
        }
    };
    materialize_serializer(f, array.iter(), offset, take)
}

fn float_serializer<'a, T>(
    array: &'a PrimitiveArray<T>,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync>
where
    T: num_traits::Float + NativeType + ryu::Float,
{
    let f = |x: Option<&T>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            if T::is_nan(*x) || T::is_infinite(*x) {
                buf.extend(b"null")
            } else {
                write_float(buf, *x)
            }
        } else {
            buf.extend(b"null")
        }
    };

    materialize_serializer(f, array.iter(), offset, take)
}

#[cfg(feature = "dtype-decimal")]
fn decimal_serializer<'a>(
    array: &'a PrimitiveArray<i128>,
    scale: usize,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let trim_zeros = get_trim_decimal_zeros();
    let f = move |x: Option<&i128>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            utf8::write_str(buf, format_decimal(*x, scale, trim_zeros).as_str()).unwrap()
        } else {
            buf.extend(b"null")
        }
    };

    materialize_serializer(f, array.iter(), offset, take)
}

fn dictionary_utf8view_serializer<'a, K: DictionaryKey>(
    array: &'a DictionaryArray<K>,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let iter = array.iter_typed::<Utf8ViewArray>().unwrap().skip(offset);
    let f = |x: Option<&str>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            utf8::write_str(buf, x).unwrap();
        } else {
            buf.extend_from_slice(b"null")
        }
    };
    materialize_serializer(f, iter, offset, take)
}

fn utf8_serializer<'a, O: Offset>(
    array: &'a Utf8Array<O>,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let f = |x: Option<&str>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            utf8::write_str(buf, x).unwrap();
        } else {
            buf.extend_from_slice(b"null")
        }
    };
    materialize_serializer(f, array.iter(), offset, take)
}

fn utf8view_serializer<'a>(
    array: &'a Utf8ViewArray,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let f = |x: Option<&str>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            utf8::write_str(buf, x).unwrap();
        } else {
            buf.extend_from_slice(b"null")
        }
    };
    materialize_serializer(f, array.iter(), offset, take)
}

fn struct_serializer<'a>(
    array: &'a StructArray,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    // {"a": [1, 2, 3], "b": [a, b, c], "c": {"a": [1, 2, 3]}}
    // [
    //  {"a": 1, "b": a, "c": {"a": 1}},
    //  {"a": 2, "b": b, "c": {"a": 2}},
    //  {"a": 3, "b": c, "c": {"a": 3}},
    // ]
    //
    let mut serializers = array
        .values()
        .iter()
        .map(|x| x.as_ref())
        .map(|arr| new_serializer(arr, offset, take))
        .collect::<Vec<_>>();

    Box::new(BufStreamingIterator::new(
        ZipValidity::new_with_validity(0..array.len(), array.validity()),
        move |maybe, buf| {
            if maybe.is_some() {
                let names = array.fields().iter().map(|f| f.name.as_str());
                serialize_item(
                    buf,
                    names.zip(
                        serializers
                            .iter_mut()
                            .map(|serializer| serializer.next().unwrap()),
                    ),
                    true,
                );
            } else {
                serializers.iter_mut().for_each(|iter| {
                    let _ = iter.next();
                });
                buf.extend(b"null");
            }
        },
        vec![],
    ))
}

fn list_serializer<'a, O: Offset>(
    array: &'a ListArray<O>,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    // [[1, 2], [3]]
    // [
    //  [1, 2],
    //  [3]
    // ]
    //
    let offsets = array.offsets().as_slice();
    let start = offsets[0].to_usize();
    let end = offsets.last().unwrap().to_usize();
    let mut serializer = new_serializer(array.values().as_ref(), start, end - start);

    let f = move |offset: Option<&[O]>, buf: &mut Vec<u8>| {
        if let Some(offset) = offset {
            let length = (offset[1] - offset[0]).to_usize();
            buf.push(b'[');
            let mut is_first_row = true;
            for _ in 0..length {
                if !is_first_row {
                    buf.push(b',');
                }
                is_first_row = false;
                buf.extend(serializer.next().unwrap());
            }
            buf.push(b']');
        } else {
            buf.extend(b"null");
        }
    };

    let iter =
        ZipValidity::new_with_validity(array.offsets().buffer().windows(2), array.validity());
    materialize_serializer(f, iter, offset, take)
}

fn fixed_size_list_serializer<'a>(
    array: &'a FixedSizeListArray,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    let mut serializer = new_serializer(array.values().as_ref(), offset, take);

    Box::new(BufStreamingIterator::new(
        ZipValidity::new(0..array.len(), array.validity().map(|x| x.iter())),
        move |ix, buf| {
            if ix.is_some() {
                let length = array.size();
                buf.push(b'[');
                let mut is_first_row = true;
                for _ in 0..length {
                    if !is_first_row {
                        buf.push(b',');
                    }
                    is_first_row = false;
                    buf.extend(serializer.next().unwrap());
                }
                buf.push(b']');
            } else {
                buf.extend(b"null");
            }
        },
        vec![],
    ))
}

fn date_serializer<'a, T, F>(
    array: &'a PrimitiveArray<T>,
    convert: F,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync>
where
    T: NativeType,
    F: Fn(T) -> NaiveDate + 'static + Send + Sync,
{
    let f = move |x: Option<&T>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            let nd = convert(*x);
            write!(buf, "\"{nd}\"").unwrap();
        } else {
            buf.extend_from_slice(b"null")
        }
    };

    materialize_serializer(f, array.iter(), offset, take)
}

fn duration_serializer<'a, T, F>(
    array: &'a PrimitiveArray<T>,
    convert: F,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync>
where
    T: NativeType,
    F: Fn(T) -> Duration + 'static + Send + Sync,
{
    let f = move |x: Option<&T>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            let duration = convert(*x);
            write!(buf, "\"{duration}\"").unwrap();
        } else {
            buf.extend_from_slice(b"null")
        }
    };

    materialize_serializer(f, array.iter(), offset, take)
}

fn timestamp_serializer<'a, F>(
    array: &'a PrimitiveArray<i64>,
    convert: F,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync>
where
    F: Fn(i64) -> NaiveDateTime + 'static + Send + Sync,
{
    let f = move |x: Option<&i64>, buf: &mut Vec<u8>| {
        if let Some(x) = x {
            let ndt = convert(*x);
            write!(buf, "\"{ndt}\"").unwrap();
        } else {
            buf.extend_from_slice(b"null")
        }
    };
    materialize_serializer(f, array.iter(), offset, take)
}

fn timestamp_tz_serializer<'a>(
    array: &'a PrimitiveArray<i64>,
    time_unit: TimeUnit,
    tz: &str,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    match parse_offset(tz) {
        Ok(parsed_tz) => {
            let f = move |x: Option<&i64>, buf: &mut Vec<u8>| {
                if let Some(x) = x {
                    let dt_str = timestamp_to_datetime(*x, time_unit, &parsed_tz).to_rfc3339();
                    write!(buf, "\"{dt_str}\"").unwrap();
                } else {
                    buf.extend_from_slice(b"null")
                }
            };

            materialize_serializer(f, array.iter(), offset, take)
        },
        #[cfg(feature = "timezones")]
        _ => match parse_offset_tz(tz) {
            Ok(parsed_tz) => {
                let f = move |x: Option<&i64>, buf: &mut Vec<u8>| {
                    if let Some(x) = x {
                        let dt_str = timestamp_to_datetime(*x, time_unit, &parsed_tz).to_rfc3339();
                        write!(buf, "\"{dt_str}\"").unwrap();
                    } else {
                        buf.extend_from_slice(b"null")
                    }
                };

                materialize_serializer(f, array.iter(), offset, take)
            },
            _ => {
                panic!("Timezone {} is invalid or not supported", tz);
            },
        },
        #[cfg(not(feature = "timezones"))]
        _ => {
            panic!("Invalid Offset format (must be [-]00:00) or timezones feature not active");
        },
    }
}

pub(crate) fn new_serializer<'a>(
    array: &'a dyn Array,
    offset: usize,
    take: usize,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync> {
    match array.data_type().to_logical_type() {
        ArrowDataType::Boolean => {
            boolean_serializer(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Int8 => {
            primitive_serializer::<i8>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Int16 => {
            primitive_serializer::<i16>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Int32 => {
            primitive_serializer::<i32>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Int64 => {
            primitive_serializer::<i64>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::UInt8 => {
            primitive_serializer::<u8>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::UInt16 => {
            primitive_serializer::<u16>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::UInt32 => {
            primitive_serializer::<u32>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::UInt64 => {
            primitive_serializer::<u64>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Float32 => {
            float_serializer::<f32>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Float64 => {
            float_serializer::<f64>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        #[cfg(feature = "dtype-decimal")]
        ArrowDataType::Decimal(_, scale) => {
            decimal_serializer(array.as_any().downcast_ref().unwrap(), *scale, offset, take)
        },
        ArrowDataType::LargeUtf8 => {
            utf8_serializer::<i64>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Utf8View => {
            utf8view_serializer(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Struct(_) => {
            struct_serializer(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::FixedSizeList(_, _) => {
            fixed_size_list_serializer(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::LargeList(_) => {
            list_serializer::<i64>(array.as_any().downcast_ref().unwrap(), offset, take)
        },
        ArrowDataType::Dictionary(k, v, _) => match (k, &**v) {
            (IntegerType::UInt32, ArrowDataType::Utf8View) => {
                let array = array
                    .as_any()
                    .downcast_ref::<DictionaryArray<u32>>()
                    .unwrap();
                dictionary_utf8view_serializer::<u32>(array, offset, take)
            },
            _ => {
                // Not produced by polars
                unreachable!()
            },
        },
        ArrowDataType::Date32 => date_serializer(
            array.as_any().downcast_ref().unwrap(),
            date32_to_date,
            offset,
            take,
        ),
        ArrowDataType::Timestamp(tu, None) => {
            let convert = match tu {
                TimeUnit::Nanosecond => timestamp_ns_to_datetime,
                TimeUnit::Microsecond => timestamp_us_to_datetime,
                TimeUnit::Millisecond => timestamp_ms_to_datetime,
                TimeUnit::Second => timestamp_s_to_datetime,
            };
            timestamp_serializer(
                array.as_any().downcast_ref().unwrap(),
                convert,
                offset,
                take,
            )
        },
        ArrowDataType::Timestamp(time_unit, Some(tz)) => timestamp_tz_serializer(
            array.as_any().downcast_ref().unwrap(),
            *time_unit,
            tz,
            offset,
            take,
        ),
        ArrowDataType::Duration(tu) => {
            let convert = match tu {
                TimeUnit::Nanosecond => duration_ns_to_duration,
                TimeUnit::Microsecond => duration_us_to_duration,
                TimeUnit::Millisecond => duration_ms_to_duration,
                TimeUnit::Second => duration_s_to_duration,
            };
            duration_serializer(
                array.as_any().downcast_ref().unwrap(),
                convert,
                offset,
                take,
            )
        },
        ArrowDataType::Null => null_serializer(array.len(), offset, take),
        other => todo!("Writing {:?} to JSON", other),
    }
}

fn serialize_item<'a>(
    buffer: &mut Vec<u8>,
    record: impl Iterator<Item = (&'a str, &'a [u8])>,
    is_first_row: bool,
) {
    if !is_first_row {
        buffer.push(b',');
    }
    buffer.push(b'{');
    let mut first_item = true;
    for (key, value) in record {
        if !first_item {
            buffer.push(b',');
        }
        first_item = false;
        utf8::write_str(buffer, key).unwrap();
        buffer.push(b':');
        buffer.extend(value);
    }
    buffer.push(b'}');
}

/// Serializes `array` to a valid JSON to `buffer`
/// # Implementation
/// This operation is CPU-bounded
pub(crate) fn serialize(array: &dyn Array, buffer: &mut Vec<u8>) {
    let mut serializer = new_serializer(array, 0, usize::MAX);

    (0..array.len()).for_each(|i| {
        if i != 0 {
            buffer.push(b',');
        }
        buffer.extend_from_slice(serializer.next().unwrap());
    });
}
