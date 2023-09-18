use lexical_core::ToLexical;

use crate::datatypes::IntegerType;
use crate::temporal_conversions;
use crate::types::NativeType;
use crate::util::lexical_to_bytes_mut;
use crate::{
    array::{
        Array, BinaryArray, BooleanArray, DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array,
    },
    datatypes::{DataType, TimeUnit},
    error::Result,
    offset::Offset,
};

use super::super::super::iterator::{BufStreamingIterator, StreamingIterator};
use csv_core::WriteResult;
use std::any::Any;
use std::fmt::{Debug, Write};

/// Options to serialize logical types to CSV
/// The default is to format times and dates as `chrono` crate formats them.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SerializeOptions {
    /// used for [`DataType::Date32`]
    pub date32_format: Option<String>,
    /// used for [`DataType::Date64`]
    pub date64_format: Option<String>,
    /// used for [`DataType::Time32`]
    pub time32_format: Option<String>,
    /// used for [`DataType::Time64`]
    pub time64_format: Option<String>,
    /// used for [`DataType::Timestamp`]
    pub timestamp_format: Option<String>,
    /// used as separator/delimiter
    pub delimiter: u8,
    /// quoting character
    pub quote: u8,
}

impl Default for SerializeOptions {
    fn default() -> Self {
        SerializeOptions {
            date32_format: None,
            date64_format: None,
            time32_format: None,
            time64_format: None,
            timestamp_format: None,
            delimiter: b',',
            quote: b'"',
        }
    }
}

/// Utility to write to `&mut Vec<u8>` buffer
struct StringWrap<'a>(pub &'a mut Vec<u8>);

impl<'a> Write for StringWrap<'a> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

fn primitive_write<'a, T: NativeType + ToLexical>(
    array: &'a PrimitiveArray<T>,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a> {
    Box::new(BufStreamingIterator::new(
        array.iter(),
        |x, buf| {
            if let Some(x) = x {
                lexical_to_bytes_mut(*x, buf)
            }
        },
        vec![],
    ))
}

macro_rules! dyn_primitive {
    ($ty:ty, $array:expr) => {{
        let array = $array.as_any().downcast_ref().unwrap();
        primitive_write::<$ty>(array)
    }};
}

macro_rules! dyn_date {
    ($ty:ident, $fn:expr, $array:expr, $format:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .unwrap();
        if let Some(format) = $format {
            Box::new(BufStreamingIterator::new(
                array.iter(),
                move |x, buf| {
                    if let Some(x) = x {
                        let dt = ($fn)(*x).format(format);
                        let _ = write!(StringWrap(buf), "{}", dt);
                    }
                },
                vec![],
            ))
        } else {
            Box::new(BufStreamingIterator::new(
                array.iter(),
                move |x, buf| {
                    if let Some(x) = x {
                        let dt = ($fn)(*x);
                        let _ = write!(StringWrap(buf), "{}", dt);
                    }
                },
                vec![],
            ))
        }
    }};
}

fn timestamp_with_tz_default<'a>(
    array: &'a PrimitiveArray<i64>,
    time_unit: TimeUnit,
    tz: &str,
) -> Result<Box<dyn StreamingIterator<Item = [u8]> + 'a>> {
    let timezone = temporal_conversions::parse_offset(tz);
    Ok(match timezone {
        Ok(timezone) => Box::new(BufStreamingIterator::new(
            array.iter(),
            move |x, buf| {
                if let Some(x) = x {
                    let dt = temporal_conversions::timestamp_to_datetime(*x, time_unit, &timezone);
                    let _ = write!(StringWrap(buf), "{dt}");
                }
            },
            vec![],
        )),
        #[cfg(feature = "chrono-tz")]
        _ => {
            let timezone = temporal_conversions::parse_offset_tz(tz)?;
            Box::new(BufStreamingIterator::new(
                array.iter(),
                move |x, buf| {
                    if let Some(x) = x {
                        let dt =
                            temporal_conversions::timestamp_to_datetime(*x, time_unit, &timezone);
                        let _ = write!(StringWrap(buf), "{dt}");
                    }
                },
                vec![],
            ))
        }
        #[cfg(not(feature = "chrono-tz"))]
        _ => {
            return Err(crate::error::Error::InvalidArgumentError(
                "Invalid Offset format (must be [-]00:00) or chrono-tz feature not active"
                    .to_string(),
            ))
        }
    })
}

fn timestamp_with_tz_with_format<'a>(
    array: &'a PrimitiveArray<i64>,
    time_unit: TimeUnit,
    tz: &str,
    format: &'a str,
) -> Result<Box<dyn StreamingIterator<Item = [u8]> + 'a>> {
    let timezone = temporal_conversions::parse_offset(tz);
    Ok(match timezone {
        Ok(timezone) => Box::new(BufStreamingIterator::new(
            array.iter(),
            move |x, buf| {
                if let Some(x) = x {
                    let dt = temporal_conversions::timestamp_to_datetime(*x, time_unit, &timezone)
                        .format(format);
                    let _ = write!(StringWrap(buf), "{dt}");
                }
            },
            vec![],
        )),
        #[cfg(feature = "chrono-tz")]
        _ => {
            let timezone = temporal_conversions::parse_offset_tz(tz)?;
            Box::new(BufStreamingIterator::new(
                array.iter(),
                move |x, buf| {
                    if let Some(x) = x {
                        let dt =
                            temporal_conversions::timestamp_to_datetime(*x, time_unit, &timezone)
                                .format(format);
                        let _ = write!(StringWrap(buf), "{dt}");
                    }
                },
                vec![],
            ))
        }
        #[cfg(not(feature = "chrono-tz"))]
        _ => {
            return Err(crate::error::Error::InvalidArgumentError(
                "Invalid Offset format (must be [-]00:00) or chrono-tz feature not active"
                    .to_string(),
            ))
        }
    })
}

fn timestamp_with_tz<'a>(
    array: &'a PrimitiveArray<i64>,
    time_unit: TimeUnit,
    tz: &str,
    format: Option<&'a str>,
) -> Result<Box<dyn StreamingIterator<Item = [u8]> + 'a>> {
    if let Some(format) = format {
        timestamp_with_tz_with_format(array, time_unit, tz, format)
    } else {
        timestamp_with_tz_default(array, time_unit, tz)
    }
}

fn new_utf8_serializer<'a, O: Offset>(
    array: &'a Utf8Array<O>,
    options: &'a SerializeOptions,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a> {
    let mut local_buf = vec![0u8; 64];
    let mut ser_writer = csv_core::WriterBuilder::new()
        .quote(options.quote)
        .delimiter(options.delimiter)
        .build();

    let resize = |local_buf: &mut Vec<u8>, additional: usize| {
        local_buf.extend(std::iter::repeat(0u8).take(additional))
    };

    Box::new(BufStreamingIterator::new(
        array.iter(),
        move |x, buf| {
            match x {
                // Empty strings are quoted.
                // This will ensure a csv parser will not read them as missing
                // in a delimited field
                Some("") => buf.extend_from_slice(b"\"\""),
                Some(s) => {
                    if s.len() * 3 > local_buf.len() {
                        resize(&mut local_buf, s.len() * 3)
                    }
                    match ser_writer.field(s.as_bytes(), &mut local_buf) {
                        (WriteResult::InputEmpty, _, n_out) => {
                            // the writer::delimiter call writes a maximum of 2 bytes
                            if local_buf.len() - n_out < 2 {
                                resize(&mut local_buf, 2);
                            }
                            match ser_writer.delimiter(&mut local_buf[n_out..]) {
                                (WriteResult::InputEmpty, n_out_delimiter) => {
                                    // we subtract 1 because we never want to include the delimiter written by csv-core
                                    buf.extend_from_slice(
                                        &local_buf[..n_out + n_out_delimiter - 1],
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                _ => {}
            }
        },
        vec![],
    ))
}

/// Returns a [`StreamingIterator`] that yields `&[u8]` serialized from `array` according to `options`.
/// For numeric types, this serializes as usual. For dates, times and timestamps, it uses `options` to
/// Supported types:
/// * boolean
/// * numeric types (i.e. floats, int, uint)
/// * times and dates
/// * naive timestamps (timestamps without timezone information)
/// # Error
/// This function errors if any of the logical types in `batch` is not supported.
pub fn new_serializer<'a>(
    array: &'a dyn Array,
    options: &'a SerializeOptions,
) -> Result<Box<dyn StreamingIterator<Item = [u8]> + 'a>> {
    Ok(match array.data_type() {
        DataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Box::new(BufStreamingIterator::new(
                array.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        if x {
                            buf.extend_from_slice(b"true");
                        } else {
                            buf.extend_from_slice(b"false");
                        }
                    }
                },
                vec![],
            ))
        }
        DataType::UInt8 => {
            dyn_primitive!(u8, array)
        }
        DataType::UInt16 => {
            dyn_primitive!(u16, array)
        }
        DataType::UInt32 => {
            dyn_primitive!(u32, array)
        }
        DataType::UInt64 => {
            dyn_primitive!(u64, array)
        }
        DataType::Int8 => {
            dyn_primitive!(i8, array)
        }
        DataType::Int16 => {
            dyn_primitive!(i16, array)
        }
        DataType::Int32 => {
            dyn_primitive!(i32, array)
        }
        DataType::Date32 => {
            dyn_date!(
                i32,
                temporal_conversions::date32_to_datetime,
                array,
                options.date32_format.as_ref()
            )
        }
        DataType::Time32(TimeUnit::Second) => {
            dyn_date!(
                i32,
                temporal_conversions::time32s_to_time,
                array,
                options.time32_format.as_ref()
            )
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            dyn_date!(
                i32,
                temporal_conversions::time32ms_to_time,
                array,
                options.time32_format.as_ref()
            )
        }
        DataType::Int64 => {
            dyn_primitive!(i64, array)
        }
        DataType::Date64 => {
            dyn_date!(
                i64,
                temporal_conversions::date64_to_datetime,
                array,
                options.date64_format.as_ref()
            )
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            dyn_date!(
                i64,
                temporal_conversions::time64us_to_time,
                array,
                &options.time64_format
            )
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            dyn_date!(
                i64,
                temporal_conversions::time64ns_to_time,
                array,
                &options.time64_format
            )
        }
        DataType::Timestamp(TimeUnit::Second, None) => {
            dyn_date!(
                i64,
                temporal_conversions::timestamp_s_to_datetime,
                array,
                &options.timestamp_format
            )
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            dyn_date!(
                i64,
                temporal_conversions::timestamp_ms_to_datetime,
                array,
                &options.timestamp_format
            )
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            dyn_date!(
                i64,
                temporal_conversions::timestamp_us_to_datetime,
                array,
                &options.timestamp_format
            )
        }
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            dyn_date!(
                i64,
                temporal_conversions::timestamp_ns_to_datetime,
                array,
                &options.timestamp_format
            )
        }
        DataType::Timestamp(time_unit, Some(tz)) => {
            return timestamp_with_tz(
                array.as_any().downcast_ref().unwrap(),
                *time_unit,
                tz.as_ref(),
                options.timestamp_format.as_ref().map(|x| x.as_ref()),
            )
        }
        DataType::Float32 => {
            dyn_primitive!(f32, array)
        }
        DataType::Float64 => {
            dyn_primitive!(f64, array)
        }
        DataType::Utf8 => {
            let array = array.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
            new_utf8_serializer(array, options)
        }
        DataType::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            new_utf8_serializer(array, options)
        }
        DataType::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            Box::new(BufStreamingIterator::new(
                array.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.extend_from_slice(x);
                    }
                },
                vec![],
            ))
        }
        DataType::LargeBinary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            Box::new(BufStreamingIterator::new(
                array.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.extend_from_slice(x);
                    }
                },
                vec![],
            ))
        }
        DataType::Dictionary(keys_dt, values_dt, _) => match &**values_dt {
            DataType::LargeUtf8 => match *keys_dt {
                IntegerType::UInt32 => serialize_utf8_dict::<u32, i64>(array.as_any()),
                IntegerType::UInt64 => serialize_utf8_dict::<u64, i64>(array.as_any()),
                _ => todo!(),
            },
            DataType::Utf8 => match *keys_dt {
                IntegerType::UInt32 => serialize_utf8_dict::<u32, i32>(array.as_any()),
                IntegerType::UInt64 => serialize_utf8_dict::<u64, i32>(array.as_any()),
                _ => todo!(),
            },
            _ => {
                panic!("only dictionary with string values are supported by csv writer")
            }
        },
        dt => panic!("data type: {dt:?} not supported by csv writer"),
    })
}

/// Helper for serializing a dictonary array. The generic parameters are:
/// - `K` for the type of the keys of the dictionary
/// - `O` for the type of the offsets in the Utf8Array: {i32, i64}
fn serialize_utf8_dict<'a, K: DictionaryKey, O: Offset>(
    array: &'a dyn Any,
) -> Box<dyn StreamingIterator<Item = [u8]> + 'a> {
    let array = array.downcast_ref::<DictionaryArray<K>>().unwrap();
    let values = array
        .values()
        .as_any()
        .downcast_ref::<Utf8Array<O>>()
        .unwrap();
    Box::new(BufStreamingIterator::new(
        array.keys_iter(),
        move |x, buf| {
            if let Some(i) = x {
                if !values.is_null(i) {
                    let val = values.value(i);
                    buf.extend_from_slice(val.as_bytes());
                }
            }
        },
        vec![],
    ))
}
