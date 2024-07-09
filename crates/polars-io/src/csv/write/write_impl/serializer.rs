//! This file is complicated because we have complicated escape handling. We want to avoid having
//! to write down each combination of type & escaping, but we also want the compiler to optimize them
//! to efficient machine code - so no dynamic dispatch. That means a lot of generics and macros.
//!
//! We need to differentiate between several kinds of types, and several kinds of escaping we support:
//!
//!  - The simplest escaping mechanism are [`QuoteStyle::Always`] and [`QuoteStyle::Never`].
//!    For `Never` we just never quote. For `Always` we pass any serializer that never quotes
//!    to [`quote_serializer()`] then it becomes quoted properly.
//!  - [`QuoteStyle::Necessary`] (the default) is only relevant for strings, as it is the only type that
//!    can have newlines (row separators), commas (column separators) or quotes. String
//!    escaping is complicated anyway, and it is all inside [`string_serializer()`].
//!  - The real complication is [`QuoteStyle::NonNumeric`], that doesn't quote numbers and nulls,
//!    and quotes any other thing. The problem is that nulls can be within any type, so we need to handle
//!    two possibilities of quoting everywhere.
//!
//! So in case the chosen style is anything but `NonNumeric`, we statically know for each column except strings
//! whether it should be quoted (and for strings too when not `Necessary`). There we use `quote_serializer()`
//! or nothing.
//!
//! But to help with `NonNumeric`, each serializer carry the potential to distinguish between nulls and non-nulls,
//! and quote the later and not the former. But in order to not have the branch when we statically know the answer,
//! we have an option to statically disable it with a const generic flag `QUOTE_NON_NULL`. Numbers (that should never
//! be quoted with `NonNumeric`) just always disable this flag.
//!
//! So we have three possibilities:
//!
//!  1. A serializer that never quotes. This is a bare serializer with `QUOTE_NON_NULL = false`.
//!  2. A serializer that always quotes. This is a serializer wrapped with `quote_serializer()`,
//!     but also with `QUOTE_NON_NULL = false`.
//!  3. A serializer that quotes only non-nulls. This is a bare serializer with `QUOTE_NON_NULL = true`.

use std::fmt::LowerExp;
use std::io::Write;

use arrow::array::{Array, BooleanArray, NullArray, PrimitiveArray, Utf8ViewArray};
use arrow::legacy::time_zone::Tz;
use arrow::types::NativeType;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
use memchr::{memchr3, memchr_iter};
use num_traits::NumCast;
use polars_core::prelude::*;

use crate::csv::write::{QuoteStyle, SerializeOptions};

const TOO_MANY_MSG: &str = "too many items requested from CSV serializer";
const ARRAY_MISMATCH_MSG: &str = "wrong array type";

#[allow(dead_code)]
struct IgnoreFmt;
impl std::fmt::Write for IgnoreFmt {
    fn write_str(&mut self, _s: &str) -> std::fmt::Result {
        Ok(())
    }
}

pub(super) trait Serializer<'a> {
    fn serialize(&mut self, buf: &mut Vec<u8>, options: &SerializeOptions);
    // Updates the array without changing the configuration.
    fn update_array(&mut self, array: &'a dyn Array);
}

fn make_serializer<'a, T, I: Iterator<Item = Option<T>>, const QUOTE_NON_NULL: bool>(
    f: impl FnMut(T, &mut Vec<u8>, &SerializeOptions),
    iter: I,
    update_array: impl FnMut(&'a dyn Array) -> I,
) -> impl Serializer<'a> {
    struct SerializerImpl<F, I, Update, const QUOTE_NON_NULL: bool> {
        f: F,
        iter: I,
        update_array: Update,
    }

    impl<'a, T, F, I, Update, const QUOTE_NON_NULL: bool> Serializer<'a>
        for SerializerImpl<F, I, Update, QUOTE_NON_NULL>
    where
        F: FnMut(T, &mut Vec<u8>, &SerializeOptions),
        I: Iterator<Item = Option<T>>,
        Update: FnMut(&'a dyn Array) -> I,
    {
        fn serialize(&mut self, buf: &mut Vec<u8>, options: &SerializeOptions) {
            let item = self.iter.next().expect(TOO_MANY_MSG);
            match item {
                Some(item) => {
                    if QUOTE_NON_NULL {
                        buf.push(options.quote_char);
                    }
                    (self.f)(item, buf, options);
                    if QUOTE_NON_NULL {
                        buf.push(options.quote_char);
                    }
                },
                None => buf.extend_from_slice(options.null.as_bytes()),
            }
        }

        fn update_array(&mut self, array: &'a dyn Array) {
            self.iter = (self.update_array)(array);
        }
    }

    SerializerImpl::<_, _, _, QUOTE_NON_NULL> {
        f,
        iter,
        update_array,
    }
}

fn integer_serializer<I: NativeType + itoa::Integer>(array: &PrimitiveArray<I>) -> impl Serializer {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        let mut buffer = itoa::Buffer::new();
        let value = buffer.format(item);
        buf.extend_from_slice(value.as_bytes());
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<I>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

fn float_serializer_no_precision_autoformat<I: NativeType + ryu::Float>(
    array: &PrimitiveArray<I>,
) -> impl Serializer {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        let mut buffer = ryu::Buffer::new();
        let value = buffer.format(item);
        buf.extend_from_slice(value.as_bytes());
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<I>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

fn float_serializer_no_precision_scientific<I: NativeType + LowerExp>(
    array: &PrimitiveArray<I>,
) -> impl Serializer {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        // Float writing into a buffer of `Vec<u8>` cannot fail.
        let _ = write!(buf, "{item:.e}");
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<I>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

fn float_serializer_no_precision_positional<I: NativeType + NumCast>(
    array: &PrimitiveArray<I>,
) -> impl Serializer {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        let v: f64 = NumCast::from(item).unwrap();
        let value = v.to_string();
        buf.extend_from_slice(value.as_bytes());
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<I>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

fn float_serializer_with_precision_scientific<I: NativeType + LowerExp>(
    array: &PrimitiveArray<I>,
    precision: usize,
) -> impl Serializer {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        // Float writing into a buffer of `Vec<u8>` cannot fail.
        let _ = write!(buf, "{item:.precision$e}");
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<I>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

fn float_serializer_with_precision_positional<I: NativeType>(
    array: &PrimitiveArray<I>,
    precision: usize,
) -> impl Serializer {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        // Float writing into a buffer of `Vec<u8>` cannot fail.
        let _ = write!(buf, "{item:.precision$}");
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<I>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

fn null_serializer(_array: &NullArray) -> impl Serializer {
    struct NullSerializer;
    impl<'a> Serializer<'a> for NullSerializer {
        fn serialize(&mut self, buf: &mut Vec<u8>, options: &SerializeOptions) {
            buf.extend_from_slice(options.null.as_bytes());
        }
        fn update_array(&mut self, _array: &'a dyn Array) {}
    }
    NullSerializer
}

fn bool_serializer<const QUOTE_NON_NULL: bool>(array: &BooleanArray) -> impl Serializer {
    let f = move |item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        let s = if item { "true" } else { "false" };
        buf.extend_from_slice(s.as_bytes());
    };

    make_serializer::<_, _, QUOTE_NON_NULL>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

#[cfg(feature = "dtype-decimal")]
fn decimal_serializer(array: &PrimitiveArray<i128>, scale: usize) -> impl Serializer {
    let trim_zeros = arrow::compute::decimal::get_trim_decimal_zeros();

    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        let value = arrow::compute::decimal::format_decimal(item, scale, trim_zeros);
        buf.extend_from_slice(value.as_str().as_bytes());
    };

    make_serializer::<_, _, false>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<i128>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-time",
    feature = "dtype-datetime"
))]
fn callback_serializer<'a, T: NativeType, const QUOTE_NON_NULL: bool>(
    array: &'a PrimitiveArray<T>,
    mut callback: impl FnMut(T, &mut Vec<u8>) + 'a,
) -> impl Serializer + 'a {
    let f = move |&item, buf: &mut Vec<u8>, _options: &SerializeOptions| {
        callback(item, buf);
    };

    make_serializer::<_, _, QUOTE_NON_NULL>(f, array.iter(), |array| {
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect(ARRAY_MISMATCH_MSG)
            .iter()
    })
}

#[cfg(any(feature = "dtype-date", feature = "dtype-time"))]
type ChronoFormatIter<'a, 'b> = std::slice::Iter<'a, chrono::format::Item<'b>>;

#[cfg(any(feature = "dtype-date", feature = "dtype-time"))]
fn date_and_time_serializer<'a, Underlying: NativeType, T: std::fmt::Display>(
    format_str: &'a Option<String>,
    description: &str,
    array: &'a dyn Array,
    sample_value: T,
    mut convert: impl FnMut(Underlying) -> T + Send + 'a,
    mut format_fn: impl for<'b> FnMut(
            &T,
            ChronoFormatIter<'b, 'a>,
        ) -> chrono::format::DelayedFormat<ChronoFormatIter<'b, 'a>>
        + Send
        + 'a,
    options: &SerializeOptions,
) -> PolarsResult<Box<dyn Serializer<'a> + Send + 'a>> {
    let array = array.as_any().downcast_ref().unwrap();
    let serializer = match format_str {
        Some(format_str) => {
            let format = chrono::format::StrftimeItems::new(format_str).parse().map_err(
                |_| polars_err!(ComputeError: "cannot format {description} with format '{format_str}'"),
            )?;
            use std::fmt::Write;
            // Fail fast for invalid format. This return error faster to the user, and allows us to not return
            // `Result` from `serialize()`.
            write!(IgnoreFmt, "{}", format_fn(&sample_value, format.iter())).map_err(
                |_| polars_err!(ComputeError: "cannot format {description} with format '{format_str}'"),
            )?;
            let callback = move |item, buf: &mut Vec<u8>| {
                let item = convert(item);
                // We checked the format is valid above.
                let _ = write!(buf, "{}", format_fn(&item, format.iter()));
            };
            date_and_time_final_serializer(array, callback, options)
        },
        None => {
            let callback = move |item, buf: &mut Vec<u8>| {
                let item = convert(item);
                // Formatting dates into `Vec<u8>` cannot fail.
                let _ = write!(buf, "{item}");
            };
            date_and_time_final_serializer(array, callback, options)
        },
    };
    Ok(serializer)
}

#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-time",
    feature = "dtype-datetime"
))]
fn date_and_time_final_serializer<'a, T: NativeType>(
    array: &'a PrimitiveArray<T>,
    callback: impl FnMut(T, &mut Vec<u8>) + Send + 'a,
    options: &SerializeOptions,
) -> Box<dyn Serializer<'a> + Send + 'a> {
    match options.quote_style {
        QuoteStyle::Always => Box::new(quote_serializer(callback_serializer::<T, false>(
            array, callback,
        ))) as Box<dyn Serializer + Send>,
        QuoteStyle::NonNumeric => Box::new(callback_serializer::<T, true>(array, callback)),
        _ => Box::new(callback_serializer::<T, false>(array, callback)),
    }
}

pub(super) fn string_serializer<'a, Iter: Send + 'a>(
    mut f: impl FnMut(&mut Iter) -> Option<&str> + Send + 'a,
    options: &SerializeOptions,
    mut update: impl FnMut(&'a dyn Array) -> Iter + Send + 'a,
    array: &'a dyn Array,
) -> Box<dyn Serializer<'a> + 'a + Send> {
    const LF: u8 = b'\n';
    const CR: u8 = b'\r';

    struct StringSerializer<F, Iter, Update> {
        serialize: F,
        update: Update,
        iter: Iter,
    }

    impl<'a, F, Iter, Update> Serializer<'a> for StringSerializer<F, Iter, Update>
    where
        F: FnMut(&mut Iter, &mut Vec<u8>, &SerializeOptions),
        Update: FnMut(&'a dyn Array) -> Iter,
    {
        fn serialize(&mut self, buf: &mut Vec<u8>, options: &SerializeOptions) {
            (self.serialize)(&mut self.iter, buf, options);
        }

        fn update_array(&mut self, array: &'a dyn Array) {
            self.iter = (self.update)(array);
        }
    }

    fn serialize_str_escaped(buf: &mut Vec<u8>, s: &[u8], quote_char: u8, quoted: bool) {
        let mut iter = memchr_iter(quote_char, s);
        let first_quote = iter.next();
        match first_quote {
            None => buf.extend_from_slice(s),
            Some(mut quote_pos) => {
                if !quoted {
                    buf.push(quote_char);
                }
                let mut start_pos = 0;
                loop {
                    buf.extend_from_slice(&s[start_pos..quote_pos]);
                    buf.extend_from_slice(&[quote_char, quote_char]);
                    match iter.next() {
                        Some(quote) => {
                            start_pos = quote_pos + 1;
                            quote_pos = quote;
                        },
                        None => {
                            buf.extend_from_slice(&s[quote_pos + 1..]);
                            break;
                        },
                    }
                }
                if !quoted {
                    buf.push(quote_char);
                }
            },
        }
    }

    let iter = update(array);
    match options.quote_style {
        QuoteStyle::Always => {
            let serialize =
                move |iter: &mut Iter, buf: &mut Vec<u8>, options: &SerializeOptions| {
                    let quote_char = options.quote_char;
                    buf.push(quote_char);
                    let Some(s) = f(iter) else {
                        buf.extend_from_slice(options.null.as_bytes());
                        buf.push(quote_char);
                        return;
                    };
                    serialize_str_escaped(buf, s.as_bytes(), quote_char, true);
                    buf.push(quote_char);
                };
            Box::new(StringSerializer {
                serialize,
                update,
                iter,
            })
        },
        QuoteStyle::NonNumeric => {
            let serialize =
                move |iter: &mut Iter, buf: &mut Vec<u8>, options: &SerializeOptions| {
                    let Some(s) = f(iter) else {
                        buf.extend_from_slice(options.null.as_bytes());
                        return;
                    };
                    let quote_char = options.quote_char;
                    buf.push(quote_char);
                    serialize_str_escaped(buf, s.as_bytes(), quote_char, true);
                    buf.push(quote_char);
                };
            Box::new(StringSerializer {
                serialize,
                update,
                iter,
            })
        },
        QuoteStyle::Necessary => {
            let serialize =
                move |iter: &mut Iter, buf: &mut Vec<u8>, options: &SerializeOptions| {
                    let Some(s) = f(iter) else {
                        buf.extend_from_slice(options.null.as_bytes());
                        return;
                    };
                    let quote_char = options.quote_char;
                    // An empty string conflicts with null, so it is necessary to quote.
                    if s.is_empty() {
                        buf.extend_from_slice(&[quote_char, quote_char]);
                        return;
                    }
                    let needs_quote = memchr3(options.separator, LF, CR, s.as_bytes()).is_some();
                    if needs_quote {
                        buf.push(quote_char);
                    }
                    serialize_str_escaped(buf, s.as_bytes(), quote_char, needs_quote);
                    if needs_quote {
                        buf.push(quote_char);
                    }
                };
            Box::new(StringSerializer {
                serialize,
                update,
                iter,
            })
        },
        QuoteStyle::Never => {
            let serialize =
                move |iter: &mut Iter, buf: &mut Vec<u8>, options: &SerializeOptions| {
                    let Some(s) = f(iter) else {
                        buf.extend_from_slice(options.null.as_bytes());
                        return;
                    };
                    buf.extend_from_slice(s.as_bytes());
                };
            Box::new(StringSerializer {
                serialize,
                update,
                iter,
            })
        },
    }
}

fn quote_serializer<'a>(serializer: impl Serializer<'a>) -> impl Serializer<'a> {
    struct QuoteSerializer<S>(S);
    impl<'a, S: Serializer<'a>> Serializer<'a> for QuoteSerializer<S> {
        fn serialize(&mut self, buf: &mut Vec<u8>, options: &SerializeOptions) {
            buf.push(options.quote_char);
            self.0.serialize(buf, options);
            buf.push(options.quote_char);
        }

        fn update_array(&mut self, array: &'a dyn Array) {
            self.0.update_array(array);
        }
    }
    QuoteSerializer(serializer)
}

pub(super) fn serializer_for<'a>(
    array: &'a dyn Array,
    options: &'a SerializeOptions,
    dtype: &'a DataType,
    _datetime_format: &'a str,
    _time_zone: Option<Tz>,
) -> PolarsResult<Box<dyn Serializer<'a> + Send + 'a>> {
    macro_rules! quote_if_always {
        ($make_serializer:path, $($arg:tt)*) => {{
            let serializer = $make_serializer(array.as_any().downcast_ref().unwrap(), $($arg)*);
            if let QuoteStyle::Always = options.quote_style {
                Box::new(quote_serializer(serializer)) as Box<dyn Serializer + Send>
            } else {
                Box::new(serializer)
            }
        }};
        ($make_serializer:path) => { quote_if_always!($make_serializer,) };
    }

    let serializer = match dtype {
        DataType::Int8 => quote_if_always!(integer_serializer::<i8>),
        DataType::UInt8 => quote_if_always!(integer_serializer::<u8>),
        DataType::Int16 => quote_if_always!(integer_serializer::<i16>),
        DataType::UInt16 => quote_if_always!(integer_serializer::<u16>),
        DataType::Int32 => quote_if_always!(integer_serializer::<i32>),
        DataType::UInt32 => quote_if_always!(integer_serializer::<u32>),
        DataType::Int64 => quote_if_always!(integer_serializer::<i64>),
        DataType::UInt64 => quote_if_always!(integer_serializer::<u64>),
        DataType::Float32 => match options.float_precision {
            Some(precision) => match options.float_scientific {
                Some(true) => {
                    quote_if_always!(float_serializer_with_precision_scientific::<f32>, precision)
                },
                _ => quote_if_always!(float_serializer_with_precision_positional::<f32>, precision),
            },
            None => match options.float_scientific {
                Some(true) => quote_if_always!(float_serializer_no_precision_scientific::<f32>),
                Some(false) => quote_if_always!(float_serializer_no_precision_positional::<f32>),
                None => quote_if_always!(float_serializer_no_precision_autoformat::<f32>),
            },
        },
        DataType::Float64 => match options.float_precision {
            Some(precision) => match options.float_scientific {
                Some(true) => {
                    quote_if_always!(float_serializer_with_precision_scientific::<f64>, precision)
                },
                _ => quote_if_always!(float_serializer_with_precision_positional::<f64>, precision),
            },
            None => match options.float_scientific {
                Some(true) => quote_if_always!(float_serializer_no_precision_scientific::<f64>),
                Some(false) => quote_if_always!(float_serializer_no_precision_positional::<f64>),
                None => quote_if_always!(float_serializer_no_precision_autoformat::<f64>),
            },
        },
        DataType::Null => quote_if_always!(null_serializer),
        DataType::Boolean => {
            let array = array.as_any().downcast_ref().unwrap();
            match options.quote_style {
                QuoteStyle::Always => Box::new(quote_serializer(bool_serializer::<false>(array)))
                    as Box<dyn Serializer + Send>,
                QuoteStyle::NonNumeric => Box::new(bool_serializer::<true>(array)),
                _ => Box::new(bool_serializer::<false>(array)),
            }
        },
        #[cfg(feature = "dtype-date")]
        DataType::Date => date_and_time_serializer(
            &options.date_format,
            "NaiveDate",
            array,
            chrono::NaiveDate::MAX,
            arrow::temporal_conversions::date32_to_date,
            |date, items| date.format_with_items(items),
            options,
        )?,
        #[cfg(feature = "dtype-time")]
        DataType::Time => date_and_time_serializer(
            &options.time_format,
            "NaiveTime",
            array,
            chrono::NaiveTime::MIN,
            arrow::temporal_conversions::time64ns_to_time,
            |time, items| time.format_with_items(items),
            options,
        )?,
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(time_unit, _) => {
            let format = chrono::format::StrftimeItems::new(_datetime_format)
                .parse()
                .map_err(|_| {
                    polars_err!(
                        ComputeError: "cannot format {} with format '{_datetime_format}'",
                        if _time_zone.is_some() { "DateTime" } else { "NaiveDateTime" },
                    )
                })?;
            use std::fmt::Write;
            let sample_datetime = match _time_zone {
                #[cfg(feature = "timezones")]
                Some(time_zone) => time_zone
                    .from_utc_datetime(&chrono::NaiveDateTime::MAX)
                    .format_with_items(format.iter()),
                #[cfg(not(feature = "timezones"))]
                Some(_) => panic!("activate 'timezones' feature"),
                None => chrono::NaiveDateTime::MAX.format_with_items(format.iter()),
            };
            // Fail fast for invalid format. This return error faster to the user, and allows us to not return
            // `Result` from `serialize()`.
            write!(IgnoreFmt, "{sample_datetime}").map_err(|_| {
                polars_err!(
                    ComputeError: "cannot format {} with format '{_datetime_format}'",
                    if _time_zone.is_some() { "DateTime" } else { "NaiveDateTime" },
                )
            })?;

            let array = array.as_any().downcast_ref().unwrap();

            macro_rules! time_unit_serializer {
                ($convert:ident) => {
                    match _time_zone {
                        #[cfg(feature = "timezones")]
                        Some(time_zone) => {
                            let callback = move |item, buf: &mut Vec<u8>| {
                                let item = arrow::temporal_conversions::$convert(item);
                                let item = time_zone.from_utc_datetime(&item);
                                // We checked the format is valid above.
                                let _ = write!(buf, "{}", item.format_with_items(format.iter()));
                            };
                            date_and_time_final_serializer(array, callback, options)
                        },
                        #[cfg(not(feature = "timezones"))]
                        Some(_) => panic!("activate 'timezones' feature"),
                        None => {
                            let callback = move |item, buf: &mut Vec<u8>| {
                                let item = arrow::temporal_conversions::$convert(item);
                                // We checked the format is valid above.
                                let _ = write!(buf, "{}", item.format_with_items(format.iter()));
                            };
                            date_and_time_final_serializer(array, callback, options)
                        },
                    }
                };
            }

            match time_unit {
                TimeUnit::Nanoseconds => time_unit_serializer!(timestamp_ns_to_datetime),
                TimeUnit::Microseconds => time_unit_serializer!(timestamp_us_to_datetime),
                TimeUnit::Milliseconds => time_unit_serializer!(timestamp_ms_to_datetime),
            }
        },
        DataType::String => string_serializer(
            |iter| Iterator::next(iter).expect(TOO_MANY_MSG),
            options,
            |arr| {
                arr.as_any()
                    .downcast_ref::<Utf8ViewArray>()
                    .expect(ARRAY_MISMATCH_MSG)
                    .iter()
            },
            array,
        ),
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(rev_map, _) | DataType::Enum(rev_map, _) => {
            let rev_map = rev_map.as_deref().unwrap();
            string_serializer(
                |iter| {
                    let &idx: &u32 = Iterator::next(iter).expect(TOO_MANY_MSG)?;
                    Some(rev_map.get(idx))
                },
                options,
                |arr| {
                    arr.as_any()
                        .downcast_ref::<PrimitiveArray<u32>>()
                        .expect(ARRAY_MISMATCH_MSG)
                        .iter()
                },
                array,
            )
        },
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, scale) => {
            let array = array.as_any().downcast_ref().unwrap();
            match options.quote_style {
                QuoteStyle::Never => Box::new(decimal_serializer(array, scale.unwrap_or(0)))
                    as Box<dyn Serializer + Send>,
                _ => Box::new(quote_serializer(decimal_serializer(
                    array,
                    scale.unwrap_or(0),
                ))),
            }
        },
        _ => polars_bail!(ComputeError: "datatype {dtype} cannot be written to csv"),
    };
    Ok(serializer)
}

#[cfg(test)]
mod test {
    use arrow::array::NullArray;
    use polars_core::prelude::ArrowDataType;

    use super::string_serializer;
    use crate::csv::write::options::{QuoteStyle, SerializeOptions};

    // It is the most complex serializer with most edge cases, it definitely needs a comprehensive test.
    #[test]
    fn test_string_serializer() {
        #[track_caller]
        fn check_string_serialization(options: &SerializeOptions, s: Option<&str>, expected: &str) {
            let fake_array = NullArray::new(ArrowDataType::Null, 0);
            let mut serializer = string_serializer(|s| *s, options, |_| s, &fake_array);
            let mut buf = Vec::new();
            serializer.serialize(&mut buf, options);
            let serialized = std::str::from_utf8(&buf).unwrap();
            // Don't use `assert_eq!()` because it prints debug format and it's hard to read with all the escapes.
            if serialized != expected {
                panic!("CSV string {s:?} wasn't serialized correctly: expected: `{expected}`, got: `{serialized}`");
            }
        }

        let always_quote = SerializeOptions {
            quote_style: QuoteStyle::Always,
            ..SerializeOptions::default()
        };
        check_string_serialization(&always_quote, None, r#""""#);
        check_string_serialization(&always_quote, Some(""), r#""""#);
        check_string_serialization(&always_quote, Some("a"), r#""a""#);
        check_string_serialization(&always_quote, Some("\""), r#""""""#);
        check_string_serialization(&always_quote, Some("a\"\"b"), r#""a""""b""#);

        let necessary_quote = SerializeOptions {
            quote_style: QuoteStyle::Necessary,
            ..SerializeOptions::default()
        };
        check_string_serialization(&necessary_quote, None, r#""#);
        check_string_serialization(&necessary_quote, Some(""), r#""""#);
        check_string_serialization(&necessary_quote, Some("a"), r#"a"#);
        check_string_serialization(&necessary_quote, Some("\""), r#""""""#);
        check_string_serialization(&necessary_quote, Some("a\"\"b"), r#""a""""b""#);
        check_string_serialization(&necessary_quote, Some("a b"), r#"a b"#);
        check_string_serialization(&necessary_quote, Some("a,b"), r#""a,b""#);
        check_string_serialization(&necessary_quote, Some("a\nb"), "\"a\nb\"");
        check_string_serialization(&necessary_quote, Some("a\rb"), "\"a\rb\"");

        let never_quote = SerializeOptions {
            quote_style: QuoteStyle::Never,
            ..SerializeOptions::default()
        };
        check_string_serialization(&never_quote, None, "");
        check_string_serialization(&never_quote, Some(""), "");
        check_string_serialization(&never_quote, Some("a"), "a");
        check_string_serialization(&never_quote, Some("\""), "\"");
        check_string_serialization(&never_quote, Some("a\"\"b"), "a\"\"b");
        check_string_serialization(&never_quote, Some("a b"), "a b");
        check_string_serialization(&never_quote, Some("a,b"), "a,b");
        check_string_serialization(&never_quote, Some("a\nb"), "a\nb");
        check_string_serialization(&never_quote, Some("a\rb"), "a\rb");

        let non_numeric_quote = SerializeOptions {
            quote_style: QuoteStyle::NonNumeric,
            ..SerializeOptions::default()
        };
        check_string_serialization(&non_numeric_quote, None, "");
        check_string_serialization(&non_numeric_quote, Some(""), r#""""#);
        check_string_serialization(&non_numeric_quote, Some("a"), r#""a""#);
        check_string_serialization(&non_numeric_quote, Some("\""), r#""""""#);
        check_string_serialization(&non_numeric_quote, Some("a\"\"b"), r#""a""""b""#);
        check_string_serialization(&non_numeric_quote, Some("a b"), r#""a b""#);
        check_string_serialization(&non_numeric_quote, Some("a,b"), r#""a,b""#);
        check_string_serialization(&non_numeric_quote, Some("a\nb"), "\"a\nb\"");
        check_string_serialization(&non_numeric_quote, Some("a\rb"), "\"a\rb\"");
    }
}
