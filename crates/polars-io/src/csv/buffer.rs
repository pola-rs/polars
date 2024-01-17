use arrow::array::{MutableBinaryViewArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::legacy::prelude::FromDataUtf8;
use polars_core::prelude::*;
use polars_error::to_compute_err;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::chunkedarray::string::Pattern;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::prelude::string::infer::{
    infer_pattern_single, DatetimeInfer, StrpTimeParser, TryFromWithUnit,
};

use crate::csv::parser::{is_whitespace, skip_whitespace};
use crate::csv::utils::escape_field;
use crate::csv::CsvEncoding;

pub(crate) trait PrimitiveParser: PolarsNumericType {
    fn parse(bytes: &[u8]) -> Option<Self::Native>;
}

impl PrimitiveParser for Float32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f32> {
        fast_float::parse(bytes).ok()
    }
}
impl PrimitiveParser for Float64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f64> {
        fast_float::parse(bytes).ok()
    }
}

impl PrimitiveParser for UInt32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u32> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for UInt64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u64> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i32> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i64> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}

trait ParsedBuffer {
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()>;
}

impl<T> ParsedBuffer for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType + PrimitiveParser,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            self.append_null()
        } else {
            let bytes = if needs_escaping {
                &bytes[1..bytes.len() - 1]
            } else {
                bytes
            };

            // legacy comment (remember this if you decide to use Results again):
            // its faster to work on options.
            // if we need to throw an error, we parse again to be able to throw the error

            match T::parse(bytes) {
                Some(value) => self.append_value(value),
                None => {
                    // try again without whitespace
                    if !bytes.is_empty() && is_whitespace(bytes[0]) {
                        let bytes = skip_whitespace(bytes);
                        return self.parse_bytes(
                            bytes,
                            ignore_errors,
                            false, // escaping was already done
                            _missing_is_null,
                            None,
                        );
                    }
                    polars_ensure!(
                        bytes.is_empty() || ignore_errors,
                        ComputeError: "remaining bytes non-empty",
                    );
                    self.append_null()
                },
            };
        }
        Ok(())
    }
}

pub(crate) struct Utf8Field {
    name: String,
    mutable: MutableBinaryViewArray<str>,
    scratch: Vec<u8>,
    quote_char: u8,
    encoding: CsvEncoding,
    ignore_errors: bool,
}

impl Utf8Field {
    fn new(
        name: &str,
        capacity: usize,
        quote_char: Option<u8>,
        encoding: CsvEncoding,
        ignore_errors: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            mutable: MutableBinaryViewArray::with_capacity(capacity),
            scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            encoding,
            ignore_errors,
        }
    }
}

#[inline]
fn validate_utf8(bytes: &[u8]) -> bool {
    simdutf8::basic::from_utf8(bytes).is_ok()
}

impl ParsedBuffer for Utf8Field {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            if missing_is_null {
                self.mutable.push_null()
            } else {
                self.mutable.push(Some(""))
            }
            return Ok(());
        }

        let parse_result = validate_utf8(bytes);
        let data_len = self.mutable.len();

        // check if field fits in the str data buffer
        let remaining_capacity = self.mutable.capacity() - data_len;
        if remaining_capacity < bytes.len() {
            // exponential growth strategy
            self.mutable
                .reserve(std::cmp::max(self.mutable.capacity(), bytes.len()))
        }

        // note that one branch writes without updating the length, so we must do that later.
        let bytes = if needs_escaping {
            self.scratch.clear();
            self.scratch.reserve(bytes.len());
            polars_ensure!(bytes.len() > 1, ComputeError: "invalid csv file\n\nField `{}` is not properly escaped.", std::str::from_utf8(bytes).map_err(to_compute_err)?);
            // Safety:
            // we just allocated enough capacity and data_len is correct.
            unsafe { escape_field(bytes, self.quote_char, self.scratch.spare_capacity_mut()) };
            self.scratch.as_slice()
        } else {
            bytes
        };

        match parse_result {
            true => {
                let value= unsafe { std::str::from_utf8_unchecked(bytes) };
                self.mutable.push_value(value)
            },
            false => {
                if matches!(self.encoding, CsvEncoding::LossyUtf8) {
                    // TODO! do this without allocating
                    let s = String::from_utf8_lossy(bytes);
                    self.mutable.push_value(s.as_ref())
                } else if ignore_errors {
                    self.mutable.push_null()
                } else {
                    polars_bail!(ComputeError: "invalid utf-8 sequence");
                }
            },
        }

        Ok(())
    }
}

#[cfg(not(feature = "dtype-categorical"))]
pub(crate) struct CategoricalField {
    phantom: std::marker::PhantomData<u8>,
}

#[cfg(feature = "dtype-categorical")]
pub(crate) struct CategoricalField {
    escape_scratch: Vec<u8>,
    quote_char: u8,
    builder: CategoricalChunkedBuilder,
}

#[cfg(feature = "dtype-categorical")]
impl CategoricalField {
    fn new(
        name: &str,
        capacity: usize,
        quote_char: Option<u8>,
        ordering: CategoricalOrdering,
    ) -> Self {
        let builder = CategoricalChunkedBuilder::new(name, capacity, ordering);

        Self {
            escape_scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            builder,
        }
    }

    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            self.builder.append_null();
            return Ok(());
        }

        if validate_utf8(bytes) {
            if needs_escaping {
                polars_ensure!(bytes.len() > 1, ComputeError: "invalid csv file\n\nField `{}` is not properly escaped.", std::str::from_utf8(bytes).map_err(to_compute_err)?);
                self.escape_scratch.clear();
                self.escape_scratch.reserve(bytes.len());
                // Safety:
                // we just allocated enough capacity and data_len is correct.
                unsafe {
                    let n_written = escape_field(
                        bytes,
                        self.quote_char,
                        self.escape_scratch.spare_capacity_mut(),
                    );
                    self.escape_scratch.set_len(n_written);
                }

                // safety:
                // just did utf8 check
                let key = unsafe { std::str::from_utf8_unchecked(&self.escape_scratch) };
                self.builder.append_value(key);
            } else {
                // safety:
                // just did utf8 check
                unsafe {
                    self.builder
                        .append_value(std::str::from_utf8_unchecked(bytes))
                }
            }
        } else if ignore_errors {
            self.builder.append_null()
        } else {
            polars_bail!(ComputeError: "invalid utf-8 sequence");
        }
        Ok(())
    }
}

impl ParsedBuffer for BooleanChunkedBuilder {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        let bytes = if needs_escaping {
            &bytes[1..bytes.len() - 1]
        } else {
            bytes
        };
        if bytes.eq_ignore_ascii_case(b"false") {
            self.append_value(false);
        } else if bytes.eq_ignore_ascii_case(b"true") {
            self.append_value(true);
        } else if ignore_errors || bytes.is_empty() {
            self.append_null();
        } else {
            polars_bail!(
                ComputeError: "error while parsing value {} as boolean",
                String::from_utf8_lossy(bytes),
            );
        }
        Ok(())
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
pub(crate) struct DatetimeField<T: PolarsNumericType> {
    compiled: Option<DatetimeInfer<T>>,
    builder: PrimitiveChunkedBuilder<T>,
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
impl<T: PolarsNumericType> DatetimeField<T> {
    fn new(name: &str, capacity: usize) -> Self {
        let builder = PrimitiveChunkedBuilder::<T>::new(name, capacity);
        Self {
            compiled: None,
            builder,
        }
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
fn slow_datetime_parser<T>(
    buf: &mut DatetimeField<T>,
    bytes: &[u8],
    time_unit: Option<TimeUnit>,
    ignore_errors: bool,
) -> PolarsResult<()>
where
    T: PolarsNumericType,
    DatetimeInfer<T>: TryFromWithUnit<Pattern>,
{
    let val = if bytes.is_ascii() {
        // Safety:
        // we just checked it is ascii
        unsafe { std::str::from_utf8_unchecked(bytes) }
    } else {
        match std::str::from_utf8(bytes) {
            Ok(val) => val,
            Err(_) => {
                if ignore_errors {
                    buf.builder.append_null();
                    return Ok(());
                } else {
                    polars_bail!(ComputeError: "invalid utf-8 sequence");
                }
            },
        }
    };

    let pattern = match &buf.compiled {
        Some(compiled) => compiled.pattern,
        None => match infer_pattern_single(val) {
            Some(pattern) => pattern,
            None => {
                if ignore_errors {
                    buf.builder.append_null();
                    return Ok(());
                } else {
                    polars_bail!(ComputeError: "could not find a 'date/datetime' pattern for {}", val)
                }
            },
        },
    };
    match DatetimeInfer::try_from_with_unit(pattern, time_unit) {
        Ok(mut infer) => {
            let parsed = infer.parse(val);
            buf.compiled = Some(infer);
            buf.builder.append_option(parsed);
            Ok(())
        },
        Err(err) => {
            if ignore_errors {
                buf.builder.append_null();
                Ok(())
            } else {
                Err(err)
            }
        },
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
impl<T> ParsedBuffer for DatetimeField<T>
where
    T: PolarsNumericType,
    DatetimeInfer<T>: TryFromWithUnit<Pattern> + StrpTimeParser<T::Native>,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        mut bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if needs_escaping && bytes.len() > 2 {
            bytes = &bytes[1..bytes.len() - 1]
        }

        match &mut self.compiled {
            None => slow_datetime_parser(self, bytes, time_unit, ignore_errors),
            Some(compiled) => {
                match compiled.parse_bytes(bytes, time_unit) {
                    Some(parsed) => {
                        self.builder.append_value(parsed);
                        Ok(())
                    },
                    // fall back on chrono parser
                    // this is a lot slower, we need to do utf8 checking and use
                    // the slower parser
                    None => slow_datetime_parser(self, bytes, time_unit, ignore_errors),
                }
            },
        }
    }
}

pub(crate) fn init_buffers(
    projection: &[usize],
    capacity: usize,
    schema: &Schema,
    quote_char: Option<u8>,
    encoding: CsvEncoding,
    ignore_errors: bool,
) -> PolarsResult<Vec<Buffer>> {
    projection
        .iter()
        .map(|&i| {
            let (name, dtype) = schema.get_at_index(i).unwrap();
            let builder = match dtype {
                &DataType::Boolean => Buffer::Boolean(BooleanChunkedBuilder::new(name, capacity)),
                &DataType::Int32 => Buffer::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Int64 => Buffer::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt32 => Buffer::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt64 => Buffer::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Float32 => Buffer::Float32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Float64 => Buffer::Float64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::String => Buffer::Utf8(Utf8Field::new(
                    name,
                    capacity,
                    quote_char,
                    encoding,
                    ignore_errors,
                )),
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(time_unit, time_zone) => Buffer::Datetime {
                    buf: DatetimeField::new(name, capacity),
                    time_unit: *time_unit,
                    time_zone: time_zone.clone(),
                },
                #[cfg(feature = "dtype-date")]
                &DataType::Date => Buffer::Date(DatetimeField::new(name, capacity)),
                #[cfg(feature = "dtype-categorical")]
                DataType::Categorical(rev_map,ordering) => {
                    if let Some(rev_map) = &rev_map {
                        polars_ensure!(!rev_map.is_enum(),InvalidOperation: "user defined categoricals are not supported when reading csv")
                    }

                    Buffer::Categorical(CategoricalField::new(name, capacity, quote_char,*ordering))
                },
                dt => polars_bail!(
                    ComputeError: "unsupported data type when reading CSV: {} when reading CSV", dt,
                ),
            };
            Ok(builder)
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
    Utf8(Utf8Field),
    #[cfg(feature = "dtype-datetime")]
    Datetime {
        buf: DatetimeField<Int64Type>,
        time_unit: TimeUnit,
        time_zone: Option<TimeZone>,
    },
    #[cfg(feature = "dtype-date")]
    Date(DatetimeField<Int32Type>),
    #[allow(dead_code)]
    Categorical(CategoricalField),
}

impl Buffer {
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
            Buffer::Datetime {
                buf,
                time_unit,
                time_zone,
            } => buf
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Datetime(time_unit, time_zone))
                .unwrap(),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(v) => v
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Date)
                .unwrap(),

            Buffer::Utf8(mut v) => {
                let arr = v.mutable.freeze();
                StringChunked::with_chunk(v.name.as_str(), arr).into_series()
            },
            #[allow(unused_variables)]
            Buffer::Categorical(buf) => {
                #[cfg(feature = "dtype-categorical")]
                {
                    buf.builder.finish().into_series()
                }
                #[cfg(not(feature = "dtype-categorical"))]
                {
                    panic!("activate 'dtype-categorical' feature")
                }
            },
        };
        Ok(s)
    }

    pub(crate) fn add_null(&mut self, valid: bool) {
        match self {
            Buffer::Boolean(v) => v.append_null(),
            Buffer::Int32(v) => v.append_null(),
            Buffer::Int64(v) => v.append_null(),
            Buffer::UInt32(v) => v.append_null(),
            Buffer::UInt64(v) => v.append_null(),
            Buffer::Float32(v) => v.append_null(),
            Buffer::Float64(v) => v.append_null(),
            Buffer::Utf8(v) => {
                v.mutable.push_null()
            },
            #[cfg(feature = "dtype-datetime")]
            Buffer::Datetime { buf, .. } => buf.builder.append_null(),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(v) => v.builder.append_null(),
            #[allow(unused_variables)]
            Buffer::Categorical(cat_builder) => {
                #[cfg(feature = "dtype-categorical")]
                {
                    cat_builder.builder.append_null()
                }
                #[cfg(not(feature = "dtype-categorical"))]
                {
                    panic!("activate 'dtype-categorical' feature")
                }
            },
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
            Buffer::Utf8(_) => DataType::String,
            #[cfg(feature = "dtype-datetime")]
            Buffer::Datetime { time_unit, .. } => DataType::Datetime(*time_unit, None),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(_) => DataType::Date,
            Buffer::Categorical(_) => {
                #[cfg(feature = "dtype-categorical")]
                {
                    DataType::Categorical(None, Default::default())
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    panic!("activate 'dtype-categorical' feature")
                }
            },
        }
    }

    #[inline]
    pub(crate) fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        missing_is_null: bool,
    ) -> PolarsResult<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => <BooleanChunkedBuilder as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int32(buf) => <PrimitiveChunkedBuilder<Int32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int64(buf) => <PrimitiveChunkedBuilder<Int64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt64(buf) => <PrimitiveChunkedBuilder<UInt64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt32(buf) => <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Float32(buf) => <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Float64(buf) => <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Utf8(buf) => <Utf8Field as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-datetime")]
            Datetime { buf, time_unit, .. } => {
                <DatetimeField<Int64Type> as ParsedBuffer>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    Some(*time_unit),
                )
            },
            #[cfg(feature = "dtype-date")]
            Date(buf) => <DatetimeField<Int32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[allow(unused_variables)]
            Categorical(buf) => {
                #[cfg(feature = "dtype-categorical")]
                {
                    buf.parse_bytes(bytes, ignore_errors, needs_escaping, missing_is_null, None)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    panic!("activate 'dtype-categorical' feature")
                }
            },
        }
    }
}
