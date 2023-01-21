use arrow::array::Utf8Array;
use arrow::bitmap::MutableBitmap;
use polars_arrow::prelude::FromDataUtf8;
use polars_core::prelude::*;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::chunkedarray::utf8::Pattern;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::prelude::utf8::infer::{infer_pattern_single, DatetimeInfer};

use crate::csv::parser::{is_whitespace, skip_whitespace};
use crate::csv::read_impl::RunningSize;
use crate::csv::utils::escape_field;
use crate::csv::CsvEncoding;

pub(crate) trait PrimitiveParser: PolarsNumericType {
    fn parse(bytes: &[u8]) -> Option<Self::Native>;
}

impl PrimitiveParser for Float32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f32> {
        lexical::parse(bytes).ok()
    }
}
impl PrimitiveParser for Float64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f64> {
        lexical::parse(bytes).ok()
    }
}

impl PrimitiveParser for UInt32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u32> {
        lexical::parse(bytes).ok()
    }
}
impl PrimitiveParser for UInt64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u64> {
        lexical::parse(bytes).ok()
    }
}
impl PrimitiveParser for Int32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i32> {
        lexical::parse(bytes).ok()
    }
}
impl PrimitiveParser for Int64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i64> {
        lexical::parse(bytes).ok()
    }
}

trait ParsedBuffer {
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _needs_escaping: bool,
        _missing_is_null: bool,
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
                            needs_escaping,
                            _missing_is_null,
                        );
                    }
                    if ignore_errors || bytes.is_empty() {
                        self.append_null()
                    } else {
                        return Err(PolarsError::ComputeError("".into()));
                    }
                }
            };
        }
        Ok(())
    }
}

pub(crate) struct Utf8Field {
    name: String,
    // buffer that holds the string data
    data: Vec<u8>,
    // offsets in the string data buffer
    offsets: Vec<i64>,
    validity: MutableBitmap,
    quote_char: u8,
    encoding: CsvEncoding,
    ignore_errors: bool,
}

impl Utf8Field {
    fn new(
        name: &str,
        capacity: usize,
        str_capacity: usize,
        quote_char: Option<u8>,
        encoding: CsvEncoding,
        ignore_errors: bool,
    ) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            name: name.to_string(),
            data: Vec::with_capacity(str_capacity),
            offsets,
            validity: MutableBitmap::with_capacity(capacity),
            quote_char: quote_char.unwrap_or(b'"'),
            encoding,
            ignore_errors,
        }
    }
}

/// We delay validation if we expect utf8 and no errors
/// In case of `ignore-error`
#[inline]
fn delay_utf8_validation(encoding: CsvEncoding, ignore_errors: bool) -> bool {
    !(matches!(encoding, CsvEncoding::LossyUtf8) || ignore_errors)
}

#[inline]
fn validate_utf8(bytes: &[u8]) -> bool {
    bytes.is_ascii() || simdutf8::basic::from_utf8(bytes).is_ok()
}

impl ParsedBuffer for Utf8Field {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        missing_is_null: bool,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            // append null
            self.offsets.push(self.data.len() as i64);
            self.validity.push(!missing_is_null);
            return Ok(());
        }

        // Only for lossy utf8 we check utf8 now. Otherwise we check all utf8 at the end.
        let parse_result = if delay_utf8_validation(self.encoding, ignore_errors) {
            true
        } else {
            validate_utf8(bytes)
        };
        let data_len = self.data.len();

        // check if field fits in the str data buffer
        let remaining_capacity = self.data.capacity() - data_len;
        if remaining_capacity < bytes.len() {
            // exponential growth strategy
            self.data
                .reserve(std::cmp::max(self.data.capacity(), bytes.len()))
        }

        // note that one branch writes without updating the length, so we must do that later.
        let n_written = if needs_escaping {
            // Safety:
            // we just allocated enough capacity and data_len is correct.
            unsafe { escape_field(bytes, self.quote_char, self.data.spare_capacity_mut()) }
        } else {
            self.data.extend_from_slice(bytes);
            bytes.len()
        };

        match parse_result {
            true => {
                // Soundness
                // the n_written from csv-core are now valid bytes so we can update the length.
                unsafe { self.data.set_len(data_len + n_written) }
                self.offsets.push(self.data.len() as i64);
                self.validity.push(true);
            }
            false => {
                if matches!(self.encoding, CsvEncoding::LossyUtf8) {
                    // Safety:
                    // we extended to data_len + n_writen
                    // so the bytes are initialized
                    debug_assert!(self.data.capacity() >= data_len + n_written);
                    let slice = unsafe {
                        self.data
                            .as_slice()
                            .get_unchecked(data_len..data_len + n_written)
                    };
                    let s = String::from_utf8_lossy(slice).into_owned();
                    let b = s.as_bytes();
                    // Make sure that we extend at the proper location,
                    // otherwise we append valid bytes to invalid utf8 bytes.
                    unsafe { self.data.set_len(data_len) }
                    self.data.extend_from_slice(b);
                    self.offsets.push(self.data.len() as i64);
                    self.validity.push(true);
                } else if ignore_errors {
                    // append null
                    self.offsets.push(self.data.len() as i64);
                    self.validity.push(false);
                } else {
                    return Err(PolarsError::ComputeError("invalid utf8 data".into()));
                }
            }
        }

        Ok(())
    }
}

#[cfg(not(feature = "dtype-categorical"))]
pub(crate) struct CategoricalField<'a> {
    phantom: std::marker::PhantomData<&'a u8>,
}

#[cfg(feature = "dtype-categorical")]
pub(crate) struct CategoricalField<'a> {
    escape_scratch: Vec<u8>,
    quote_char: u8,
    builder: CategoricalChunkedBuilder<'a>,
    owned_strings: Vec<String>,
}

#[cfg(feature = "dtype-categorical")]
impl<'a> CategoricalField<'a> {
    fn new(name: &str, capacity: usize, quote_char: Option<u8>) -> Self {
        let builder = CategoricalChunkedBuilder::new(name, capacity);

        Self {
            escape_scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            builder,
            owned_strings: vec![],
        }
    }

    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &'a [u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            self.builder.append_null();
            return Ok(());
        }

        if validate_utf8(bytes) {
            if needs_escaping {
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

                // now it gets a bit complicated
                // the categorical map has keys that have a lifetime in the `&bytes`
                // but we just wrote to a `escape_scratch`. The string values
                // there will be cleared next iteration/call, so we cannot use the
                // `key` naively
                //
                // if the `key` does not exist yet, we allocate a `String` and we store that in a
                // `Vec` that may grow. If the `Vec` reallocates, the pointers to the `String` will
                // still be valid.
                //
                // if the `key` does exist, we can simply insert the value, because the pointer of
                // the key will not be stored by the builder and may be short-lived
                if self.builder.exits(key) {
                    // Safety:
                    // extend lifetime, see rationale from above
                    let key = unsafe { std::mem::transmute::<&str, &'a str>(key) };
                    self.builder.append_value(key)
                } else {
                    let key_owned = key.to_string();

                    // ptr to the string value on the heap
                    let heap_ptr = key_owned.as_str().as_ptr();
                    let len = key_owned.len();
                    self.owned_strings.push(key_owned);
                    unsafe {
                        let str_slice = std::slice::from_raw_parts(heap_ptr, len);
                        let key = std::str::from_utf8_unchecked(str_slice);
                        // Safety:
                        // extend lifetime, see rationale from above
                        let key = std::mem::transmute::<&str, &'a str>(key);
                        self.builder.append_value(key)
                    }
                }
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
            return Err(PolarsError::ComputeError("invalid utf8 data".into()));
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
            return Err(PolarsError::ComputeError(
                format!(
                    "Error while parsing value {} as boolean",
                    String::from_utf8_lossy(bytes)
                )
                .into(),
            ));
        }
        Ok(())
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
pub(crate) struct DatetimeField<T: PolarsNumericType> {
    compiled: Option<DatetimeInfer<T::Native>>,
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
    ignore_errors: bool,
) -> PolarsResult<()>
where
    T: PolarsNumericType,
    DatetimeInfer<T::Native>: TryFrom<Pattern>,
{
    let val = if bytes.is_ascii() {
        // Safety:
        // we just checked it is ascii
        unsafe { std::str::from_utf8_unchecked(bytes) }
    } else if ignore_errors {
        buf.builder.append_null();
        return Ok(());
    } else if !ignore_errors && std::str::from_utf8(bytes).is_err() {
        return Err(PolarsError::ComputeError("invalid utf8".into()));
    } else {
        buf.builder.append_null();
        return Ok(());
    };

    match infer_pattern_single(val) {
        None => {
            buf.builder.append_null();
            Ok(())
        }
        Some(pattern) => match DatetimeInfer::<T::Native>::try_from(pattern) {
            Ok(mut infer) => {
                let parsed = infer.parse(val);
                buf.compiled = Some(infer);
                buf.builder.append_option(parsed);
                Ok(())
            }
            Err(_) => {
                buf.builder.append_null();
                Ok(())
            }
        },
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
impl<T> ParsedBuffer for DatetimeField<T>
where
    T: PolarsNumericType,
    DatetimeInfer<T::Native>: TryFrom<Pattern>,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        mut bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
    ) -> PolarsResult<()> {
        if needs_escaping && bytes.len() > 2 {
            bytes = &bytes[1..bytes.len() - 1]
        }

        match &mut self.compiled {
            None => slow_datetime_parser(self, bytes, ignore_errors),
            Some(compiled) => {
                match compiled.parse_bytes(bytes) {
                    Some(parsed) => {
                        self.builder.append_value(parsed);
                        Ok(())
                    }
                    // fall back on chrono parser
                    // this is a lot slower, we need to do utf8 checking and use
                    // the slower parser
                    None => slow_datetime_parser(self, bytes, ignore_errors),
                }
            }
        }
    }
}

pub(crate) fn init_buffers<'a>(
    projection: &[usize],
    capacity: usize,
    schema: &Schema,
    // The running statistic of the amount of bytes we must allocate per str column
    str_capacities: &[RunningSize],
    quote_char: Option<u8>,
    encoding: CsvEncoding,
    ignore_errors: bool,
) -> PolarsResult<Vec<Buffer<'a>>> {
    // we keep track of the string columns we have seen so that we can increment the index
    let mut str_index = 0;

    projection
        .iter()
        .map(|&i| {
            let (name, dtype) = schema.get_index(i).unwrap();
            let mut str_capacity = 0;
            // determine the needed capacity for this column
            if dtype == &DataType::Utf8 {
                str_capacity = str_capacities[str_index].size_hint();
                str_index += 1;
            }

            let builder = match dtype {
                &DataType::Boolean => Buffer::Boolean(BooleanChunkedBuilder::new(name, capacity)),
                &DataType::Int32 => Buffer::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Int64 => Buffer::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt32 => Buffer::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt64 => Buffer::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Float32 => Buffer::Float32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Float64 => Buffer::Float64(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Utf8 => Buffer::Utf8(Utf8Field::new(
                    name,
                    capacity,
                    str_capacity,
                    quote_char,
                    encoding,
                    ignore_errors,
                )),
                #[cfg(feature = "dtype-datetime")]
                &DataType::Datetime(tu, _) => Buffer::Datetime {
                    buf: DatetimeField::new(name, capacity),
                    tu,
                },
                #[cfg(feature = "dtype-date")]
                &DataType::Date => Buffer::Date(DatetimeField::new(name, capacity)),
                #[cfg(feature = "dtype-categorical")]
                &DataType::Categorical(_) => {
                    Buffer::Categorical(CategoricalField::new(name, capacity, quote_char))
                }
                other => {
                    return Err(PolarsError::ComputeError(
                        format!("Unsupported data type {other:?} when reading a csv").into(),
                    ))
                }
            };
            Ok(builder)
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
    /// Stores the Utf8 fields and the total string length seen for that column
    Utf8(Utf8Field),
    #[cfg(feature = "dtype-datetime")]
    Datetime {
        buf: DatetimeField<Int64Type>,
        tu: TimeUnit,
    },
    #[cfg(feature = "dtype-date")]
    Date(DatetimeField<Int32Type>),
    #[allow(dead_code)]
    Categorical(CategoricalField<'a>),
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
            Buffer::Datetime { buf, tu } => buf
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Datetime(tu, None))
                .unwrap(),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(v) => v
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Date)
                .unwrap(),
            // Safety:
            // We already checked utf8 validity during parsing
            Buffer::Utf8(mut v) => unsafe {
                v.offsets.shrink_to_fit();
                v.data.shrink_to_fit();

                let mut valid_utf8 = true;
                if delay_utf8_validation(v.encoding, v.ignore_errors) {
                    // check whole buffer for utf8
                    // this alone is not enough
                    // we must also check byte starts
                    // see: https://github.com/jorgecarleitao/arrow2/pull/823
                    simdutf8::basic::from_utf8(&v.data).map_err(|_| {
                        PolarsError::ComputeError("invalid utf8 data in csv".into())
                    })?;

                    for i in (0..v.offsets.len() - 1).step_by(2) {
                        // Safety:
                        // we iterate over offsets.len()
                        let start = *v.offsets.get_unchecked(i) as usize;

                        let first = v.data.get(start);

                        // A valid code-point iff it does not start with 0b10xxxxxx
                        // Bit-magic taken from `std::str::is_char_boundary`
                        if let Some(&b) = first {
                            if (b as i8) < -0x40 {
                                valid_utf8 = false;
                                break;
                            }
                        }
                    }

                    if !valid_utf8 {
                        return Err(PolarsError::ComputeError("invalid utf8 data in csv".into()));
                    }
                }

                let arr = Utf8Array::<i64>::from_data_unchecked_default(
                    v.offsets.into(),
                    v.data.into(),
                    Some(v.validity.into()),
                );
                let ca = Utf8Chunked::from_chunks(&v.name, vec![Box::new(arr)]);
                ca.into_series()
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
            }
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
                v.offsets.push(v.data.len() as i64);
                v.validity.push(valid);
            }
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
            }
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
            #[cfg(feature = "dtype-datetime")]
            Buffer::Datetime { tu, .. } => DataType::Datetime(*tu, None),
            #[cfg(feature = "dtype-date")]
            Buffer::Date(_) => DataType::Date,
            Buffer::Categorical(_) => {
                #[cfg(feature = "dtype-categorical")]
                {
                    DataType::Categorical(None)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    panic!("activate 'dtype-categorical' feature")
                }
            }
        }
    }

    #[inline]
    pub(crate) fn add(
        &mut self,
        bytes: &'a [u8],
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
            ),
            Int32(buf) => <PrimitiveChunkedBuilder<Int32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            Int64(buf) => <PrimitiveChunkedBuilder<Int64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            UInt64(buf) => <PrimitiveChunkedBuilder<UInt64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            UInt32(buf) => <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            Float32(buf) => <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            Float64(buf) => <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            Utf8(buf) => <Utf8Field as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            #[cfg(feature = "dtype-datetime")]
            Datetime { buf, .. } => <DatetimeField<Int64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            #[cfg(feature = "dtype-date")]
            Date(buf) => <DatetimeField<Int32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
            ),
            #[allow(unused_variables)]
            Categorical(buf) => {
                #[cfg(feature = "dtype-categorical")]
                {
                    buf.parse_bytes(bytes, ignore_errors, needs_escaping, missing_is_null)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    panic!("activate 'dtype-categorical' feature")
                }
            }
        }
    }
}
