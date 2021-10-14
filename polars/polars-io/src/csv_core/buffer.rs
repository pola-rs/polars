use crate::csv::CsvEncoding;
use crate::csv_core::csv::RunningSize;
use crate::csv_core::parser::drop_quotes;
use crate::csv_core::utils::escape_field;
use arrow::array::Utf8Array;
use arrow::bitmap::MutableBitmap;
use polars_arrow::prelude::FromDataUtf8;
use polars_core::prelude::*;
use std::fmt::Debug;

trait ToPolarsError: Debug {
    fn to_polars_err(&self) -> PolarsError {
        PolarsError::ComputeError(
            format!(
                "Could not parse primitive type during csv parsing: {:?}.\
                This can occur when a column was inferred as integer type but we stumbled upon a floating point value\
                You could pass a predefined schema or set `with_ignore_parser_errors` to `true`",
                self
            )
            .into(),
        )
    }
}

impl ToPolarsError for lexical::Error {}

pub(crate) trait PrimitiveParser: PolarsNumericType {
    fn parse(bytes: &[u8]) -> Result<Self::Native>;
}

impl PrimitiveParser for Float32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<f32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Float64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<f64> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}

impl PrimitiveParser for UInt32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<u32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<u64> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<i32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<i64> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}

trait ParsedBuffer<T> {
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        start_pos: usize,
        _needs_escaping: bool,
    ) -> Result<()>;
}

impl<T> ParsedBuffer<T> for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType + PrimitiveParser,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _start_pos: usize,
        _needs_escaping: bool,
    ) -> Result<()> {
        if bytes.is_empty() {
            self.append_null()
        } else {
            let bytes = drop_quotes(bytes);
            // its faster to work on options.
            // if we need to throw an error, we parse again to be able to throw the error
            let result = T::parse(bytes).ok();

            match (result, ignore_errors) {
                (Some(value), _) => self.append_value(value),
                (None, true) => self.append_null(),
                (None, _) => {
                    T::parse(bytes)?;
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

fn delay_utf8_validation(encoding: CsvEncoding, ignore_errors: bool) -> bool {
    !matches!((encoding, ignore_errors), (CsvEncoding::LossyUtf8, true))
}

impl ParsedBuffer<Utf8Type> for Utf8Field {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _start_pos: usize,
        needs_escaping: bool,
    ) -> Result<()> {
        // Only for lossy utf8 we check utf8 now. Otherwise we check all utf8 at the end.
        let parse_result = if delay_utf8_validation(self.encoding, ignore_errors) {
            true
        } else {
            simdutf8::basic::from_utf8(bytes).is_ok()
        };
        let data_len = self.data.len();

        // check if field fits in the str data buffer
        let remaining_capacity = self.data.capacity() - data_len;
        if remaining_capacity < bytes.len() {
            // exponential growth strategy
            self.data
                .reserve(std::cmp::max(self.data.capacity(), bytes.len()))
        }
        let n_written = if needs_escaping {
            // Safety:
            // we just allocated enough capacity and data_len is correct.
            unsafe { escape_field(bytes, self.quote_char, &mut self.data[data_len..]) }
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
                    let s = String::from_utf8_lossy(
                        &self.data.as_slice()[data_len..data_len + n_written],
                    )
                    .into_owned();
                    let b = s.as_bytes();
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

impl ParsedBuffer<BooleanType> for BooleanChunkedBuilder {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        start_pos: usize,
        _needs_escaping: bool,
    ) -> Result<()> {
        if bytes.eq_ignore_ascii_case(b"false") {
            self.append_value(false);
        } else if bytes.eq_ignore_ascii_case(b"true") {
            self.append_value(true);
        } else if ignore_errors || bytes.is_empty() {
            self.append_null();
        } else {
            return Err(PolarsError::ComputeError(
                format!(
                    "Error while parsing value {} at byte position {} as boolean",
                    start_pos,
                    String::from_utf8_lossy(bytes)
                )
                .into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn init_buffers(
    projection: &[usize],
    capacity: usize,
    schema: &Schema,
    // The running statistic of the amount of bytes we must allocate per str column
    str_capacities: &[RunningSize],
    quote_char: Option<u8>,
    encoding: CsvEncoding,
    ignore_errors: bool,
) -> Result<Vec<Buffer>> {
    // we keep track of the string columns we have seen so that we can increment the index
    let mut str_index = 0;

    projection
        .iter()
        .map(|&i| {
            let field = schema.field(i).unwrap();
            let mut str_capacity = 0;
            // determine the needed capacity for this column
            if field.data_type() == &DataType::Utf8 {
                str_capacity = str_capacities[str_index].size_hint();
                str_index += 1;
            }

            let builder = match field.data_type() {
                &DataType::Boolean => {
                    Buffer::Boolean(BooleanChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::Int32 => {
                    Buffer::Int32(PrimitiveChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::Int64 => {
                    Buffer::Int64(PrimitiveChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::UInt32 => {
                    Buffer::UInt32(PrimitiveChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::UInt64 => {
                    Buffer::UInt64(PrimitiveChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::Float32 => {
                    Buffer::Float32(PrimitiveChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::Float64 => {
                    Buffer::Float64(PrimitiveChunkedBuilder::new(field.name(), capacity))
                }
                &DataType::Utf8 => Buffer::Utf8(Utf8Field::new(
                    field.name(),
                    capacity,
                    str_capacity,
                    quote_char,
                    encoding,
                    ignore_errors,
                )),
                other => {
                    return Err(PolarsError::ComputeError(
                        format!("Unsupported data type {:?} when reading a csv", other).into(),
                    ))
                }
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
            // Safety:
            // We already checked utf8 validity during parsing
            Buffer::Utf8(mut v) => unsafe {
                v.offsets.shrink_to_fit();
                v.data.shrink_to_fit();

                if delay_utf8_validation(v.encoding, v.ignore_errors) {
                    simdutf8::basic::from_utf8(v.data.as_slice()).map_err(|_| {
                        PolarsError::ComputeError("invalid utf8 data in csv".into())
                    })?;
                }

                let arr = Utf8Array::<i64>::from_data_unchecked_default(
                    v.offsets.into(),
                    v.data.into(),
                    Some(v.validity.into()),
                );
                let ca = Utf8Chunked::new_from_chunks(&v.name, vec![Arc::new(arr)]);
                ca.into_series()
            },
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
            Buffer::Utf8(v) => {
                v.offsets.push(v.data.len() as i64);
                v.validity.push(false);
            }
        };
    }

    #[inline]
    pub(crate) fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        start_pos: usize,
        needs_escaping: bool,
    ) -> Result<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => <BooleanChunkedBuilder as ParsedBuffer<BooleanType>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
                needs_escaping,
            ),
            Int32(buf) => {
                <PrimitiveChunkedBuilder<Int32Type> as ParsedBuffer<Int32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    needs_escaping,
                )
            }
            Int64(buf) => {
                <PrimitiveChunkedBuilder<Int64Type> as ParsedBuffer<Int64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    needs_escaping,
                )
            }
            UInt64(buf) => {
                <PrimitiveChunkedBuilder<UInt64Type> as ParsedBuffer<UInt64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    needs_escaping,
                )
            }
            UInt32(buf) => {
                <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuffer<UInt32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    needs_escaping,
                )
            }
            Float32(buf) => {
                <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer<Float32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    needs_escaping,
                )
            }
            Float64(buf) => {
                <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer<Float64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    needs_escaping,
                )
            }
            Utf8(buf) => <Utf8Field as ParsedBuffer<Utf8Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
                needs_escaping,
            ),
        }
    }
}
