use crate::csv::CsvEncoding;
use crate::csv_core::csv::RunningSize;
use crate::csv_core::parser::{drop_quotes, skip_whitespace};
use polars_core::prelude::*;
use std::fmt::Debug;

trait ToPolarsError: Debug {
    fn to_polars_err(&self) -> PolarsError {
        PolarsError::Other(
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
impl ToPolarsError for fast_float::Error {}

pub(crate) trait PrimitiveParser: ArrowPrimitiveType {
    fn parse(bytes: &[u8]) -> Result<Self::Native>;
}

impl PrimitiveParser for Float32Type {
    fn parse(bytes: &[u8]) -> Result<f32> {
        let a = fast_float::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Float64Type {
    fn parse(bytes: &[u8]) -> Result<f64> {
        let a = fast_float::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}

impl PrimitiveParser for UInt32Type {
    fn parse(bytes: &[u8]) -> Result<u32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt64Type {
    fn parse(bytes: &[u8]) -> Result<u64> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int32Type {
    fn parse(bytes: &[u8]) -> Result<i32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int64Type {
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
        encoding: CsvEncoding,
    ) -> Result<()>;
}

impl<T> ParsedBuffer<T> for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType + PrimitiveParser,
{
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _start_pos: usize,
        _encoding: CsvEncoding,
    ) -> Result<()> {
        let (bytes, _) = skip_whitespace(bytes);
        let bytes = drop_quotes(bytes);
        let result = T::parse(bytes);

        match (result, ignore_errors) {
            (Ok(value), _) => self.append_value(value),
            (Err(_), true) => self.append_null(),
            (Err(err), _) => {
                if bytes.is_empty() {
                    self.append_null()
                } else {
                    return Err(err);
                }
            }
        };
        Ok(())
    }
}

pub(crate) struct Utf8Field {
    builder: Utf8ChunkedBuilder,
    // buffer that is used as output buffer for csv-core
    string_buf: Vec<u8>,
    rdr: csv_core::Reader,
}

impl Utf8Field {
    fn new(name: &str, capacity: usize, str_capacity: usize, delimiter: u8) -> Self {
        Self {
            builder: Utf8ChunkedBuilder::new(name, capacity, str_capacity),
            string_buf: vec![0; 256],
            rdr: csv_core::ReaderBuilder::new().delimiter(delimiter).build(),
        }
    }
}

impl ParsedBuffer<Utf8Type> for Utf8Field {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _start_pos: usize,
        encoding: CsvEncoding,
    ) -> Result<()> {
        let bytes = unsafe {
            // csv core expects the delimiter for its state machine, but we already split on that.
            // so we extend the slice by one to include the delimiter
            // Safety:
            // A field always has a delimiter OR a end of line char so we can extend the field
            // without accessing unowned memory
            let ptr = bytes.as_ptr();
            std::slice::from_raw_parts(ptr, bytes.len() + 1)
        };

        if bytes.len() > self.string_buf.capacity() {
            self.string_buf
                .resize(std::cmp::max(bytes.len(), self.string_buf.capacity()), 0);
        }
        let (_, _, n_end) = self.rdr.read_field(bytes, &mut self.string_buf);

        let bytes = unsafe {
            // SAFETY
            // we know that n_end never will be larger than our output buffer
            self.string_buf.get_unchecked(..n_end)
        };

        let parse_result =
            std::str::from_utf8(bytes).map_err(|_| PolarsError::Other("invalid utf8 data".into()));
        match parse_result {
            Ok(s) => {
                self.builder.append_value(s);
            }
            Err(err) => {
                if matches!(encoding, CsvEncoding::LossyUtf8) {
                    let s = String::from_utf8_lossy(bytes);
                    self.builder.append_value(s.as_ref());
                } else if ignore_errors {
                    self.builder.append_null();
                } else {
                    return Err(err);
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
        _encoding: CsvEncoding,
    ) -> Result<()> {
        if bytes.eq_ignore_ascii_case(b"false") {
            self.append_value(false);
        } else if bytes.eq_ignore_ascii_case(b"true") {
            self.append_value(true);
        } else if ignore_errors || bytes.is_empty() {
            self.append_null();
        } else {
            return Err(PolarsError::Other(
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
    schema: &SchemaRef,
    // The running statistic of the amount of bytes we must allocate per str column
    str_capacities: &[RunningSize],
    delimiter: u8,
) -> Result<Vec<Buffer>> {
    // we keep track of the string columns we have seen so that we can increment the index
    let mut str_index = 0;

    projection
        .iter()
        .map(|&i| {
            let field = schema.field(i).unwrap();
            let mut str_capacity = 0;
            // determine the needed capacity for this column
            // we overallocate 20%
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
                #[cfg(feature = "dtype-u64")]
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
                    delimiter,
                )),
                other => {
                    return Err(PolarsError::Other(
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
    #[cfg(feature = "dtype-u64")]
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    /// Stores the Utf8 fields and the total string length seen for that column
    Utf8(Utf8Field),
}

impl Buffer {
    pub(crate) fn into_series(self) -> Series {
        match self {
            Buffer::Boolean(v) => v.finish().into_series(),
            Buffer::Int32(v) => v.finish().into_series(),
            Buffer::Int64(v) => v.finish().into_series(),
            Buffer::UInt32(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-u64")]
            Buffer::UInt64(v) => v.finish().into_series(),
            Buffer::Float32(v) => v.finish().into_series(),
            Buffer::Float64(v) => v.finish().into_series(),
            Buffer::Utf8(v) => v.builder.finish().into_series(),
        }
    }

    pub(crate) fn add_null(&mut self) {
        match self {
            Buffer::Boolean(v) => v.append_null(),
            Buffer::Int32(v) => v.append_null(),
            Buffer::Int64(v) => v.append_null(),
            Buffer::UInt32(v) => v.append_null(),
            #[cfg(feature = "dtype-u64")]
            Buffer::UInt64(v) => v.append_null(),
            Buffer::Float32(v) => v.append_null(),
            Buffer::Float64(v) => v.append_null(),
            Buffer::Utf8(v) => v.builder.append_null(),
        };
    }

    #[inline]
    pub(crate) fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        start_pos: usize,
        encoding: CsvEncoding,
    ) -> Result<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => <BooleanChunkedBuilder as ParsedBuffer<BooleanType>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
                encoding,
            ),
            Int32(buf) => {
                <PrimitiveChunkedBuilder<Int32Type> as ParsedBuffer<Int32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                )
            }
            Int64(buf) => {
                <PrimitiveChunkedBuilder<Int64Type> as ParsedBuffer<Int64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                )
            }
            #[cfg(feature = "dtype-u64")]
            UInt64(buf) => {
                <PrimitiveChunkedBuilder<UInt64Type> as ParsedBuffer<UInt64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                )
            }
            UInt32(buf) => {
                <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuffer<UInt32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                )
            }
            Float32(buf) => {
                <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer<Float32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                )
            }
            Float64(buf) => {
                <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer<Float64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                )
            }
            Utf8(buf) => <Utf8Field as ParsedBuffer<Utf8Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
                encoding,
            ),
        }
    }
}
