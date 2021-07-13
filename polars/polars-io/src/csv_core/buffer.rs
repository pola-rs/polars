use crate::csv::CsvEncoding;
use crate::csv_core::csv::RunningSize;
use crate::csv_core::parser::{drop_quotes, skip_whitespace};
use arrow::array::{ArrayData, LargeStringArray};
use polars_arrow::builder::BooleanBufferBuilder;
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
    #[inline]
    fn parse(bytes: &[u8]) -> Result<f32> {
        let a = fast_float::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Float64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Result<f64> {
        let a = fast_float::parse(bytes).map_err(|e| e.to_polars_err())?;
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
        encoding: CsvEncoding,
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
        _encoding: CsvEncoding,
        _needs_escaping: bool,
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
    name: String,
    // buffer that holds the string data
    data: AlignedVec<u8>,
    // offsets in the string data buffer
    offsets: AlignedVec<i64>,
    validity: BooleanBufferBuilder,
    rdr: csv_core::Reader,
}

impl Utf8Field {
    fn new(name: &str, capacity: usize, str_capacity: usize, delimiter: u8) -> Self {
        let mut offsets = AlignedVec::with_capacity_aligned(capacity + 1);
        offsets.push(0);
        Self {
            name: name.to_string(),
            data: AlignedVec::with_capacity_aligned(str_capacity),
            offsets,
            validity: BooleanBufferBuilder::new(capacity),
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
        needs_escaping: bool,
    ) -> Result<()> {
        // first check utf8 validity
        #[cfg(feature = "simdutf8")]
        let parse_result = simdutf8::basic::from_utf8(bytes)
            .map_err(|_| PolarsError::Other("invalid utf8 data".into()));
        #[cfg(not(feature = "simdutf8"))]
        let parse_result =
            std::str::from_utf8(bytes).map_err(|_| PolarsError::Other("invalid utf8 data".into()));
        let data_len = self.data.len();

        // check if field fits in the str data buffer
        let remaining_capacity = self.data.capacity() - data_len;
        if remaining_capacity < bytes.len() {
            // exponential growth strategy
            self.data
                .reserve(std::cmp::max(self.data.capacity(), bytes.len()))
        }
        let n_written = if needs_escaping {
            // Write bytes to string buffer, but don't update the length just yet.
            // We do that after we are sure its valid utf8.
            // Or in case of LossyUtf8 and invalid utf8, we overwrite it with lossy parsed data and then
            // set the length.
            let bytes = unsafe {
                // csv core expects the delimiter for its state machine, but we already split on that.
                // so we extend the slice by one to include the delimiter
                // Safety:
                // A field always has a delimiter OR a end of line char so we can extend the field
                // without accessing unowned memory
                let ptr = bytes.as_ptr();
                std::slice::from_raw_parts(ptr, bytes.len() + 1)
            };

            // Safety:
            // we just allocated enough capacity and data_len is correct.
            let out_buf = unsafe {
                std::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(data_len), bytes.len())
            };
            let (_, _, n_written) = self.rdr.read_field(bytes, out_buf);
            n_written
        } else {
            self.data.extend_from_slice(bytes);
            bytes.len()
        };

        match parse_result {
            Ok(_) => {
                // Soundness
                // the n_written from csv-core are now valid bytes so we can update the length.
                unsafe { self.data.set_len(data_len + n_written) }
                self.offsets.push(self.data.len() as i64);
                self.validity.append(true);
            }
            Err(err) => {
                if matches!(encoding, CsvEncoding::LossyUtf8) {
                    let s = String::from_utf8_lossy(
                        &self.data.as_slice()[data_len..data_len + n_written],
                    )
                    .into_owned();
                    self.data.extend_from_slice(s.as_bytes());
                } else if ignore_errors {
                    // append null
                    self.offsets.push(self.data.len() as i64);
                    self.validity.append(false);
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
        _needs_escaping: bool,
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
            Buffer::Utf8(mut v) => {
                v.offsets.shrink_to_fit();
                v.data.shrink_to_fit();
                let array_data = ArrayData::builder(ArrowDataType::LargeUtf8)
                    .len(v.offsets.len() - 1)
                    .add_buffer(v.offsets.into_arrow_buffer())
                    .add_buffer(v.data.into_arrow_buffer())
                    .null_bit_buffer(v.validity.finish())
                    .build();

                let arr = LargeStringArray::from(array_data);
                let ca = Utf8Chunked::new_from_chunks(&v.name, vec![Arc::new(arr)]);
                ca.into_series()
            }
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
            Buffer::Utf8(v) => {
                v.offsets.push(v.data.len() as i64);
                v.validity.append(false);
            }
        };
    }

    #[inline]
    pub(crate) fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        start_pos: usize,
        encoding: CsvEncoding,
        needs_escaping: bool,
    ) -> Result<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => <BooleanChunkedBuilder as ParsedBuffer<BooleanType>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
                encoding,
                needs_escaping,
            ),
            Int32(buf) => {
                <PrimitiveChunkedBuilder<Int32Type> as ParsedBuffer<Int32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                    needs_escaping,
                )
            }
            Int64(buf) => {
                <PrimitiveChunkedBuilder<Int64Type> as ParsedBuffer<Int64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                    needs_escaping,
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
                    needs_escaping,
                )
            }
            UInt32(buf) => {
                <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuffer<UInt32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                    needs_escaping,
                )
            }
            Float32(buf) => {
                <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer<Float32Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                    needs_escaping,
                )
            }
            Float64(buf) => {
                <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer<Float64Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                    encoding,
                    needs_escaping,
                )
            }
            Utf8(buf) => <Utf8Field as ParsedBuffer<Utf8Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
                encoding,
                needs_escaping,
            ),
        }
    }
}
