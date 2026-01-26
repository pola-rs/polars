use arrow::array::MutableBinaryViewArray;
#[cfg(feature = "dtype-decimal")]
use polars_compute::decimal::str_to_dec128;
#[cfg(feature = "dtype-categorical")]
use polars_core::chunked_array::builder::CategoricalChunkedBuilder;
use polars_core::prelude::*;
use polars_error::to_compute_err;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::chunkedarray::string::Pattern;
#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
use polars_time::prelude::string::infer::{
    DatetimeInfer, StrpTimeParser, TryFromWithUnit, infer_pattern_single,
};
#[cfg(feature = "dtype-f16")]
use polars_utils::float16::pf16;
use polars_utils::vec::PushUnchecked;

use super::options::CsvEncoding;
use super::parser::{could_be_whitespace_fast, skip_whitespace};
use super::utils::escape_field;

pub(crate) trait PrimitiveParser: PolarsNumericType {
    fn parse(bytes: &[u8]) -> Option<Self::Native>;
}

#[cfg(feature = "dtype-f16")]
impl PrimitiveParser for Float16Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<pf16> {
        use num_traits::FromPrimitive;

        pf16::from_f32(fast_float2::parse(bytes).ok()?)
    }
}

impl PrimitiveParser for Float32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f32> {
        fast_float2::parse(bytes).ok()
    }
}
impl PrimitiveParser for Float64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f64> {
        fast_float2::parse(bytes).ok()
    }
}

#[cfg(feature = "dtype-u8")]
impl PrimitiveParser for UInt8Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u8> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
#[cfg(feature = "dtype-u16")]
impl PrimitiveParser for UInt16Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u16> {
        atoi_simd::parse_skipped(bytes).ok()
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
#[cfg(feature = "dtype-u128")]
impl PrimitiveParser for UInt128Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u128> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
#[cfg(feature = "dtype-i8")]
impl PrimitiveParser for Int8Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i8> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
#[cfg(feature = "dtype-i16")]
impl PrimitiveParser for Int16Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i16> {
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
#[cfg(feature = "dtype-i128")]
impl PrimitiveParser for Int128Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i128> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}

trait ParsedBuilder {
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()>;
}

impl<T> ParsedBuilder for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType + PrimitiveParser,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        mut bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if !bytes.is_empty() && needs_escaping {
            bytes = &bytes[1..bytes.len() - 1];
        }

        if !bytes.is_empty() && could_be_whitespace_fast(bytes[0]) {
            bytes = skip_whitespace(bytes);
        }

        if bytes.is_empty() {
            self.append_null();
            return Ok(());
        }

        match T::parse(bytes) {
            Some(value) => self.append_value(value),
            None => {
                if ignore_errors {
                    self.append_null()
                } else {
                    polars_bail!(ComputeError: "invalid primitive value found during CSV parsing")
                }
            },
        }
        Ok(())
    }
}

pub struct Utf8Field {
    name: PlSmallStr,
    mutable: MutableBinaryViewArray<[u8]>,
    scratch: Vec<u8>,
    quote_char: u8,
    encoding: CsvEncoding,
}

impl Utf8Field {
    fn new(
        name: PlSmallStr,
        capacity: usize,
        quote_char: Option<u8>,
        encoding: CsvEncoding,
    ) -> Self {
        Self {
            name,
            mutable: MutableBinaryViewArray::with_capacity(capacity),
            scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            encoding,
        }
    }
}

#[inline]
pub fn validate_utf8(bytes: &[u8]) -> bool {
    simdutf8::basic::from_utf8(bytes).is_ok()
}

impl ParsedBuilder for Utf8Field {
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
                self.mutable.push(Some([]))
            }
            return Ok(());
        }

        // note that one branch writes without updating the length, so we must do that later.
        let escaped_bytes = if needs_escaping {
            self.scratch.clear();
            self.scratch.reserve(bytes.len());
            polars_ensure!(bytes.len() > 1 && bytes.last() == Some(&self.quote_char), ComputeError: "invalid csv file\n\nField `{}` is not properly escaped.", std::str::from_utf8(bytes).map_err(to_compute_err)?);

            // SAFETY:
            // we just allocated enough capacity and data_len is correct.
            unsafe {
                let n_written =
                    escape_field(bytes, self.quote_char, self.scratch.spare_capacity_mut());
                self.scratch.set_len(n_written);
            }

            self.scratch.as_slice()
        } else {
            bytes
        };

        if matches!(self.encoding, CsvEncoding::LossyUtf8) | ignore_errors {
            // It is important that this happens after escaping, as invalid escaped string can produce
            // invalid utf8.
            let parse_result = validate_utf8(escaped_bytes);

            match parse_result {
                true => {
                    let value = escaped_bytes;
                    self.mutable.push_value(value)
                },
                false => {
                    if matches!(self.encoding, CsvEncoding::LossyUtf8) {
                        // TODO! do this without allocating
                        let s = String::from_utf8_lossy(escaped_bytes);
                        self.mutable.push_value(s.as_ref().as_bytes())
                    } else if ignore_errors {
                        self.mutable.push_null()
                    } else {
                        // If field before escaping is valid utf8, the escaping is incorrect.
                        if needs_escaping && validate_utf8(bytes) {
                            polars_bail!(ComputeError: "string field is not properly escaped");
                        } else {
                            polars_bail!(ComputeError: "invalid utf-8 sequence");
                        }
                    }
                },
            }
        } else {
            self.mutable.push_value(escaped_bytes)
        }

        Ok(())
    }
}

#[cfg(feature = "dtype-categorical")]
pub struct CategoricalField<T: PolarsCategoricalType> {
    escape_scratch: Vec<u8>,
    quote_char: u8,
    builder: CategoricalChunkedBuilder<T>,
}

#[cfg(feature = "dtype-categorical")]
impl<T: PolarsCategoricalType> CategoricalField<T> {
    fn new(name: PlSmallStr, capacity: usize, quote_char: Option<u8>, dtype: DataType) -> Self {
        let mut builder = CategoricalChunkedBuilder::new(name, dtype);
        builder.reserve(capacity);

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
                // SAFETY:
                // we just allocated enough capacity and data_len is correct.
                unsafe {
                    let n_written = escape_field(
                        bytes,
                        self.quote_char,
                        self.escape_scratch.spare_capacity_mut(),
                    );
                    self.escape_scratch.set_len(n_written);
                }

                // SAFETY:
                // just did utf8 check
                let key = unsafe { std::str::from_utf8_unchecked(&self.escape_scratch) };
                self.builder.append_str(key)?;
            } else {
                // SAFETY:
                // just did utf8 check
                let key = unsafe { std::str::from_utf8_unchecked(bytes) };
                self.builder.append_str(key)?;
            }
        } else if ignore_errors {
            self.builder.append_null()
        } else {
            polars_bail!(ComputeError: "invalid utf-8 sequence");
        }
        Ok(())
    }
}

impl ParsedBuilder for BooleanChunkedBuilder {
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

#[cfg(feature = "dtype-decimal")]
pub struct DecimalField {
    builder: PrimitiveChunkedBuilder<Int128Type>,
    precision: usize,
    scale: usize,
    decimal_comma: bool,
}

#[cfg(feature = "dtype-decimal")]
impl DecimalField {
    fn new(
        name: PlSmallStr,
        capacity: usize,
        precision: usize,
        scale: usize,
        decimal_comma: bool,
    ) -> Self {
        let builder = PrimitiveChunkedBuilder::<Int128Type>::new(name, capacity);
        Self {
            builder,
            precision,
            scale,
            decimal_comma,
        }
    }
}

#[cfg(feature = "dtype-decimal")]
impl ParsedBuilder for DecimalField {
    #[inline]
    fn parse_bytes(
        &mut self,
        mut bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if !bytes.is_empty() && needs_escaping {
            bytes = &bytes[1..bytes.len() - 1];
        }

        if !bytes.is_empty() && could_be_whitespace_fast(bytes[0]) {
            bytes = skip_whitespace(bytes);
        }

        if bytes.is_empty() {
            self.builder.append_null();
            return Ok(());
        }

        match str_to_dec128(bytes, self.precision, self.scale, self.decimal_comma) {
            Some(value) => self.builder.append_value(value),
            None => {
                if ignore_errors {
                    self.builder.append_null()
                } else {
                    polars_bail!(ComputeError: "invalid decimal value found during CSV parsing")
                }
            },
        }

        Ok(())
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
pub struct DatetimeField<T: PolarsNumericType> {
    compiled: Option<DatetimeInfer<T>>,
    builder: PrimitiveChunkedBuilder<T>,
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-date"))]
impl<T: PolarsNumericType> DatetimeField<T> {
    fn new(name: PlSmallStr, capacity: usize) -> Self {
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
        // SAFETY:
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
                    polars_bail!(ComputeError: "could not find a 'date/datetime' pattern for '{}'", val)
                }
            },
        },
    };
    match DatetimeInfer::try_from_with_unit(pattern, time_unit) {
        Ok(mut infer) => {
            let parsed = infer.parse(val);
            let Some(parsed) = parsed else {
                if ignore_errors {
                    buf.builder.append_null();
                    return Ok(());
                } else {
                    polars_bail!(ComputeError: "could not parse '{}' with pattern '{:?}'", val, pattern)
                }
            };

            buf.compiled = Some(infer);
            buf.builder.append_value(parsed);
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
impl<T> ParsedBuilder for DatetimeField<T>
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
        if needs_escaping && bytes.len() >= 2 {
            bytes = &bytes[1..bytes.len() - 1]
        }

        if bytes.is_empty() {
            // for types other than string `_missing_is_null` is irrelevant; we always append null
            self.builder.append_null();
            return Ok(());
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

pub fn init_builders(
    projection: &[usize],
    capacity: usize,
    schema: &Schema,
    quote_char: Option<u8>,
    encoding: CsvEncoding,
    decimal_comma: bool,
) -> PolarsResult<Vec<Builder>> {
    projection
        .iter()
        .map(|&i| {
            let (name, dtype) = schema.get_at_index(i).unwrap();
            let name = name.clone();
            let builder = match dtype {
                &DataType::Boolean => Builder::Boolean(BooleanChunkedBuilder::new(name, capacity)),
                #[cfg(feature = "dtype-i8")]
                &DataType::Int8 => Builder::Int8(PrimitiveChunkedBuilder::new(name, capacity)),
                #[cfg(feature = "dtype-i16")]
                &DataType::Int16 => Builder::Int16(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Int32 => Builder::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::Int64 => Builder::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
                #[cfg(feature = "dtype-i128")]
                &DataType::Int128 => Builder::Int128(PrimitiveChunkedBuilder::new(name, capacity)),
                #[cfg(feature = "dtype-u8")]
                &DataType::UInt8 => Builder::UInt8(PrimitiveChunkedBuilder::new(name, capacity)),
                #[cfg(feature = "dtype-u16")]
                &DataType::UInt16 => Builder::UInt16(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt32 => Builder::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
                &DataType::UInt64 => Builder::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
                #[cfg(feature = "dtype-u128")]
                &DataType::UInt128 => {
                    Builder::UInt128(PrimitiveChunkedBuilder::new(name, capacity))
                },
                #[cfg(feature = "dtype-f16")]
                &DataType::Float16 => {
                    if decimal_comma {
                        Builder::DecimalFloat16(
                            PrimitiveChunkedBuilder::new(name, capacity),
                            Default::default(),
                        )
                    } else {
                        Builder::Float16(PrimitiveChunkedBuilder::new(name, capacity))
                    }
                },
                &DataType::Float32 => {
                    if decimal_comma {
                        Builder::DecimalFloat32(
                            PrimitiveChunkedBuilder::new(name, capacity),
                            Default::default(),
                        )
                    } else {
                        Builder::Float32(PrimitiveChunkedBuilder::new(name, capacity))
                    }
                },
                &DataType::Float64 => {
                    if decimal_comma {
                        Builder::DecimalFloat64(
                            PrimitiveChunkedBuilder::new(name, capacity),
                            Default::default(),
                        )
                    } else {
                        Builder::Float64(PrimitiveChunkedBuilder::new(name, capacity))
                    }
                },
                #[cfg(feature = "dtype-decimal")]
                &DataType::Decimal(precision, scale) => Builder::Decimal(DecimalField::new(
                    name,
                    capacity,
                    precision,
                    scale,
                    decimal_comma,
                )),
                &DataType::String => {
                    Builder::Utf8(Utf8Field::new(name, capacity, quote_char, encoding))
                },
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(time_unit, time_zone) => Builder::Datetime {
                    buf: DatetimeField::new(name, capacity),
                    time_unit: *time_unit,
                    time_zone: time_zone.clone(),
                },
                #[cfg(feature = "dtype-date")]
                &DataType::Date => Builder::Date(DatetimeField::new(name, capacity)),
                #[cfg(feature = "dtype-categorical")]
                DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                    match dtype.cat_physical().unwrap() {
                        CategoricalPhysical::U8 => {
                            Builder::Categorical8(CategoricalField::<Categorical8Type>::new(
                                name,
                                capacity,
                                quote_char,
                                dtype.clone(),
                            ))
                        },
                        CategoricalPhysical::U16 => {
                            Builder::Categorical16(CategoricalField::<Categorical16Type>::new(
                                name,
                                capacity,
                                quote_char,
                                dtype.clone(),
                            ))
                        },
                        CategoricalPhysical::U32 => {
                            Builder::Categorical32(CategoricalField::<Categorical32Type>::new(
                                name,
                                capacity,
                                quote_char,
                                dtype.clone(),
                            ))
                        },
                    }
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
pub enum Builder {
    Boolean(BooleanChunkedBuilder),
    #[cfg(feature = "dtype-i8")]
    Int8(PrimitiveChunkedBuilder<Int8Type>),
    #[cfg(feature = "dtype-i16")]
    Int16(PrimitiveChunkedBuilder<Int16Type>),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    #[cfg(feature = "dtype-i128")]
    Int128(PrimitiveChunkedBuilder<Int128Type>),
    #[cfg(feature = "dtype-u8")]
    UInt8(PrimitiveChunkedBuilder<UInt8Type>),
    #[cfg(feature = "dtype-u16")]
    UInt16(PrimitiveChunkedBuilder<UInt16Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    #[cfg(feature = "dtype-u128")]
    UInt128(PrimitiveChunkedBuilder<UInt128Type>),
    #[cfg(feature = "dtype-f16")]
    Float16(PrimitiveChunkedBuilder<Float16Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    #[cfg(feature = "dtype-decimal")]
    Decimal(DecimalField),
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
    #[cfg(feature = "dtype-categorical")]
    Categorical8(CategoricalField<Categorical8Type>),
    #[cfg(feature = "dtype-categorical")]
    Categorical16(CategoricalField<Categorical16Type>),
    #[cfg(feature = "dtype-categorical")]
    Categorical32(CategoricalField<Categorical32Type>),
    #[cfg(feature = "dtype-f16")]
    DecimalFloat16(PrimitiveChunkedBuilder<Float16Type>, Vec<u8>),
    DecimalFloat32(PrimitiveChunkedBuilder<Float32Type>, Vec<u8>),
    DecimalFloat64(PrimitiveChunkedBuilder<Float64Type>, Vec<u8>),
}

impl Builder {
    pub fn into_series(self) -> PolarsResult<Series> {
        let s = match self {
            Builder::Boolean(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-i8")]
            Builder::Int8(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-i16")]
            Builder::Int16(v) => v.finish().into_series(),
            Builder::Int32(v) => v.finish().into_series(),
            Builder::Int64(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-i128")]
            Builder::Int128(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-u8")]
            Builder::UInt8(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-u16")]
            Builder::UInt16(v) => v.finish().into_series(),
            Builder::UInt32(v) => v.finish().into_series(),
            Builder::UInt64(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-u128")]
            Builder::UInt128(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-f16")]
            Builder::Float16(v) => v.finish().into_series(),
            Builder::Float32(v) => v.finish().into_series(),
            Builder::Float64(v) => v.finish().into_series(),
            #[cfg(feature = "dtype-f16")]
            Builder::DecimalFloat16(v, _) => v.finish().into_series(),
            Builder::DecimalFloat32(v, _) => v.finish().into_series(),
            Builder::DecimalFloat64(v, _) => v.finish().into_series(),
            #[cfg(feature = "dtype-decimal")]
            Builder::Decimal(DecimalField {
                builder,
                precision,
                scale,
                ..
            }) => unsafe {
                builder
                    .finish()
                    .into_series()
                    .from_physical_unchecked(&DataType::Decimal(precision, scale))
                    .unwrap()
            },
            #[cfg(feature = "dtype-datetime")]
            Builder::Datetime {
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
            Builder::Date(v) => v
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Date)
                .unwrap(),

            Builder::Utf8(v) => {
                let arr = v.mutable.freeze();
                StringChunked::with_chunk(v.name, unsafe { arr.to_utf8view_unchecked() })
                    .into_series()
            },
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical8(buf) => buf.builder.finish().into_series(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical16(buf) => buf.builder.finish().into_series(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical32(buf) => buf.builder.finish().into_series(),
        };
        Ok(s)
    }

    pub fn add_null(&mut self, valid: bool) {
        match self {
            Builder::Boolean(v) => v.append_null(),
            #[cfg(feature = "dtype-i8")]
            Builder::Int8(v) => v.append_null(),
            #[cfg(feature = "dtype-i16")]
            Builder::Int16(v) => v.append_null(),
            Builder::Int32(v) => v.append_null(),
            Builder::Int64(v) => v.append_null(),
            #[cfg(feature = "dtype-i128")]
            Builder::Int128(v) => v.append_null(),
            #[cfg(feature = "dtype-u8")]
            Builder::UInt8(v) => v.append_null(),
            #[cfg(feature = "dtype-u16")]
            Builder::UInt16(v) => v.append_null(),
            Builder::UInt32(v) => v.append_null(),
            Builder::UInt64(v) => v.append_null(),
            #[cfg(feature = "dtype-u128")]
            Builder::UInt128(v) => v.append_null(),
            #[cfg(feature = "dtype-f16")]
            Builder::Float16(v) => v.append_null(),
            Builder::Float32(v) => v.append_null(),
            Builder::Float64(v) => v.append_null(),
            #[cfg(feature = "dtype-decimal")]
            Builder::Decimal(buf) => buf.builder.append_null(),
            #[cfg(feature = "dtype-f16")]
            Builder::DecimalFloat16(v, _) => v.append_null(),
            Builder::DecimalFloat32(v, _) => v.append_null(),
            Builder::DecimalFloat64(v, _) => v.append_null(),
            Builder::Utf8(v) => {
                if valid {
                    v.mutable.push_value("")
                } else {
                    v.mutable.push_null()
                }
            },
            #[cfg(feature = "dtype-datetime")]
            Builder::Datetime { buf, .. } => buf.builder.append_null(),
            #[cfg(feature = "dtype-date")]
            Builder::Date(v) => v.builder.append_null(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical8(buf) => buf.builder.append_null(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical16(buf) => buf.builder.append_null(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical32(buf) => buf.builder.append_null(),
        };
    }

    pub fn dtype(&self) -> DataType {
        match self {
            Builder::Boolean(_) => DataType::Boolean,
            #[cfg(feature = "dtype-i8")]
            Builder::Int8(_) => DataType::Int8,
            #[cfg(feature = "dtype-i16")]
            Builder::Int16(_) => DataType::Int16,
            Builder::Int32(_) => DataType::Int32,
            Builder::Int64(_) => DataType::Int64,
            #[cfg(feature = "dtype-i128")]
            Builder::Int128(_) => DataType::Int128,
            #[cfg(feature = "dtype-u8")]
            Builder::UInt8(_) => DataType::UInt8,
            #[cfg(feature = "dtype-u16")]
            Builder::UInt16(_) => DataType::UInt16,
            Builder::UInt32(_) => DataType::UInt32,
            Builder::UInt64(_) => DataType::UInt64,
            #[cfg(feature = "dtype-u128")]
            Builder::UInt128(_) => DataType::UInt128,
            #[cfg(feature = "dtype-f16")]
            Builder::Float16(_) | Builder::DecimalFloat16(_, _) => DataType::Float16,
            Builder::Float32(_) | Builder::DecimalFloat32(_, _) => DataType::Float32,
            Builder::Float64(_) | Builder::DecimalFloat64(_, _) => DataType::Float64,
            #[cfg(feature = "dtype-decimal")]
            Builder::Decimal(DecimalField {
                precision, scale, ..
            }) => DataType::Decimal(*precision, *scale),
            Builder::Utf8(_) => DataType::String,
            #[cfg(feature = "dtype-datetime")]
            Builder::Datetime { time_unit, .. } => DataType::Datetime(*time_unit, None),
            #[cfg(feature = "dtype-date")]
            Builder::Date(_) => DataType::Date,
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical8(buf) => buf.builder.dtype().clone(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical16(buf) => buf.builder.dtype().clone(),
            #[cfg(feature = "dtype-categorical")]
            Builder::Categorical32(buf) => buf.builder.dtype().clone(),
        }
    }

    #[inline]
    pub fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        missing_is_null: bool,
    ) -> PolarsResult<()> {
        use Builder::*;
        match self {
            Boolean(buf) => <BooleanChunkedBuilder as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-i8")]
            Int8(buf) => <PrimitiveChunkedBuilder<Int8Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-i16")]
            Int16(buf) => <PrimitiveChunkedBuilder<Int16Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int32(buf) => <PrimitiveChunkedBuilder<Int32Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int64(buf) => <PrimitiveChunkedBuilder<Int64Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-i128")]
            Int128(buf) => <PrimitiveChunkedBuilder<Int128Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-u8")]
            UInt8(buf) => <PrimitiveChunkedBuilder<UInt8Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-u16")]
            UInt16(buf) => <PrimitiveChunkedBuilder<UInt16Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt32(buf) => <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt64(buf) => <PrimitiveChunkedBuilder<UInt64Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-u128")]
            UInt128(buf) => <PrimitiveChunkedBuilder<UInt128Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-f16")]
            Float16(buf) => <PrimitiveChunkedBuilder<Float16Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Float32(buf) => <PrimitiveChunkedBuilder<Float32Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Float64(buf) => <PrimitiveChunkedBuilder<Float64Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-f16")]
            DecimalFloat16(buf, scratch) => {
                prepare_decimal_comma(bytes, scratch);
                <PrimitiveChunkedBuilder<Float16Type> as ParsedBuilder>::parse_bytes(
                    buf,
                    scratch,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    None,
                )
            },
            DecimalFloat32(buf, scratch) => {
                prepare_decimal_comma(bytes, scratch);
                <PrimitiveChunkedBuilder<Float32Type> as ParsedBuilder>::parse_bytes(
                    buf,
                    scratch,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    None,
                )
            },
            DecimalFloat64(buf, scratch) => {
                prepare_decimal_comma(bytes, scratch);
                <PrimitiveChunkedBuilder<Float64Type> as ParsedBuilder>::parse_bytes(
                    buf,
                    scratch,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    None,
                )
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(buf) => <DecimalField as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Utf8(buf) => <Utf8Field as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-datetime")]
            Datetime { buf, time_unit, .. } => {
                <DatetimeField<Int64Type> as ParsedBuilder>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    Some(*time_unit),
                )
            },
            #[cfg(feature = "dtype-date")]
            Date(buf) => <DatetimeField<Int32Type> as ParsedBuilder>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            #[cfg(feature = "dtype-categorical")]
            Categorical8(buf) => {
                buf.parse_bytes(bytes, ignore_errors, needs_escaping, missing_is_null, None)
            },
            #[cfg(feature = "dtype-categorical")]
            Categorical16(buf) => {
                buf.parse_bytes(bytes, ignore_errors, needs_escaping, missing_is_null, None)
            },
            #[cfg(feature = "dtype-categorical")]
            Categorical32(buf) => {
                buf.parse_bytes(bytes, ignore_errors, needs_escaping, missing_is_null, None)
            },
        }
    }
}

#[inline]
fn prepare_decimal_comma(bytes: &[u8], scratch: &mut Vec<u8>) {
    scratch.clear();
    scratch.reserve(bytes.len());

    // SAFETY: we pre-allocated.
    for &byte in bytes {
        if byte == b',' {
            unsafe { scratch.push_unchecked(b'.') }
        } else {
            unsafe { scratch.push_unchecked(byte) }
        }
    }
}
