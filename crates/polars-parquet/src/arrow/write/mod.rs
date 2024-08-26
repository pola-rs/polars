//! APIs to write to Parquet format.
//!
//! # Arrow/Parquet Interoperability
//! As of [parquet-format v2.9](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md)
//! there are Arrow [DataTypes](arrow::datatypes::ArrowDataType) which do not have a parquet
//! representation. These include but are not limited to:
//! * `ArrowDataType::Timestamp(TimeUnit::Second, _)`
//! * `ArrowDataType::Int64`
//! * `ArrowDataType::Duration`
//! * `ArrowDataType::Date64`
//! * `ArrowDataType::Time32(TimeUnit::Second)`
//!
//! The use of these arrow types will result in no logical type being stored within a parquet file.

mod binary;
mod binview;
mod boolean;
mod dictionary;
mod file;
mod fixed_len_bytes;
mod nested;
mod pages;
mod primitive;
mod row_group;
mod schema;
#[cfg(feature = "async")]
mod sink;
mod utils;

use arrow::array::*;
use arrow::datatypes::*;
use arrow::types::{days_ms, i256, NativeType};
pub use nested::{num_values, write_rep_and_def};
pub use pages::{to_leaves, to_nested, to_parquet_leaves};
pub use utils::write_def_levels;

pub use crate::parquet::compression::{BrotliLevel, CompressionOptions, GzipLevel, ZstdLevel};
pub use crate::parquet::encoding::Encoding;
pub use crate::parquet::metadata::{
    Descriptor, FileMetaData, KeyValue, SchemaDescriptor, ThriftFileMetaData,
};
pub use crate::parquet::page::{CompressedDataPage, CompressedPage, Page};
use crate::parquet::schema::types::PrimitiveType as ParquetPrimitiveType;
pub use crate::parquet::schema::types::{
    FieldInfo, ParquetType, PhysicalType as ParquetPhysicalType,
};
pub use crate::parquet::write::{
    compress, write_metadata_sidecar, Compressor, DynIter, DynStreamingIterator,
    RowGroupIterColumns, Version,
};
pub use crate::parquet::{fallible_streaming_iterator, FallibleStreamingIterator};

/// The statistics to write
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StatisticsOptions {
    pub min_value: bool,
    pub max_value: bool,
    pub distinct_count: bool,
    pub null_count: bool,
}

impl Default for StatisticsOptions {
    fn default() -> Self {
        Self {
            min_value: true,
            max_value: true,
            distinct_count: false,
            null_count: true,
        }
    }
}

/// Options to encode an array
#[derive(Clone, Copy)]
pub enum EncodeNullability {
    Required,
    Optional,
}

/// Currently supported options to write to parquet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    /// Whether to write statistics
    pub statistics: StatisticsOptions,
    /// The page and file version to use
    pub version: Version,
    /// The compression to apply to every page
    pub compression: CompressionOptions,
    /// The size to flush a page, defaults to 1024 * 1024 if None
    pub data_page_size: Option<usize>,
}

use arrow::compute::aggregate::estimated_bytes_size;
use arrow::match_integer_type;
pub use file::FileWriter;
pub use pages::{array_to_columns, arrays_to_columns, Nested};
use polars_error::{polars_bail, PolarsResult};
pub use row_group::{row_group_iter, RowGroupIterator};
pub use schema::to_parquet_type;
#[cfg(feature = "async")]
pub use sink::FileSink;

use self::pages::{FixedSizeListNested, PrimitiveNested, StructNested};
use crate::write::dictionary::encode_as_dictionary_optional;

impl StatisticsOptions {
    pub fn empty() -> Self {
        Self {
            min_value: false,
            max_value: false,
            distinct_count: false,
            null_count: false,
        }
    }

    pub fn full() -> Self {
        Self {
            min_value: true,
            max_value: true,
            distinct_count: true,
            null_count: true,
        }
    }

    pub fn is_empty(&self) -> bool {
        !(self.min_value || self.max_value || self.distinct_count || self.null_count)
    }

    pub fn is_full(&self) -> bool {
        self.min_value && self.max_value && self.distinct_count && self.null_count
    }
}

impl WriteOptions {
    pub fn has_statistics(&self) -> bool {
        !self.statistics.is_empty()
    }
}

impl EncodeNullability {
    const fn new(is_optional: bool) -> Self {
        if is_optional {
            Self::Optional
        } else {
            Self::Required
        }
    }

    fn is_optional(self) -> bool {
        matches!(self, Self::Optional)
    }
}

/// returns offset and length to slice the leaf values
pub fn slice_nested_leaf(nested: &[Nested]) -> (usize, usize) {
    // find the deepest recursive dremel structure as that one determines how many values we must
    // take
    let mut out = (0, 0);
    for nested in nested.iter().rev() {
        match nested {
            Nested::LargeList(l_nested) => {
                let start = *l_nested.offsets.first();
                let end = *l_nested.offsets.last();
                return (start as usize, (end - start) as usize);
            },
            Nested::List(l_nested) => {
                let start = *l_nested.offsets.first();
                let end = *l_nested.offsets.last();
                return (start as usize, (end - start) as usize);
            },
            Nested::FixedSizeList(nested) => return (0, nested.length * nested.width),
            Nested::Primitive(nested) => out = (0, nested.length),
            Nested::Struct(_) => {},
        }
    }
    out
}

fn decimal_length_from_precision(precision: usize) -> usize {
    // digits = floor(log_10(2^(8*n - 1) - 1))
    // ceil(digits) = log10(2^(8*n - 1) - 1)
    // 10^ceil(digits) = 2^(8*n - 1) - 1
    // 10^ceil(digits) + 1 = 2^(8*n - 1)
    // log2(10^ceil(digits) + 1) = (8*n - 1)
    // log2(10^ceil(digits) + 1) + 1 = 8*n
    // (log2(10^ceil(a) + 1) + 1) / 8 = n
    (((10.0_f64.powi(precision as i32) + 1.0).log2() + 1.0) / 8.0).ceil() as usize
}

/// Creates a parquet [`SchemaDescriptor`] from a [`ArrowSchema`].
pub fn to_parquet_schema(schema: &ArrowSchema) -> PolarsResult<SchemaDescriptor> {
    let parquet_types = schema
        .fields
        .iter()
        .map(to_parquet_type)
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(SchemaDescriptor::new("root".to_string(), parquet_types))
}

/// Slices the [`Array`] to `Box<dyn Array>` and `Vec<Nested>`.
pub fn slice_parquet_array(
    primitive_array: &mut dyn Array,
    nested: &mut [Nested],
    mut current_offset: usize,
    mut current_length: usize,
) {
    for nested in nested.iter_mut() {
        match nested {
            Nested::LargeList(l_nested) => {
                l_nested.offsets.slice(current_offset, current_length + 1);
                if let Some(validity) = l_nested.validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };

                // Update the offset/ length so that the Primitive is sliced properly.
                current_length = l_nested.offsets.range() as usize;
                current_offset = *l_nested.offsets.first() as usize;
            },
            Nested::List(l_nested) => {
                l_nested.offsets.slice(current_offset, current_length + 1);
                if let Some(validity) = l_nested.validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };

                // Update the offset/ length so that the Primitive is sliced properly.
                current_length = l_nested.offsets.range() as usize;
                current_offset = *l_nested.offsets.first() as usize;
            },
            Nested::Struct(StructNested {
                validity, length, ..
            }) => {
                *length = current_length;
                if let Some(validity) = validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };
            },
            Nested::Primitive(PrimitiveNested {
                validity, length, ..
            }) => {
                *length = current_length;
                if let Some(validity) = validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };
                primitive_array.slice(current_offset, current_length);
            },
            Nested::FixedSizeList(FixedSizeListNested {
                validity,
                length,
                width,
                ..
            }) => {
                if let Some(validity) = validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };
                *length = current_length;
                // Update the offset/ length so that the Primitive is sliced properly.
                current_length *= *width;
                current_offset *= *width;
            },
        }
    }
}

/// Get the length of [`Array`] that should be sliced.
pub fn get_max_length(nested: &[Nested]) -> usize {
    let mut length = 0;
    for nested in nested.iter() {
        match nested {
            Nested::LargeList(l_nested) => length += l_nested.offsets.range() as usize,
            Nested::List(l_nested) => length += l_nested.offsets.range() as usize,
            Nested::FixedSizeList(nested) => length += nested.length * nested.width,
            _ => {},
        }
    }
    length
}

/// Returns an iterator of [`Page`].
pub fn array_to_pages(
    primitive_array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    mut encoding: Encoding,
) -> PolarsResult<DynIter<'static, PolarsResult<Page>>> {
    if let ArrowDataType::Dictionary(key_type, _, _) = primitive_array.data_type().to_logical_type()
    {
        return match_integer_type!(key_type, |$T| {
            dictionary::array_to_pages::<$T>(
                primitive_array.as_any().downcast_ref().unwrap(),
                type_,
                &nested,
                options,
                encoding,
            )
        });
    };
    if let Encoding::RleDictionary = encoding {
        // Only take this path for primitive columns
        if matches!(nested.first(), Some(Nested::Primitive(_))) {
            if let Some(result) =
                encode_as_dictionary_optional(primitive_array, nested, type_.clone(), options)
            {
                return result;
            }
        }

        // We didn't succeed, fallback to plain
        encoding = Encoding::Plain;
    }

    let nested = nested.to_vec();

    let number_of_rows = nested[0].len();

    // note: this is not correct if the array is sliced - the estimation should happen on the
    // primitive after sliced for parquet
    let byte_size = estimated_bytes_size(primitive_array);

    const DEFAULT_PAGE_SIZE: usize = 1024 * 1024;
    let max_page_size = options.data_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
    let max_page_size = max_page_size.min(2usize.pow(31) - 2usize.pow(25)); // allowed maximum page size
    let bytes_per_row = if number_of_rows == 0 {
        0
    } else {
        ((byte_size as f64) / (number_of_rows as f64)) as usize
    };
    let rows_per_page = (max_page_size / (bytes_per_row + 1)).max(1);

    let row_iter = (0..number_of_rows)
        .step_by(rows_per_page)
        .map(move |offset| {
            let length = if offset + rows_per_page > number_of_rows {
                number_of_rows - offset
            } else {
                rows_per_page
            };
            (offset, length)
        });

    let primitive_array = primitive_array.to_boxed();

    let pages = row_iter.map(move |(offset, length)| {
        let mut right_array = primitive_array.clone();
        let mut right_nested = nested.clone();
        slice_parquet_array(right_array.as_mut(), &mut right_nested, offset, length);

        array_to_page(
            right_array.as_ref(),
            type_.clone(),
            &right_nested,
            options,
            encoding,
        )
    });
    Ok(DynIter::new(pages))
}

/// Converts an [`Array`] to a [`CompressedPage`] based on options, descriptor and `encoding`.
pub fn array_to_page(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<Page> {
    if nested.len() == 1 {
        // special case where validity == def levels
        return array_to_page_simple(array, type_, options, encoding);
    }
    array_to_page_nested(array, type_, nested, options, encoding)
}

/// Converts an [`Array`] to a [`CompressedPage`] based on options, descriptor and `encoding`.
pub fn array_to_page_simple(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<Page> {
    let data_type = array.data_type();

    match data_type.to_logical_type() {
        ArrowDataType::Boolean => boolean::array_to_page(
            array.as_any().downcast_ref().unwrap(),
            options,
            type_,
            encoding,
        ),
        // casts below MUST match the casts done at the metadata (field -> parquet type).
        ArrowDataType::UInt8 => {
            return primitive::array_to_page_integer::<u8, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::UInt16 => {
            return primitive::array_to_page_integer::<u16, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::UInt32 => {
            return primitive::array_to_page_integer::<u32, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::UInt64 => {
            return primitive::array_to_page_integer::<u64, i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::Int8 => {
            return primitive::array_to_page_integer::<i8, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::Int16 => {
            return primitive::array_to_page_integer::<i16, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::Int32 | ArrowDataType::Date32 | ArrowDataType::Time32(_) => {
            return primitive::array_to_page_integer::<i32, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::Int64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_) => {
            return primitive::array_to_page_integer::<i64, i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::Float32 => primitive::array_to_page_plain::<f32, f32>(
            array.as_any().downcast_ref().unwrap(),
            options,
            type_,
        ),
        ArrowDataType::Float64 => primitive::array_to_page_plain::<f64, f64>(
            array.as_any().downcast_ref().unwrap(),
            options,
            type_,
        ),
        ArrowDataType::LargeUtf8 => {
            let array =
                arrow::compute::cast::cast(array, &ArrowDataType::LargeBinary, Default::default())
                    .unwrap();
            return binary::array_to_page::<i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::LargeBinary => {
            return binary::array_to_page::<i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::BinaryView => {
            return binview::array_to_page(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            )
        },
        ArrowDataType::Utf8View => {
            let array =
                arrow::compute::cast::cast(array, &ArrowDataType::BinaryView, Default::default())
                    .unwrap();
            return binview::array_to_page(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Null => {
            let array = Int32Array::new_null(ArrowDataType::Int32, array.len());
            primitive::array_to_page_plain::<i32, i32>(&array, options, type_)
        },
        ArrowDataType::Interval(IntervalUnit::YearMonth) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            let mut values = Vec::<u8>::with_capacity(12 * array.len());
            array.values().iter().for_each(|x| {
                let bytes = &x.to_le_bytes();
                values.extend_from_slice(bytes);
                values.extend_from_slice(&[0; 8]);
            });
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(12),
                values.into(),
                array.validity().cloned(),
            );
            let statistics = if options.has_statistics() {
                Some(fixed_len_bytes::build_statistics(
                    &array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };
            fixed_len_bytes::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::Interval(IntervalUnit::DayTime) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<days_ms>>()
                .unwrap();
            let mut values = Vec::<u8>::with_capacity(12 * array.len());
            array.values().iter().for_each(|x| {
                let bytes = &x.to_le_bytes();
                values.extend_from_slice(&[0; 4]); // months
                values.extend_from_slice(bytes); // days and seconds
            });
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(12),
                values.into(),
                array.validity().cloned(),
            );
            let statistics = if options.has_statistics() {
                Some(fixed_len_bytes::build_statistics(
                    &array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };
            fixed_len_bytes::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::FixedSizeBinary(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                Some(fixed_len_bytes::build_statistics(
                    array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };

            fixed_len_bytes::array_to_page(array, options, type_, statistics)
        },
        ArrowDataType::Decimal256(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i256>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i32())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i32, i32>(
                    &array, options, type_, encoding,
                );
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i64())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i64, i64>(
                    &array, options, type_, encoding,
                );
            } else if precision <= 38 {
                let size = decimal_length_from_precision(precision);
                let statistics = if options.has_statistics() {
                    let stats = fixed_len_bytes::build_statistics_decimal256_with_i128(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.0.low().to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_len_bytes::array_to_page(&array, options, type_, statistics)
            } else {
                let size = 32;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i256>>()
                    .unwrap();
                let statistics = if options.has_statistics() {
                    let stats = fixed_len_bytes::build_statistics_decimal256(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };
                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes();
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );

                fixed_len_bytes::array_to_page(&array, options, type_, statistics)
            }
        },
        ArrowDataType::Decimal(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i32, i32>(
                    &array, options, type_, encoding,
                );
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i64, i64>(
                    &array, options, type_, encoding,
                );
            } else {
                let size = decimal_length_from_precision(precision);

                let statistics = if options.has_statistics() {
                    let stats = fixed_len_bytes::build_statistics_decimal(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_len_bytes::array_to_page(&array, options, type_, statistics)
            }
        },
        other => polars_bail!(nyi = "Writing parquet pages for data type {other:?}"),
    }
    .map(Page::Data)
}

fn array_to_page_nested(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    _encoding: Encoding,
) -> PolarsResult<Page> {
    use ArrowDataType::*;
    match array.data_type().to_logical_type() {
        Null => {
            let array = Int32Array::new_null(ArrowDataType::Int32, array.len());
            primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
        },
        Boolean => {
            let array = array.as_any().downcast_ref().unwrap();
            boolean::nested_array_to_page(array, options, type_, nested)
        },
        LargeUtf8 => {
            let array =
                arrow::compute::cast::cast(array, &LargeBinary, Default::default()).unwrap();
            let array = array.as_any().downcast_ref().unwrap();
            binary::nested_array_to_page::<i64>(array, options, type_, nested)
        },
        LargeBinary => {
            let array = array.as_any().downcast_ref().unwrap();
            binary::nested_array_to_page::<i64>(array, options, type_, nested)
        },
        BinaryView => {
            let array = array.as_any().downcast_ref().unwrap();
            binview::nested_array_to_page(array, options, type_, nested)
        },
        Utf8View => {
            let array = arrow::compute::cast::cast(array, &BinaryView, Default::default()).unwrap();
            let array = array.as_any().downcast_ref().unwrap();
            binview::nested_array_to_page(array, options, type_, nested)
        },
        UInt8 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u8, i32>(array, options, type_, nested)
        },
        UInt16 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u16, i32>(array, options, type_, nested)
        },
        UInt32 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u32, i32>(array, options, type_, nested)
        },
        UInt64 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u64, i64>(array, options, type_, nested)
        },
        Int8 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i8, i32>(array, options, type_, nested)
        },
        Int16 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i16, i32>(array, options, type_, nested)
        },
        Int32 | Date32 | Time32(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i32, i32>(array, options, type_, nested)
        },
        Int64 | Date64 | Time64(_) | Timestamp(_, _) | Duration(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i64, i64>(array, options, type_, nested)
        },
        Float32 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<f32, f32>(array, options, type_, nested)
        },
        Float64 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<f64, f64>(array, options, type_, nested)
        },
        Decimal(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i64, i64>(&array, options, type_, nested)
            } else {
                let size = decimal_length_from_precision(precision);

                let statistics = if options.has_statistics() {
                    let stats = fixed_len_bytes::build_statistics_decimal(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_len_bytes::array_to_page(&array, options, type_, statistics)
            }
        },
        Decimal256(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i256>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i32())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i64())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i64, i64>(&array, options, type_, nested)
            } else if precision <= 38 {
                let size = decimal_length_from_precision(precision);
                let statistics = if options.has_statistics() {
                    let stats = fixed_len_bytes::build_statistics_decimal256_with_i128(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.0.low().to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_len_bytes::array_to_page(&array, options, type_, statistics)
            } else {
                let size = 32;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i256>>()
                    .unwrap();
                let statistics = if options.has_statistics() {
                    let stats = fixed_len_bytes::build_statistics_decimal256(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };
                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes();
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );

                fixed_len_bytes::array_to_page(&array, options, type_, statistics)
            }
        },
        other => polars_bail!(nyi = "Writing nested parquet pages for data type {other:?}"),
    }
    .map(Page::Data)
}

fn transverse_recursive<T, F: Fn(&ArrowDataType) -> T + Clone>(
    data_type: &ArrowDataType,
    map: F,
    encodings: &mut Vec<T>,
) {
    use arrow::datatypes::PhysicalType::*;
    match data_type.to_physical_type() {
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | Dictionary(_) | LargeUtf8 | BinaryView | Utf8View => encodings.push(map(data_type)),
        List | FixedSizeList | LargeList => {
            let a = data_type.to_logical_type();
            if let ArrowDataType::List(inner) = a {
                transverse_recursive(&inner.data_type, map, encodings)
            } else if let ArrowDataType::LargeList(inner) = a {
                transverse_recursive(&inner.data_type, map, encodings)
            } else if let ArrowDataType::FixedSizeList(inner, _) = a {
                transverse_recursive(&inner.data_type, map, encodings)
            } else {
                unreachable!()
            }
        },
        Struct => {
            if let ArrowDataType::Struct(fields) = data_type.to_logical_type() {
                for field in fields {
                    transverse_recursive(&field.data_type, map.clone(), encodings)
                }
            } else {
                unreachable!()
            }
        },
        Map => {
            if let ArrowDataType::Map(field, _) = data_type.to_logical_type() {
                if let ArrowDataType::Struct(fields) = field.data_type.to_logical_type() {
                    for field in fields {
                        transverse_recursive(&field.data_type, map.clone(), encodings)
                    }
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
        Union => todo!(),
    }
}

/// Transverses the `data_type` up to its (parquet) columns and returns a vector of
/// items based on `map`.
///
/// This is used to assign an [`Encoding`] to every parquet column based on the columns' type (see example)
pub fn transverse<T, F: Fn(&ArrowDataType) -> T + Clone>(
    data_type: &ArrowDataType,
    map: F,
) -> Vec<T> {
    let mut encodings = vec![];
    transverse_recursive(data_type, map, &mut encodings);
    encodings
}
