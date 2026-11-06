//! APIs to write to Parquet format.
//!
//! # Arrow/Parquet Interoperability
//! As of [parquet-format v2.9](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md)
//! there are Arrow [DataTypes](polars_arrow::datatypes::ArrowDataType) which do not have a parquet
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
mod fixed_size_binary;
mod nested;
mod pages;
mod primitive;
mod row_group;
mod schema;
mod utils;

pub use nested::write_rep_and_def;
pub use pages::{to_leaves, to_nested, to_parquet_leaves};
use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::*;
use polars_arrow::types::{NativeType, days_ms, i256};
use polars_config::config;
use polars_utils::float16::pf16;
use polars_utils::pl_str::PlSmallStr;
pub use utils::write_def_levels;

pub use crate::parquet::compression::{BrotliLevel, CompressionOptions, GzipLevel, ZstdLevel};
pub use crate::parquet::encoding::Encoding;
pub use crate::parquet::metadata::{
    Descriptor, FileMetadata, KeyValue, SchemaDescriptor, ThriftFileMetadata,
};
pub use crate::parquet::page::{CompressedDataPage, CompressedPage, Page};
use crate::parquet::schema::Repetition;
use crate::parquet::schema::types::PrimitiveType as ParquetPrimitiveType;
pub use crate::parquet::schema::types::{
    FieldInfo, ParquetType, PhysicalType as ParquetPhysicalType,
};
pub use crate::parquet::write::{
    Compressor, DynIter, DynStreamingIterator, RowGroupIterColumns, Version, compress,
    write_metadata_sidecar,
};
pub use crate::parquet::{FallibleStreamingIterator, fallible_streaming_iterator};
use crate::write::fixed_size_binary::build_statistics_float16;

/// The statistics to write
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct StatisticsOptions {
    pub min_value: bool,
    pub max_value: bool,
    pub distinct_count: bool,
    pub null_count: bool,
    /// Target byte length for binary/string statistics truncation. Set to
    /// `Some(0)` to disable truncation.
    pub binary_statistics_truncate_length: Option<u64>,
}

impl Default for StatisticsOptions {
    fn default() -> Self {
        Self {
            min_value: true,
            max_value: true,
            distinct_count: false,
            null_count: true,
            binary_statistics_truncate_length: None,
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

pub use file::FileWriter;
pub use pages::{Nested, array_to_columns};
use polars_arrow::compute::aggregate::estimated_bytes_size;
use polars_arrow::match_integer_type;
use polars_error::{PolarsResult, polars_bail};
pub use row_group::{RowGroupIterator, row_group_iter};
pub use schema::{schema_to_metadata_key, to_parquet_type};

use self::pages::{FixedSizeListNested, PrimitiveNested, StructNested};
use crate::write::dictionary::encode_as_dictionary_optional;

impl StatisticsOptions {
    pub fn empty() -> Self {
        Self {
            min_value: false,
            max_value: false,
            distinct_count: false,
            null_count: false,
            binary_statistics_truncate_length: None,
        }
    }

    pub fn full() -> Self {
        Self {
            min_value: true,
            max_value: true,
            distinct_count: true,
            null_count: true,
            binary_statistics_truncate_length: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        !(self.min_value || self.max_value || self.distinct_count || self.null_count)
    }

    pub fn is_full(&self) -> bool {
        self.min_value && self.max_value && self.distinct_count && self.null_count
    }

    /// Truncate statistics for binary columns to this length.
    pub fn binary_statistics_truncate_length(&self) -> Option<u64> {
        let len = self
            .binary_statistics_truncate_length
            .unwrap_or_else(|| config().parquet_binary_statistics_truncate_length());
        (len > 0).then_some(len)
    }

    pub fn binary_statistics_truncate_length_usize(&self) -> Option<usize> {
        self.binary_statistics_truncate_length()
            .and_then(|x| usize::try_from(x).ok())
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

/// `data_page_size`: Set a target threshold for the approximate encoded size of data
/// pages within a column chunk (in bytes). If None, use the default data page size of 1MByte.
/// See: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
pub(crate) fn row_slice_ranges(
    number_of_rows: usize,
    byte_size: usize,
    options: WriteOptions,
) -> impl Iterator<Item = (usize, usize)> {
    let max_page_size = max_page_size(options);

    let bytes_per_row = if number_of_rows == 0 {
        0
    } else {
        ((byte_size as f64) / (number_of_rows as f64)) as usize
    };
    let rows_per_page = (max_page_size / (bytes_per_row + 1)).max(1);

    (0..number_of_rows)
        .step_by(rows_per_page)
        .map(move |offset| {
            let length = (offset + rows_per_page).min(number_of_rows) - offset;
            (offset, length)
        })
}

fn max_page_size(options: WriteOptions) -> usize {
    const DEFAULT_PAGE_SIZE: usize = 1024 * 1024; // 1 MB
    let max_page_size = options.data_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
    max_page_size.min(2usize.pow(31) - 2usize.pow(25)) // allowed maximum page size
}

/// Parquet's Plain encoding prefixes variable-length values with 4 byte lengths
const LENGTH_PREFIX: usize = size_of::<u32>();

enum LeafSizes<'a> {
    Offsets(&'a [i64]),
    Views {
        views: &'a [View],
        /// The total length of the values, if known without a scan.
        total_bytes_len: Option<usize>,
    },
}

impl<'a> LeafSizes<'a> {
    /// `None` if all values of `array` have the same size.
    fn new(array: &'a dyn Array) -> Option<Self> {
        let any = array.as_any();
        Some(match array.dtype().to_storage() {
            ArrowDataType::LargeBinary => Self::Offsets(
                any.downcast_ref::<BinaryArray<i64>>()
                    .unwrap()
                    .offsets()
                    .as_slice(),
            ),
            ArrowDataType::LargeUtf8 => Self::Offsets(
                any.downcast_ref::<Utf8Array<i64>>()
                    .unwrap()
                    .offsets()
                    .as_slice(),
            ),
            ArrowDataType::BinaryView => {
                let array = any.downcast_ref::<BinaryViewArray>().unwrap();
                Self::Views {
                    views: array.views(),
                    total_bytes_len: array.try_total_bytes_len(),
                }
            },
            ArrowDataType::Utf8View => {
                let array = any.downcast_ref::<Utf8ViewArray>().unwrap();
                Self::Views {
                    views: array.views(),
                    total_bytes_len: array.try_total_bytes_len(),
                }
            },
            _ => return None,
        })
    }

    /// Splits the rows into page ranges of `(offset, length)`, cutting before the row that would
    /// take the page over `max_page_size` plain-encoded bytes. A single row larger than the max
    /// gets a page of its own.
    fn slice_ranges(
        &self,
        nested: &[Nested],
        number_of_rows: usize,
        max_page_size: usize,
    ) -> Vec<(usize, usize)> {
        match *self {
            Self::Offsets(offsets) => {
                let bytes_until = |i: usize| (offsets[i] - offsets[0]) as usize + i * LENGTH_PREFIX;
                search_slice_ranges(number_of_rows, max_page_size, |row| {
                    bytes_until(leaf_index(nested, row))
                })
            },
            Self::Views {
                views,
                total_bytes_len,
            } => {
                // An upper bound for the rows, which may only cover part of the leaf.
                let total = total_bytes_len.map(|len| len + views.len() * LENGTH_PREFIX);
                if number_of_rows > 0 && total.is_some_and(|total| total <= max_page_size) {
                    return vec![(0, number_of_rows)];
                }
                let view_size = |v: &View| v.length as usize + LENGTH_PREFIX;
                let is_flat = nested
                    .iter()
                    .all(|n| matches!(n, Nested::Struct(_) | Nested::Primitive(_)));
                if is_flat {
                    let row_sizes = views[..number_of_rows].iter().map(view_size);
                    slice_ranges_from_row_sizes(max_page_size, row_sizes, number_of_rows)
                } else {
                    let mut leaf_start = leaf_index(nested, 0);
                    let row_sizes = (1..=number_of_rows).map(|row| {
                        let leaf_end = leaf_index(nested, row);
                        let size = views[leaf_start..leaf_end].iter().map(view_size).sum();
                        leaf_start = leaf_end;
                        size
                    });
                    slice_ranges_from_row_sizes(max_page_size, row_sizes, number_of_rows)
                }
            },
        }
    }
}

/// Page ranges for rows whose cumulative size `bytes_until(row)` can be computed for any row.
/// Each page end is found with a binary search, which takes O(log rows) lookups.
///
/// `bytes_until` must be non-decreasing.
fn search_slice_ranges(
    number_of_rows: usize,
    max_page_size: usize,
    bytes_until: impl Fn(usize) -> usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < number_of_rows {
        let limit = bytes_until(start) + max_page_size;
        let fits = |end: usize| bytes_until(end) <= limit;

        // The page ends at the largest `end` that fits, but always holds at least one row.
        let end = if fits(number_of_rows) {
            number_of_rows
        } else {
            // `lo` is a valid end and `hi` does not fit.
            let mut lo = start + 1;
            let mut hi = number_of_rows;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if fits(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        ranges.push((start, end - start));
        start = end;
    }
    ranges
}

/// Page ranges for rows whose sizes are only known one row at a time.
fn slice_ranges_from_row_sizes(
    max_page_size: usize,
    row_sizes: impl Iterator<Item = usize>,
    number_of_rows: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut page_bytes = 0;
    for (row, size) in row_sizes.enumerate() {
        if page_bytes + size > max_page_size && row > start {
            ranges.push((start, row - start));
            start = row;
            page_bytes = 0;
        }
        page_bytes += size;
    }
    if start < number_of_rows {
        ranges.push((start, number_of_rows - start));
    }
    ranges
}

/// The index in the leaf array of the first value of top-level row `row`.
fn leaf_index(nested: &[Nested], mut row: usize) -> usize {
    for nested in nested {
        match nested {
            Nested::List(l) => row = l.offsets.buffer()[row] as usize,
            Nested::LargeList(l) => row = l.offsets.buffer()[row] as usize,
            Nested::FixedSizeList(l) => row *= l.width,
            Nested::Struct(_) | Nested::Primitive(_) => {},
        }
    }
    row
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
        .iter_values()
        .map(to_parquet_type)
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(SchemaDescriptor::new(
        PlSmallStr::from_static("root"),
        parquet_types,
    ))
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

/// Returns an iterator of [`Page`].
pub fn array_to_pages(
    primitive_array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    mut encoding: Encoding,
) -> PolarsResult<DynIter<'static, PolarsResult<Page>>> {
    if let ArrowDataType::Dictionary(key_type, _, _) = primitive_array.dtype().to_storage() {
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
    // Variable-length values can differ wildly in size, so the average size of a row does not
    // bound the size of a page.
    let ranges = match LeafSizes::new(primitive_array) {
        Some(sizes) => sizes.slice_ranges(&nested, number_of_rows, max_page_size(options)),
        None => {
            // note: this is not correct if the array is sliced - the estimation should happen on
            // the primitive after sliced for parquet
            let byte_size = estimated_bytes_size(primitive_array);
            row_slice_ranges(number_of_rows, byte_size, options).collect()
        },
    };
    let primitive_array = primitive_array.to_boxed();

    let pages = ranges.into_iter().map(move |(offset, length)| {
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
    let dtype = array.dtype();

    if type_.field_info.repetition == Repetition::Required && array.null_count() > 0 {
        polars_bail!(InvalidOperation: "writing a missing value to required parquet column '{}'", type_.field_info.name);
    }

    match dtype {
        // Map empty struct to boolean array with same validity.
        ArrowDataType::Struct(fs) if fs.is_empty() => boolean::array_to_page(
            &BooleanArray::new(
                ArrowDataType::Boolean,
                Bitmap::new_zeroed(array.len()),
                array.validity().cloned(),
            ),
            options,
            type_,
            encoding,
        ),

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
            );
        },
        ArrowDataType::UInt16 => {
            return primitive::array_to_page_integer::<u16, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::UInt32 => {
            return primitive::array_to_page_integer::<u32, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::UInt64 => {
            return primitive::array_to_page_integer::<u64, i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int8 => {
            return primitive::array_to_page_integer::<i8, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int16 => {
            return primitive::array_to_page_integer::<i16, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int32 | ArrowDataType::Date32 | ArrowDataType::Time32(_) => {
            return primitive::array_to_page_integer::<i32, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
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
            );
        },
        ArrowDataType::Float16 => {
            let array: &PrimitiveArray<pf16> = array.as_any().downcast_ref().unwrap();
            let statistics = options
                .has_statistics()
                .then(|| build_statistics_float16(array, type_.clone(), &options.statistics));
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(2),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
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
                polars_compute::cast::cast(array, &ArrowDataType::LargeBinary, Default::default())
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
            );
        },
        ArrowDataType::BinaryView => {
            return binview::array_to_page(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Utf8View => {
            let array =
                polars_compute::cast::cast(array, &ArrowDataType::BinaryView, Default::default())
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
                Some(fixed_size_binary::build_statistics(
                    &array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
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
                Some(fixed_size_binary::build_statistics(
                    &array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::FixedSizeBinary(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                Some(fixed_size_binary::build_statistics(
                    array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };

            fixed_size_binary::array_to_page(array, options, type_, statistics)
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
                    let stats = fixed_size_binary::build_statistics_i256_big_endian_low(
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
                fixed_size_binary::array_to_page(&array, options, type_, statistics)
            } else {
                let size = 32;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i256>>()
                    .unwrap();
                // Can't write min/max statistics for an unannotated 256-bit decimal, see #25965.
                let mut no_mm_options = options;
                no_mm_options.statistics.min_value = false;
                no_mm_options.statistics.max_value = false;
                let statistics = if no_mm_options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_i256_big_endian(
                        array,
                        type_.clone(),
                        size,
                        &no_mm_options.statistics,
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

                fixed_size_binary::array_to_page(&array, no_mm_options, type_, statistics)
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
                    let stats = fixed_size_binary::build_statistics_big_endian(
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
                fixed_size_binary::array_to_page(&array, options, type_, statistics)
            }
        },
        ArrowDataType::UInt128 => {
            let array: &PrimitiveArray<u128> = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_big_endian(
                    array,
                    type_.clone(),
                    16,
                    &options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::Int128 => {
            let array: &PrimitiveArray<i128> = array.as_any().downcast_ref().unwrap();
            // Can't write min/max statistics for signed 128-bit integer, see #25965.
            let mut no_mm_options = options;
            no_mm_options.statistics.min_value = false;
            no_mm_options.statistics.max_value = false;
            let statistics = if no_mm_options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_big_endian(
                    array,
                    type_.clone(),
                    16,
                    &no_mm_options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::array_to_page(&array, no_mm_options, type_, statistics)
        },
        ArrowDataType::Extension(ext) => {
            let mut boxed = array.to_boxed();
            assert!(matches!(boxed.dtype(), ArrowDataType::Extension(ext2) if ext2 == ext));
            *boxed.dtype_mut() = ext.inner.clone();
            return array_to_page_simple(boxed.as_ref(), type_, options, encoding);
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
    if type_.field_info.repetition == Repetition::Required
        && array.validity().is_some_and(|v| v.unset_bits() > 0)
    {
        polars_bail!(InvalidOperation: "writing a missing value to required parquet column '{}'", type_.field_info.name);
    }

    use ArrowDataType::*;
    match array.dtype().to_storage() {
        Null => {
            let array = Int32Array::new_null(ArrowDataType::Int32, array.len());
            primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
        },
        // Map empty struct to boolean array with same validity.
        Struct(fs) if fs.is_empty() => {
            let array = BooleanArray::new(
                ArrowDataType::Boolean,
                Bitmap::new_zeroed(array.len()),
                array.validity().cloned(),
            );
            boolean::nested_array_to_page(&array, options, type_, nested)
        },
        Boolean => {
            let array = array.as_any().downcast_ref().unwrap();
            boolean::nested_array_to_page(array, options, type_, nested)
        },
        LargeUtf8 => {
            let array =
                polars_compute::cast::cast(array, &LargeBinary, Default::default()).unwrap();
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
            let array = polars_compute::cast::cast(array, &BinaryView, Default::default()).unwrap();
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
        Float16 => {
            let array: &PrimitiveArray<pf16> = array.as_any().downcast_ref().unwrap();
            let statistics = options
                .has_statistics()
                .then(|| build_statistics_float16(array, type_.clone(), &options.statistics));
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(2),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
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
                    let stats = fixed_size_binary::build_statistics_big_endian(
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
                fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
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
                    let stats = fixed_size_binary::build_statistics_i256_big_endian_low(
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
                fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
            } else {
                let size = 32;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i256>>()
                    .unwrap();
                // Can't write min/max statistics for an unannotated 256-bit decimal, see #25965.
                let mut no_mm_options = options;
                no_mm_options.statistics.min_value = false;
                no_mm_options.statistics.max_value = false;
                let statistics = if no_mm_options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_i256_big_endian(
                        array,
                        type_.clone(),
                        size,
                        &no_mm_options.statistics,
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

                fixed_size_binary::nested_array_to_page(
                    &array,
                    no_mm_options,
                    type_,
                    nested,
                    statistics,
                )
            }
        },
        Int128 => {
            let array: &PrimitiveArray<i128> = array.as_any().downcast_ref().unwrap();
            // Can't write min/max statistics for signed 128-bit integer, see #25965.
            let mut no_mm_options = options;
            no_mm_options.statistics.min_value = false;
            no_mm_options.statistics.max_value = false;
            let statistics = if no_mm_options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_big_endian(
                    array,
                    type_.clone(),
                    16,
                    &no_mm_options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::nested_array_to_page(
                &array,
                no_mm_options,
                type_,
                nested,
                statistics,
            )
        },
        UInt128 => {
            let array: &PrimitiveArray<u128> = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_big_endian(
                    array,
                    type_.clone(),
                    16,
                    &options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
        },
        other => polars_bail!(nyi = "Writing nested parquet pages for data type {other:?}"),
    }
    .map(Page::Data)
}

fn get_encodings_recursive(dtype: &ArrowDataType, encodings: &mut Vec<Encoding>) {
    use polars_arrow::datatypes::PhysicalType::*;
    match dtype.to_physical_type() {
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | Dictionary(_) | LargeUtf8 | BinaryView | Utf8View => {
            encodings.push(get_primitive_dtype_encoding(dtype))
        },
        List | FixedSizeList | LargeList => {
            let a = dtype.to_storage();
            if let ArrowDataType::List(inner) = a {
                get_encodings_recursive(&inner.dtype, encodings)
            } else if let ArrowDataType::LargeList(inner) = a {
                get_encodings_recursive(&inner.dtype, encodings)
            } else if let ArrowDataType::FixedSizeList(inner, _) = a {
                get_encodings_recursive(&inner.dtype, encodings)
            } else {
                unreachable!()
            }
        },
        Struct => {
            if let ArrowDataType::Struct(fields) = dtype.to_storage() {
                if fields.is_empty() {
                    // 0-field struct writes as a boolean column representing outer validity.
                    encodings.push(Encoding::Rle)
                }

                for field in fields {
                    get_encodings_recursive(&field.dtype, encodings)
                }
            } else {
                unreachable!()
            }
        },
        Map => {
            if let ArrowDataType::Map(field, _) = dtype.to_storage() {
                if let ArrowDataType::Struct(fields) = field.dtype.to_storage() {
                    for field in fields {
                        get_encodings_recursive(&field.dtype, encodings)
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

/// Traverses the `dtype` up to its (parquet) columns and returns a vector of
/// items based on `map`.
///
/// This is used to assign an [`Encoding`] to every parquet column based on the columns' type (see example)
pub fn get_dtype_encoding(dtype: &ArrowDataType) -> Vec<Encoding> {
    let mut encodings = vec![];
    get_encodings_recursive(dtype, &mut encodings);
    encodings
}

fn get_primitive_dtype_encoding(dtype: &ArrowDataType) -> Encoding {
    match dtype.to_physical_type() {
        PhysicalType::Dictionary(_)
        | PhysicalType::LargeBinary
        | PhysicalType::LargeUtf8
        | PhysicalType::Utf8View
        | PhysicalType::BinaryView
        | PhysicalType::Primitive(_) => Encoding::RleDictionary,
        // remaining is plain
        _ => Encoding::Plain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::write::pages::to_nested;

    type Ranges = Vec<(usize, usize)>;

    /// The straightforward greedy pass that [`LeafSizes::slice_ranges`] must match.
    fn reference_slice_ranges(
        number_of_rows: usize,
        max_page_size: usize,
        bytes_until: impl Fn(usize) -> usize,
    ) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut start_bytes = bytes_until(0);
        let mut prev_bytes = start_bytes;
        for row in 1..=number_of_rows {
            let bytes = bytes_until(row);
            if bytes - start_bytes > max_page_size && row - 1 > start {
                ranges.push((start, row - 1 - start));
                start = row - 1;
                start_bytes = prev_bytes;
            }
            prev_bytes = bytes;
        }
        if start < number_of_rows {
            ranges.push((start, number_of_rows - start));
        }
        ranges
    }

    /// A small deterministic generator (splitmix64), so failures reproduce.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> usize {
            (self.next() % n) as usize
        }
    }

    /// Mostly short values, with occasional runs of large ones.
    fn skewed_values(rng: &mut Rng, len: usize) -> Vec<Vec<u8>> {
        let mut values = Vec::with_capacity(len);
        while values.len() < len {
            let (run, max_len) = match rng.below(20) {
                0 => (1 + rng.below(4), 5000),
                1 => (1, 0),
                _ => (1, 40),
            };
            for _ in 0..run.min(len - values.len()) {
                values.push(vec![b'x'; rng.below(max_len as u64 + 1)]);
            }
        }
        values
    }

    fn large_list(rng: &mut Rng, values: Box<dyn Array>) -> Box<dyn Array> {
        let mut offsets = vec![0i64];
        while (*offsets.last().unwrap() as usize) < values.len() {
            let remaining = values.len() - *offsets.last().unwrap() as usize;
            offsets.push(*offsets.last().unwrap() + rng.below(4).min(remaining) as i64);
        }
        let dtype = ArrowDataType::LargeList(Box::new(Field::new(
            "item".into(),
            values.dtype().clone(),
            false,
        )));
        ListArray::<i64>::new(dtype, offsets.try_into().unwrap(), values, None).boxed()
    }

    fn arrays(rng: &mut Rng, len: usize) -> Vec<Box<dyn Array>> {
        let values = skewed_values(rng, len);
        let strings: Vec<String> = values
            .iter()
            .map(|v| String::from_utf8(v.clone()).unwrap())
            .collect();
        let flat: Vec<Box<dyn Array>> = vec![
            BinaryArray::<i64>::from_slice(&values).boxed(),
            Utf8Array::<i64>::from_slice(&strings).boxed(),
            BinaryViewArray::from_slice_values(&values).boxed(),
            Utf8ViewArray::from_slice_values(&strings).boxed(),
        ];
        let mut arrays = flat.clone();
        for leaf in &flat[..] {
            arrays.push(large_list(rng, leaf.clone()));
        }
        // Sliced arrays: `offsets[0] != 0`, list offsets that do not start at 0, and views without
        // a known total length.
        for array in arrays.clone() {
            if array.len() > 2 {
                let offset = 1 + rng.below(array.len() as u64 / 2);
                let length = rng.below((array.len() - offset) as u64 + 1);
                arrays.push(array.sliced(offset, length));
            }
        }
        arrays
    }

    /// The new ranges and the reference ranges for `array`.
    fn ranges(array: &dyn Array, max_page_size: usize) -> (Ranges, Ranges) {
        let type_ = to_parquet_type(&Field::new("a".into(), array.dtype().clone(), false)).unwrap();
        let nested = to_nested(array, &type_).unwrap().remove(0);
        let mut leaves = Vec::new();
        to_leaves(array, &mut leaves);
        let leaf = leaves.remove(0);
        let number_of_rows = nested[0].len();

        let sizes = LeafSizes::new(leaf.as_ref()).unwrap();
        let lengths: Vec<usize> = match &sizes {
            LeafSizes::Offsets(offsets) => {
                offsets.windows(2).map(|w| (w[1] - w[0]) as usize).collect()
            },
            LeafSizes::Views { views, .. } => views.iter().map(|v| v.length as usize).collect(),
        };
        let mut prefix = vec![0];
        for len in lengths {
            prefix.push(prefix.last().unwrap() + len + LENGTH_PREFIX);
        }

        let actual = sizes.slice_ranges(&nested, number_of_rows, max_page_size);
        let expected = reference_slice_ranges(number_of_rows, max_page_size, |row| {
            prefix[leaf_index(&nested, row)]
        });
        (actual, expected)
    }

    #[test]
    fn test_slice_ranges_match_reference() {
        let mut rng = Rng(30312);
        for len in [0, 1, 2, 3, 10, 100, 1000, 5000] {
            for array in arrays(&mut rng, len) {
                for max_page_size in [0, 1, 4, 5, 37, 100, 1000, 5004, 10_000, 1 << 20] {
                    let (actual, expected) = ranges(array.as_ref(), max_page_size);
                    assert_eq!(
                        actual,
                        expected,
                        "{:?}, {} rows, max_page_size {max_page_size}",
                        array.dtype(),
                        array.len(),
                    );
                }
            }
        }
    }

    #[test]
    fn test_slice_ranges_large_rows() {
        // Several rows over the limit, alone and in a run, and empty values.
        let lengths = [0, 3, 50, 2, 60, 70, 80, 1, 0, 0, 90, 4];
        let values: Vec<Vec<u8>> = lengths.iter().map(|&n| vec![b'x'; n]).collect();
        for array in [
            BinaryArray::<i64>::from_slice(&values).boxed(),
            BinaryViewArray::from_slice_values(&values).boxed(),
        ] {
            let (actual, expected) = ranges(array.as_ref(), 20);
            assert_eq!(actual, expected);
            assert_eq!(
                actual,
                [
                    (0, 2),
                    (2, 1),
                    (3, 1),
                    (4, 1),
                    (5, 1),
                    (6, 1),
                    (7, 3),
                    (10, 1),
                    (11, 1)
                ]
            );
        }
    }

    #[test]
    fn test_view_chunk_under_limit_is_one_page() {
        let values: Vec<Vec<u8>> = (0..100).map(|i| vec![b'x'; i % 30]).collect();
        let array = BinaryViewArray::from_slice_values(&values);
        let sliced = array.clone().sliced(10, 50);
        assert!(array.try_total_bytes_len().is_some());
        assert!(sliced.try_total_bytes_len().is_none());

        for array in [array.boxed(), sliced.boxed()] {
            let (actual, expected) = ranges(array.as_ref(), 1 << 20);
            assert_eq!(actual, [(0, array.len())]);
            assert_eq!(actual, expected);
        }
    }
}
