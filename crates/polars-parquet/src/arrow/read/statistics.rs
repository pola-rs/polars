//! APIs exposing `crate::parquet`'s statistics as arrow's statistics.

mod bounds;

use polars_arrow::array::{
    Array, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, MutableBinaryViewArray,
    MutableBooleanArray, MutableFixedSizeBinaryArray, MutablePrimitiveArray, NullArray,
    PrimitiveArray, Utf8ViewArray, new_null_array,
};
use polars_arrow::datatypes::{ArrowDataType, Field, IntegerType};
use polars_arrow::types::{NativeType, i256};
use polars_utils::IdxSize;
use polars_utils::float16::pf16;
use polars_utils::pl_str::PlSmallStr;

pub use self::bounds::{ArrowBound, BoundConversion, PhysicalBound};
use super::{FileMetadata, RowGroupMetadata};
use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::ColumnOrder;
pub use crate::parquet::metadata::RawBounds;
use crate::parquet::schema::types::{PhysicalType as ParquetPhysicalType, PrimitiveType};
use crate::parquet::statistics::Statistics as ParquetStatistics;
use crate::read::ColumnChunkMetadata;

/// Parquet statistics for a nesting level
#[derive(Debug, PartialEq)]
pub enum Statistics {
    Column(Box<ColumnStatistics>),

    List(Option<Box<Statistics>>),
    FixedSizeList(Option<Box<Statistics>>, usize),

    Struct(Box<[Option<Statistics>]>),
    Dictionary(IntegerType, Option<Box<Statistics>>, bool),
}

/// Arrow-deserialized parquet statistics of a leaf-column
#[derive(Debug, PartialEq)]
pub struct ColumnStatistics {
    field: Field,

    primitive_type: PrimitiveType,
    column_order: ColumnOrder,

    /// Statistics of the leaf array of the column
    statistics: ParquetStatistics,
}

/// Arrow-deserialized parquet statistics of a leaf-column
#[derive(Debug, PartialEq)]
pub struct ArrowColumnStatistics {
    pub null_count: Option<u64>,
    pub distinct_count: Option<u64>,

    // While these two are Box<dyn Array>, they will only ever contain one valid value. This might
    // seems dumb, and don't get me wrong it is, but polars_arrow::Scalar is basically useless.
    pub min_value: Option<Box<dyn Array>>,
    pub max_value: Option<Box<dyn Array>>,
}

/// Arrow-deserialized parquet statistics of a leaf-column
pub struct ArrowColumnStatisticsArrays {
    pub null_count: PrimitiveArray<IdxSize>,
    pub distinct_count: PrimitiveArray<IdxSize>,
    pub min_value: Box<dyn Array>,
    pub max_value: Box<dyn Array>,
}

impl ColumnStatistics {
    pub fn into_arrow(self) -> ParquetResult<ArrowColumnStatistics> {
        let null_count = self.statistics.null_count().map(|v| v as u64);
        let distinct_count = self.statistics.distinct_count().map(|v| v as u64);

        let conversion =
            BoundConversion::new(self.field.dtype(), &self.primitive_type, self.column_order)?;
        let (min_value, max_value) = match conversion {
            Some(conversion) => {
                let (min, max) = self.statistics.bounds();
                let (min, max) = conversion.convert_bounds(min, max)?;
                let array =
                    |bound: Option<ArrowBound>| bound.map(|b| b.into_array(self.field.dtype()));
                (array(min), array(max))
            },
            None => (None, None),
        };

        Ok(ArrowColumnStatistics {
            null_count,
            distinct_count,
            min_value,
            max_value,
        })
    }
}

impl ArrowBound<'_> {
    /// The bound as a one-element array of `dtype`.
    fn into_array(self, dtype: &ArrowDataType) -> Box<dyn Array> {
        fn primitive<T: NativeType>(dtype: &ArrowDataType, value: T) -> Box<dyn Array> {
            PrimitiveArray::new(dtype.clone(), vec![value].into(), None).boxed()
        }
        match self {
            Self::Boolean(v) => {
                BooleanArray::new(ArrowDataType::Boolean, vec![v].into(), None).boxed()
            },
            Self::Int8(v) => primitive(dtype, v),
            Self::Int16(v) => primitive(dtype, v),
            Self::Int32(v) => primitive(dtype, v),
            Self::Int64(v) => primitive(dtype, v),
            Self::UInt8(v) => primitive(dtype, v),
            Self::UInt16(v) => primitive(dtype, v),
            Self::UInt32(v) => primitive(dtype, v),
            Self::UInt64(v) => primitive(dtype, v),
            Self::Float16(v) => primitive(dtype, v),
            Self::Float32(v) => primitive(dtype, v),
            Self::Float64(v) => primitive(dtype, v),
            Self::Int128(v) => primitive(dtype, v),
            Self::Int256(v) => primitive(dtype, v),
            Self::Bytes(v) if matches!(dtype, ArrowDataType::FixedSizeBinary(_)) => {
                FixedSizeBinaryArray::new(dtype.clone(), v.to_vec().into(), None).boxed()
            },
            Self::Bytes(v) => BinaryViewArray::from_slice([Some(v)]).boxed(),
            Self::Str(v) => Utf8ViewArray::from_slice([Some(v)]).boxed(),
        }
    }
}

/// Null bounds for every row group, with the null counts the chunks report.
fn unavailable_bounds(
    field: &Field,
    row_groups: &[RowGroupMetadata],
    field_idx: usize,
) -> ArrowColumnStatisticsArrays {
    let mut null_count = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
    let mut distinct_count = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
    for rg in row_groups {
        let column = &rg.parquet_columns()[field_idx];
        null_count.push(column.null_count().map(|v| v as IdxSize));
        distinct_count.push(column.distinct_count().map(|v| v as IdxSize));
    }
    let nulls = || new_null_array(field.dtype().clone(), row_groups.len());
    ArrowColumnStatisticsArrays {
        null_count: null_count.freeze(),
        distinct_count: distinct_count.freeze(),
        min_value: nulls(),
        max_value: nulls(),
    }
}

/// Deserializes the statistics in the column chunks from a single `row_group`
/// into [`Statistics`] associated from `field`'s name.
///
/// # Errors
/// This function errors if the deserialization of the statistics fails (e.g. invalid utf8)
pub fn deserialize_all(
    field: &Field,
    row_groups: &[RowGroupMetadata],
    field_idx: usize,
    column_order: ColumnOrder,
    footer_buf: &[u8],
) -> ParquetResult<Option<ArrowColumnStatisticsArrays>> {
    assert!(!row_groups.is_empty());
    use ArrowDataType as D;
    match field.dtype() {
        // @TODO: These are all a bit more complex, skip for now.
        D::List(..) | D::LargeList(..) | D::Map(..) => Ok(None),
        D::Dictionary(..) => Ok(None),
        D::FixedSizeList(..) => Ok(None),
        D::Struct(..) => Ok(None),

        D::Null => {
            let mut null_count = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
            let mut distinct_count =
                MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
            for rg in row_groups {
                null_count.push(Some(rg.num_rows() as IdxSize));
                distinct_count.push(Some(0));
            }
            Ok(Some(ArrowColumnStatisticsArrays {
                null_count: null_count.freeze(),
                distinct_count: distinct_count.freeze(),
                min_value: NullArray::new(ArrowDataType::Null, row_groups.len()).to_boxed(),
                max_value: NullArray::new(ArrowDataType::Null, row_groups.len()).to_boxed(),
            }))
        },

        _ => {
            let primitive_type = &row_groups[0].parquet_columns()[field_idx]
                .descriptor()
                .descriptor
                .primitive_type;
            let physical_type = primitive_type.physical_type;

            let Some(conversion) =
                BoundConversion::new(field.dtype(), primitive_type, column_order)?
            else {
                return Ok(Some(unavailable_bounds(field, row_groups, field_idx)));
            };

            let mut null_count = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
            let mut distinct_count =
                MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());

            // Decodes every group's bounds into a pair of `$arr`s, whose values the
            // `$variant` of [`ArrowBound`] carries.
            macro_rules! collect {
                ($arr:expr, $variant:ident) => {{
                    let mut min_arr = $arr;
                    let mut max_arr = $arr;
                    macro_rules! value {
                        ($bound:expr) => {
                            match $bound {
                                Some(ArrowBound::$variant(v)) => Some(v),
                                Some(_) => unreachable!(),
                                None => None,
                            }
                        };
                    }
                    for rg in row_groups {
                        let column = &rg.parquet_columns()[field_idx];
                        let (min, max) = match column.raw_bounds(footer_buf) {
                            Some(raw) => conversion.decode_bounds(physical_type, raw)?,
                            None => (None, None),
                        };
                        min_arr.push(value!(min));
                        max_arr.push(value!(max));
                        null_count.push(column.null_count().map(|v| v as IdxSize));
                        distinct_count.push(column.distinct_count().map(|v| v as IdxSize));
                    }
                    (min_arr.freeze().to_boxed(), max_arr.freeze().to_boxed())
                }};
            }

            let n = row_groups.len();
            use BoundConversion as C;
            let (min_value, max_value) = match conversion {
                C::Boolean => collect!(MutableBooleanArray::with_capacity(n), Boolean),
                C::Int8 => collect!(MutablePrimitiveArray::<i8>::with_capacity(n), Int8),
                C::Int16 => collect!(MutablePrimitiveArray::<i16>::with_capacity(n), Int16),
                C::Int32 => collect!(MutablePrimitiveArray::<i32>::with_capacity(n), Int32),
                C::Int64 | C::DaysToMillis | C::Timestamp { .. } => {
                    collect!(MutablePrimitiveArray::<i64>::with_capacity(n), Int64)
                },
                C::UInt8 => collect!(MutablePrimitiveArray::<u8>::with_capacity(n), UInt8),
                C::UInt16 => collect!(MutablePrimitiveArray::<u16>::with_capacity(n), UInt16),
                C::UInt32 => collect!(MutablePrimitiveArray::<u32>::with_capacity(n), UInt32),
                C::UInt64 => collect!(MutablePrimitiveArray::<u64>::with_capacity(n), UInt64),
                C::Float16 => collect!(MutablePrimitiveArray::<pf16>::with_capacity(n), Float16),
                C::Float32 => collect!(MutablePrimitiveArray::<f32>::with_capacity(n), Float32),
                C::Float64 => collect!(MutablePrimitiveArray::<f64>::with_capacity(n), Float64),
                C::Decimal128 => collect!(MutablePrimitiveArray::<i128>::with_capacity(n), Int128),
                C::Decimal256 => collect!(MutablePrimitiveArray::<i256>::with_capacity(n), Int256),
                C::Binary => match field.dtype() {
                    D::FixedSizeBinary(width) => {
                        collect!(MutableFixedSizeBinaryArray::with_capacity(*width, n), Bytes)
                    },
                    _ => collect!(MutableBinaryViewArray::<[u8]>::with_capacity(n), Bytes),
                },
                C::Utf8 => collect!(MutableBinaryViewArray::<str>::with_capacity(n), Str),
            };

            Ok(Some(ArrowColumnStatisticsArrays {
                null_count: null_count.freeze(),
                distinct_count: distinct_count.freeze(),
                min_value,
                max_value,
            }))
        },
    }
}

/// Deserializes the statistics in the column chunks from a single `row_group`
/// into [`Statistics`] associated from `field`'s name.
///
/// # Errors
/// This function errors if the deserialization of the statistics fails (e.g. invalid utf8)
pub fn deserialize<'a>(
    field: &Field,
    columns: &mut impl ExactSizeIterator<Item = &'a ColumnChunkMetadata>,
    metadata: &FileMetadata,
) -> ParquetResult<Option<Statistics>> {
    use ArrowDataType as D;
    match field.dtype() {
        D::List(field) | D::LargeList(field) | D::Map(field, _) => Ok(Some(Statistics::List(
            deserialize(field.as_ref(), columns, metadata)?.map(Box::new),
        ))),
        D::Dictionary(key, dtype, ordered) => Ok(Some(Statistics::Dictionary(
            *key,
            deserialize(
                &Field::new(PlSmallStr::EMPTY, dtype.as_ref().clone(), true),
                columns,
                metadata,
            )?
            .map(Box::new),
            *ordered,
        ))),
        D::FixedSizeList(field, width) => Ok(Some(Statistics::FixedSizeList(
            deserialize(field.as_ref(), columns, metadata)?.map(Box::new),
            *width,
        ))),
        D::Struct(fields) => {
            let field_columns = fields
                .iter()
                .map(|f| deserialize(f, columns, metadata))
                .collect::<ParquetResult<_>>()?;
            Ok(Some(Statistics::Struct(field_columns)))
        },
        _ => {
            let column = columns.next().unwrap();

            Ok(column
                .statistics(&metadata.footer_buf)
                .transpose()?
                .map(|statistics| {
                    let primitive_type = &column.descriptor().descriptor.primitive_type;

                    Statistics::Column(Box::new(ColumnStatistics {
                        field: field.clone(),
                        primitive_type: primitive_type.clone(),
                        column_order: metadata.column_order(column.leaf_index()),

                        statistics,
                    }))
                }))
        },
    }
}

/// Reads the bounds of one leaf straight from the footer bytes, for a field polars
/// decodes as `dtype`, without materialising statistics arrays.
pub struct LeafBounds {
    field_idx: usize,
    physical_type: ParquetPhysicalType,
    conversion: BoundConversion,
}

impl LeafBounds {
    /// `None` when the leaf's statistics cannot bound the values polars decodes
    /// as `field`'s type.
    pub fn new(field: &Field, metadata: &FileMetadata, field_idx: usize) -> Option<Self> {
        let primitive_type = &metadata.schema_descr.columns()[field_idx]
            .descriptor
            .primitive_type;
        let conversion = BoundConversion::new(
            field.dtype(),
            primitive_type,
            metadata.column_order(field_idx),
        )
        .ok()??;
        Some(Self {
            field_idx,
            physical_type: primitive_type.physical_type,
            conversion,
        })
    }

    pub fn conversion(&self) -> BoundConversion {
        self.conversion
    }

    /// The min and max of the leaf in `row_group`. Either is `None` when the
    /// chunk does not give it, does not give it exactly, or gives it malformed.
    pub fn bounds<'a>(
        &self,
        row_group: &RowGroupMetadata,
        footer_buf: &'a [u8],
    ) -> (Option<ArrowBound<'a>>, Option<ArrowBound<'a>>) {
        let column = &row_group.parquet_columns()[self.field_idx];
        column
            .raw_bounds(footer_buf)
            .and_then(|raw| self.conversion.decode_bounds(self.physical_type, raw).ok())
            .unwrap_or((None, None))
    }
}
