//! APIs exposing `crate::parquet`'s statistics as arrow's statistics.
use arrow::array::{
    Array, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, PrimitiveArray, Utf8ViewArray,
};
use arrow::datatypes::{ArrowDataType, Field, IntegerType, IntervalUnit, TimeUnit};
use arrow::types::{f16, i256, NativeType};
use ethnum::I256;
use polars_utils::pl_str::PlSmallStr;

use super::ParquetTimeUnit;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::types::PhysicalType as ParquetPhysicalType;
use crate::parquet::statistics::Statistics as ParquetStatistics;
use crate::read::{
    convert_days_ms, convert_i128, convert_i256, convert_year_month, int96_to_i64_ns,
    ColumnChunkMetadata, PrimitiveLogicalType,
};

/// Parquet statistics for a nesting level
#[derive(Debug, PartialEq)]
pub enum Statistics {
    Column(ColumnStatistics),

    List(Option<Box<Statistics>>),
    FixedSizeList(Option<Box<Statistics>>, usize),

    Struct(Box<[Option<Statistics>]>),
    Dictionary(IntegerType, Option<Box<Statistics>>, bool),
}

/// Arrow-deserialized parquet statistics of a leaf-column
#[derive(Debug, PartialEq)]
pub struct ColumnStatistics {
    field: Field,

    logical_type: Option<PrimitiveLogicalType>,
    physical_type: ParquetPhysicalType,

    /// Statistics of the leaf array of the column
    statistics: ParquetStatistics,
}

#[derive(Debug, PartialEq)]
pub enum ColumnPathSegment {
    List { is_large: bool },
    FixedSizeList { width: usize },
    Dictionary { key: IntegerType, is_sorted: bool },
    Struct { column_idx: usize },
}

/// Arrow-deserialized parquet statistics of a leaf-column
#[derive(Debug, PartialEq)]
pub struct ArrowColumnStatistics {
    pub null_count: Option<u64>,
    pub distinct_count: Option<u64>,

    // While these two are Box<dyn Array>, they will only ever contain one valid value. This might
    // seems dumb, and don't get me wrong it is, but arrow::Scalar is basically useless.
    pub min_value: Option<Box<dyn Array>>,
    pub max_value: Option<Box<dyn Array>>,
}

fn timestamp(logical_type: Option<&PrimitiveLogicalType>, time_unit: TimeUnit, x: i64) -> i64 {
    let unit = if let Some(PrimitiveLogicalType::Timestamp { unit, .. }) = logical_type {
        unit
    } else {
        return x;
    };

    match (unit, time_unit) {
        (ParquetTimeUnit::Milliseconds, TimeUnit::Second) => x / 1_000,
        (ParquetTimeUnit::Microseconds, TimeUnit::Second) => x / 1_000_000,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Second) => x * 1_000_000_000,

        (ParquetTimeUnit::Milliseconds, TimeUnit::Millisecond) => x,
        (ParquetTimeUnit::Microseconds, TimeUnit::Millisecond) => x / 1_000,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Millisecond) => x / 1_000_000,

        (ParquetTimeUnit::Milliseconds, TimeUnit::Microsecond) => x * 1_000,
        (ParquetTimeUnit::Microseconds, TimeUnit::Microsecond) => x,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Microsecond) => x / 1_000,

        (ParquetTimeUnit::Milliseconds, TimeUnit::Nanosecond) => x * 1_000_000,
        (ParquetTimeUnit::Microseconds, TimeUnit::Nanosecond) => x * 1_000,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Nanosecond) => x,
    }
}

impl ColumnStatistics {
    pub fn into_arrow(self) -> ParquetResult<ArrowColumnStatistics> {
        use ParquetStatistics as S;
        let (null_count, distinct_count) = match &self.statistics {
            S::Binary(s) => (s.null_count, s.distinct_count),
            S::Boolean(s) => (s.null_count, s.distinct_count),
            S::FixedLen(s) => (s.null_count, s.distinct_count),
            S::Int32(s) => (s.null_count, s.distinct_count),
            S::Int64(s) => (s.null_count, s.distinct_count),
            S::Int96(s) => (s.null_count, s.distinct_count),
            S::Float(s) => (s.null_count, s.distinct_count),
            S::Double(s) => (s.null_count, s.distinct_count),
        };

        let null_count = null_count.map(|v| v as u64);
        let distinct_count = distinct_count.map(|v| v as u64);

        macro_rules! rmap {
            ($expect:ident, $map:expr) => {{
                let s = self.statistics.$expect();

                let min = s.min_value;
                let max = s.max_value;

                let min = ($map)(min)?.map(|x| Box::new(x) as Box<dyn Array>);
                let max = ($map)(max)?.map(|x| Box::new(x) as Box<dyn Array>);

                (min, max)
            }};
            ($expect:ident, @prim $from:ty $(as $to:ty)? $(, $map:expr)?) => {{
                rmap!(
                    $expect,
                    |x: Option<$from>| {
                        $(
                        let x = x.map(|x| x as $to);
                        )?
                        $(
                        let x = x.map($map);
                        )?
                        ParquetResult::Ok(x.map(|x| PrimitiveArray::$(<$to>::)?new(
                            self.field.dtype().clone(),
                            vec![x].into(),
                            None,
                        )))
                    }
                )
            }};
            (@binary $(, $map:expr)?) => {{
                rmap!(
                    expect_binary,
                    |x: Option<Vec<u8>>| {
                        $(
                        let x = x.map($map);
                        )?
                        ParquetResult::Ok(x.map(|x| BinaryViewArray::from_slice([Some(x)])))
                    }
                )
            }};
            (@string) => {{
                rmap!(
                    expect_binary,
                    |x: Option<Vec<u8>>| {
                        let x = x.map(String::from_utf8).transpose().map_err(|_| {
                            ParquetError::oos("Invalid UTF8 in Statistics")
                        })?;
                        ParquetResult::Ok(x.map(|x| Utf8ViewArray::from_slice([Some(x)])))
                    }
                )
            }};
        }

        use {ArrowDataType as D, ParquetPhysicalType as PPT};
        let (min_value, max_value) = match (self.field.dtype(), &self.physical_type) {
            (D::Null, _) => (None, None),

            (D::Boolean, _) => rmap!(expect_boolean, |x: Option<bool>| ParquetResult::Ok(
                x.map(|x| BooleanArray::new(ArrowDataType::Boolean, vec![x].into(), None,))
            )),

            (D::Int8, _) => rmap!(expect_int32, @prim i32 as i8),
            (D::Int16, _) => rmap!(expect_int32, @prim i32 as i16),
            (D::Int32 | D::Date32 | D::Time32(_), _) => rmap!(expect_int32, @prim i32 as i32),

            // some implementations of parquet write arrow's date64 into i32.
            (D::Date64, PPT::Int32) => rmap!(expect_int32, @prim i32 as i64, |x| x * 86400000),

            (D::Int64 | D::Time64(_) | D::Duration(_), _) | (D::Date64, PPT::Int64) => {
                rmap!(expect_int64, @prim i64 as i64)
            },

            (D::Interval(IntervalUnit::YearMonth), _) => rmap!(
                expect_binary,
                @prim Vec<u8>,
                |x| convert_year_month(&x)
            ),
            (D::Interval(IntervalUnit::DayTime), _) => rmap!(
                expect_binary,
                @prim Vec<u8>,
                |x| convert_days_ms(&x)
            ),

            (D::UInt8, _) => rmap!(expect_int32, @prim i32 as u8),
            (D::UInt16, _) => rmap!(expect_int32, @prim i32 as u16),
            (D::UInt32, PPT::Int32) => rmap!(expect_int32, @prim i32 as u32),

            // some implementations of parquet write arrow's u32 into i64.
            (D::UInt32, PPT::Int64) => rmap!(expect_int64, @prim i64 as u32),
            (D::UInt64, _) => rmap!(expect_int64, @prim i64 as u64),

            (D::Timestamp(time_unit, _), PPT::Int96) => {
                rmap!(expect_int96, @prim [u32; 3], |x| {
                    timestamp(self.logical_type.as_ref(), *time_unit, int96_to_i64_ns(x))
                })
            },
            (D::Timestamp(time_unit, _), PPT::Int64) => {
                rmap!(expect_int64, @prim i64, |x| {
                    timestamp(self.logical_type.as_ref(), *time_unit, x)
                })
            },

            // Read Float16, since we don't have a f16 type in Polars we read it to a Float32.
            (_, PPT::FixedLenByteArray(2))
                if matches!(
                    self.logical_type.as_ref(),
                    Some(PrimitiveLogicalType::Float16)
                ) =>
            {
                rmap!(expect_fixedlen, @prim Vec<u8>, |v| f16::from_le_bytes([v[0], v[1]]).to_f32())
            },
            (D::Float32, _) => rmap!(expect_float, @prim f32),
            (D::Float64, _) => rmap!(expect_double, @prim f64),

            (D::Decimal(_, _), PPT::Int32) => rmap!(expect_int32, @prim i32 as i128),
            (D::Decimal(_, _), PPT::Int64) => rmap!(expect_int64, @prim i64 as i128),
            (D::Decimal(_, _), PPT::FixedLenByteArray(n)) if *n > 16 => {
                return Err(ParquetError::not_supported(format!(
                    "Can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}",
                )))
            },
            (D::Decimal(_, _), PPT::FixedLenByteArray(n)) => rmap!(
                expect_fixedlen,
                @prim Vec<u8>,
                |x| convert_i128(&x, *n)
            ),
            (D::Decimal256(_, _), PPT::Int32) => {
                rmap!(expect_int32, @prim i32, |x: i32| i256(I256::new(x.into())))
            },
            (D::Decimal256(_, _), PPT::Int64) => {
                rmap!(expect_int64, @prim i64, |x: i64| i256(I256::new(x.into())))
            },
            (D::Decimal256(_, _), PPT::FixedLenByteArray(n)) if *n > 16 => {
                return Err(ParquetError::not_supported(format!(
                    "Can't decode Decimal256 type from Fixed Size Byte Array of len {n:?}",
                )))
            },
            (D::Decimal256(_, _), PPT::FixedLenByteArray(_)) => rmap!(
                expect_fixedlen,
                @prim Vec<u8>,
                |x| convert_i256(&x)
            ),
            (D::Binary, _) => rmap!(@binary),
            (D::LargeBinary, _) => rmap!(@binary),
            (D::Utf8, _) => rmap!(@string),
            (D::LargeUtf8, _) => rmap!(@string),

            (D::BinaryView, _) => rmap!(@binary),
            (D::Utf8View, _) => rmap!(@string),

            (D::FixedSizeBinary(_), _) => {
                rmap!(expect_fixedlen, |x: Option<Vec<u8>>| ParquetResult::Ok(
                    x.map(|x| FixedSizeBinaryArray::new(
                        self.field.dtype().clone(),
                        x.into(),
                        None
                    ))
                ))
            },

            other => todo!("{:?}", other),
        };

        Ok(ArrowColumnStatistics {
            null_count,
            distinct_count,

            min_value,
            max_value,
        })
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
) -> ParquetResult<Option<Statistics>> {
    use ArrowDataType as D;
    match field.dtype() {
        D::List(field) | D::LargeList(field) => Ok(Some(Statistics::List(
            deserialize(field.as_ref(), columns)?.map(Box::new),
        ))),
        D::Dictionary(key, dtype, is_sorted) => Ok(Some(Statistics::Dictionary(
            *key,
            deserialize(
                &Field::new(PlSmallStr::EMPTY, dtype.as_ref().clone(), true),
                columns,
            )?
            .map(Box::new),
            *is_sorted,
        ))),
        D::FixedSizeList(field, width) => Ok(Some(Statistics::FixedSizeList(
            deserialize(field.as_ref(), columns)?.map(Box::new),
            *width,
        ))),
        D::Struct(fields) => {
            let field_columns = fields
                .iter()
                .map(|f| deserialize(f, columns))
                .collect::<ParquetResult<_>>()?;
            Ok(Some(Statistics::Struct(field_columns)))
        },
        _ => {
            let column = columns.next().unwrap();

            Ok(column.statistics().transpose()?.map(|statistics| {
                let primitive_type = &column.descriptor().descriptor.primitive_type;

                Statistics::Column(ColumnStatistics {
                    field: field.clone(),

                    logical_type: primitive_type.logical_type,
                    physical_type: primitive_type.physical_type,

                    statistics,
                })
            }))
        },
    }
}
