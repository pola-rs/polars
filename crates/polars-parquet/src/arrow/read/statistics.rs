//! APIs exposing `crate::parquet`'s statistics as arrow's statistics.

use ethnum::I256;
use num_traits::{AsPrimitive, FromBytes};
use polars_arrow::array::{
    Array, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, MutableBinaryViewArray,
    MutableBooleanArray, MutableFixedSizeBinaryArray, MutablePrimitiveArray, NullArray,
    PrimitiveArray, Utf8ViewArray, new_null_array,
};
use polars_arrow::datatypes::{ArrowDataType, Field, IntegerType, IntervalUnit, TimeUnit};
use polars_arrow::types::{days_ms, i256};
use polars_utils::IdxSize;
use polars_utils::float16::pf16;
use polars_utils::pl_str::PlSmallStr;

use super::{FileMetadata, ParquetTimeUnit, RowGroupMetadata};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnOrder, SortOrder};
use crate::parquet::schema::types::PhysicalType as ParquetPhysicalType;
use crate::parquet::statistics::Statistics as ParquetStatistics;
use crate::read::{
    ColumnChunkMetadata, PrimitiveLogicalType, convert_days_ms, convert_i128, convert_i256,
    convert_year_month,
};

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

    logical_type: Option<PrimitiveLogicalType>,
    physical_type: ParquetPhysicalType,
    column_order: ColumnOrder,

    /// Statistics of the leaf array of the column
    statistics: ParquetStatistics,
}

/// The sort order the statistics of a leaf must have been collected with for its
/// `min_value` and `max_value` to bound the values polars decodes as `dtype`, or `None`
/// when no such order exists.
fn required_sort_order(
    dtype: &ArrowDataType,
    physical_type: &ParquetPhysicalType,
) -> Option<SortOrder> {
    use ArrowDataType as D;
    use ParquetPhysicalType as P;
    use SortOrder as O;
    Some(match (dtype, physical_type) {
        (D::Boolean, P::Boolean) => O::Unsigned,
        (D::Int8 | D::Int16 | D::Int32 | D::Date32 | D::Time32(_), P::Int32) => O::Signed,
        (D::Int64 | D::Time64(_) | D::Duration(_) | D::Timestamp(..), P::Int64) => O::Signed,
        (D::Date64, P::Int32 | P::Int64) => O::Signed,
        (D::UInt8 | D::UInt16 | D::UInt32, P::Int32) => O::Unsigned,
        (D::UInt32 | D::UInt64, P::Int64) => O::Unsigned,
        (D::Float16, P::FixedLenByteArray(2)) => O::Signed,
        (D::Float32, P::Float) | (D::Float64, P::Double) => O::Signed,
        (D::Decimal(..) | D::Decimal256(..), P::Int32 | P::Int64 | P::FixedLenByteArray(_)) => {
            O::Signed
        },
        (
            D::Binary | D::LargeBinary | D::BinaryView | D::Utf8 | D::LargeUtf8 | D::Utf8View,
            P::ByteArray,
        ) => O::Unsigned,
        (D::FixedSizeBinary(width), P::FixedLenByteArray(len)) if width == len => O::Unsigned,
        _ => return None,
    })
}

/// Whether the `min_value` and `max_value` of a leaf bound the values polars decodes
/// as `dtype`: the file declares an order for the leaf, and it is the one polars
/// compares the decoded values with.
fn bounds_are_usable(
    column_order: ColumnOrder,
    physical_type: &ParquetPhysicalType,
    dtype: &ArrowDataType,
) -> bool {
    let Some(required) = required_sort_order(dtype, physical_type) else {
        return false;
    };
    match column_order {
        ColumnOrder::TypeDefinedOrder(order) => order == required,
        // Total order agrees with the float comparison on every non-NaN value.
        ColumnOrder::IEEE754TotalOrder => {
            matches!(
                physical_type,
                ParquetPhysicalType::Float | ParquetPhysicalType::Double
            )
        },
        ColumnOrder::Unsupported | ColumnOrder::Undefined => false,
    }
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
                        let x = x.map(|x| AsPrimitive::<$to>::as_(x));
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

        if !bounds_are_usable(self.column_order, &self.physical_type, self.field.dtype()) {
            return Ok(ArrowColumnStatistics {
                null_count,
                distinct_count,
                min_value: None,
                max_value: None,
            });
        }

        use ArrowDataType as D;
        use ParquetPhysicalType as PPT;
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

            (D::Timestamp(time_unit, _), PPT::Int64) => {
                rmap!(expect_int64, @prim i64, |x| {
                    timestamp(self.logical_type.as_ref(), *time_unit, x)
                })
            },

            (D::Float16, PPT::FixedLenByteArray(2))
                if matches!(
                    self.logical_type.as_ref(),
                    Some(PrimitiveLogicalType::Float16)
                ) =>
            {
                rmap!(expect_fixedlen, @prim Vec<u8>, |v| pf16::from_le_bytes(&[v[0], v[1]]))
            },
            (D::Float32, _) => rmap!(expect_float, @prim f32),
            (D::Float64, _) => rmap!(expect_double, @prim f64),

            (D::Decimal(_, _), PPT::Int32) => rmap!(expect_int32, @prim i32 as i128),
            (D::Decimal(_, _), PPT::Int64) => rmap!(expect_int64, @prim i64 as i128),
            (D::Decimal(_, _), PPT::FixedLenByteArray(n)) if *n > 16 => {
                return Err(ParquetError::not_supported(format!(
                    "Can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}",
                )));
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
                )));
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

            _ => (None, None),
        };

        Ok(ArrowColumnStatistics {
            null_count,
            distinct_count,

            min_value,
            max_value,
        })
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

        _ => {
            let primitive_type = &row_groups[0].parquet_columns()[field_idx]
                .descriptor()
                .descriptor
                .primitive_type;

            let logical_type = &primitive_type.logical_type;
            let physical_type = &primitive_type.physical_type;

            if !matches!(field.dtype(), D::Null)
                && !bounds_are_usable(column_order, physical_type, field.dtype())
            {
                return Ok(Some(unavailable_bounds(field, row_groups, field_idx)));
            }

            let mut null_count = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
            let mut distinct_count =
                MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());

            macro_rules! rmap {
                ($expect:ident, $map:expr, $arr:ty$(, $arg:expr)?) => {{
                    let mut min_arr = <$arr>::with_capacity(row_groups.len()$(, $arg)?);
                    let mut max_arr = <$arr>::with_capacity(row_groups.len()$(, $arg)?);

                    for rg in row_groups {
                        let column = &rg.parquet_columns()[field_idx];
                        let s = column.statistics(footer_buf).transpose()?;

                        let (v_min, v_max, v_null_count, v_distinct_count) = match s {
                            None => (None, None, None, None),
                            Some(s) => {
                                let s = s.$expect();

                                let min = s.min_value;
                                let max = s.max_value;

                                let min = ($map)(min)?;
                                let max = ($map)(max)?;

                                (
                                min,
                                max,
                                s.null_count.map(|v| v as IdxSize),
                                s.distinct_count.map(|v| v as IdxSize),
                                )
                            }
                        };

                        min_arr.push(v_min);
                        max_arr.push(v_max);
                        null_count.push(v_null_count);
                        distinct_count.push(v_distinct_count);
                    }

                    (min_arr.freeze().to_boxed(), max_arr.freeze().to_boxed())
                }};
                ($expect:ident, $arr:ty, @prim $from:ty $(as $to:ty)? $(, $map:expr)?) => {{
                    rmap!(
                        $expect,
                        |x: Option<$from>| {
                            $(
                            let x = x.map(|x| AsPrimitive::<$to>::as_(x));
                            )?
                            $(
                            let x = x.map($map);
                            )?
                            ParquetResult::Ok(x)
                        },
                        $arr
                    )
                }};
                (@binary $(, $map:expr)?) => {{
                    rmap!(
                        expect_binary,
                        |x: Option<Vec<u8>>| {
                            $(
                            let x = x.map($map);
                            )?
                            ParquetResult::Ok(x)
                        },
                        MutableBinaryViewArray<[u8]>
                    )
                }};
                (@string) => {{
                    rmap!(
                        expect_binary,
                        |x: Option<Vec<u8>>| {
                            let x = x.map(String::from_utf8).transpose().map_err(|_| {
                                ParquetError::oos("Invalid UTF8 in Statistics")
                            })?;
                            ParquetResult::Ok(x)
                        },
                        MutableBinaryViewArray<str>
                    )
                }};
            }

            use ArrowDataType as D;
            use ParquetPhysicalType as PPT;
            let (min_value, max_value) = match (field.dtype(), physical_type) {
                (D::Null, _) => {
                    for rg in row_groups {
                        null_count.push(Some(rg.num_rows() as IdxSize));
                        distinct_count.push(Some(0));
                    }
                    (
                        NullArray::new(ArrowDataType::Null, row_groups.len()).to_boxed(),
                        NullArray::new(ArrowDataType::Null, row_groups.len()).to_boxed(),
                    )
                },

                (D::Boolean, _) => rmap!(
                    expect_boolean,
                    |x: Option<bool>| ParquetResult::Ok(x),
                    MutableBooleanArray
                ),

                (D::Int8, _) => rmap!(expect_int32, MutablePrimitiveArray::<i8>, @prim i32 as i8),
                (D::Int16, _) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<i16>, @prim i32 as i16)
                },
                (D::Int32 | D::Date32 | D::Time32(_), _) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<i32>, @prim i32 as i32)
                },

                // some implementations of parquet write arrow's date64 into i32.
                (D::Date64, PPT::Int32) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<i64>, @prim i32 as i64, |x| x * 86400000)
                },

                (D::Int64 | D::Time64(_) | D::Duration(_), _) | (D::Date64, PPT::Int64) => {
                    rmap!(expect_int64, MutablePrimitiveArray::<i64>, @prim i64 as i64)
                },

                (D::Interval(IntervalUnit::YearMonth), _) => rmap!(
                    expect_binary,
                    MutablePrimitiveArray::<i32>,
                    @prim Vec<u8>,
                    |x| convert_year_month(&x)
                ),
                (D::Interval(IntervalUnit::DayTime), _) => rmap!(
                    expect_binary,
                    MutablePrimitiveArray::<days_ms>,
                    @prim Vec<u8>,
                    |x| convert_days_ms(&x)
                ),

                (D::UInt8, _) => rmap!(expect_int32, MutablePrimitiveArray::<u8>, @prim i32 as u8),
                (D::UInt16, _) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<u16>, @prim i32 as u16)
                },
                (D::UInt32, PPT::Int32) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<u32>, @prim i32 as u32)
                },

                // some implementations of parquet write arrow's u32 into i64.
                (D::UInt32, PPT::Int64) => {
                    rmap!(expect_int64, MutablePrimitiveArray::<u32>, @prim i64 as u32)
                },
                (D::UInt64, _) => {
                    rmap!(expect_int64, MutablePrimitiveArray::<u64>, @prim i64 as u64)
                },

                (D::Timestamp(time_unit, _), PPT::Int64) => {
                    rmap!(expect_int64, MutablePrimitiveArray::<i64>, @prim i64, |x| {
                        timestamp(logical_type.as_ref(), *time_unit, x)
                    })
                },

                (D::Float16, _) => {
                    rmap!(expect_fixedlen, MutablePrimitiveArray::<pf16>, @prim Vec<u8>, |v| {
                        let le_bytes: [u8; 2] = [v[0], v[1]];
                        pf16::from_le_bytes(&le_bytes)
                    })
                },
                (D::Float32, _) => rmap!(expect_float, MutablePrimitiveArray::<f32>, @prim f32),
                (D::Float64, _) => rmap!(expect_double, MutablePrimitiveArray::<f64>, @prim f64),

                (D::Decimal(_, _), PPT::Int32) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<i128>, @prim i32 as i128)
                },
                (D::Decimal(_, _), PPT::Int64) => {
                    rmap!(expect_int64, MutablePrimitiveArray::<i128>, @prim i64 as i128)
                },
                (D::Decimal(_, _), PPT::FixedLenByteArray(n)) if *n > 16 => {
                    return Err(ParquetError::not_supported(format!(
                        "Can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}",
                    )));
                },
                (D::Decimal(_, _), PPT::FixedLenByteArray(n)) => rmap!(
                    expect_fixedlen,
                    MutablePrimitiveArray::<i128>,
                    @prim Vec<u8>,
                    |x| convert_i128(&x, *n)
                ),
                (D::Decimal256(_, _), PPT::Int32) => {
                    rmap!(expect_int32, MutablePrimitiveArray::<i256>, @prim i32, |x: i32| i256(I256::new(x.into())))
                },
                (D::Decimal256(_, _), PPT::Int64) => {
                    rmap!(expect_int64, MutablePrimitiveArray::<i256>, @prim i64, |x: i64| i256(I256::new(x.into())))
                },
                (D::Decimal256(_, _), PPT::FixedLenByteArray(n)) if *n > 16 => {
                    return Err(ParquetError::not_supported(format!(
                        "Can't decode Decimal256 type from Fixed Size Byte Array of len {n:?}",
                    )));
                },
                (D::Decimal256(_, _), PPT::FixedLenByteArray(_)) => rmap!(
                    expect_fixedlen,
                    MutablePrimitiveArray::<i256>,
                    @prim Vec<u8>,
                    |x| convert_i256(&x)
                ),
                (D::Binary, _) => rmap!(@binary),
                (D::LargeBinary, _) => rmap!(@binary),
                (D::Utf8, _) => rmap!(@string),
                (D::LargeUtf8, _) => rmap!(@string),

                (D::BinaryView, _) => rmap!(@binary),
                (D::Utf8View, _) => rmap!(@string),

                (D::FixedSizeBinary(width), _) => {
                    struct FixedSizeBinaryArray2;

                    impl FixedSizeBinaryArray2 {
                        fn with_capacity(
                            row_groups_len: usize,
                            row_width: usize,
                        ) -> MutableFixedSizeBinaryArray {
                            MutableFixedSizeBinaryArray::with_capacity(row_width, row_groups_len)
                        }
                    }

                    rmap!(
                        expect_fixedlen,
                        |x: Option<Vec<u8>>| ParquetResult::Ok(x),
                        FixedSizeBinaryArray2,
                        *width
                    )
                },

                _ => return Ok(Some(unavailable_bounds(field, row_groups, field_idx))),
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

                        logical_type: primitive_type.logical_type,
                        physical_type: primitive_type.physical_type,
                        column_order: metadata.column_order(column.leaf_index()),

                        statistics,
                    }))
                }))
        },
    }
}

fn seconds_to_millis(unit: &TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1_000,
        _ => 1,
    }
}

fn nanoseconds_per(unit: &TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// A bound of a leaf column decoded from its statistics: every integer-like
/// value as one integer, in the unit polars decodes the column with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StatBound<'a> {
    Int(i128),
    Bytes(&'a [u8]),
}

/// Reads the min and max of one leaf column straight from the footer bytes, for
/// a field polars decodes as `dtype`, without materialising statistics arrays.
pub struct LeafBounds {
    field_idx: usize,
    dtype: ArrowDataType,
    physical_type: ParquetPhysicalType,
    logical_type: Option<PrimitiveLogicalType>,
}

impl LeafBounds {
    /// `None` when the leaf's statistics cannot bound the values polars decodes
    /// as `field`'s type, or the type has no decoding here.
    pub fn new(field: &Field, metadata: &FileMetadata, field_idx: usize) -> Option<Self> {
        let primitive_type = &metadata.schema_descr.columns()[field_idx]
            .descriptor
            .primitive_type;
        let physical_type = primitive_type.physical_type;
        let column_order = metadata.column_order(field_idx);
        if !bounds_are_usable(column_order, &physical_type, field.dtype()) {
            return None;
        }
        use ArrowDataType as D;
        use ParquetPhysicalType as P;
        let supported = matches!(
            (field.dtype(), &physical_type),
            (D::Boolean, P::Boolean)
                | (
                    D::Int8
                        | D::Int16
                        | D::Int32
                        | D::Date32
                        | D::Time32(_)
                        | D::UInt8
                        | D::UInt16
                        | D::UInt32
                        | D::Date64,
                    P::Int32
                )
                | (
                    D::Int64
                        | D::Time64(_)
                        | D::Duration(_)
                        | D::Timestamp(..)
                        | D::UInt32
                        | D::UInt64
                        | D::Date64,
                    P::Int64
                )
                | (D::Decimal(..), P::Int32 | P::Int64)
                | (
                    D::Binary
                        | D::LargeBinary
                        | D::BinaryView
                        | D::Utf8
                        | D::LargeUtf8
                        | D::Utf8View,
                    P::ByteArray
                )
        ) || matches!(
            (field.dtype(), &physical_type),
            (D::Decimal(..), P::FixedLenByteArray(n)) if *n <= 16
        ) || matches!(
            (field.dtype(), &physical_type),
            (D::FixedSizeBinary(width), P::FixedLenByteArray(len)) if width == len
        );
        supported.then_some(Self {
            field_idx,
            dtype: field.dtype().clone(),
            physical_type,
            logical_type: primitive_type.logical_type,
        })
    }

    /// The min and max of the leaf in `row_group`. Either is `None` when the
    /// chunk does not give it or does not give it exactly.
    pub fn bounds<'a>(
        &self,
        row_group: &RowGroupMetadata,
        footer_buf: &'a [u8],
    ) -> (Option<StatBound<'a>>, Option<StatBound<'a>>) {
        let column = &row_group.parquet_columns()[self.field_idx];
        let Some(stats) = &column.compact_metadata().statistics else {
            return (None, None);
        };
        let min = stats
            .min_value
            .filter(|_| !stats.is_min_value_exact.is_some_and(|exact| !exact))
            .and_then(|range| self.decode(range.resolve(footer_buf)));
        let max = stats
            .max_value
            .filter(|_| !stats.is_max_value_exact.is_some_and(|exact| !exact))
            .and_then(|range| self.decode(range.resolve(footer_buf)));
        (min, max)
    }

    fn decode<'a>(&self, bytes: &'a [u8]) -> Option<StatBound<'a>> {
        use ArrowDataType as D;
        use ParquetPhysicalType as P;
        let int32 = || Some(i32::from_le_bytes(bytes.try_into().ok()?));
        let int64 = || Some(i64::from_le_bytes(bytes.try_into().ok()?));
        let value = match (&self.dtype, &self.physical_type) {
            (D::Boolean, P::Boolean) => (bytes.len() == 1).then(|| (bytes[0] != 0) as i128)?,
            (D::Int8, P::Int32) => int32()? as i8 as i128,
            (D::Int16, P::Int32) => int32()? as i16 as i128,
            (D::Int32 | D::Date32, P::Int32) => int32()? as i128,
            // Polars keeps time of day in nanoseconds.
            (D::Time32(unit), P::Int32) => int32()? as i128 * nanoseconds_per(unit),
            (D::Time64(unit), P::Int64) => int64()? as i128 * nanoseconds_per(unit),
            (D::Date64, P::Int32) => int32()? as i128 * 86400000,
            (D::UInt8, P::Int32) => int32()? as u8 as i128,
            (D::UInt16, P::Int32) => int32()? as u16 as i128,
            (D::UInt32, P::Int32) => int32()? as u32 as i128,
            (D::UInt32, P::Int64) => int64()? as u32 as i128,
            (D::UInt64, P::Int64) => int64()? as u64 as i128,
            (D::Int64 | D::Date64, P::Int64) => int64()? as i128,
            // Polars has no second resolution; seconds become milliseconds.
            (D::Duration(unit), P::Int64) => int64()? as i128 * seconds_to_millis(unit),
            (D::Timestamp(time_unit, _), P::Int64) => {
                timestamp(self.logical_type.as_ref(), *time_unit, int64()?) as i128
                    * seconds_to_millis(time_unit)
            },
            (D::Decimal(..), P::Int32) => int32()? as i128,
            (D::Decimal(..), P::Int64) => int64()? as i128,
            (D::Decimal(..), P::FixedLenByteArray(n)) => {
                (bytes.len() == *n).then(|| convert_i128(bytes, *n))?
            },
            (
                D::Binary | D::LargeBinary | D::BinaryView | D::Utf8 | D::LargeUtf8 | D::Utf8View,
                P::ByteArray,
            ) => return Some(StatBound::Bytes(bytes)),
            (D::FixedSizeBinary(width), P::FixedLenByteArray(_)) => {
                return (bytes.len() == *width).then_some(StatBound::Bytes(bytes));
            },
            _ => return None,
        };
        Some(StatBound::Int(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_bounds_decode_as_polars_does() {
        use ArrowDataType as D;
        use ParquetPhysicalType as P;
        let leaf = |dtype: D, physical_type: P, logical_type| LeafBounds {
            field_idx: 0,
            dtype,
            physical_type,
            logical_type,
        };
        let int = |v: i128| Some(StatBound::Int(v));

        assert_eq!(
            leaf(D::Int8, P::Int32, None).decode(&(-3i32).to_le_bytes()),
            int(-3)
        );
        assert_eq!(
            leaf(D::UInt8, P::Int32, None).decode(&255i32.to_le_bytes()),
            int(255)
        );
        assert_eq!(
            leaf(D::UInt32, P::Int64, None).decode(&(u32::MAX as i64).to_le_bytes()),
            int(u32::MAX as i128)
        );
        assert_eq!(
            leaf(D::UInt64, P::Int64, None).decode(&(-1i64).to_le_bytes()),
            int(u64::MAX as i128)
        );
        assert_eq!(leaf(D::Boolean, P::Boolean, None).decode(&[1]), int(1));
        assert_eq!(
            leaf(D::Time32(TimeUnit::Millisecond), P::Int32, None).decode(&7i32.to_le_bytes()),
            int(7_000_000)
        );
        assert_eq!(
            leaf(D::Duration(TimeUnit::Second), P::Int64, None).decode(&7i64.to_le_bytes()),
            int(7_000)
        );
        assert_eq!(
            leaf(D::Timestamp(TimeUnit::Second, None), P::Int64, None).decode(&7i64.to_le_bytes()),
            int(7_000)
        );
        assert_eq!(
            leaf(D::Time64(TimeUnit::Microsecond), P::Int64, None).decode(&7i64.to_le_bytes()),
            int(7_000)
        );
        assert_eq!(
            leaf(D::Date64, P::Int32, None).decode(&2i32.to_le_bytes()),
            int(2 * 86400000)
        );
        let micros = Some(PrimitiveLogicalType::Timestamp {
            unit: ParquetTimeUnit::Microseconds,
            is_adjusted_to_utc: false,
        });
        assert_eq!(
            leaf(D::Timestamp(TimeUnit::Millisecond, None), P::Int64, micros)
                .decode(&5_000i64.to_le_bytes()),
            int(5)
        );
        assert_eq!(
            leaf(D::Decimal(10, 2), P::FixedLenByteArray(2), None).decode(&[0xff, 0xfe]),
            int(-2)
        );
        assert_eq!(
            leaf(D::Utf8View, P::ByteArray, None).decode(b"abc"),
            Some(StatBound::Bytes(b"abc"))
        );
        // Malformed lengths give no bound.
        assert_eq!(leaf(D::Int32, P::Int32, None).decode(&[1, 2]), None);
        assert_eq!(
            leaf(D::FixedSizeBinary(4), P::FixedLenByteArray(4), None).decode(&[1, 2]),
            None
        );
        assert_eq!(
            leaf(D::Decimal(10, 2), P::FixedLenByteArray(2), None).decode(&[1]),
            None
        );
    }

    #[test]
    fn bounds_need_the_order_polars_compares_with() {
        use ArrowDataType as D;
        use ParquetPhysicalType as P;
        let typed = |order| ColumnOrder::TypeDefinedOrder(order);

        assert!(bounds_are_usable(
            typed(SortOrder::Signed),
            &P::Int64,
            &D::Int64
        ));
        assert!(bounds_are_usable(
            typed(SortOrder::Unsigned),
            &P::Int64,
            &D::UInt64
        ));
        assert!(bounds_are_usable(
            typed(SortOrder::Unsigned),
            &P::ByteArray,
            &D::Utf8View
        ));
        assert!(bounds_are_usable(
            typed(SortOrder::Signed),
            &P::Int32,
            &D::Date32
        ));
        assert!(bounds_are_usable(
            typed(SortOrder::Signed),
            &P::FixedLenByteArray(16),
            &D::Decimal(38, 2)
        ));
        assert!(bounds_are_usable(
            ColumnOrder::IEEE754TotalOrder,
            &P::Double,
            &D::Float64
        ));

        // Unsigned values compared as signed, and the other way round.
        assert!(!bounds_are_usable(
            typed(SortOrder::Signed),
            &P::Int64,
            &D::UInt64
        ));
        assert!(!bounds_are_usable(
            typed(SortOrder::Unsigned),
            &P::Int32,
            &D::Int32
        ));
        // A total order on integers, and orders the file does not declare.
        assert!(!bounds_are_usable(
            ColumnOrder::IEEE754TotalOrder,
            &P::Int64,
            &D::Int64
        ));
        assert!(!bounds_are_usable(
            ColumnOrder::Undefined,
            &P::Int64,
            &D::Int64
        ));
        assert!(!bounds_are_usable(
            ColumnOrder::Unsupported,
            &P::Int64,
            &D::Int64
        ));
        // Types polars cannot bound from statistics, or that do not match the leaf.
        assert!(!bounds_are_usable(
            typed(SortOrder::Signed),
            &P::Int96,
            &D::Timestamp(TimeUnit::Nanosecond, None)
        ));
        assert!(!bounds_are_usable(
            typed(SortOrder::Unsigned),
            &P::FixedLenByteArray(16),
            &D::Int128
        ));
        assert!(!bounds_are_usable(
            typed(SortOrder::Signed),
            &P::Int32,
            &D::Int64
        ));
        assert!(!bounds_are_usable(
            typed(SortOrder::Undefined),
            &P::FixedLenByteArray(12),
            &D::Interval(IntervalUnit::DayTime)
        ));
    }
}
