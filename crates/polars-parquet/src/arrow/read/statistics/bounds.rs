//! Decoding of the bounds a leaf's statistics store to the Arrow type polars
//! reads the leaf as.

use ethnum::I256;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::i256;
use polars_utils::float16::pf16;

use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnOrder, RawBounds, SortOrder};
use crate::parquet::schema::types::{PhysicalType, PrimitiveType};
use crate::parquet::statistics::Statistics as ParquetStatistics;
use crate::parquet::types::{self, NativeType};
use crate::read::deserialize::unify_timestamp_unit;
use crate::read::{convert_i128, convert_i256};

/// A bound as the file stores it: a plain-encoded value of the leaf's physical type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhysicalBound<'a> {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Int96([u32; 3]),
    Float(f32),
    Double(f64),
    /// A `ByteArray` or `FixedLenByteArray` value.
    Bytes(&'a [u8]),
}

impl<'a> PhysicalBound<'a> {
    /// Decodes the plain-encoded `bytes` of a bound of a `physical_type` leaf.
    pub fn decode(physical_type: PhysicalType, bytes: &'a [u8]) -> ParquetResult<Self> {
        fn fixed<T: NativeType>(bytes: &[u8]) -> ParquetResult<T> {
            if bytes.len() != size_of::<T>() {
                return Err(plain_encoding_error());
            }
            Ok(types::decode(bytes))
        }
        Ok(match physical_type {
            PhysicalType::Boolean => match bytes {
                [v] => Self::Boolean(*v != 0),
                _ => return Err(plain_encoding_error()),
            },
            PhysicalType::Int32 => Self::Int32(fixed(bytes)?),
            PhysicalType::Int64 => Self::Int64(fixed(bytes)?),
            PhysicalType::Int96 => Self::Int96(fixed(bytes)?),
            PhysicalType::Float => Self::Float(fixed(bytes)?),
            PhysicalType::Double => Self::Double(fixed(bytes)?),
            PhysicalType::ByteArray => Self::Bytes(bytes),
            PhysicalType::FixedLenByteArray(len) => {
                if bytes.len() != len {
                    return Err(plain_encoding_error());
                }
                Self::Bytes(bytes)
            },
        })
    }
}

fn plain_encoding_error() -> ParquetError {
    ParquetError::oos("The min_value and max_value of statistics MUST be plain encoded")
}

impl ParquetStatistics {
    /// The min and max the statistics hold, borrowed.
    pub fn bounds(&self) -> (Option<PhysicalBound<'_>>, Option<PhysicalBound<'_>>) {
        macro_rules! bounds {
            ($s:expr, $variant:ident) => {
                (
                    $s.min_value.as_ref().map(|v| PhysicalBound::$variant(*v)),
                    $s.max_value.as_ref().map(|v| PhysicalBound::$variant(*v)),
                )
            };
            ($s:expr, @bytes) => {
                (
                    $s.min_value.as_deref().map(PhysicalBound::Bytes),
                    $s.max_value.as_deref().map(PhysicalBound::Bytes),
                )
            };
        }
        match self {
            Self::Boolean(s) => bounds!(s, Boolean),
            Self::Int32(s) => bounds!(s, Int32),
            Self::Int64(s) => bounds!(s, Int64),
            Self::Int96(s) => bounds!(s, Int96),
            Self::Float(s) => bounds!(s, Float),
            Self::Double(s) => bounds!(s, Double),
            Self::Binary(s) => bounds!(s, @bytes),
            Self::FixedLen(s) => bounds!(s, @bytes),
        }
    }
}

/// A bound converted to the Arrow type polars reads the leaf as.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArrowBound<'a> {
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float16(pf16),
    Float32(f32),
    Float64(f64),
    Int128(i128),
    Int256(i256),
    Bytes(&'a [u8]),
    Str(&'a str),
}

/// The sort order the statistics of a leaf must have been collected with for its
/// `min_value` and `max_value` to bound the values polars decodes as `dtype`, or `None`
/// when no such order exists.
fn required_sort_order(dtype: &ArrowDataType, physical_type: &PhysicalType) -> Option<SortOrder> {
    use ArrowDataType as D;
    use PhysicalType as P;
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
    physical_type: &PhysicalType,
    dtype: &ArrowDataType,
) -> bool {
    let Some(required) = required_sort_order(dtype, physical_type) else {
        return false;
    };
    match column_order {
        ColumnOrder::TypeDefinedOrder(order) => order == required,
        // Total order agrees with the float comparison on every non-NaN value.
        ColumnOrder::IEEE754TotalOrder => {
            matches!(physical_type, PhysicalType::Float | PhysicalType::Double)
        },
        ColumnOrder::Unsupported | ColumnOrder::Undefined => false,
    }
}

/// How the bounds of a leaf convert to the Arrow type polars reads the leaf as,
/// resolved once per column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundConversion {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    /// Arrow `Date64` some writers store as `Int32` days.
    DaysToMillis,
    /// An `Int64` timestamp in the file's unit, scaled to the Arrow unit.
    Timestamp {
        factor: i64,
        multiply: bool,
    },
    Float16,
    Float32,
    Float64,
    Decimal128,
    Decimal256,
    Binary,
    Utf8,
}

impl BoundConversion {
    /// `None` when the leaf's statistics cannot bound the values polars decodes as
    /// `dtype`: no conversion exists, or the file did not collect them in the
    /// order polars compares with.
    pub fn new(
        dtype: &ArrowDataType,
        primitive_type: &PrimitiveType,
        column_order: ColumnOrder,
    ) -> ParquetResult<Option<Self>> {
        use ArrowDataType as D;
        use PhysicalType as P;
        let physical_type = &primitive_type.physical_type;
        if !bounds_are_usable(column_order, physical_type, dtype) {
            return Ok(None);
        }
        Ok(Some(match (dtype, physical_type) {
            (D::Boolean, _) => Self::Boolean,
            (D::Int8, _) => Self::Int8,
            (D::Int16, _) => Self::Int16,
            (D::Int32 | D::Date32 | D::Time32(_), _) => Self::Int32,
            (D::Date64, P::Int32) => Self::DaysToMillis,
            (D::Int64 | D::Time64(_) | D::Duration(_) | D::Date64, _) => Self::Int64,
            (D::UInt8, _) => Self::UInt8,
            (D::UInt16, _) => Self::UInt16,
            (D::UInt32, _) => Self::UInt32,
            (D::UInt64, _) => Self::UInt64,
            (D::Timestamp(time_unit, _), _) => {
                let (factor, multiply) =
                    unify_timestamp_unit(&primitive_type.logical_type, *time_unit);
                Self::Timestamp { factor, multiply }
            },
            (D::Float16, _) => Self::Float16,
            (D::Float32, _) => Self::Float32,
            (D::Float64, _) => Self::Float64,
            (D::Decimal(..), P::FixedLenByteArray(n)) if *n > 16 => {
                return Err(ParquetError::not_supported(format!(
                    "Can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}",
                )));
            },
            (D::Decimal(..), _) => Self::Decimal128,
            (D::Decimal256(..), P::FixedLenByteArray(n)) if *n > 16 => {
                return Err(ParquetError::not_supported(format!(
                    "Can't decode Decimal256 type from Fixed Size Byte Array of len {n:?}",
                )));
            },
            (D::Decimal256(..), _) => Self::Decimal256,
            (D::Utf8 | D::LargeUtf8 | D::Utf8View, _) => Self::Utf8,
            (D::Binary | D::LargeBinary | D::BinaryView | D::FixedSizeBinary(_), _) => Self::Binary,
            _ => return Ok(None),
        }))
    }

    /// Decodes and converts the bounds of a chunk of the leaf. Malformed bounds
    /// are errors whether exact or not; inexact ones then bound nothing.
    pub fn decode_bounds<'a>(
        self,
        physical_type: PhysicalType,
        raw: RawBounds<'a>,
    ) -> ParquetResult<(Option<ArrowBound<'a>>, Option<ArrowBound<'a>>)> {
        let decode = |bytes: Option<&'a [u8]>, is_exact: bool| -> ParquetResult<_> {
            let bound = bytes
                .map(|b| PhysicalBound::decode(physical_type, b))
                .transpose()?;
            Ok(bound.filter(|_| is_exact))
        };
        self.convert_bounds(
            decode(raw.min, raw.min_is_exact)?,
            decode(raw.max, raw.max_is_exact)?,
        )
    }

    /// Converts the bounds of a chunk of the leaf. A timestamp the reader wraps
    /// leaves the chunk's values without order, so neither bound holds then.
    pub fn convert_bounds<'a>(
        self,
        min: Option<PhysicalBound<'a>>,
        max: Option<PhysicalBound<'a>>,
    ) -> ParquetResult<(Option<ArrowBound<'a>>, Option<ArrowBound<'a>>)> {
        if [min, max].into_iter().flatten().any(|b| self.wraps(b)) {
            return Ok((None, None));
        }
        let convert = |bound: Option<_>| bound.map(|b| self.convert(b)).transpose();
        Ok((convert(min)?.flatten(), convert(max)?.flatten()))
    }

    fn wraps(self, bound: PhysicalBound) -> bool {
        match (self, bound) {
            (Self::Timestamp { factor, multiply }, PhysicalBound::Int64(v)) => {
                multiply && v.checked_mul(factor).is_none()
            },
            _ => false,
        }
    }

    /// Converts a bound of the leaf. `None` when the value bounds nothing: a NaN,
    /// or a timestamp the reader wraps.
    pub fn convert<'a>(self, bound: PhysicalBound<'a>) -> ParquetResult<Option<ArrowBound<'a>>> {
        use ArrowBound as A;
        use PhysicalBound as P;
        Ok(Some(match (self, bound) {
            (Self::Boolean, P::Boolean(v)) => A::Boolean(v),
            (Self::Int8, P::Int32(v)) => A::Int8(v as i8),
            (Self::Int16, P::Int32(v)) => A::Int16(v as i16),
            (Self::Int32, P::Int32(v)) => A::Int32(v),
            (Self::Int64, P::Int64(v)) => A::Int64(v),
            (Self::UInt8, P::Int32(v)) => A::UInt8(v as u8),
            (Self::UInt16, P::Int32(v)) => A::UInt16(v as u16),
            (Self::UInt32, P::Int32(v)) => A::UInt32(v as u32),
            (Self::UInt32, P::Int64(v)) => A::UInt32(v as u32),
            (Self::UInt64, P::Int64(v)) => A::UInt64(v as u64),
            (Self::DaysToMillis, P::Int32(v)) => A::Int64(i64::from(v) * 86400000),
            (
                Self::Timestamp {
                    factor,
                    multiply: false,
                },
                P::Int64(v),
            ) => A::Int64(v / factor),
            (
                Self::Timestamp {
                    factor,
                    multiply: true,
                },
                P::Int64(v),
            ) => {
                return Ok(v.checked_mul(factor).map(A::Int64));
            },
            (Self::Float16, P::Bytes(v)) => A::Float16(pf16::from_le_bytes([v[0], v[1]])),
            // Parquet Format:
            // > - If the min is a NaN, it should be ignored.
            // > - If the max is a NaN, it should be ignored.
            (Self::Float32, P::Float(v)) => return Ok((!v.is_nan()).then_some(A::Float32(v))),
            (Self::Float64, P::Double(v)) => return Ok((!v.is_nan()).then_some(A::Float64(v))),
            (Self::Decimal128, P::Int32(v)) => A::Int128(v as i128),
            (Self::Decimal128, P::Int64(v)) => A::Int128(v as i128),
            (Self::Decimal128, P::Bytes(v)) => A::Int128(convert_i128(v, v.len())),
            (Self::Decimal256, P::Int32(v)) => A::Int256(i256(I256::new(v.into()))),
            (Self::Decimal256, P::Int64(v)) => A::Int256(i256(I256::new(v.into()))),
            (Self::Decimal256, P::Bytes(v)) => A::Int256(convert_i256(v)),
            (Self::Binary, P::Bytes(v)) => A::Bytes(v),
            (Self::Utf8, P::Bytes(v)) => A::Str(
                std::str::from_utf8(v)
                    .map_err(|_| ParquetError::oos("Invalid UTF8 in Statistics"))?,
            ),
            _ => {
                return Err(ParquetError::oos(
                    "Statistics do not have the physical type of the column",
                ));
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use polars_arrow::datatypes::TimeUnit;

    use super::*;
    use crate::parquet::schema::types::{PrimitiveLogicalType, TimeUnit as ParquetTimeUnit};

    fn conversion(dtype: ArrowDataType, primitive_type: &PrimitiveType) -> BoundConversion {
        let order = ColumnOrder::TypeDefinedOrder(
            required_sort_order(&dtype, &primitive_type.physical_type).unwrap(),
        );
        BoundConversion::new(&dtype, primitive_type, order)
            .unwrap()
            .unwrap()
    }

    fn leaf(physical_type: PhysicalType) -> PrimitiveType {
        PrimitiveType::from_physical("leaf".into(), physical_type)
    }

    fn timestamp_leaf(unit: ParquetTimeUnit) -> PrimitiveType {
        let mut leaf = leaf(PhysicalType::Int64);
        leaf.logical_type = Some(PrimitiveLogicalType::Timestamp {
            unit,
            is_adjusted_to_utc: false,
        });
        leaf
    }

    fn try_decode<'a>(
        dtype: ArrowDataType,
        primitive_type: &PrimitiveType,
        bytes: &'a [u8],
    ) -> ParquetResult<Option<ArrowBound<'a>>> {
        let raw = RawBounds {
            min: Some(bytes),
            min_is_exact: true,
            ..Default::default()
        };
        Ok(conversion(dtype, primitive_type)
            .decode_bounds(primitive_type.physical_type, raw)?
            .0)
    }

    fn decode<'a>(
        dtype: ArrowDataType,
        primitive_type: &PrimitiveType,
        bytes: &'a [u8],
    ) -> Option<ArrowBound<'a>> {
        try_decode(dtype, primitive_type, bytes).unwrap()
    }

    #[test]
    fn bounds_decode_as_the_reader_does() {
        use ArrowDataType as D;
        use PhysicalType as P;
        let int32 = leaf(P::Int32);
        let int64 = leaf(P::Int64);

        assert_eq!(
            decode(D::Int8, &int32, &(-3i32).to_le_bytes()),
            Some(ArrowBound::Int8(-3))
        );
        assert_eq!(
            decode(D::UInt8, &int32, &255i32.to_le_bytes()),
            Some(ArrowBound::UInt8(255))
        );
        assert_eq!(
            decode(D::UInt32, &int64, &(u32::MAX as i64).to_le_bytes()),
            Some(ArrowBound::UInt32(u32::MAX))
        );
        assert_eq!(
            decode(D::UInt64, &int64, &(-1i64).to_le_bytes()),
            Some(ArrowBound::UInt64(u64::MAX))
        );
        assert_eq!(
            decode(D::Boolean, &leaf(P::Boolean), &[1]),
            Some(ArrowBound::Boolean(true))
        );
        assert_eq!(
            decode(
                D::Time32(TimeUnit::Millisecond),
                &int32,
                &7i32.to_le_bytes()
            ),
            Some(ArrowBound::Int32(7))
        );
        assert_eq!(
            decode(D::Duration(TimeUnit::Second), &int64, &7i64.to_le_bytes()),
            Some(ArrowBound::Int64(7))
        );
        assert_eq!(
            decode(D::Date64, &int32, &2i32.to_le_bytes()),
            Some(ArrowBound::Int64(2 * 86400000))
        );
        assert_eq!(
            decode(
                D::Timestamp(TimeUnit::Millisecond, None),
                &timestamp_leaf(ParquetTimeUnit::Microseconds),
                &(-5_001i64).to_le_bytes()
            ),
            Some(ArrowBound::Int64(-5))
        );
        assert_eq!(
            decode(
                D::Timestamp(TimeUnit::Second, None),
                &timestamp_leaf(ParquetTimeUnit::Nanoseconds),
                &3_000_000_000i64.to_le_bytes()
            ),
            Some(ArrowBound::Int64(3))
        );
        assert_eq!(
            decode(
                D::Timestamp(TimeUnit::Nanosecond, None),
                &timestamp_leaf(ParquetTimeUnit::Milliseconds),
                &3i64.to_le_bytes()
            ),
            Some(ArrowBound::Int64(3_000_000))
        );
        assert_eq!(
            decode(
                D::Decimal(10, 2),
                &leaf(P::FixedLenByteArray(2)),
                &[0xff, 0xfe]
            ),
            Some(ArrowBound::Int128(-2))
        );
        assert_eq!(
            decode(D::Decimal(10, 2), &int32, &(-7i32).to_le_bytes()),
            Some(ArrowBound::Int128(-7))
        );
        assert_eq!(
            decode(D::Utf8View, &leaf(P::ByteArray), b"abc"),
            Some(ArrowBound::Str("abc"))
        );
        assert_eq!(
            decode(D::FixedSizeBinary(2), &leaf(P::FixedLenByteArray(2)), b"ab"),
            Some(ArrowBound::Bytes(b"ab"))
        );
        assert_eq!(
            decode(D::Float32, &leaf(P::Float), &f32::NAN.to_le_bytes()),
            None
        );
        assert_eq!(
            decode(D::Float64, &leaf(P::Double), &(-0.0f64).to_le_bytes()),
            Some(ArrowBound::Float64(-0.0))
        );
    }

    #[test]
    fn wrapped_timestamps_bound_nothing() {
        use ArrowDataType as D;
        let conversion = conversion(
            D::Timestamp(TimeUnit::Nanosecond, None),
            &timestamp_leaf(ParquetTimeUnit::Milliseconds),
        );
        let (min, max) = conversion
            .convert_bounds(
                Some(PhysicalBound::Int64(0)),
                Some(PhysicalBound::Int64(i64::MAX / 1_000_000 + 1)),
            )
            .unwrap();
        assert_eq!((min, max), (None, None));
        let (min, max) = conversion
            .convert_bounds(
                Some(PhysicalBound::Int64(-1)),
                Some(PhysicalBound::Int64(1)),
            )
            .unwrap();
        assert_eq!(min, Some(ArrowBound::Int64(-1_000_000)));
        assert_eq!(max, Some(ArrowBound::Int64(1_000_000)));
    }

    #[test]
    fn inexact_bounds_are_validated_then_dropped() {
        use ArrowDataType as D;
        use PhysicalType as P;
        let int32 = leaf(P::Int32);
        let conversion = conversion(D::Int32, &int32);
        let seven = 7i32.to_le_bytes();
        let inexact = |bytes: &'static [u8]| RawBounds {
            min: Some(bytes),
            min_is_exact: false,
            max: Some(&seven[..]),
            max_is_exact: true,
        };
        assert!(
            conversion
                .decode_bounds(P::Int32, inexact(&[1, 2]))
                .is_err()
        );
        assert_eq!(
            conversion
                .decode_bounds(P::Int32, inexact(&[1, 2, 3, 4]))
                .unwrap(),
            (None, Some(ArrowBound::Int32(7)))
        );
        // The owned statistics apply the same order.
        use crate::parquet::statistics::ParquetStatistics as Thrift;
        let thrift = |min: Vec<u8>| Thrift {
            min_value: Some(min),
            is_min_value_exact: Some(false),
            max_value: Some(7i32.to_le_bytes().to_vec()),
            is_max_value_exact: None,
            min: None,
            max: None,
            null_count: None,
            distinct_count: None,
        };
        assert!(ParquetStatistics::deserialize(&thrift(vec![1, 2]), int32.clone()).is_err());
        let owned = ParquetStatistics::deserialize(&thrift(vec![1, 2, 3, 4]), int32).unwrap();
        let (min, max) = owned.bounds();
        assert_eq!(
            conversion.convert_bounds(min, max).unwrap(),
            (None, Some(ArrowBound::Int32(7)))
        );
    }

    #[test]
    fn malformed_bounds_are_errors() {
        use ArrowDataType as D;
        use PhysicalType as P;
        assert!(try_decode(D::Int32, &leaf(P::Int32), &[1, 2]).is_err());
        assert!(
            try_decode(
                D::FixedSizeBinary(4),
                &leaf(P::FixedLenByteArray(4)),
                &[1, 2]
            )
            .is_err()
        );
        assert!(try_decode(D::Decimal(10, 2), &leaf(P::FixedLenByteArray(2)), &[1]).is_err());
        assert!(try_decode(D::Utf8View, &leaf(P::ByteArray), &[0xff]).is_err());
        assert!(try_decode(D::Boolean, &leaf(P::Boolean), &[]).is_err());
        assert!(
            BoundConversion::new(
                &D::Decimal(38, 0),
                &leaf(P::FixedLenByteArray(17)),
                ColumnOrder::TypeDefinedOrder(SortOrder::Signed)
            )
            .is_err()
        );
    }

    #[test]
    fn bounds_need_the_order_polars_compares_with() {
        use ArrowDataType as D;
        use PhysicalType as P;
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
            &D::Interval(polars_arrow::datatypes::IntervalUnit::DayTime)
        ));
    }
}
