use std::cmp::Ordering;

use polars_utils::float16::pf16;

use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{SortOrder, get_sort_order};
use crate::parquet::schema::types::{PhysicalType, PrimitiveLogicalType, PrimitiveType};
use crate::parquet::statistics::*;
use crate::parquet::types::NativeType;

/// How the byte encoded bounds of a column are compared while merging statistics.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BoundsOrder {
    /// Unsigned comparison of the bytes.
    Bytes,
    /// Comparison as big-endian two's complement values.
    SignedBytes,
    /// Comparison as little-endian half floats.
    Float16,
    /// The bounds have no defined order and must be dropped.
    Undefined,
}

fn bounds_order(primitive_type: &PrimitiveType) -> BoundsOrder {
    if primitive_type.logical_type == Some(PrimitiveLogicalType::Float16) {
        return BoundsOrder::Float16;
    }

    match get_sort_order(
        &primitive_type.logical_type,
        &primitive_type.converted_type,
        &primitive_type.physical_type,
    ) {
        // Float and double are never encoded as bytes, so their total order is not reachable here.
        SortOrder::Unsigned | SortOrder::IEEE754TotalOrder => BoundsOrder::Bytes,
        SortOrder::Signed => BoundsOrder::SignedBytes,
        SortOrder::Undefined => BoundsOrder::Undefined,
    }
}

/// Compares two big-endian two's complement values, which may have different lengths.
fn compare_signed_bytes(x: &[u8], y: &[u8]) -> Ordering {
    let is_negative = |v: &[u8]| v.first().is_some_and(|b| b & 0x80 != 0);
    match (is_negative(x), is_negative(y)) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {},
    }

    // Sign-extend the shorter value to the length of the longer one.
    let fill = if is_negative(x) { 0xff } else { 0x00 };
    let len = x.len().max(y.len());
    let byte_at = |v: &[u8], i: usize| {
        let offset = len - v.len();
        if i < offset { fill } else { v[i - offset] }
    };
    (0..len)
        .map(|i| byte_at(x, i).cmp(&byte_at(y, i)))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

fn compare_bounds(order: BoundsOrder, x: &[u8], y: &[u8]) -> Ordering {
    match order {
        BoundsOrder::Bytes | BoundsOrder::Undefined => x.cmp(y),
        BoundsOrder::SignedBytes => compare_signed_bytes(x, y),
        BoundsOrder::Float16 => {
            let to_f16 =
                |v: &[u8]| <pf16 as NativeType>::from_le_bytes(v.try_into().unwrap_or_default());
            to_f16(x).ord(&to_f16(y))
        },
    }
}

#[inline]
fn reduce_single<T, F: Fn(T, T) -> T>(lhs: Option<T>, rhs: Option<T>, op: F) -> Option<T> {
    match (lhs, rhs) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some(op(x, y)),
    }
}

#[inline]
fn reduce_vec8(
    lhs: Option<Vec<u8>>,
    rhs: &Option<Vec<u8>>,
    max: bool,
    order: BoundsOrder,
) -> Option<Vec<u8>> {
    if order == BoundsOrder::Undefined {
        return None;
    }

    let take_min = !max;
    match (lhs, rhs) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(x)) => Some(x.clone()),
        (Some(x), Some(y)) => Some(if compare_bounds(order, &x, y).is_le() == take_min {
            x
        } else {
            y.clone()
        }),
    }
}

pub fn reduce(stats: &[&Option<Statistics>]) -> ParquetResult<Option<Statistics>> {
    if stats.is_empty() {
        return Ok(None);
    }
    let stats = stats
        .iter()
        .filter_map(|x| x.as_ref())
        .collect::<Vec<&Statistics>>();
    if stats.is_empty() {
        return Ok(None);
    };

    let same_type = stats
        .iter()
        .skip(1)
        .all(|x| x.physical_type() == stats[0].physical_type());
    if !same_type {
        return Err(ParquetError::oos(
            "The statistics do not have the same dtype",
        ));
    };

    use PhysicalType as T;
    let stats = match stats[0].physical_type() {
        T::Boolean => reduce_boolean(stats.iter().map(|x| x.expect_as_boolean())).into(),
        T::Int32 => reduce_primitive::<i32, _>(stats.iter().map(|x| x.expect_as_int32())).into(),
        T::Int64 => reduce_primitive(stats.iter().map(|x| x.expect_as_int64())).into(),
        T::Float => reduce_primitive(stats.iter().map(|x| x.expect_as_float())).into(),
        T::Double => reduce_primitive(stats.iter().map(|x| x.expect_as_double())).into(),
        T::ByteArray => {
            let order = bounds_order(&stats[0].expect_as_binary().primitive_type);
            reduce_binary(stats.iter().map(|x| x.expect_as_binary()), order).into()
        },
        T::FixedLenByteArray(_) => {
            let order = bounds_order(&stats[0].expect_as_fixedlen().primitive_type);
            reduce_fix_len_binary(stats.iter().map(|x| x.expect_as_fixedlen()), order).into()
        },
        _ => todo!(),
    };

    Ok(Some(stats))
}

fn reduce_binary<'a, I: Iterator<Item = &'a BinaryStatistics>>(
    mut stats: I,
    order: BoundsOrder,
) -> BinaryStatistics {
    let initial = stats.next().unwrap().clone();
    stats.fold(initial, |mut acc, new| {
        acc.min_value = reduce_vec8(acc.min_value, &new.min_value, false, order);
        acc.max_value = reduce_vec8(acc.max_value, &new.max_value, true, order);
        acc.null_count = reduce_single(acc.null_count, new.null_count, |x, y| x + y);
        acc.distinct_count = None;
        acc
    })
}

fn reduce_fix_len_binary<'a, I: Iterator<Item = &'a FixedLenStatistics>>(
    mut stats: I,
    order: BoundsOrder,
) -> FixedLenStatistics {
    let initial = stats.next().unwrap().clone();
    stats.fold(initial, |mut acc, new| {
        acc.min_value = reduce_vec8(acc.min_value, &new.min_value, false, order);
        acc.max_value = reduce_vec8(acc.max_value, &new.max_value, true, order);
        acc.null_count = reduce_single(acc.null_count, new.null_count, |x, y| x + y);
        acc.distinct_count = None;
        acc
    })
}

fn reduce_boolean<'a, I: Iterator<Item = &'a BooleanStatistics>>(
    mut stats: I,
) -> BooleanStatistics {
    let initial = stats.next().unwrap().clone();
    stats.fold(initial, |mut acc, new| {
        acc.min_value = reduce_single(
            acc.min_value,
            new.min_value,
            |x, y| if x & !(y) { y } else { x },
        );
        acc.max_value = reduce_single(
            acc.max_value,
            new.max_value,
            |x, y| if x & !(y) { x } else { y },
        );
        acc.null_count = reduce_single(acc.null_count, new.null_count, |x, y| x + y);
        acc.distinct_count = None;
        acc
    })
}

fn reduce_primitive<
    'a,
    T: NativeType + std::cmp::PartialOrd,
    I: Iterator<Item = &'a PrimitiveStatistics<T>>,
>(
    mut stats: I,
) -> PrimitiveStatistics<T> {
    let initial = stats.next().unwrap().clone();
    let mut result = stats.fold(initial, |mut acc, new| {
        acc.min_value = reduce_single(
            acc.min_value,
            new.min_value,
            |x, y| if x > y { y } else { x },
        );
        acc.max_value = reduce_single(
            acc.max_value,
            new.max_value,
            |x, y| if x > y { x } else { y },
        );
        acc.null_count = reduce_single(acc.null_count, new.null_count, |x, y| x + y);
        acc.distinct_count = None;
        acc
    });
    result.min_value = result.min_value.map(|v| v.norm_min());
    result.max_value = result.max_value.map(|v| v.norm_max());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary() -> ParquetResult<()> {
        let iter = [
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::ByteArray),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![3, 4]),
            },
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::ByteArray),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![4, 5]),
                max_value: None,
            },
        ];
        let a = reduce_binary(iter.iter(), BoundsOrder::Bytes);

        assert_eq!(
            a,
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::ByteArray,),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![3, 4]),
            },
        );

        Ok(())
    }

    #[test]
    fn fixed_len_binary() -> ParquetResult<()> {
        let iter = [
            FixedLenStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".into(),
                    PhysicalType::FixedLenByteArray(2),
                ),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![3, 4]),
            },
            FixedLenStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".into(),
                    PhysicalType::FixedLenByteArray(2),
                ),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![4, 5]),
                max_value: None,
            },
        ];
        let a = reduce_fix_len_binary(iter.iter(), BoundsOrder::Bytes);

        assert_eq!(
            a,
            FixedLenStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".into(),
                    PhysicalType::FixedLenByteArray(2),
                ),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![3, 4]),
            },
        );

        Ok(())
    }

    #[test]
    fn boolean() -> ParquetResult<()> {
        let iter = [
            BooleanStatistics {
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(false),
                max_value: Some(false),
            },
            BooleanStatistics {
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(true),
                max_value: Some(true),
            },
        ];
        let a = reduce_boolean(iter.iter());

        assert_eq!(
            a,
            BooleanStatistics {
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(false),
                max_value: Some(true),
            },
        );

        Ok(())
    }

    #[test]
    fn primitive() -> ParquetResult<()> {
        let iter = [PrimitiveStatistics {
            null_count: Some(2),
            distinct_count: None,
            min_value: Some(30),
            max_value: Some(70),
            primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::Int32),
        }];
        let a = reduce_primitive(iter.iter());

        assert_eq!(
            a,
            PrimitiveStatistics {
                null_count: Some(2),
                distinct_count: None,
                min_value: Some(30),
                max_value: Some(70),
                primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::Int32,),
            },
        );

        Ok(())
    }

    #[test]
    fn binary_prefix_ordering() -> ParquetResult<()> {
        // Here [1, 2] is a prefix of [1, 2, 0].
        // Lexicographically: [1, 2] < [1, 2, 0],
        // so min must be [1, 2] and max must be [1, 2, 0].
        let iter = [
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::ByteArray),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![1, 2]),
            },
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical("bla".into(), PhysicalType::ByteArray),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2, 0]),
                max_value: Some(vec![1, 2, 0]),
            },
        ];

        let a = reduce_binary(iter.iter(), BoundsOrder::Bytes);

        assert_eq!(a.min_value, Some(vec![1, 2]));
        assert_eq!(a.max_value, Some(vec![1, 2, 0]));
        assert_eq!(a.null_count, Some(0));
        assert_eq!(a.distinct_count, None);

        Ok(())
    }

    #[test]
    fn test_reduce_vec8_equal_prefix_min_max() -> ParquetResult<()> {
        let a = vec![1, 2];
        let b = vec![1, 2, 0];

        // For max=true, we expect the longer (lexicographically larger) value.
        let max_val =
            reduce_vec8(Some(a.clone()), &Some(b.clone()), true, BoundsOrder::Bytes).unwrap();
        assert_eq!(max_val, b);

        // For max=false, we expect the shorter (lexicographically smaller) value.
        let min_val = reduce_vec8(Some(a.clone()), &Some(b), false, BoundsOrder::Bytes).unwrap();
        assert_eq!(min_val, a);

        Ok(())
    }
}
