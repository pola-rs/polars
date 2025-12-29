use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::types::PhysicalType;
use crate::parquet::statistics::*;
use crate::parquet::types::NativeType;

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
fn reduce_vec8(lhs: Option<Vec<u8>>, rhs: &Option<Vec<u8>>, max: bool) -> Option<Vec<u8>> {
    let take_min = !max;

    match (lhs, rhs) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(x)) => Some(x.clone()),
        (Some(x), Some(y)) => Some(if (&x <= y) == take_min { x } else { y.clone() }),
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
        T::ByteArray => reduce_binary(stats.iter().map(|x| x.expect_as_binary())).into(),
        T::FixedLenByteArray(_) => {
            reduce_fix_len_binary(stats.iter().map(|x| x.expect_as_fixedlen())).into()
        },
        _ => todo!(),
    };

    Ok(Some(stats))
}

fn reduce_binary<'a, I: Iterator<Item = &'a BinaryStatistics>>(mut stats: I) -> BinaryStatistics {
    let initial = stats.next().unwrap().clone();
    stats.fold(initial, |mut acc, new| {
        acc.min_value = reduce_vec8(acc.min_value, &new.min_value, false);
        acc.max_value = reduce_vec8(acc.max_value, &new.max_value, true);
        acc.null_count = reduce_single(acc.null_count, new.null_count, |x, y| x + y);
        acc.distinct_count = None;
        acc
    })
}

fn reduce_fix_len_binary<'a, I: Iterator<Item = &'a FixedLenStatistics>>(
    mut stats: I,
) -> FixedLenStatistics {
    let initial = stats.next().unwrap().clone();
    stats.fold(initial, |mut acc, new| {
        acc.min_value = reduce_vec8(acc.min_value, &new.min_value, false);
        acc.max_value = reduce_vec8(acc.max_value, &new.max_value, true);
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
    stats.fold(initial, |mut acc, new| {
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::schema::types::PrimitiveType;

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
        let a = reduce_binary(iter.iter());

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
        let a = reduce_fix_len_binary(iter.iter());

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

        let a = reduce_binary(iter.iter());

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
        let max_val = reduce_vec8(Some(a.clone()), &Some(b.clone()), true).unwrap();
        assert_eq!(max_val, b);

        // For max=false, we expect the shorter (lexicographically smaller) value.
        let min_val = reduce_vec8(Some(a.clone()), &Some(b), false).unwrap();
        assert_eq!(min_val, a);

        Ok(())
    }
}
