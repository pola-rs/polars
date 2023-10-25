use std::sync::Arc;

use crate::error::{Error, Result};
use crate::schema::types::PhysicalType;
use crate::statistics::*;
use crate::types::NativeType;

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
    match (lhs, rhs) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(x)) => Some(x.clone()),
        (Some(x), Some(y)) => Some(ord_binary(x, y.clone(), max)),
    }
}

pub fn reduce(stats: &[&Option<Arc<dyn Statistics>>]) -> Result<Option<Arc<dyn Statistics>>> {
    if stats.is_empty() {
        return Ok(None);
    }
    let stats = stats
        .iter()
        .filter_map(|x| x.as_ref())
        .map(|x| x.as_ref())
        .collect::<Vec<&dyn Statistics>>();
    if stats.is_empty() {
        return Ok(None);
    };

    let same_type = stats
        .iter()
        .skip(1)
        .all(|x| x.physical_type() == stats[0].physical_type());
    if !same_type {
        return Err(Error::oos("The statistics do not have the same data_type"));
    };
    Ok(match stats[0].physical_type() {
        PhysicalType::Boolean => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_boolean(stats)))
        }
        PhysicalType::Int32 => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_primitive::<i32, _>(stats)))
        }
        PhysicalType::Int64 => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_primitive::<i64, _>(stats)))
        }
        PhysicalType::Float => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_primitive::<f32, _>(stats)))
        }
        PhysicalType::Double => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_primitive::<f64, _>(stats)))
        }
        PhysicalType::ByteArray => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_binary(stats)))
        }
        PhysicalType::FixedLenByteArray(_) => {
            let stats = stats.iter().map(|x| x.as_any().downcast_ref().unwrap());
            Some(Arc::new(reduce_fix_len_binary(stats)))
        }
        _ => todo!(),
    })
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

fn ord_binary(a: Vec<u8>, b: Vec<u8>, max: bool) -> Vec<u8> {
    for (v1, v2) in a.iter().zip(b.iter()) {
        match v1.cmp(v2) {
            std::cmp::Ordering::Greater => {
                if max {
                    return a;
                } else {
                    return b;
                }
            }
            std::cmp::Ordering::Less => {
                if max {
                    return b;
                } else {
                    return a;
                }
            }
            _ => {}
        }
    }
    a
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
    use crate::schema::types::PrimitiveType;

    use super::*;

    #[test]
    fn binary() -> Result<()> {
        let iter = vec![
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".to_string(),
                    PhysicalType::ByteArray,
                ),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![3, 4]),
            },
            BinaryStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".to_string(),
                    PhysicalType::ByteArray,
                ),
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
                primitive_type: PrimitiveType::from_physical(
                    "bla".to_string(),
                    PhysicalType::ByteArray,
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
    fn fixed_len_binary() -> Result<()> {
        let iter = vec![
            FixedLenStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".to_string(),
                    PhysicalType::FixedLenByteArray(2),
                ),
                null_count: Some(0),
                distinct_count: None,
                min_value: Some(vec![1, 2]),
                max_value: Some(vec![3, 4]),
            },
            FixedLenStatistics {
                primitive_type: PrimitiveType::from_physical(
                    "bla".to_string(),
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
                    "bla".to_string(),
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
    fn boolean() -> Result<()> {
        let iter = vec![
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
    fn primitive() -> Result<()> {
        let iter = vec![PrimitiveStatistics {
            null_count: Some(2),
            distinct_count: None,
            min_value: Some(30),
            max_value: Some(70),
            primitive_type: PrimitiveType::from_physical("bla".to_string(), PhysicalType::Int32),
        }];
        let a = reduce_primitive(iter.iter());

        assert_eq!(
            a,
            PrimitiveStatistics {
                null_count: Some(2),
                distinct_count: None,
                min_value: Some(30),
                max_value: Some(70),
                primitive_type: PrimitiveType::from_physical(
                    "bla".to_string(),
                    PhysicalType::Int32,
                ),
            },
        );

        Ok(())
    }
}
