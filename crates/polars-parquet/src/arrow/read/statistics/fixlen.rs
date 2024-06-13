use arrow::array::*;
use arrow::types::{days_ms, i256};
use ethnum::I256;
use polars_error::PolarsResult;

use super::super::{convert_days_ms, convert_i128};
use crate::arrow::read::convert_i256;
use crate::parquet::statistics::FixedLenStatistics;

pub(super) fn push_i128(
    from: Option<&FixedLenStatistics>,
    n: usize,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i128>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i128>>()
        .unwrap();

    min.push(from.and_then(|s| s.min_value.as_deref().map(|x| convert_i128(x, n))));
    max.push(from.and_then(|s| s.max_value.as_deref().map(|x| convert_i128(x, n))));

    Ok(())
}

pub(super) fn push_i256_with_i128(
    from: Option<&FixedLenStatistics>,
    n: usize,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i256>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i256>>()
        .unwrap();

    min.push(from.and_then(|s| {
        s.min_value
            .as_deref()
            .map(|x| i256(I256::new(convert_i128(x, n))))
    }));
    max.push(from.and_then(|s| {
        s.max_value
            .as_deref()
            .map(|x| i256(I256::new(convert_i128(x, n))))
    }));

    Ok(())
}

pub(super) fn push_i256(
    from: Option<&FixedLenStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i256>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i256>>()
        .unwrap();

    min.push(from.and_then(|s| s.min_value.as_deref().map(convert_i256)));
    max.push(from.and_then(|s| s.max_value.as_deref().map(convert_i256)));

    Ok(())
}

pub(super) fn push(
    from: Option<&FixedLenStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutableFixedSizeBinaryArray>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutableFixedSizeBinaryArray>()
        .unwrap();

    min.push(from.and_then(|s| s.min_value.as_ref()));
    max.push(from.and_then(|s| s.max_value.as_ref()));

    Ok(())
}

fn convert_year_month(value: &[u8]) -> i32 {
    i32::from_le_bytes(value[..4].try_into().unwrap())
}

pub(super) fn push_year_month(
    from: Option<&FixedLenStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i32>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<i32>>()
        .unwrap();

    min.push(from.and_then(|s| s.min_value.as_deref().map(convert_year_month)));
    max.push(from.and_then(|s| s.max_value.as_deref().map(convert_year_month)));

    Ok(())
}

pub(super) fn push_days_ms(
    from: Option<&FixedLenStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<days_ms>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<days_ms>>()
        .unwrap();

    min.push(from.and_then(|s| s.min_value.as_deref().map(convert_days_ms)));
    max.push(from.and_then(|s| s.max_value.as_deref().map(convert_days_ms)));

    Ok(())
}
