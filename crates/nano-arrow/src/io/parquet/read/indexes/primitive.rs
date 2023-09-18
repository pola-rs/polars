use ethnum::I256;
use parquet2::indexes::PageIndex;
use parquet2::schema::types::{PrimitiveLogicalType, PrimitiveType, TimeUnit as ParquetTimeUnit};
use parquet2::types::int96_to_i64_ns;

use crate::array::{Array, MutablePrimitiveArray, PrimitiveArray};
use crate::datatypes::{DataType, TimeUnit};
use crate::trusted_len::TrustedLen;
use crate::types::{i256, NativeType};

use super::ColumnPageStatistics;

#[inline]
fn deserialize_int32<I: TrustedLen<Item = Option<i32>>>(
    iter: I,
    data_type: DataType,
) -> Box<dyn Array> {
    use DataType::*;
    match data_type.to_logical_type() {
        UInt8 => Box::new(
            PrimitiveArray::<u8>::from_trusted_len_iter(iter.map(|x| x.map(|x| x as u8)))
                .to(data_type),
        ) as _,
        UInt16 => Box::new(
            PrimitiveArray::<u16>::from_trusted_len_iter(iter.map(|x| x.map(|x| x as u16)))
                .to(data_type),
        ),
        UInt32 => Box::new(
            PrimitiveArray::<u32>::from_trusted_len_iter(iter.map(|x| x.map(|x| x as u32)))
                .to(data_type),
        ),
        Decimal(_, _) => Box::new(
            PrimitiveArray::<i128>::from_trusted_len_iter(iter.map(|x| x.map(|x| x as i128)))
                .to(data_type),
        ),
        Decimal256(_, _) => Box::new(
            PrimitiveArray::<i256>::from_trusted_len_iter(
                iter.map(|x| x.map(|x| i256(I256::new(x.into())))),
            )
            .to(data_type),
        ) as _,
        _ => Box::new(PrimitiveArray::<i32>::from_trusted_len_iter(iter).to(data_type)),
    }
}

#[inline]
fn timestamp(
    array: &mut MutablePrimitiveArray<i64>,
    time_unit: TimeUnit,
    logical_type: Option<PrimitiveLogicalType>,
) {
    let unit = if let Some(PrimitiveLogicalType::Timestamp { unit, .. }) = logical_type {
        unit
    } else {
        return;
    };

    match (unit, time_unit) {
        (ParquetTimeUnit::Milliseconds, TimeUnit::Second) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000),
        (ParquetTimeUnit::Microseconds, TimeUnit::Second) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000_000),
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Second) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000_000_000),

        (ParquetTimeUnit::Milliseconds, TimeUnit::Millisecond) => {}
        (ParquetTimeUnit::Microseconds, TimeUnit::Millisecond) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000),
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Millisecond) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000_000),

        (ParquetTimeUnit::Milliseconds, TimeUnit::Microsecond) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x *= 1_000),
        (ParquetTimeUnit::Microseconds, TimeUnit::Microsecond) => {}
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Microsecond) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000),

        (ParquetTimeUnit::Milliseconds, TimeUnit::Nanosecond) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x *= 1_000_000),
        (ParquetTimeUnit::Microseconds, TimeUnit::Nanosecond) => array
            .values_mut_slice()
            .iter_mut()
            .for_each(|x| *x /= 1_000),
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Nanosecond) => {}
    }
}

#[inline]
fn deserialize_int64<I: TrustedLen<Item = Option<i64>>>(
    iter: I,
    primitive_type: &PrimitiveType,
    data_type: DataType,
) -> Box<dyn Array> {
    use DataType::*;
    match data_type.to_logical_type() {
        UInt64 => Box::new(
            PrimitiveArray::<u64>::from_trusted_len_iter(iter.map(|x| x.map(|x| x as u64)))
                .to(data_type),
        ) as _,
        Decimal(_, _) => Box::new(
            PrimitiveArray::<i128>::from_trusted_len_iter(iter.map(|x| x.map(|x| x as i128)))
                .to(data_type),
        ) as _,
        Decimal256(_, _) => Box::new(
            PrimitiveArray::<i256>::from_trusted_len_iter(
                iter.map(|x| x.map(|x| i256(I256::new(x.into())))),
            )
            .to(data_type),
        ) as _,
        Timestamp(time_unit, _) => {
            let mut array =
                MutablePrimitiveArray::<i64>::from_trusted_len_iter(iter).to(data_type.clone());

            timestamp(&mut array, *time_unit, primitive_type.logical_type);

            let array: PrimitiveArray<i64> = array.into();

            Box::new(array)
        }
        _ => Box::new(PrimitiveArray::<i64>::from_trusted_len_iter(iter).to(data_type)),
    }
}

#[inline]
fn deserialize_int96<I: TrustedLen<Item = Option<[u32; 3]>>>(
    iter: I,
    data_type: DataType,
) -> Box<dyn Array> {
    Box::new(
        PrimitiveArray::<i64>::from_trusted_len_iter(iter.map(|x| x.map(int96_to_i64_ns)))
            .to(data_type),
    )
}

#[inline]
fn deserialize_id_s<T: NativeType, I: TrustedLen<Item = Option<T>>>(
    iter: I,
    data_type: DataType,
) -> Box<dyn Array> {
    Box::new(PrimitiveArray::<T>::from_trusted_len_iter(iter).to(data_type))
}

pub fn deserialize_i32(indexes: &[PageIndex<i32>], data_type: DataType) -> ColumnPageStatistics {
    ColumnPageStatistics {
        min: deserialize_int32(indexes.iter().map(|index| index.min), data_type.clone()),
        max: deserialize_int32(indexes.iter().map(|index| index.max), data_type),
        null_count: PrimitiveArray::from_trusted_len_iter(
            indexes
                .iter()
                .map(|index| index.null_count.map(|x| x as u64)),
        ),
    }
}

pub fn deserialize_i64(
    indexes: &[PageIndex<i64>],
    primitive_type: &PrimitiveType,
    data_type: DataType,
) -> ColumnPageStatistics {
    ColumnPageStatistics {
        min: deserialize_int64(
            indexes.iter().map(|index| index.min),
            primitive_type,
            data_type.clone(),
        ),
        max: deserialize_int64(
            indexes.iter().map(|index| index.max),
            primitive_type,
            data_type,
        ),
        null_count: PrimitiveArray::from_trusted_len_iter(
            indexes
                .iter()
                .map(|index| index.null_count.map(|x| x as u64)),
        ),
    }
}

pub fn deserialize_i96(
    indexes: &[PageIndex<[u32; 3]>],
    data_type: DataType,
) -> ColumnPageStatistics {
    ColumnPageStatistics {
        min: deserialize_int96(indexes.iter().map(|index| index.min), data_type.clone()),
        max: deserialize_int96(indexes.iter().map(|index| index.max), data_type),
        null_count: PrimitiveArray::from_trusted_len_iter(
            indexes
                .iter()
                .map(|index| index.null_count.map(|x| x as u64)),
        ),
    }
}

pub fn deserialize_id<T: NativeType>(
    indexes: &[PageIndex<T>],
    data_type: DataType,
) -> ColumnPageStatistics {
    ColumnPageStatistics {
        min: deserialize_id_s(indexes.iter().map(|index| index.min), data_type.clone()),
        max: deserialize_id_s(indexes.iter().map(|index| index.max), data_type),
        null_count: PrimitiveArray::from_trusted_len_iter(
            indexes
                .iter()
                .map(|index| index.null_count.map(|x| x as u64)),
        ),
    }
}
