use arrow::array::{Array, DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray};
use arrow::datatypes::{ArrowDataType, IntervalUnit, TimeUnit};
use arrow::match_integer_type;
use arrow::types::{days_ms, i256};
use ethnum::I256;
use polars_error::{polars_bail, PolarsResult};

use super::utils::filter::Filter;
use super::{
    boolean, dictionary, fixed_size_binary, null, primitive, BasicDecompressor, ParquetResult,
};
use crate::parquet::error::ParquetError;
use crate::parquet::schema::types::{
    PhysicalType, PrimitiveLogicalType, PrimitiveType, TimeUnit as ParquetTimeUnit,
};
use crate::parquet::types::int96_to_i64_ns;
use crate::read::deserialize::binview::{self, BinViewDecoder};
use crate::read::deserialize::utils::PageDecoder;

/// An iterator adapter that maps an iterator of Pages a boxed [`Array`] of [`ArrowDataType`]
/// `data_type` with a maximum of `num_rows` elements.
pub fn page_iter_to_array(
    pages: BasicDecompressor,
    type_: &PrimitiveType,
    data_type: ArrowDataType,
    filter: Option<Filter>,
) -> PolarsResult<Box<dyn Array>> {
    use ArrowDataType::*;

    let physical_type = &type_.physical_type;
    let logical_type = &type_.logical_type;

    Ok(match (physical_type, data_type.to_logical_type()) {
        (_, Null) => null::iter_to_arrays(pages, data_type, filter)?,
        (PhysicalType::Boolean, Boolean) => {
            Box::new(PageDecoder::new(pages, data_type, boolean::BooleanDecoder)?.collect_n(filter)?)
        },
        (PhysicalType::Int32, UInt8) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, u8, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int32, UInt16) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, u16, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int32, UInt32) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, u32, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64, UInt32) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i64, u32, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int32, Int8) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, i8, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int32, Int16) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, i16, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int32, Int32 | Date32 | Time32(_)) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, _, _>::unit(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64 | PhysicalType::Int96, Timestamp(time_unit, _)) => {
            let time_unit = *time_unit;
            return timestamp(
                pages,
                physical_type,
                logical_type,
                data_type,
                filter,
                time_unit,
            );
        },
        (PhysicalType::FixedLenByteArray(_), FixedSizeBinary(_)) => {
            let size = FixedSizeBinaryArray::get_size(&data_type);

            Box::new(PageDecoder::new(pages, data_type, fixed_size_binary::BinaryDecoder { size })?
                .collect_n(filter)?)
        },
        (PhysicalType::FixedLenByteArray(12), Interval(IntervalUnit::YearMonth)) => {
            // @TODO: Make a separate decoder for this

            let n = 12;
            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(|value: &[u8]| i32::from_le_bytes(value[..4].try_into().unwrap()))
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            Box::new(PrimitiveArray::<i32>::try_new(
                data_type.clone(),
                values.into(),
                validity,
            )?)
        },
        (PhysicalType::FixedLenByteArray(12), Interval(IntervalUnit::DayTime)) => {
            // @TODO: Make a separate decoder for this

            let n = 12;
            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(super::super::convert_days_ms)
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            Box::new(PrimitiveArray::<days_ms>::try_new(
                data_type.clone(),
                values.into(),
                validity,
            )?)
        },
        (PhysicalType::Int32, Decimal(_, _)) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i32, i128, _>::cast_into(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64, Decimal(_, _)) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i64, i128, _>::cast_into(),
        )?
        .collect_n(filter)?),
        (PhysicalType::FixedLenByteArray(n), Decimal(_, _)) if *n > 16 => {
            polars_bail!(ComputeError:
                "not implemented: can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}"
            )
        },
        (PhysicalType::FixedLenByteArray(n), Decimal(_, _)) => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(|value: &[u8]| super::super::convert_i128(value, n))
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            Box::new(PrimitiveArray::<i128>::try_new(
                data_type.clone(),
                values.into(),
                validity,
            )?)
        },
        (PhysicalType::Int32, Decimal256(_, _)) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::closure(|x: i32| i256(I256::new(x as i128))),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64, Decimal256(_, _)) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::closure(|x: i64| i256(I256::new(x as i128))),
        )?
        .collect_n(filter)?),
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n <= 16 => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(|value: &[u8]| i256(I256::new(super::super::convert_i128(value, n))))
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            Box::new(PrimitiveArray::<i256>::try_new(
                data_type.clone(),
                values.into(),
                validity,
            )?)
        },
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n <= 32 => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(super::super::convert_i256)
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            Box::new(PrimitiveArray::<i256>::try_new(
                data_type.clone(),
                values.into(),
                validity,
            )?)
        },
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n > 32 => {
            polars_bail!(ComputeError:
                "Can't decode Decimal256 type from Fixed Size Byte Array of len {n:?}"
            )
        },
        (PhysicalType::Int32, Date64) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::closure(|x: i32| i64::from(x) * 86400000),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64, Date64) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i64, _, _>::unit(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64, Int64 | Time64(_) | Duration(_)) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i64, _, _>::unit(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Int64, UInt64) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::<i64, u64, _>::cast_as(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Float, Float32) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::FloatDecoder::<f32, _, _>::unit(),
        )?
        .collect_n(filter)?),
        (PhysicalType::Double, Float64) => Box::new(PageDecoder::new(
            pages,
            data_type,
            primitive::FloatDecoder::<f64, _, _>::unit(),
        )?
        .collect_n(filter)?),
        // Don't compile this code with `i32` as we don't use this in polars
        (PhysicalType::ByteArray, LargeBinary | LargeUtf8) => {
            PageDecoder::new(pages, data_type, binview::BinViewDecoder::default())?
                .collect_n(filter)?
        },
        (_, Binary | Utf8) => unreachable!(),
        (PhysicalType::ByteArray, BinaryView | Utf8View) => {
            PageDecoder::new(pages, data_type, binview::BinViewDecoder::default())?
                .collect_n(filter)?
        },
        (_, Dictionary(key_type, _, _)) => {
            return match_integer_type!(key_type, |$K| {
                dict_read::<$K>(pages, physical_type, logical_type, data_type, filter).map(|v| Box::new(v) as Box<_>)
            }).map_err(Into::into)
        },
        (from, to) => {
            polars_bail!(ComputeError:
                "not implemented: reading parquet type {from:?} to {to:?} still not implemented"
            )
        },
    })
}

/// Unify the timestamp unit from parquet TimeUnit into arrow's TimeUnit
/// Returns (a int64 factor, is_multiplier)
fn unify_timestamp_unit(
    logical_type: &Option<PrimitiveLogicalType>,
    time_unit: TimeUnit,
) -> (i64, bool) {
    if let Some(PrimitiveLogicalType::Timestamp { unit, .. }) = logical_type {
        match (*unit, time_unit) {
            (ParquetTimeUnit::Milliseconds, TimeUnit::Millisecond)
            | (ParquetTimeUnit::Microseconds, TimeUnit::Microsecond)
            | (ParquetTimeUnit::Nanoseconds, TimeUnit::Nanosecond) => (1, true),

            (ParquetTimeUnit::Milliseconds, TimeUnit::Second)
            | (ParquetTimeUnit::Microseconds, TimeUnit::Millisecond)
            | (ParquetTimeUnit::Nanoseconds, TimeUnit::Microsecond) => (1000, false),

            (ParquetTimeUnit::Microseconds, TimeUnit::Second)
            | (ParquetTimeUnit::Nanoseconds, TimeUnit::Millisecond) => (1_000_000, false),

            (ParquetTimeUnit::Nanoseconds, TimeUnit::Second) => (1_000_000_000, false),

            (ParquetTimeUnit::Milliseconds, TimeUnit::Microsecond)
            | (ParquetTimeUnit::Microseconds, TimeUnit::Nanosecond) => (1_000, true),

            (ParquetTimeUnit::Milliseconds, TimeUnit::Nanosecond) => (1_000_000, true),
        }
    } else {
        (1, true)
    }
}

#[inline]
pub fn int96_to_i64_us(value: [u32; 3]) -> i64 {
    const JULIAN_DAY_OF_EPOCH: i64 = 2_440_588;
    const SECONDS_PER_DAY: i64 = 86_400;
    const MICROS_PER_SECOND: i64 = 1_000_000;

    let day = value[2] as i64;
    let microseconds = (((value[1] as i64) << 32) + value[0] as i64) / 1_000;
    let seconds = (day - JULIAN_DAY_OF_EPOCH) * SECONDS_PER_DAY;

    seconds * MICROS_PER_SECOND + microseconds
}

#[inline]
pub fn int96_to_i64_ms(value: [u32; 3]) -> i64 {
    const JULIAN_DAY_OF_EPOCH: i64 = 2_440_588;
    const SECONDS_PER_DAY: i64 = 86_400;
    const MILLIS_PER_SECOND: i64 = 1_000;

    let day = value[2] as i64;
    let milliseconds = (((value[1] as i64) << 32) + value[0] as i64) / 1_000_000;
    let seconds = (day - JULIAN_DAY_OF_EPOCH) * SECONDS_PER_DAY;

    seconds * MILLIS_PER_SECOND + milliseconds
}

#[inline]
pub fn int96_to_i64_s(value: [u32; 3]) -> i64 {
    const JULIAN_DAY_OF_EPOCH: i64 = 2_440_588;
    const SECONDS_PER_DAY: i64 = 86_400;

    let day = value[2] as i64;
    let seconds = (((value[1] as i64) << 32) + value[0] as i64) / 1_000_000_000;
    let day_seconds = (day - JULIAN_DAY_OF_EPOCH) * SECONDS_PER_DAY;

    day_seconds + seconds
}

fn timestamp(
    pages: BasicDecompressor,
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
    data_type: ArrowDataType,
    filter: Option<Filter>,
    time_unit: TimeUnit,
) -> PolarsResult<Box<dyn Array>> {
    if physical_type == &PhysicalType::Int96 {
        return match time_unit {
            TimeUnit::Nanosecond => Ok(Box::new(
                PageDecoder::new(
                    pages,
                    data_type,
                    primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_ns(x)),
                )?
                .collect_n(filter)?,
            )),
            TimeUnit::Microsecond => Ok(Box::new(
                PageDecoder::new(
                    pages,
                    data_type,
                    primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_us(x)),
                )?
                .collect_n(filter)?,
            )),
            TimeUnit::Millisecond => Ok(Box::new(
                PageDecoder::new(
                    pages,
                    data_type,
                    primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_ms(x)),
                )?
                .collect_n(filter)?,
            )),
            TimeUnit::Second => Ok(Box::new(
                PageDecoder::new(
                    pages,
                    data_type,
                    primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_s(x)),
                )?
                .collect_n(filter)?,
            )),
        };
    };

    if physical_type != &PhysicalType::Int64 {
        polars_bail!(ComputeError:
            "not implemented: can't decode a timestamp from a non-int64 parquet type",
        );
    }

    let (factor, is_multiplier) = unify_timestamp_unit(logical_type, time_unit);
    Ok(match (factor, is_multiplier) {
        (1, _) => Box::new(
            PageDecoder::new(pages, data_type, primitive::IntDecoder::<i64, _, _>::unit())?
                .collect_n(filter)?,
        ),
        (a, true) => Box::new(
            PageDecoder::new(
                pages,
                data_type,
                primitive::IntDecoder::closure(|x: i64| x * a),
            )?
            .collect_n(filter)?,
        ),
        (a, false) => Box::new(
            PageDecoder::new(
                pages,
                data_type,
                primitive::IntDecoder::closure(|x: i64| x / a),
            )?
            .collect_n(filter)?,
        ),
    })
}

fn timestamp_dict<K: DictionaryKey>(
    pages: BasicDecompressor,
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
    data_type: ArrowDataType,
    filter: Option<Filter>,
    time_unit: TimeUnit,
) -> ParquetResult<DictionaryArray<K>> {
    if physical_type == &PhysicalType::Int96 {
        let logical_type = PrimitiveLogicalType::Timestamp {
            unit: ParquetTimeUnit::Nanoseconds,
            is_adjusted_to_utc: false,
        };
        let (factor, is_multiplier) = unify_timestamp_unit(&Some(logical_type), time_unit);
        return match (factor, is_multiplier) {
            (a, true) => PageDecoder::new(
                pages,
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
                dictionary::DictionaryDecoder::<K, _>::new(primitive::FloatDecoder::closure(
                    |x: [u32; 3]| int96_to_i64_ns(x) * a,
                )),
            )?
            .collect_n(filter),
            (a, false) => PageDecoder::new(
                pages,
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
                dictionary::DictionaryDecoder::<K, _>::new(primitive::FloatDecoder::closure(
                    |x: [u32; 3]| int96_to_i64_ns(x) / a,
                )),
            )?
            .collect_n(filter),
        };
    };

    let (factor, is_multiplier) = unify_timestamp_unit(logical_type, time_unit);
    match (factor, is_multiplier) {
        (a, true) => PageDecoder::new(
            pages,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::closure(|x: i64| x * a)),
        )?
        .collect_n(filter),
        (a, false) => PageDecoder::new(
            pages,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::closure(|x: i64| x / a)),
        )?
        .collect_n(filter),
    }
}

fn dict_read<K: DictionaryKey>(
    iter: BasicDecompressor,
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
    data_type: ArrowDataType,
    filter: Option<Filter>,
) -> ParquetResult<DictionaryArray<K>> {
    use ArrowDataType::*;
    let values_data_type = if let Dictionary(_, v, _) = &data_type {
        v.as_ref()
    } else {
        panic!()
    };

    Ok(match (physical_type, values_data_type.to_logical_type()) {
        (PhysicalType::Int32, UInt8) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i32, u8, _>::cast_as()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Int32, UInt16) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i32, u16, _>::cast_as()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Int32, UInt32) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i32, u32, _>::cast_as()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Int64, UInt64) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i64, u64, _>::cast_as()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Int32, Int8) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i32, i8, _>::cast_as()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Int32, Int16) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i32, i16, _>::cast_as()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Int32, Int32 | Date32 | Time32(_) | Interval(IntervalUnit::YearMonth)) => {
            PageDecoder::new(
                iter,
                data_type,
                dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i32, _, _>::unit()),
            )?
            .collect_n(filter)?
        },

        (PhysicalType::Int64, Timestamp(time_unit, _)) => {
            let time_unit = *time_unit;
            return timestamp_dict::<K>(
                iter,
                physical_type,
                logical_type,
                data_type,
                filter,
                time_unit,
            );
        },

        (PhysicalType::Int64, Int64 | Date64 | Time64(_) | Duration(_)) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<i64, _, _>::unit()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Float, Float32) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<f32, _, _>::unit()),
        )?
        .collect_n(filter)?,
        (PhysicalType::Double, Float64) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(primitive::FloatDecoder::<f64, _, _>::unit()),
        )?
        .collect_n(filter)?,
        (_, LargeUtf8 | LargeBinary | Utf8 | Binary) => unreachable!(),
        (PhysicalType::ByteArray, Utf8View | BinaryView) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(BinViewDecoder::default()),
        )?
        .collect_n(filter)?,
        (PhysicalType::FixedLenByteArray(size), FixedSizeBinary(_)) => PageDecoder::new(
            iter,
            data_type,
            dictionary::DictionaryDecoder::new(fixed_size_binary::BinaryDecoder { size: *size }),
        )?
        .collect_n(filter)?,
        other => {
            return Err(ParquetError::FeatureNotSupported(format!(
                "Reading dictionaries of type {other:?}"
            )));
        },
    })
}
