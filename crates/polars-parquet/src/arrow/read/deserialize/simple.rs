use arrow::array::{Array, DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray};
use arrow::datatypes::{ArrowDataType, IntervalUnit, TimeUnit};
use arrow::match_integer_type;
use arrow::types::{days_ms, i256};
use ethnum::I256;
use polars_error::{polars_bail, PolarsResult};

use self::primitive::UnitDecoderFunction;
use super::primitive::PrimitiveDictArrayDecoder;
use super::utils::PageDictArrayDecoder;
use super::{
    binary, boolean, fixed_size_binary, null, primitive, BasicDecompressor, CompressedPagesIter,
    ParquetResult,
};
use crate::parquet::error::ParquetError;
use crate::parquet::schema::types::{
    PhysicalType, PrimitiveLogicalType, PrimitiveType, TimeUnit as ParquetTimeUnit,
};
use crate::parquet::types::int96_to_i64_ns;
use crate::read::deserialize::binary::BinaryDictArrayDecoder;
use crate::read::deserialize::binview::{self, BinViewDictArrayDecoder};
use crate::read::deserialize::fixed_size_binary::FixedSizeBinaryDictArrayDecoder;
use crate::read::deserialize::primitive::{
    AsDecoderFunction, IntoDecoderFunction, PrimitiveDecoder,
};
use crate::read::deserialize::utils::PageDecoder;

/// An iterator adapter that maps an iterator of Pages into an iterator of Arrays
/// of [`ArrowDataType`] `data_type` and length `chunk_size`.
pub fn page_iter_to_arrays<'a, I: CompressedPagesIter + 'a>(
    pages: BasicDecompressor<I>,
    type_: &PrimitiveType,
    data_type: ArrowDataType,
    num_rows: usize,
) -> PolarsResult<Box<dyn Array>> {
    use ArrowDataType::*;

    let physical_type = &type_.physical_type;
    let logical_type = &type_.logical_type;

    Ok(match (physical_type, data_type.to_logical_type()) {
        (_, Null) => null::iter_to_arrays(pages, data_type, num_rows),
        (PhysicalType::Boolean, Boolean) => {
            PageDecoder::new(pages, data_type, boolean::BooleanDecoder)?.collect_n(num_rows)?
        },
        (PhysicalType::Int32, UInt8) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i32, u8>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, UInt16) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i32, u16>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, UInt32) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i32, u32>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, UInt32) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i64, u32>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, Int8) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i32, i8>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, Int16) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i32, i16>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, Int32 | Date32 | Time32(_)) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(UnitDecoderFunction::<i32>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64 | PhysicalType::Int96, Timestamp(time_unit, _)) => {
            let time_unit = *time_unit;
            return timestamp(
                pages,
                physical_type,
                logical_type,
                data_type,
                num_rows,
                time_unit,
            );
        },
        (PhysicalType::FixedLenByteArray(_), FixedSizeBinary(_)) => {
            let size = FixedSizeBinaryArray::get_size(&data_type);

            PageDecoder::new(pages, data_type, fixed_size_binary::BinaryDecoder { size })?
                .collect_n(num_rows)?
        },
        (PhysicalType::FixedLenByteArray(12), Interval(IntervalUnit::YearMonth)) => {
            // @TODO: Make a separate decoder for this

            let n = 12;
            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(num_rows)?;

            let values = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
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
            .collect_n(num_rows)?;

            let values = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
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
        (PhysicalType::Int32, Decimal(_, _)) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(IntoDecoderFunction::<i32, i128>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, Decimal(_, _)) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(IntoDecoderFunction::<i64, i128>::default()),
        )?
        .collect_n(num_rows)?,
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
            .collect_n(num_rows)?;

            let values = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
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
        (PhysicalType::Int32, Decimal256(_, _)) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(
                decoder_fn!((x) => <i32, i256> => i256(I256::new(x as i128))),
            ),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, Decimal256(_, _)) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(
                decoder_fn!((x) => <i64, i256> => i256(I256::new(x as i128))),
            ),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n <= 16 => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let array = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect_n(num_rows)?;

            let values = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
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
            .collect_n(num_rows)?;

            let values = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
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
        (PhysicalType::Int32, Date64) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(decoder_fn!((x) => <i32, i64> => i64::from(x) * 86400000)),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, Date64) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(UnitDecoderFunction::<i64>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, Int64 | Time64(_) | Duration(_)) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(UnitDecoderFunction::<i64>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, UInt64) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(AsDecoderFunction::<i64, u64>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Float, Float32) => PageDecoder::new(
            pages,
            data_type,
            PrimitiveDecoder::new(UnitDecoderFunction::<f32>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Double, Float64) => PageDecoder::new(
            pages,
            data_type,
            PrimitiveDecoder::new(UnitDecoderFunction::<f64>::default()),
        )?
        .collect_n(num_rows)?,
        // Don't compile this code with `i32` as we don't use this in polars
        (PhysicalType::ByteArray, LargeBinary | LargeUtf8) => {
            PageDecoder::new(pages, data_type, binary::BinaryDecoder::<i64>::default())?
                .collect_n(num_rows)?
        },
        (PhysicalType::ByteArray, BinaryView | Utf8View) => {
            PageDecoder::new(pages, data_type, binview::BinViewDecoder::default())?
                .collect_n(num_rows)?
        },
        (_, Dictionary(key_type, _, _)) => {
            return match_integer_type!(key_type, |$K| {
                dict_read::<$K, _>(pages, physical_type, logical_type, data_type, num_rows).map(|v| Box::new(v) as Box<_>)
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

fn timestamp<I: CompressedPagesIter>(
    pages: BasicDecompressor<I>,
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
    data_type: ArrowDataType,
    num_rows: usize,
    time_unit: TimeUnit,
) -> PolarsResult<Box<dyn Array>> {
    if physical_type == &PhysicalType::Int96 {
        return match time_unit {
            TimeUnit::Nanosecond => Ok(PageDecoder::new(
                pages,
                data_type,
                PrimitiveDecoder::new(decoder_fn!((x) => <[u32; 3], i64> => int96_to_i64_ns(x))),
            )?
            .collect_n(num_rows)?),
            TimeUnit::Microsecond => Ok(PageDecoder::new(
                pages,
                data_type,
                PrimitiveDecoder::new(decoder_fn!((x) => <[u32; 3], i64> => int96_to_i64_us(x))),
            )?
            .collect_n(num_rows)?),
            TimeUnit::Millisecond => Ok(PageDecoder::new(
                pages,
                data_type,
                PrimitiveDecoder::new(decoder_fn!((x) => <[u32; 3], i64> => int96_to_i64_ms(x))),
            )?
            .collect_n(num_rows)?),
            TimeUnit::Second => Ok(PageDecoder::new(
                pages,
                data_type,
                PrimitiveDecoder::new(decoder_fn!((x) => <[u32; 3], i64> => int96_to_i64_s(x))),
            )?
            .collect_n(num_rows)?),
        };
    };

    if physical_type != &PhysicalType::Int64 {
        polars_bail!(ComputeError:
            "not implemented: can't decode a timestamp from a non-int64 parquet type",
        );
    }

    let (factor, is_multiplier) = unify_timestamp_unit(logical_type, time_unit);
    Ok(match (factor, is_multiplier) {
        (1, _) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(UnitDecoderFunction::<i64>::default()),
        )?
        .collect_n(num_rows)?,
        (a, true) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(decoder_fn!((x, a: i64) => <i64, i64> => x * a)),
        )?
        .collect_n(num_rows)?,
        (a, false) => PageDecoder::new(
            pages,
            data_type,
            primitive::IntDecoder::new(decoder_fn!((x, a: i64) => <i64, i64> => x / a)),
        )?
        .collect_n(num_rows)?,
    })
}

fn timestamp_dict<K: DictionaryKey, I: CompressedPagesIter>(
    pages: BasicDecompressor<I>,
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
    data_type: ArrowDataType,
    num_rows: usize,
    time_unit: TimeUnit,
) -> ParquetResult<DictionaryArray<K>> {
    if physical_type == &PhysicalType::Int96 {
        let logical_type = PrimitiveLogicalType::Timestamp {
            unit: ParquetTimeUnit::Nanoseconds,
            is_adjusted_to_utc: false,
        };
        let (factor, is_multiplier) = unify_timestamp_unit(&Some(logical_type), time_unit);
        return match (factor, is_multiplier) {
            (a, true) => PageDictArrayDecoder::<_, K, _>::new(
                pages,
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
                PrimitiveDictArrayDecoder::new(
                    decoder_fn!((x, a: i64) => <[u32; 3], i64> => int96_to_i64_ns(x) * a),
                ),
            )?
            .collect_n(num_rows),
            (a, false) => PageDictArrayDecoder::<_, K, _>::new(
                pages,
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
                PrimitiveDictArrayDecoder::new(
                    decoder_fn!((x, a: i64) => <[u32; 3], i64> => int96_to_i64_ns(x) / a),
                ),
            )?
            .collect_n(num_rows),
        };
    };

    let (factor, is_multiplier) = unify_timestamp_unit(logical_type, time_unit);
    match (factor, is_multiplier) {
        (a, true) => PageDictArrayDecoder::<_, K, _>::new(
            pages,
            data_type,
            PrimitiveDictArrayDecoder::new(decoder_fn!((x, a: i64) => <i64, i64> => x * a)),
        )?
        .collect_n(num_rows),
        (a, false) => PageDictArrayDecoder::<_, K, _>::new(
            pages,
            data_type,
            PrimitiveDictArrayDecoder::new(decoder_fn!((x, a: i64) => <i64, i64> => x / a)),
        )?
        .collect_n(num_rows),
    }
}

fn dict_read<K: DictionaryKey, I: CompressedPagesIter>(
    iter: BasicDecompressor<I>,
    physical_type: &PhysicalType,
    logical_type: &Option<PrimitiveLogicalType>,
    data_type: ArrowDataType,
    num_rows: usize,
) -> ParquetResult<DictionaryArray<K>> {
    use ArrowDataType::*;
    let values_data_type = if let Dictionary(_, v, _) = &data_type {
        v.as_ref()
    } else {
        panic!()
    };

    Ok(match (physical_type, values_data_type.to_logical_type()) {
        (PhysicalType::Int32, UInt8) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(AsDecoderFunction::<i32, u8>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, UInt16) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(AsDecoderFunction::<i32, u16>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, UInt32) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(AsDecoderFunction::<i32, u32>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int64, UInt64) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(AsDecoderFunction::<i64, u64>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, Int8) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(AsDecoderFunction::<i32, i8>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, Int16) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(AsDecoderFunction::<i32, i16>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Int32, Int32 | Date32 | Time32(_) | Interval(IntervalUnit::YearMonth)) => {
            PageDictArrayDecoder::<_, K, _>::new(
                iter,
                data_type,
                PrimitiveDictArrayDecoder::new(UnitDecoderFunction::<i32>::default()),
            )?
            .collect_n(num_rows)?
        },

        (PhysicalType::Int64, Timestamp(time_unit, _)) => {
            let time_unit = *time_unit;
            return timestamp_dict::<K, _>(
                iter,
                physical_type,
                logical_type,
                data_type,
                num_rows,
                time_unit,
            );
        },

        (PhysicalType::Int64, Int64 | Date64 | Time64(_) | Duration(_)) => {
            PageDictArrayDecoder::<_, K, _>::new(
                iter,
                data_type,
                PrimitiveDictArrayDecoder::new(UnitDecoderFunction::<i64>::default()),
            )?
            .collect_n(num_rows)?
        },
        (PhysicalType::Float, Float32) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(UnitDecoderFunction::<f32>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::Double, Float64) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            PrimitiveDictArrayDecoder::new(UnitDecoderFunction::<f64>::default()),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::ByteArray, LargeUtf8 | LargeBinary) => PageDictArrayDecoder::<_, K, _>::new(
            iter,
            data_type,
            BinaryDictArrayDecoder::<i64>::default(),
        )?
        .collect_n(num_rows)?,
        (PhysicalType::ByteArray, Utf8View | BinaryView) => {
            PageDictArrayDecoder::<_, K, _>::new(iter, data_type, BinViewDictArrayDecoder)?
                .collect_n(num_rows)?
        },
        (PhysicalType::FixedLenByteArray(size), FixedSizeBinary(_)) => {
            PageDictArrayDecoder::<_, K, _>::new(
                iter,
                data_type,
                FixedSizeBinaryDictArrayDecoder { size: *size },
            )?
            .collect_n(num_rows)?
        },
        other => {
            return Err(ParquetError::FeatureNotSupported(format!(
                "Reading dictionaries of type {other:?}"
            )));
        },
    })
}
