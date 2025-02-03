use arrow::array::{Array, FixedSizeBinaryArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{
    ArrowDataType, Field, IntegerType, IntervalUnit, TimeUnit, DTYPE_CATEGORICAL, DTYPE_ENUM_VALUES,
};
use arrow::types::{days_ms, i256, NativeType};
use ethnum::I256;
use polars_compute::cast::CastOptionsImpl;

use super::utils::filter::Filter;
use super::{boolean, fixed_size_binary, null, primitive, BasicDecompressor};
use crate::parquet::error::ParquetResult;
use crate::parquet::schema::types::{
    PhysicalType, PrimitiveLogicalType, PrimitiveType, TimeUnit as ParquetTimeUnit,
};
use crate::parquet::types::int96_to_i64_ns;
use crate::read::deserialize::binview;
use crate::read::deserialize::categorical::CategoricalDecoder;
use crate::read::deserialize::utils::PageDecoder;
use crate::read::ParquetError;

/// An iterator adapter that maps an iterator of Pages a boxed [`Array`] of [`ArrowDataType`]
/// `dtype` with a maximum of `num_rows` elements.
pub fn page_iter_to_array(
    pages: BasicDecompressor,
    type_: &PrimitiveType,
    field: Field,
    filter: Option<Filter>,
) -> ParquetResult<(Box<dyn Array>, Bitmap)> {
    use ArrowDataType::*;

    let physical_type = &type_.physical_type;
    let logical_type = &type_.logical_type;
    let dtype = field.dtype;

    Ok(match (physical_type, dtype.to_logical_type()) {
        (_, Null) => PageDecoder::new(pages, dtype, null::NullDecoder)?.collect_boxed(filter)?,
        (PhysicalType::Boolean, Boolean) => {
            PageDecoder::new(pages, dtype, boolean::BooleanDecoder)?.collect_boxed(filter)?
        },
        (PhysicalType::Int32, UInt8) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::<i32, u8, _>::cast_as())?
                .collect_boxed(filter)?
        },
        (PhysicalType::Int32, UInt16) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i32, u16, _>::cast_as(),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int32, UInt32) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i32, u32, _>::cast_as(),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int64, UInt32) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i64, u32, _>::cast_as(),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int32, Int8) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::<i32, i8, _>::cast_as())?
                .collect_boxed(filter)?
        },
        (PhysicalType::Int32, Int16) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i32, i16, _>::cast_as(),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int32, Int32 | Date32 | Time32(_)) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::<i32, _, _>::unit())?
                .collect_boxed(filter)?
        },
        (PhysicalType::Int64 | PhysicalType::Int96, Timestamp(time_unit, _)) => {
            let time_unit = *time_unit;
            return timestamp(pages, physical_type, logical_type, dtype, filter, time_unit);
        },
        (PhysicalType::FixedLenByteArray(_), FixedSizeBinary(_)) => {
            let size = FixedSizeBinaryArray::get_size(&dtype);

            PageDecoder::new(pages, dtype, fixed_size_binary::BinaryDecoder { size })?
                .collect_boxed(filter)?
        },
        (PhysicalType::FixedLenByteArray(12), Interval(IntervalUnit::YearMonth)) => {
            // @TODO: Make a separate decoder for this

            let n = 12;
            let (array, ptm) = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(|value: &[u8]| i32::from_le_bytes(value[..4].try_into().unwrap()))
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            (
                PrimitiveArray::<i32>::try_new(dtype.clone(), values.into(), validity)?.to_boxed(),
                ptm,
            )
        },
        (PhysicalType::FixedLenByteArray(12), Interval(IntervalUnit::DayTime)) => {
            // @TODO: Make a separate decoder for this

            let n = 12;
            let (array, ptm) = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(super::super::convert_days_ms)
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            (
                PrimitiveArray::<days_ms>::try_new(dtype.clone(), values.into(), validity)?
                    .to_boxed(),
                ptm,
            )
        },
        (PhysicalType::Int32, Decimal(_, _)) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i32, i128, _>::cast_into(),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int64, Decimal(_, _)) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i64, i128, _>::cast_into(),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::FixedLenByteArray(n), Decimal(_, _)) if *n > 16 => {
            return Err(ParquetError::not_supported(format!(
                "not implemented: can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}"
            )));
        },
        (PhysicalType::FixedLenByteArray(n), Decimal(_, _)) => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let (array, ptm) = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(|value: &[u8]| super::super::convert_i128(value, n))
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            (
                PrimitiveArray::<i128>::try_new(dtype.clone(), values.into(), validity)?.to_boxed(),
                ptm,
            )
        },
        (PhysicalType::Int32, Decimal256(_, _)) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::closure(|x: i32| i256(I256::new(x as i128))),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int64, Decimal256(_, _)) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::closure(|x: i64| i256(I256::new(x as i128))),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n <= 16 => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let (array, ptm) = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(|value: &[u8]| i256(I256::new(super::super::convert_i128(value, n))))
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            (
                PrimitiveArray::<i256>::try_new(dtype.clone(), values.into(), validity)?.to_boxed(),
                ptm,
            )
        },
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n <= 32 => {
            // @TODO: Make a separate decoder for this

            let n = *n;

            let (array, ptm) = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(n),
                fixed_size_binary::BinaryDecoder { size: n },
            )?
            .collect(filter)?;

            let values = array
                .values()
                .chunks_exact(n)
                .map(super::super::convert_i256)
                .collect::<Vec<_>>();
            let validity = array.validity().cloned();

            (
                PrimitiveArray::<i256>::try_new(dtype.clone(), values.into(), validity)?.to_boxed(),
                ptm,
            )
        },
        (PhysicalType::FixedLenByteArray(n), Decimal256(_, _)) if *n > 32 => {
            return Err(ParquetError::not_supported(format!(
                "Can't decode Decimal256 type from Fixed Size Byte Array of len {n:?}",
            )));
        },
        (PhysicalType::Int32, Date64) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::closure(|x: i32| i64::from(x) * 86400000),
        )?
        .collect_boxed(filter)?,
        (PhysicalType::Int64, Date64) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::<i64, _, _>::unit())?
                .collect_boxed(filter)?
        },
        (PhysicalType::Int64, Int64 | Time64(_) | Duration(_)) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::<i64, _, _>::unit())?
                .collect_boxed(filter)?
        },

        (PhysicalType::Int64, UInt64) => PageDecoder::new(
            pages,
            dtype,
            primitive::IntDecoder::<i64, u64, _>::cast_as(),
        )?
        .collect_boxed(filter)?,

        // Float16
        (PhysicalType::FixedLenByteArray(2), Float32) => {
            // @NOTE: To reduce code bloat, we just use the FixedSizeBinary decoder.

            let (mut fsb_array, ptm) = PageDecoder::new(
                pages,
                ArrowDataType::FixedSizeBinary(2),
                fixed_size_binary::BinaryDecoder { size: 2 },
            )?
            .collect(filter)?;

            let validity = fsb_array.take_validity();
            let values = fsb_array.values().as_slice();
            assert_eq!(values.len() % 2, 0);
            let values = values.chunks_exact(2);
            let values = values
                .map(|v| {
                    // SAFETY: We know that `v` is always of size two.
                    let le_bytes: [u8; 2] = unsafe { v.try_into().unwrap_unchecked() };
                    let v = arrow::types::f16::from_le_bytes(le_bytes);
                    v.to_f32()
                })
                .collect();

            (
                PrimitiveArray::<f32>::new(dtype, values, validity).to_boxed(),
                ptm,
            )
        },

        (PhysicalType::Float, Float32) => {
            PageDecoder::new(pages, dtype, primitive::FloatDecoder::<f32, _, _>::unit())?
                .collect_boxed(filter)?
        },
        (PhysicalType::Double, Float64) => {
            PageDecoder::new(pages, dtype, primitive::FloatDecoder::<f64, _, _>::unit())?
                .collect_boxed(filter)?
        },
        // Don't compile this code with `i32` as we don't use this in polars
        (PhysicalType::ByteArray, LargeBinary | LargeUtf8) => {
            let is_string = matches!(dtype, LargeUtf8);
            PageDecoder::new(pages, dtype, binview::BinViewDecoder { is_string })?
                .collect(filter)?
        },
        (_, Binary | Utf8) => unreachable!(),
        (PhysicalType::ByteArray, BinaryView | Utf8View) => {
            let is_string = matches!(dtype, Utf8View);
            PageDecoder::new(pages, dtype, binview::BinViewDecoder { is_string })?
                .collect(filter)?
        },
        (_, Dictionary(key_type, value_type, _)) => {
            // @NOTE: This should only hit in two cases:
            // - Polars enum's and categorical's
            // - Int -> String which can be turned into categoricals
            assert_eq!(value_type.as_ref(), &ArrowDataType::Utf8View);

            if field.metadata.is_none_or(|md| {
                !md.contains_key(DTYPE_ENUM_VALUES) && !md.contains_key(DTYPE_CATEGORICAL)
            }) {
                let (array, ptm) = PageDecoder::new(
                    pages,
                    ArrowDataType::Utf8View,
                    binview::BinViewDecoder::new_string(),
                )?
                .collect(filter)?;

                (
                    polars_compute::cast::cast(array.as_ref(), &dtype, CastOptionsImpl::default())
                        .unwrap(),
                    ptm,
                )
            } else {
                assert_eq!(key_type, &IntegerType::UInt32);
                PageDecoder::new(pages, dtype, CategoricalDecoder::new())?.collect_boxed(filter)?
            }
        },
        (from, to) => {
            return Err(ParquetError::not_supported(format!(
                "reading parquet type {from:?} to {to:?} still not implemented",
            )));
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
    dtype: ArrowDataType,
    filter: Option<Filter>,
    time_unit: TimeUnit,
) -> ParquetResult<(Box<dyn Array>, Bitmap)> {
    if physical_type == &PhysicalType::Int96 {
        return match time_unit {
            TimeUnit::Nanosecond => PageDecoder::new(
                pages,
                dtype,
                primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_ns(x)),
            )?
            .collect_boxed(filter),
            TimeUnit::Microsecond => PageDecoder::new(
                pages,
                dtype,
                primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_us(x)),
            )?
            .collect_boxed(filter),
            TimeUnit::Millisecond => PageDecoder::new(
                pages,
                dtype,
                primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_ms(x)),
            )?
            .collect_boxed(filter),
            TimeUnit::Second => PageDecoder::new(
                pages,
                dtype,
                primitive::FloatDecoder::closure(|x: [u32; 3]| int96_to_i64_s(x)),
            )?
            .collect_boxed(filter),
        };
    };

    if physical_type != &PhysicalType::Int64 {
        return Err(ParquetError::not_supported(
            "can't decode a timestamp from a non-int64 parquet type",
        ));
    }

    let (factor, is_multiplier) = unify_timestamp_unit(logical_type, time_unit);
    match (factor, is_multiplier) {
        (1, _) => PageDecoder::new(pages, dtype, primitive::IntDecoder::<i64, _, _>::unit())?
            .collect_boxed(filter),
        (a, true) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::closure(|x: i64| x * a))?
                .collect_boxed(filter)
        },
        (a, false) => {
            PageDecoder::new(pages, dtype, primitive::IntDecoder::closure(|x: i64| x / a))?
                .collect_boxed(filter)
        },
    }
}
