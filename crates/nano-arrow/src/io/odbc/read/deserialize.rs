use chrono::{NaiveDate, NaiveDateTime};
use odbc_api::buffers::{BinColumnView, TextColumnView};
use odbc_api::Bit;

use crate::array::{Array, BinaryArray, BooleanArray, PrimitiveArray, Utf8Array};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::buffer::Buffer;
use crate::datatypes::{DataType, TimeUnit};
use crate::offset::{Offsets, OffsetsBuffer};
use crate::types::NativeType;

use super::super::api::buffers::AnyColumnView;

/// Deserializes a [`AnyColumnView`] into an array of [`DataType`].
/// This is CPU-bounded
pub fn deserialize(column: AnyColumnView, data_type: DataType) -> Box<dyn Array> {
    match column {
        AnyColumnView::Text(view) => Box::new(utf8(data_type, view)) as _,
        AnyColumnView::WText(_) => todo!(),
        AnyColumnView::Binary(view) => Box::new(binary(data_type, view)) as _,
        AnyColumnView::Date(values) => Box::new(date(data_type, values)) as _,
        AnyColumnView::Time(values) => Box::new(time(data_type, values)) as _,
        AnyColumnView::Timestamp(values) => Box::new(timestamp(data_type, values)) as _,
        AnyColumnView::F64(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::F32(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::I8(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::I16(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::I32(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::I64(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::U8(values) => Box::new(primitive(data_type, values)) as _,
        AnyColumnView::Bit(values) => Box::new(bool(data_type, values)) as _,
        AnyColumnView::NullableDate(slice) => Box::new(date_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableTime(slice) => Box::new(time_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableTimestamp(slice) => Box::new(timestamp_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableF64(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableF32(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableI8(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableI16(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableI32(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableI64(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableU8(slice) => Box::new(primitive_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
        AnyColumnView::NullableBit(slice) => Box::new(bool_optional(
            data_type,
            slice.raw_values().0,
            slice.raw_values().1,
        )) as _,
    }
}

fn bitmap(values: &[isize]) -> Option<Bitmap> {
    MutableBitmap::from_trusted_len_iter(values.iter().map(|x| *x != -1)).into()
}

fn primitive<T: NativeType>(data_type: DataType, values: &[T]) -> PrimitiveArray<T> {
    PrimitiveArray::new(data_type, values.to_vec().into(), None)
}

fn primitive_optional<T: NativeType>(
    data_type: DataType,
    values: &[T],
    indicators: &[isize],
) -> PrimitiveArray<T> {
    let validity = bitmap(indicators);
    PrimitiveArray::new(data_type, values.to_vec().into(), validity)
}

fn bool(data_type: DataType, values: &[Bit]) -> BooleanArray {
    let values = values.iter().map(|x| x.as_bool());
    let values = Bitmap::from_trusted_len_iter(values);
    BooleanArray::new(data_type, values, None)
}

fn bool_optional(data_type: DataType, values: &[Bit], indicators: &[isize]) -> BooleanArray {
    let validity = bitmap(indicators);
    let values = values.iter().map(|x| x.as_bool());
    let values = Bitmap::from_trusted_len_iter(values);
    BooleanArray::new(data_type, values, validity)
}

fn binary_generic<'a>(
    iter: impl Iterator<Item = Option<&'a [u8]>>,
) -> (OffsetsBuffer<i32>, Buffer<u8>, Option<Bitmap>) {
    let length = iter.size_hint().0;
    let mut validity = MutableBitmap::with_capacity(length);
    let mut values = Vec::<u8>::with_capacity(0);

    let mut offsets = Offsets::<i32>::with_capacity(length);
    for item in iter {
        if let Some(item) = item {
            values.extend_from_slice(item);
            offsets
                .try_push_usize(item.len())
                .expect("List to contain less than i32::MAX items.");
            validity.push(true);
        } else {
            offsets.extend_constant(1);
            validity.push(false);
        }
    }

    (offsets.into(), values.into(), validity.into())
}

fn binary(data_type: DataType, view: BinColumnView) -> BinaryArray<i32> {
    let (offsets, values, validity) = binary_generic(view.iter());
    BinaryArray::new(data_type, offsets, values, validity)
}

fn utf8(data_type: DataType, view: TextColumnView<u8>) -> Utf8Array<i32> {
    let (offsets, values, validity) = binary_generic(view.iter());

    // this O(N) check is necessary for the utf8 validity
    Utf8Array::new(data_type, offsets, values, validity)
}

fn date(data_type: DataType, values: &[odbc_api::sys::Date]) -> PrimitiveArray<i32> {
    let values = values.iter().map(days_since_epoch).collect::<Vec<_>>();
    PrimitiveArray::new(data_type, values.into(), None)
}

fn date_optional(
    data_type: DataType,
    values: &[odbc_api::sys::Date],
    indicators: &[isize],
) -> PrimitiveArray<i32> {
    let values = values.iter().map(days_since_epoch).collect::<Vec<_>>();
    let validity = bitmap(indicators);
    PrimitiveArray::new(data_type, values.into(), validity)
}

fn days_since_epoch(date: &odbc_api::sys::Date) -> i32 {
    let unix_epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("invalid or out-of-range date");
    let date = NaiveDate::from_ymd_opt(date.year as i32, date.month as u32, date.day as u32)
        .unwrap_or(unix_epoch);
    let duration = date.signed_duration_since(unix_epoch);
    duration.num_days().try_into().unwrap_or(i32::MAX)
}

fn time(data_type: DataType, values: &[odbc_api::sys::Time]) -> PrimitiveArray<i32> {
    let values = values.iter().map(time_since_midnight).collect::<Vec<_>>();
    PrimitiveArray::new(data_type, values.into(), None)
}

fn time_since_midnight(date: &odbc_api::sys::Time) -> i32 {
    (date.hour as i32) * 60 * 60 + (date.minute as i32) * 60 + date.second as i32
}

fn time_optional(
    data_type: DataType,
    values: &[odbc_api::sys::Time],
    indicators: &[isize],
) -> PrimitiveArray<i32> {
    let values = values.iter().map(time_since_midnight).collect::<Vec<_>>();
    let validity = bitmap(indicators);
    PrimitiveArray::new(data_type, values.into(), validity)
}

fn timestamp(data_type: DataType, values: &[odbc_api::sys::Timestamp]) -> PrimitiveArray<i64> {
    let unit = if let DataType::Timestamp(unit, _) = &data_type {
        unit
    } else {
        unreachable!()
    };
    let values = match unit {
        TimeUnit::Second => values.iter().map(timestamp_s).collect::<Vec<_>>(),
        TimeUnit::Millisecond => values.iter().map(timestamp_ms).collect::<Vec<_>>(),
        TimeUnit::Microsecond => values.iter().map(timestamp_us).collect::<Vec<_>>(),
        TimeUnit::Nanosecond => values.iter().map(timestamp_ns).collect::<Vec<_>>(),
    };
    PrimitiveArray::new(data_type, values.into(), None)
}

fn timestamp_optional(
    data_type: DataType,
    values: &[odbc_api::sys::Timestamp],
    indicators: &[isize],
) -> PrimitiveArray<i64> {
    let unit = if let DataType::Timestamp(unit, _) = &data_type {
        unit
    } else {
        unreachable!()
    };
    let values = match unit {
        TimeUnit::Second => values.iter().map(timestamp_s).collect::<Vec<_>>(),
        TimeUnit::Millisecond => values.iter().map(timestamp_ms).collect::<Vec<_>>(),
        TimeUnit::Microsecond => values.iter().map(timestamp_us).collect::<Vec<_>>(),
        TimeUnit::Nanosecond => values.iter().map(timestamp_ns).collect::<Vec<_>>(),
    };
    let validity = bitmap(indicators);
    PrimitiveArray::new(data_type, values.into(), validity)
}

fn timestamp_to_naive(timestamp: &odbc_api::sys::Timestamp) -> Option<NaiveDateTime> {
    NaiveDate::from_ymd_opt(
        timestamp.year as i32,
        timestamp.month as u32,
        timestamp.day as u32,
    )
    .and_then(|x| {
        x.and_hms_nano_opt(
            timestamp.hour as u32,
            timestamp.minute as u32,
            timestamp.second as u32,
            /*
            https://docs.microsoft.com/en-us/sql/odbc/reference/appendixes/c-data-types?view=sql-server-ver15
            [b] The value of the fraction field is [...] for a billionth of a second (one nanosecond) is 1.
            */
            timestamp.fraction,
        )
    })
}

fn timestamp_s(timestamp: &odbc_api::sys::Timestamp) -> i64 {
    timestamp_to_naive(timestamp)
        .map(|x| x.timestamp())
        .unwrap_or(0)
}

fn timestamp_ms(timestamp: &odbc_api::sys::Timestamp) -> i64 {
    timestamp_to_naive(timestamp)
        .map(|x| x.timestamp_millis())
        .unwrap_or(0)
}

fn timestamp_us(timestamp: &odbc_api::sys::Timestamp) -> i64 {
    timestamp_to_naive(timestamp)
        .map(|x| x.timestamp_nanos_opt().unwrap() / 1000)
        .unwrap_or(0)
}

fn timestamp_ns(timestamp: &odbc_api::sys::Timestamp) -> i64 {
    timestamp_to_naive(timestamp)
        .map(|x| x.timestamp_nanos_opt().unwrap())
        .unwrap_or(0)
}
