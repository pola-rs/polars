use arrow::array::*;
use arrow::datatypes::*;
use arrow::error::{ArrowError, Result};

use TimeUnit::*;

/// Concatenate multiple `ArrayRef` with the same type.
///
/// Returns a new ArrayRef.
pub fn concat(array_list: &[ArrayRef]) -> Result<ArrayRef> {
    if array_list.is_empty() {
        return Err(ArrowError::ComputeError(
            "concat requires input of at least one array".to_string(),
        ));
    }
    let array_data_list = &array_list
        .iter()
        .map(|a| a.data_ref().clone())
        .collect::<Vec<ArrayDataRef>>();

    match array_data_list[0].data_type() {
        DataType::LargeUtf8 => {
            let mut builder = LargeStringBuilder::new(0);
            builder.append_data(array_data_list)?;
            Ok(ArrayBuilder::finish(&mut builder))
        }
        DataType::Boolean => {
            let mut builder = PrimitiveArray::<BooleanType>::builder(0);
            builder.append_data(array_data_list)?;
            Ok(ArrayBuilder::finish(&mut builder))
        }
        DataType::Int8 => concat_primitive::<Int8Type>(array_data_list),
        DataType::Int16 => concat_primitive::<Int16Type>(array_data_list),
        DataType::Int32 => concat_primitive::<Int32Type>(array_data_list),
        DataType::Int64 => concat_primitive::<Int64Type>(array_data_list),
        DataType::UInt8 => concat_primitive::<UInt8Type>(array_data_list),
        DataType::UInt16 => concat_primitive::<UInt16Type>(array_data_list),
        DataType::UInt32 => concat_primitive::<UInt32Type>(array_data_list),
        DataType::UInt64 => concat_primitive::<UInt64Type>(array_data_list),
        DataType::Float32 => concat_primitive::<Float32Type>(array_data_list),
        DataType::Float64 => concat_primitive::<Float64Type>(array_data_list),
        DataType::Date32(_) => concat_primitive::<Date32Type>(array_data_list),
        DataType::Date64(_) => concat_primitive::<Date64Type>(array_data_list),
        DataType::Time32(Second) => concat_primitive::<Time32SecondType>(array_data_list),
        DataType::Time32(Millisecond) => concat_primitive::<Time32MillisecondType>(array_data_list),
        DataType::Time64(Microsecond) => concat_primitive::<Time64MicrosecondType>(array_data_list),
        DataType::Time64(Nanosecond) => concat_primitive::<Time64NanosecondType>(array_data_list),
        DataType::Timestamp(Second, _) => concat_primitive::<TimestampSecondType>(array_data_list),
        DataType::Timestamp(Millisecond, _) => {
            concat_primitive::<TimestampMillisecondType>(array_data_list)
        }
        DataType::Timestamp(Microsecond, _) => {
            concat_primitive::<TimestampMicrosecondType>(array_data_list)
        }
        DataType::Timestamp(Nanosecond, _) => {
            concat_primitive::<TimestampNanosecondType>(array_data_list)
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            concat_primitive::<IntervalYearMonthType>(array_data_list)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            concat_primitive::<IntervalDayTimeType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Second) => {
            concat_primitive::<DurationSecondType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            concat_primitive::<DurationMillisecondType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            concat_primitive::<DurationMicrosecondType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            concat_primitive::<DurationNanosecondType>(array_data_list)
        }
        DataType::LargeList(nested_type) => concat_list(array_data_list, *nested_type.clone()),
        t => Err(ArrowError::ComputeError(format!(
            "Concat not supported for data type {:?}",
            t
        ))),
    }
}

#[inline]
fn concat_primitive<T>(array_data_list: &[ArrayDataRef]) -> Result<ArrayRef>
where
    T: ArrowNumericType,
{
    let mut builder = PrimitiveArray::<T>::builder(0);
    builder.append_data(array_data_list)?;
    Ok(ArrayBuilder::finish(&mut builder))
}

#[inline]
fn concat_primitive_list<T>(array_data_list: &[ArrayDataRef]) -> Result<ArrayRef>
where
    T: ArrowNumericType,
{
    let mut builder = LargeListBuilder::new(PrimitiveArray::<T>::builder(0));
    builder.append_data(array_data_list)?;
    Ok(ArrayBuilder::finish(&mut builder))
}

#[inline]
fn concat_list(array_data_list: &[ArrayDataRef], data_type: DataType) -> Result<ArrayRef> {
    match data_type {
        DataType::Int8 => concat_primitive_list::<Int8Type>(array_data_list),
        DataType::Int16 => concat_primitive_list::<Int16Type>(array_data_list),
        DataType::Int32 => concat_primitive_list::<Int32Type>(array_data_list),
        DataType::Int64 => concat_primitive_list::<Int64Type>(array_data_list),
        DataType::UInt8 => concat_primitive_list::<UInt8Type>(array_data_list),
        DataType::UInt16 => concat_primitive_list::<UInt16Type>(array_data_list),
        DataType::UInt32 => concat_primitive_list::<UInt32Type>(array_data_list),
        DataType::UInt64 => concat_primitive_list::<UInt64Type>(array_data_list),
        t => Err(ArrowError::ComputeError(format!(
            "Concat not supported for list with data type {:?}",
            t
        ))),
    }
}
