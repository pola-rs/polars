// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;
use std::convert::{From, TryFrom};
use std::fmt;
use std::io::Write;
use std::iter::{FromIterator, IntoIterator};
use std::mem;
use std::sync::Arc;

use chrono::prelude::*;

use super::*;
use crate::array::builder::StringDictionaryBuilder;
use crate::array::equal::JsonEqual;
use crate::buffer::{Buffer, MutableBuffer};
use crate::datatypes::DataType::Struct;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::memory;
use crate::util::bit_util;

/// Number of seconds in a day
const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
const NANOSECONDS: i64 = 1_000_000_000;

/// Trait for dealing with different types of array at runtime when the type of the
/// array is not known in advance.
pub trait Array: fmt::Debug + Send + Sync + ArrayEqual + JsonEqual {
    /// Returns the array as [`Any`](std::any::Any) so that it can be
    /// downcasted to a specific implementation.
    ///
    /// # Example:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use some::array::Int32Array;
    /// use some::datatypes::{Schema, Field, DataType};
    /// use some::record_batch::RecordBatch;
    ///
    /// # fn main() -> some::error::Result<()> {
    /// let id = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let batch = RecordBatch::try_new(
    ///     Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
    ///     vec![Arc::new(id)]
    /// )?;
    ///
    /// let int32array = batch
    ///     .column(0)
    ///     .as_any()
    ///     .downcast_ref::<Int32Array>()
    ///     .expect("Failed to downcast");
    /// # Ok(())
    /// # }
    /// ```
    fn as_any(&self) -> &Any;

    /// Returns a reference-counted pointer to the underlying data of this array.
    fn data(&self) -> ArrayDataRef;

    /// Returns a borrowed & reference-counted pointer to the underlying data of this array.
    fn data_ref(&self) -> &ArrayDataRef;

    /// Returns a reference to the [`DataType`](crate::datatypes::DataType) of this array.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::datatypes::DataType;
    /// use some::array::{Array, Int32Array};
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(*array.data_type(), DataType::Int32);
    /// ```
    fn data_type(&self) -> &DataType {
        self.data_ref().data_type()
    }

    /// Returns a zero-copy slice of this array with the indicated offset and length.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::array::{Array, Int32Array};
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// // Make slice over the values [2, 3, 4]
    /// let array_slice = array.slice(1, 3);
    ///
    /// assert!(array_slice.equals(&Int32Array::from(vec![2, 3, 4])));
    /// ```
    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        make_array(slice_data(self.data(), offset, length))
    }

    /// Returns the length (i.e., number of elements) of this array.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::array::{Array, Int32Array};
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(array.len(), 5);
    /// ```
    fn len(&self) -> usize {
        self.data().len()
    }

    /// Returns the offset into the underlying data used by this array(-slice).
    /// Note that the underlying data can be shared by many arrays.
    /// This defaults to `0`.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::array::{Array, Int32Array};
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// // Make slice over the values [2, 3, 4]
    /// let array_slice = array.slice(1, 3);
    ///
    /// assert_eq!(array.offset(), 0);
    /// assert_eq!(array_slice.offset(), 1);
    /// ```
    fn offset(&self) -> usize {
        self.data().offset()
    }

    /// Returns whether the element at `index` is null.
    /// When using this function on a slice, the index is relative to the slice.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::array::{Array, Int32Array};
    ///
    /// let array = Int32Array::from(vec![Some(1), None]);
    ///
    /// assert_eq!(array.is_null(0), false);
    /// assert_eq!(array.is_null(1), true);
    /// ```
    fn is_null(&self, index: usize) -> bool {
        self.data().is_null(self.data().offset() + index)
    }

    /// Returns whether the element at `index` is not null.
    /// When using this function on a slice, the index is relative to the slice.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::array::{Array, Int32Array};
    ///
    /// let array = Int32Array::from(vec![Some(1), None]);
    ///
    /// assert_eq!(array.is_valid(0), true);
    /// assert_eq!(array.is_valid(1), false);
    /// ```
    fn is_valid(&self, index: usize) -> bool {
        self.data().is_valid(self.data().offset() + index)
    }

    /// Returns the total number of null values in this array.
    ///
    /// # Example:
    ///
    /// ```
    /// use some::array::{Array, Int32Array};
    ///
    /// // Construct an array with values [1, NULL, NULL]
    /// let array = Int32Array::from(vec![Some(1), None, None]);
    ///
    /// assert_eq!(array.null_count(), 2);
    /// ```
    fn null_count(&self) -> usize {
        self.data().null_count()
    }
}

/// A reference-counted reference to a generic `Array`.
pub type ArrayRef = Arc<Array>;

/// Constructs an array using the input `data`.
/// Returns a reference-counted `Array` instance.
pub fn make_array(data: ArrayDataRef) -> ArrayRef {
    match data.data_type() {
        DataType::Boolean => Arc::new(BooleanArray::from(data)) as ArrayRef,
        DataType::Int8 => Arc::new(Int8Array::from(data)) as ArrayRef,
        DataType::Int16 => Arc::new(Int16Array::from(data)) as ArrayRef,
        DataType::Int32 => Arc::new(Int32Array::from(data)) as ArrayRef,
        DataType::Int64 => Arc::new(Int64Array::from(data)) as ArrayRef,
        DataType::UInt8 => Arc::new(UInt8Array::from(data)) as ArrayRef,
        DataType::UInt16 => Arc::new(UInt16Array::from(data)) as ArrayRef,
        DataType::UInt32 => Arc::new(UInt32Array::from(data)) as ArrayRef,
        DataType::UInt64 => Arc::new(UInt64Array::from(data)) as ArrayRef,
        DataType::Float16 => panic!("Float16 datatype not supported"),
        DataType::Float32 => Arc::new(Float32Array::from(data)) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(data)) as ArrayRef,
        DataType::Date32(DateUnit::Day) => Arc::new(Date32Array::from(data)) as ArrayRef,
        DataType::Date64(DateUnit::Millisecond) => {
            Arc::new(Date64Array::from(data)) as ArrayRef
        }
        DataType::Time32(TimeUnit::Second) => {
            Arc::new(Time32SecondArray::from(data)) as ArrayRef
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            Arc::new(Time32MillisecondArray::from(data)) as ArrayRef
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            Arc::new(Time64MicrosecondArray::from(data)) as ArrayRef
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            Arc::new(Time64NanosecondArray::from(data)) as ArrayRef
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            Arc::new(TimestampSecondArray::from(data)) as ArrayRef
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            Arc::new(TimestampMillisecondArray::from(data)) as ArrayRef
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            Arc::new(TimestampMicrosecondArray::from(data)) as ArrayRef
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            Arc::new(TimestampNanosecondArray::from(data)) as ArrayRef
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            Arc::new(IntervalYearMonthArray::from(data)) as ArrayRef
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            Arc::new(IntervalDayTimeArray::from(data)) as ArrayRef
        }
        DataType::Duration(TimeUnit::Second) => {
            Arc::new(DurationSecondArray::from(data)) as ArrayRef
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            Arc::new(DurationMillisecondArray::from(data)) as ArrayRef
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            Arc::new(DurationMicrosecondArray::from(data)) as ArrayRef
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            Arc::new(DurationNanosecondArray::from(data)) as ArrayRef
        }
        DataType::Binary => Arc::new(BinaryArray::from(data)) as ArrayRef,
        DataType::FixedSizeBinary(_) => {
            Arc::new(FixedSizeBinaryArray::from(data)) as ArrayRef
        }
        DataType::Utf8 => Arc::new(StringArray::from(data)) as ArrayRef,
        DataType::List(_) => Arc::new(ListArray::from(data)) as ArrayRef,
        DataType::Struct(_) => Arc::new(StructArray::from(data)) as ArrayRef,
        DataType::Union(_) => Arc::new(UnionArray::from(data)) as ArrayRef,
        DataType::FixedSizeList(_, _) => {
            Arc::new(FixedSizeListArray::from(data)) as ArrayRef
        }
        DataType::Dictionary(ref key_type, _) => match key_type.as_ref() {
            DataType::Int8 => {
                Arc::new(DictionaryArray::<Int8Type>::from(data)) as ArrayRef
            }
            DataType::Int16 => {
                Arc::new(DictionaryArray::<Int16Type>::from(data)) as ArrayRef
            }
            DataType::Int32 => {
                Arc::new(DictionaryArray::<Int32Type>::from(data)) as ArrayRef
            }
            DataType::Int64 => {
                Arc::new(DictionaryArray::<Int64Type>::from(data)) as ArrayRef
            }
            DataType::UInt8 => {
                Arc::new(DictionaryArray::<UInt8Type>::from(data)) as ArrayRef
            }
            DataType::UInt16 => {
                Arc::new(DictionaryArray::<UInt16Type>::from(data)) as ArrayRef
            }
            DataType::UInt32 => {
                Arc::new(DictionaryArray::<UInt32Type>::from(data)) as ArrayRef
            }
            DataType::UInt64 => {
                Arc::new(DictionaryArray::<UInt64Type>::from(data)) as ArrayRef
            }
            dt => panic!("Unexpected dictionary key type {:?}", dt),
        },
        DataType::Null => Arc::new(NullArray::from(data)) as ArrayRef,
        dt => panic!("Unexpected data type {:?}", dt),
    }
}

/// Creates a zero-copy slice of the array's data.
///
/// # Panics
///
/// Panics if `offset + length > data.len()`.
fn slice_data(data: ArrayDataRef, mut offset: usize, length: usize) -> ArrayDataRef {
    assert!((offset + length) <= data.len());

    let mut new_data = data.as_ref().clone();
    let len = std::cmp::min(new_data.len - offset, length);

    offset += data.offset;
    new_data.len = len;
    new_data.offset = offset;

    // Calculate the new null count based on the offset
    new_data.null_count = if let Some(bitmap) = new_data.null_bitmap() {
        let valid_bits = bitmap.bits.data();
        len.checked_sub(bit_util::count_set_bits_offset(valid_bits, offset, length))
            .unwrap()
    } else {
        0
    };

    Arc::new(new_data)
}

/// ----------------------------------------------------------------------------
/// Implementations of different array types

struct RawPtrBox<T> {
    inner: *const T,
}

impl<T> RawPtrBox<T> {
    fn new(inner: *const T) -> Self {
        Self { inner }
    }

    fn get(&self) -> *const T {
        self.inner
    }
}

unsafe impl<T> Send for RawPtrBox<T> {}
unsafe impl<T> Sync for RawPtrBox<T> {}

/// Array whose elements are of primitive types.
pub struct PrimitiveArray<T: ArrowPrimitiveType> {
    data: ArrayDataRef,
    /// Pointer to the value array. The lifetime of this must be <= to the value buffer
    /// stored in `data`, so it's safe to store.
    /// Also note that boolean arrays are bit-packed, so although the underlying pointer
    /// is of type bool it should be cast back to u8 before being used.
    /// i.e. `self.raw_values.get() as *const u8`
    raw_values: RawPtrBox<T::Native>,
}

/// Common operations for primitive types, including numeric types and boolean type.
pub trait PrimitiveArrayOps<T: ArrowPrimitiveType> {
    fn values(&self) -> Buffer;
    fn value(&self, i: usize) -> T::Native;
}

// This is necessary when caller wants to access `PrimitiveArrayOps`'s methods with
// `ArrowPrimitiveType`. It doesn't have any implementation as the actual implementations
// are delegated to that of `ArrowNumericType` and `BooleanType`.
impl<T: ArrowPrimitiveType> PrimitiveArrayOps<T> for PrimitiveArray<T> {
    default fn values(&self) -> Buffer {
        unimplemented!()
    }

    default fn value(&self, _: usize) -> T::Native {
        unimplemented!()
    }
}

impl<T: ArrowNumericType> PrimitiveArrayOps<T> for PrimitiveArray<T> {
    fn values(&self) -> Buffer {
        self.values()
    }

    fn value(&self, i: usize) -> T::Native {
        self.value(i)
    }
}

impl PrimitiveArrayOps<BooleanType> for BooleanArray {
    fn values(&self) -> Buffer {
        self.values()
    }

    fn value(&self, i: usize) -> bool {
        self.value(i)
    }
}

impl<T: ArrowPrimitiveType> Array for PrimitiveArray<T> {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

/// Implementation for primitive arrays with numeric types.
/// Boolean arrays are bit-packed and so implemented separately.
impl<T: ArrowNumericType> PrimitiveArray<T> {
    pub fn new(length: usize, values: Buffer, null_count: usize, offset: usize) -> Self {
        let array_data = ArrayData::builder(T::get_data_type())
            .len(length)
            .add_buffer(values)
            .null_count(null_count)
            .offset(offset)
            .build();
        PrimitiveArray::from(array_data)
    }

    /// Returns a `Buffer` holding all the values of this array.
    ///
    /// Note this doesn't take the offset of this array into account.
    pub fn values(&self) -> Buffer {
        self.data.buffers()[0].clone()
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a raw pointer to the values of this array.
    pub fn raw_values(&self) -> *const T::Native {
        unsafe { self.raw_values.get().add(self.data.offset()) }
    }

    /// Returns the primitive value at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    pub fn value(&self, i: usize) -> T::Native {
        unsafe { *(self.raw_values().add(i)) }
    }

    /// Returns a slice for the given offset and length
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    pub fn value_slice(&self, offset: usize, len: usize) -> &[T::Native] {
        let raw =
            unsafe { std::slice::from_raw_parts(self.raw_values().add(offset), len) };
        &raw[..]
    }

    // Returns a new primitive array builder
    pub fn builder(capacity: usize) -> PrimitiveBuilder<T> {
        PrimitiveBuilder::<T>::new(capacity)
    }
}

impl<T: ArrowTemporalType + ArrowNumericType> PrimitiveArray<T>
where
    i64: std::convert::From<T::Native>,
{
    /// Returns value as a chrono `NaiveDateTime`, handling time resolution
    ///
    /// If a data type cannot be converted to `NaiveDateTime`, a `None` is returned.
    /// A valid value is expected, thus the user should first check for validity.
    pub fn value_as_datetime(&self, i: usize) -> Option<NaiveDateTime> {
        let v = i64::from(self.value(i));
        match self.data_type() {
            DataType::Date32(_) => {
                // convert days into seconds
                Some(NaiveDateTime::from_timestamp(v as i64 * SECONDS_IN_DAY, 0))
            }
            DataType::Date64(_) => Some(NaiveDateTime::from_timestamp(
                // extract seconds from milliseconds
                v / MILLISECONDS,
                // discard extracted seconds and convert milliseconds to nanoseconds
                (v % MILLISECONDS * MICROSECONDS) as u32,
            )),
            DataType::Time32(_) | DataType::Time64(_) => None,
            DataType::Timestamp(unit, _) => match unit {
                TimeUnit::Second => Some(NaiveDateTime::from_timestamp(v, 0)),
                TimeUnit::Millisecond => Some(NaiveDateTime::from_timestamp(
                    // extract seconds from milliseconds
                    v / MILLISECONDS,
                    // discard extracted seconds and convert milliseconds to nanoseconds
                    (v % MILLISECONDS * MICROSECONDS) as u32,
                )),
                TimeUnit::Microsecond => Some(NaiveDateTime::from_timestamp(
                    // extract seconds from microseconds
                    v / MICROSECONDS,
                    // discard extracted seconds and convert microseconds to nanoseconds
                    (v % MICROSECONDS * MILLISECONDS) as u32,
                )),
                TimeUnit::Nanosecond => Some(NaiveDateTime::from_timestamp(
                    // extract seconds from nanoseconds
                    v / NANOSECONDS,
                    // discard extracted seconds
                    (v % NANOSECONDS) as u32,
                )),
            },
            // interval is not yet fully documented [ARROW-3097]
            DataType::Interval(_) => None,
            _ => None,
        }
    }

    /// Returns value as a chrono `NaiveDate` by using `Self::datetime()`
    ///
    /// If a data type cannot be converted to `NaiveDate`, a `None` is returned
    pub fn value_as_date(&self, i: usize) -> Option<NaiveDate> {
        self.value_as_datetime(i).map(|datetime| datetime.date())
    }

    /// Returns a value as a chrono `NaiveTime`
    ///
    /// `Date32` and `Date64` return UTC midnight as they do not have time resolution
    pub fn value_as_time(&self, i: usize) -> Option<NaiveTime> {
        match self.data_type() {
            DataType::Time32(unit) => {
                // safe to immediately cast to u32 as `self.value(i)` is positive i32
                let v = i64::from(self.value(i)) as u32;
                match unit {
                    TimeUnit::Second => {
                        Some(NaiveTime::from_num_seconds_from_midnight(v, 0))
                    }
                    TimeUnit::Millisecond => {
                        Some(NaiveTime::from_num_seconds_from_midnight(
                            // extract seconds from milliseconds
                            v / MILLISECONDS as u32,
                            // discard extracted seconds and convert milliseconds to
                            // nanoseconds
                            v % MILLISECONDS as u32 * MICROSECONDS as u32,
                        ))
                    }
                    _ => None,
                }
            }
            DataType::Time64(unit) => {
                let v = i64::from(self.value(i));
                match unit {
                    TimeUnit::Microsecond => {
                        Some(NaiveTime::from_num_seconds_from_midnight(
                            // extract seconds from microseconds
                            (v / MICROSECONDS) as u32,
                            // discard extracted seconds and convert microseconds to
                            // nanoseconds
                            (v % MICROSECONDS * MILLISECONDS) as u32,
                        ))
                    }
                    TimeUnit::Nanosecond => {
                        Some(NaiveTime::from_num_seconds_from_midnight(
                            // extract seconds from nanoseconds
                            (v / NANOSECONDS) as u32,
                            // discard extracted seconds
                            (v % NANOSECONDS) as u32,
                        ))
                    }
                    _ => None,
                }
            }
            DataType::Timestamp(_, _) => {
                self.value_as_datetime(i).map(|datetime| datetime.time())
            }
            DataType::Date32(_) | DataType::Date64(_) => {
                Some(NaiveTime::from_hms(0, 0, 0))
            }
            DataType::Interval(_) => None,
            _ => None,
        }
    }
}

impl<T: ArrowPrimitiveType> fmt::Debug for PrimitiveArray<T> {
    default fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PrimitiveArray<{:?}>\n[\n", T::get_data_type())?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

impl<T: ArrowNumericType> fmt::Debug for PrimitiveArray<T> {
    default fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PrimitiveArray<{:?}>\n[\n", T::get_data_type())?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

impl<T: ArrowNumericType + ArrowTemporalType> fmt::Debug for PrimitiveArray<T>
where
    i64: std::convert::From<T::Native>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PrimitiveArray<{:?}>\n[\n", T::get_data_type())?;
        print_long_array(self, f, |array, index, f| match T::get_data_type() {
            DataType::Date32(_) | DataType::Date64(_) => {
                match array.value_as_date(index) {
                    Some(date) => write!(f, "{:?}", date),
                    None => write!(f, "null"),
                }
            }
            DataType::Time32(_) | DataType::Time64(_) => {
                match array.value_as_time(index) {
                    Some(time) => write!(f, "{:?}", time),
                    None => write!(f, "null"),
                }
            }
            DataType::Timestamp(_, _) => match array.value_as_datetime(index) {
                Some(datetime) => write!(f, "{:?}", datetime),
                None => write!(f, "null"),
            },
            _ => write!(f, "null"),
        })?;
        write!(f, "]")
    }
}

/// Specific implementation for Boolean arrays due to bit-packing
impl PrimitiveArray<BooleanType> {
    pub fn new(length: usize, values: Buffer, null_count: usize, offset: usize) -> Self {
        let array_data = ArrayData::builder(DataType::Boolean)
            .len(length)
            .add_buffer(values)
            .null_count(null_count)
            .offset(offset)
            .build();
        BooleanArray::from(array_data)
    }

    /// Returns a `Buffer` holds all the values of this array.
    ///
    /// Note this doesn't take account into the offset of this array.
    pub fn values(&self) -> Buffer {
        self.data.buffers()[0].clone()
    }

    /// Returns the boolean value at index `i`.
    pub fn value(&self, i: usize) -> bool {
        assert!(i < self.data.len());
        let offset = i + self.offset();
        unsafe { bit_util::get_bit_raw(self.raw_values.get() as *const u8, offset) }
    }

    // Returns a new primitive array builder
    pub fn builder(capacity: usize) -> BooleanBuilder {
        BooleanBuilder::new(capacity)
    }
}

impl fmt::Debug for PrimitiveArray<BooleanType> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PrimitiveArray<{:?}>\n[\n", BooleanType::get_data_type())?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

// TODO: the macro is needed here because we'd get "conflicting implementations" error
// otherwise with both `From<Vec<T::Native>>` and `From<Vec<Option<T::Native>>>`.
// We should revisit this in future.
macro_rules! def_numeric_from_vec {
    ( $ty:ident, $native_ty:ident, $ty_id:expr ) => {
        impl From<Vec<$native_ty>> for PrimitiveArray<$ty> {
            fn from(data: Vec<$native_ty>) -> Self {
                let array_data = ArrayData::builder($ty_id)
                    .len(data.len())
                    .add_buffer(Buffer::from(data.to_byte_slice()))
                    .build();
                PrimitiveArray::from(array_data)
            }
        }

        // Constructs a primitive array from a vector. Should only be used for testing.
        impl From<Vec<Option<$native_ty>>> for PrimitiveArray<$ty> {
            fn from(data: Vec<Option<$native_ty>>) -> Self {
                let data_len = data.len();
                let num_bytes = bit_util::ceil(data_len, 8);
                let mut null_buf =
                    MutableBuffer::new(num_bytes).with_bitset(num_bytes, false);
                let mut val_buf =
                    MutableBuffer::new(data_len * mem::size_of::<$native_ty>());

                {
                    let null = vec![0; mem::size_of::<$native_ty>()];
                    let null_slice = null_buf.data_mut();
                    for (i, v) in data.iter().enumerate() {
                        if let Some(n) = v {
                            bit_util::set_bit(null_slice, i);
                            // unwrap() in the following should be safe here since we've
                            // made sure enough space is allocated for the values.
                            val_buf.write_all(&n.to_byte_slice()).unwrap();
                        } else {
                            val_buf.write_all(&null).unwrap();
                        }
                    }
                }

                let array_data = ArrayData::builder($ty_id)
                    .len(data_len)
                    .add_buffer(val_buf.freeze())
                    .null_bit_buffer(null_buf.freeze())
                    .build();
                PrimitiveArray::from(array_data)
            }
        }
    };
}

def_numeric_from_vec!(Int8Type, i8, DataType::Int8);
def_numeric_from_vec!(Int16Type, i16, DataType::Int16);
def_numeric_from_vec!(Int32Type, i32, DataType::Int32);
def_numeric_from_vec!(Int64Type, i64, DataType::Int64);
def_numeric_from_vec!(UInt8Type, u8, DataType::UInt8);
def_numeric_from_vec!(UInt16Type, u16, DataType::UInt16);
def_numeric_from_vec!(UInt32Type, u32, DataType::UInt32);
def_numeric_from_vec!(UInt64Type, u64, DataType::UInt64);
def_numeric_from_vec!(Float32Type, f32, DataType::Float32);
def_numeric_from_vec!(Float64Type, f64, DataType::Float64);

def_numeric_from_vec!(Date32Type, i32, DataType::Date32(DateUnit::Day));
def_numeric_from_vec!(Date64Type, i64, DataType::Date64(DateUnit::Millisecond));
def_numeric_from_vec!(Time32SecondType, i32, DataType::Time32(TimeUnit::Second));
def_numeric_from_vec!(
    Time32MillisecondType,
    i32,
    DataType::Time32(TimeUnit::Millisecond)
);
def_numeric_from_vec!(
    Time64MicrosecondType,
    i64,
    DataType::Time64(TimeUnit::Microsecond)
);
def_numeric_from_vec!(
    Time64NanosecondType,
    i64,
    DataType::Time64(TimeUnit::Nanosecond)
);
def_numeric_from_vec!(
    IntervalYearMonthType,
    i32,
    DataType::Interval(IntervalUnit::YearMonth)
);
def_numeric_from_vec!(
    IntervalDayTimeType,
    i64,
    DataType::Interval(IntervalUnit::DayTime)
);
def_numeric_from_vec!(
    DurationSecondType,
    i64,
    DataType::Duration(TimeUnit::Second)
);
def_numeric_from_vec!(
    DurationMillisecondType,
    i64,
    DataType::Duration(TimeUnit::Millisecond)
);
def_numeric_from_vec!(
    DurationMicrosecondType,
    i64,
    DataType::Duration(TimeUnit::Microsecond)
);
def_numeric_from_vec!(
    DurationNanosecondType,
    i64,
    DataType::Duration(TimeUnit::Nanosecond)
);
def_numeric_from_vec!(
    TimestampMillisecondType,
    i64,
    DataType::Timestamp(TimeUnit::Millisecond, None)
);
def_numeric_from_vec!(
    TimestampMicrosecondType,
    i64,
    DataType::Timestamp(TimeUnit::Microsecond, None)
);

impl<T: ArrowTimestampType> PrimitiveArray<T> {
    /// Construct a timestamp array from a vec of i64 values and an optional timezone
    pub fn from_vec(data: Vec<i64>, timezone: Option<Arc<String>>) -> Self {
        let array_data =
            ArrayData::builder(DataType::Timestamp(T::get_time_unit(), timezone))
                .len(data.len())
                .add_buffer(Buffer::from(data.to_byte_slice()))
                .build();
        PrimitiveArray::from(array_data)
    }
}

impl<T: ArrowTimestampType> PrimitiveArray<T> {
    /// Construct a timestamp array from a vec of Option<i64> values and an optional timezone
    pub fn from_opt_vec(data: Vec<Option<i64>>, timezone: Option<Arc<String>>) -> Self {
        // TODO: duplicated from def_numeric_from_vec! macro, it looks possible to convert to generic
        let data_len = data.len();
        let num_bytes = bit_util::ceil(data_len, 8);
        let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, false);
        let mut val_buf = MutableBuffer::new(data_len * mem::size_of::<i64>());

        {
            let null = vec![0; mem::size_of::<i64>()];
            let null_slice = null_buf.data_mut();
            for (i, v) in data.iter().enumerate() {
                if let Some(n) = v {
                    bit_util::set_bit(null_slice, i);
                    // unwrap() in the following should be safe here since we've
                    // made sure enough space is allocated for the values.
                    val_buf.write_all(&n.to_byte_slice()).unwrap();
                } else {
                    val_buf.write_all(&null).unwrap();
                }
            }
        }

        let array_data =
            ArrayData::builder(DataType::Timestamp(T::get_time_unit(), timezone))
                .len(data_len)
                .add_buffer(val_buf.freeze())
                .null_bit_buffer(null_buf.freeze())
                .build();
        PrimitiveArray::from(array_data)
    }
}

/// Constructs a boolean array from a vector. Should only be used for testing.
impl From<Vec<bool>> for BooleanArray {
    fn from(data: Vec<bool>) -> Self {
        let num_byte = bit_util::ceil(data.len(), 8);
        let mut mut_buf = MutableBuffer::new(num_byte).with_bitset(num_byte, false);
        {
            let mut_slice = mut_buf.data_mut();
            for (i, b) in data.iter().enumerate() {
                if *b {
                    bit_util::set_bit(mut_slice, i);
                }
            }
        }
        let array_data = ArrayData::builder(DataType::Boolean)
            .len(data.len())
            .add_buffer(mut_buf.freeze())
            .build();
        BooleanArray::from(array_data)
    }
}

impl From<Vec<Option<bool>>> for BooleanArray {
    fn from(data: Vec<Option<bool>>) -> Self {
        let data_len = data.len();
        let num_byte = bit_util::ceil(data_len, 8);
        let mut null_buf = MutableBuffer::new(num_byte).with_bitset(num_byte, false);
        let mut val_buf = MutableBuffer::new(num_byte).with_bitset(num_byte, false);

        {
            let null_slice = null_buf.data_mut();
            let val_slice = val_buf.data_mut();

            for (i, v) in data.iter().enumerate() {
                if let Some(b) = v {
                    bit_util::set_bit(null_slice, i);
                    if *b {
                        bit_util::set_bit(val_slice, i);
                    }
                }
            }
        }

        let array_data = ArrayData::builder(DataType::Boolean)
            .len(data_len)
            .add_buffer(val_buf.freeze())
            .null_bit_buffer(null_buf.freeze())
            .build();
        BooleanArray::from(array_data)
    }
}

/// Constructs a `PrimitiveArray` from an array data reference.
impl<T: ArrowPrimitiveType> From<ArrayDataRef> for PrimitiveArray<T> {
    default fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            1,
            "PrimitiveArray data should contain a single buffer only (values buffer)"
        );
        let raw_values = data.buffers()[0].raw_data();
        assert!(
            memory::is_aligned::<u8>(raw_values, mem::align_of::<T::Native>()),
            "memory is not aligned"
        );
        Self {
            data,
            raw_values: RawPtrBox::new(raw_values as *const T::Native),
        }
    }
}

/// Common operations for List types, currently `ListArray`, `FixedSizeListArray`, `BinaryArray`
/// `StringArray` and `DictionaryArray`
pub trait ListArrayOps {
    fn value_offset_at(&self, i: usize) -> i32;
}

impl ListArrayOps for ListArray {
    fn value_offset_at(&self, i: usize) -> i32 {
        self.value_offset_at(i)
    }
}

impl ListArrayOps for FixedSizeListArray {
    fn value_offset_at(&self, i: usize) -> i32 {
        self.value_offset_at(i)
    }
}

impl ListArrayOps for BinaryArray {
    fn value_offset_at(&self, i: usize) -> i32 {
        self.value_offset_at(i)
    }
}

impl ListArrayOps for StringArray {
    fn value_offset_at(&self, i: usize) -> i32 {
        self.value_offset_at(i)
    }
}

impl ListArrayOps for FixedSizeBinaryArray {
    fn value_offset_at(&self, i: usize) -> i32 {
        self.value_offset_at(i)
    }
}

/// A list array where each element is a variable-sized sequence of values with the same
/// type.
pub struct ListArray {
    data: ArrayDataRef,
    values: ArrayRef,
    value_offsets: RawPtrBox<i32>,
}

impl ListArray {
    /// Returns a reference to the values of this list.
    pub fn values(&self) -> ArrayRef {
        self.values.clone()
    }

    /// Returns a clone of the value type of this list.
    pub fn value_type(&self) -> DataType {
        self.values.data().data_type().clone()
    }

    /// Returns ith value of this list array.
    pub fn value(&self, i: usize) -> ArrayRef {
        self.values
            .slice(self.value_offset(i) as usize, self.value_length(i) as usize)
    }

    /// Returns the offset for value at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_offset(&self, i: usize) -> i32 {
        self.value_offset_at(self.data.offset() + i)
    }

    /// Returns the length for value at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_length(&self, mut i: usize) -> i32 {
        i += self.data.offset();
        self.value_offset_at(i + 1) - self.value_offset_at(i)
    }

    #[inline]
    fn value_offset_at(&self, i: usize) -> i32 {
        unsafe { *self.value_offsets.get().add(i) }
    }
}

/// Constructs a `ListArray` from an array data reference.
impl From<ArrayDataRef> for ListArray {
    fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            1,
            "ListArray data should contain a single buffer only (value offsets)"
        );
        assert_eq!(
            data.child_data().len(),
            1,
            "ListArray should contain a single child array (values array)"
        );
        let values = make_array(data.child_data()[0].clone());
        let raw_value_offsets = data.buffers()[0].raw_data();
        assert!(
            memory::is_aligned(raw_value_offsets, mem::align_of::<i32>()),
            "memory is not aligned"
        );
        let value_offsets = raw_value_offsets as *const i32;
        unsafe {
            assert_eq!(*value_offsets.offset(0), 0, "offsets do not start at zero");
        }
        Self {
            data,
            values,
            value_offsets: RawPtrBox::new(value_offsets),
        }
    }
}

impl Array for ListArray {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

// Helper function for printing potentially long arrays.
fn print_long_array<A, F>(array: &A, f: &mut fmt::Formatter, print_item: F) -> fmt::Result
where
    A: Array,
    F: Fn(&A, usize, &mut fmt::Formatter) -> fmt::Result,
{
    for i in 0..std::cmp::min(10, array.len()) {
        if array.is_null(i) {
            writeln!(f, "  null,")?;
        } else {
            write!(f, "  ")?;
            print_item(&array, i, f)?;
            writeln!(f, ",")?;
        }
    }
    if array.len() > 10 {
        if array.len() > 20 {
            writeln!(f, "  ...{} elements...,", array.len() - 20)?;
        }
        for i in array.len() - 10..array.len() {
            if array.is_null(i) {
                writeln!(f, "  null,")?;
            } else {
                write!(f, "  ")?;
                print_item(&array, i, f)?;
                writeln!(f, ",")?;
            }
        }
    }
    Ok(())
}

impl fmt::Debug for ListArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ListArray\n[\n")?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

/// A list array where each element is a fixed-size sequence of values with the same
/// type.
pub struct FixedSizeListArray {
    data: ArrayDataRef,
    values: ArrayRef,
    length: i32,
}

impl FixedSizeListArray {
    /// Returns a reference to the values of this list.
    pub fn values(&self) -> ArrayRef {
        self.values.clone()
    }

    /// Returns a clone of the value type of this list.
    pub fn value_type(&self) -> DataType {
        self.values.data().data_type().clone()
    }

    /// Returns ith value of this list array.
    pub fn value(&self, i: usize) -> ArrayRef {
        self.values
            .slice(self.value_offset(i) as usize, self.value_length() as usize)
    }

    /// Returns the offset for value at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_offset(&self, i: usize) -> i32 {
        self.value_offset_at(self.data.offset() + i)
    }

    /// Returns the length for value at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_length(&self) -> i32 {
        self.length
    }

    #[inline]
    fn value_offset_at(&self, i: usize) -> i32 {
        i as i32 * self.length
    }
}

/// Constructs a `FixedSizeListArray` from an array data reference.
impl From<ArrayDataRef> for FixedSizeListArray {
    fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            0,
            "FixedSizeListArray data should not contain a buffer for value offsets"
        );
        assert_eq!(
            data.child_data().len(),
            1,
            "FixedSizeListArray should contain a single child array (values array)"
        );
        let values = make_array(data.child_data()[0].clone());
        let length = match data.data_type() {
            DataType::FixedSizeList(_, len) => {
                // check that child data is multiple of length
                assert_eq!(
                    values.len() % *len as usize,
                    0,
                    "FixedSizeListArray child array length should be a multiple of {}",
                    len
                );
                *len
            }
            _ => {
                panic!("FixedSizeListArray data should contain a FixedSizeList data type")
            }
        };
        Self {
            data,
            values,
            length,
        }
    }
}

impl Array for FixedSizeListArray {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

impl fmt::Debug for FixedSizeListArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FixedSizeListArray<{}>\n[\n", self.value_length())?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

/// A type of `ListArray` whose elements are binaries.
pub struct BinaryArray {
    data: ArrayDataRef,
    value_offsets: RawPtrBox<i32>,
    value_data: RawPtrBox<u8>,
}

/// A type of `ListArray` whose elements are UTF8 strings.
pub struct StringArray {
    data: ArrayDataRef,
    value_offsets: RawPtrBox<i32>,
    value_data: RawPtrBox<u8>,
}

/// A type of `FixedSizeListArray` whose elements are binaries.
pub struct FixedSizeBinaryArray {
    data: ArrayDataRef,
    value_data: RawPtrBox<u8>,
    length: i32,
}

impl BinaryArray {
    /// Returns the element at index `i` as a byte slice.
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.data.len(), "BinaryArray out of bounds access");
        let offset = i.checked_add(self.data.offset()).unwrap();
        unsafe {
            let pos = self.value_offset_at(offset);
            std::slice::from_raw_parts(
                self.value_data.get().offset(pos as isize),
                (self.value_offset_at(offset + 1) - pos) as usize,
            )
        }
    }

    /// Returns the offset for the element at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_offset(&self, i: usize) -> i32 {
        self.value_offset_at(self.data.offset() + i)
    }

    /// Returns the length for the element at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_length(&self, mut i: usize) -> i32 {
        i += self.data.offset();
        self.value_offset_at(i + 1) - self.value_offset_at(i)
    }

    /// Returns a clone of the value offset buffer
    pub fn value_offsets(&self) -> Buffer {
        self.data.buffers()[0].clone()
    }

    /// Returns a clone of the value data buffer
    pub fn value_data(&self) -> Buffer {
        self.data.buffers()[1].clone()
    }

    #[inline]
    fn value_offset_at(&self, i: usize) -> i32 {
        unsafe { *self.value_offsets.get().add(i) }
    }

    // Returns a new binary array builder
    pub fn builder(capacity: usize) -> BinaryBuilder {
        BinaryBuilder::new(capacity)
    }
}

impl StringArray {
    /// Returns the element at index `i` as a string slice.
    pub fn value(&self, i: usize) -> &str {
        assert!(i < self.data.len(), "StringArray out of bounds access");
        let offset = i.checked_add(self.data.offset()).unwrap();
        unsafe {
            let pos = self.value_offset_at(offset);
            let slice = std::slice::from_raw_parts(
                self.value_data.get().offset(pos as isize),
                (self.value_offset_at(offset + 1) - pos) as usize,
            );

            std::str::from_utf8_unchecked(slice)
        }
    }

    /// Returns the offset for the element at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_offset(&self, i: usize) -> i32 {
        self.value_offset_at(self.data.offset() + i)
    }

    /// Returns the length for the element at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_length(&self, mut i: usize) -> i32 {
        i += self.data.offset();
        self.value_offset_at(i + 1) - self.value_offset_at(i)
    }

    /// Returns a clone of the value offset buffer
    pub fn value_offsets(&self) -> Buffer {
        self.data.buffers()[0].clone()
    }

    /// Returns a clone of the value data buffer
    pub fn value_data(&self) -> Buffer {
        self.data.buffers()[1].clone()
    }

    #[inline]
    fn value_offset_at(&self, i: usize) -> i32 {
        unsafe { *self.value_offsets.get().add(i) }
    }

    // Returns a new string array builder
    pub fn builder(capacity: usize) -> StringBuilder {
        StringBuilder::new(capacity)
    }
}

impl FixedSizeBinaryArray {
    /// Returns the element at index `i` as a byte slice.
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(
            i < self.data.len(),
            "FixedSizeBinaryArray out of bounds access"
        );
        let offset = i.checked_add(self.data.offset()).unwrap();
        unsafe {
            let pos = self.value_offset_at(offset);
            std::slice::from_raw_parts(
                self.value_data.get().offset(pos as isize),
                (self.value_offset_at(offset + 1) - pos) as usize,
            )
        }
    }

    /// Returns the offset for the element at index `i`.
    ///
    /// Note this doesn't do any bound checking, for performance reason.
    #[inline]
    pub fn value_offset(&self, i: usize) -> i32 {
        self.value_offset_at(self.data.offset() + i)
    }

    /// Returns the length for an element.
    ///
    /// All elements have the same length as the array is a fixed size.
    #[inline]
    pub fn value_length(&self) -> i32 {
        self.length
    }

    /// Returns a clone of the value data buffer
    pub fn value_data(&self) -> Buffer {
        self.data.buffers()[0].clone()
    }

    #[inline]
    fn value_offset_at(&self, i: usize) -> i32 {
        self.length * i as i32
    }
}

impl From<ArrayDataRef> for BinaryArray {
    fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            2,
            "BinaryArray data should contain 2 buffers only (offsets and values)"
        );
        let raw_value_offsets = data.buffers()[0].raw_data();
        assert!(
            memory::is_aligned(raw_value_offsets, mem::align_of::<i32>()),
            "memory is not aligned"
        );
        let value_data = data.buffers()[1].raw_data();
        Self {
            data,
            value_offsets: RawPtrBox::new(raw_value_offsets as *const i32),
            value_data: RawPtrBox::new(value_data),
        }
    }
}

impl From<ArrayDataRef> for StringArray {
    fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            2,
            "StringArray data should contain 2 buffers only (offsets and values)"
        );
        let raw_value_offsets = data.buffers()[0].raw_data();
        assert!(
            memory::is_aligned(raw_value_offsets, mem::align_of::<i32>()),
            "memory is not aligned"
        );
        let value_data = data.buffers()[1].raw_data();
        Self {
            data,
            value_offsets: RawPtrBox::new(raw_value_offsets as *const i32),
            value_data: RawPtrBox::new(value_data),
        }
    }
}

impl From<ArrayDataRef> for FixedSizeBinaryArray {
    fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            1,
            "FixedSizeBinaryArray data should contain 1 buffer only (values)"
        );
        let value_data = data.buffers()[0].raw_data();
        let length = match data.data_type() {
            DataType::FixedSizeBinary(len) => *len,
            _ => panic!("Expected data type to be FixedSizeBinary"),
        };
        Self {
            data,
            value_data: RawPtrBox::new(value_data),
            length,
        }
    }
}

impl<'a> From<Vec<&'a str>> for StringArray {
    fn from(v: Vec<&'a str>) -> Self {
        let mut offsets = Vec::with_capacity(v.len() + 1);
        let mut values = Vec::new();
        let mut length_so_far = 0;
        offsets.push(length_so_far);
        for s in &v {
            length_so_far += s.len() as i32;
            offsets.push(length_so_far as i32);
            values.extend_from_slice(s.as_bytes());
        }
        let array_data = ArrayData::builder(DataType::Utf8)
            .len(v.len())
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        StringArray::from(array_data)
    }
}

impl From<Vec<&[u8]>> for BinaryArray {
    fn from(v: Vec<&[u8]>) -> Self {
        let mut offsets = Vec::with_capacity(v.len() + 1);
        let mut values = Vec::new();
        let mut length_so_far = 0;
        offsets.push(length_so_far);
        for s in &v {
            length_so_far += s.len() as i32;
            offsets.push(length_so_far as i32);
            values.extend_from_slice(s);
        }
        let array_data = ArrayData::builder(DataType::Binary)
            .len(v.len())
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        BinaryArray::from(array_data)
    }
}

impl<'a> TryFrom<Vec<Option<&'a str>>> for StringArray {
    type Error = ArrowError;

    fn try_from(v: Vec<Option<&'a str>>) -> Result<Self> {
        let mut builder = StringBuilder::new(v.len());
        for val in v {
            if let Some(s) = val {
                builder.append_value(s)?;
            } else {
                builder.append(false)?;
            }
        }
        Ok(builder.finish())
    }
}

/// Creates a `BinaryArray` from `List<u8>` array
impl From<ListArray> for BinaryArray {
    fn from(v: ListArray) -> Self {
        assert_eq!(
            v.data().child_data()[0].child_data().len(),
            0,
            "BinaryArray can only be created from list array of u8 values \
             (i.e. List<PrimitiveArray<u8>>)."
        );
        assert_eq!(
            v.data().child_data()[0].data_type(),
            &DataType::UInt8,
            "BinaryArray can only be created from List<u8> arrays, mismatched data types."
        );

        let mut builder = ArrayData::builder(DataType::Binary)
            .len(v.len())
            .add_buffer(v.data().buffers()[0].clone())
            .add_buffer(v.data().child_data()[0].buffers()[0].clone());
        if let Some(bitmap) = v.data().null_bitmap() {
            builder = builder
                .null_count(v.data().null_count())
                .null_bit_buffer(bitmap.bits.clone())
        }

        let data = builder.build();
        Self::from(data)
    }
}

/// Creates a `StringArray` from `List<u8>` array
impl From<ListArray> for StringArray {
    fn from(v: ListArray) -> Self {
        assert_eq!(
            v.data().child_data()[0].child_data().len(),
            0,
            "StringArray can only be created from list array of u8 values \
             (i.e. List<PrimitiveArray<u8>>)."
        );
        assert_eq!(
            v.data().child_data()[0].data_type(),
            &DataType::UInt8,
            "StringArray can only be created from List<u8> arrays, mismatched data types."
        );

        let mut builder = ArrayData::builder(DataType::Utf8)
            .len(v.len())
            .add_buffer(v.data().buffers()[0].clone())
            .add_buffer(v.data().child_data()[0].buffers()[0].clone());
        if let Some(bitmap) = v.data().null_bitmap() {
            builder = builder
                .null_count(v.data().null_count())
                .null_bit_buffer(bitmap.bits.clone())
        }

        let data = builder.build();
        Self::from(data)
    }
}

/// Creates a `FixedSizeBinaryArray` from `FixedSizeList<u8>` array
impl From<FixedSizeListArray> for FixedSizeBinaryArray {
    fn from(v: FixedSizeListArray) -> Self {
        assert_eq!(
            v.data().child_data()[0].child_data().len(),
            0,
            "FixedSizeBinaryArray can only be created from list array of u8 values \
             (i.e. FixedSizeList<PrimitiveArray<u8>>)."
        );
        assert_eq!(
            v.data().child_data()[0].data_type(),
            &DataType::UInt8,
            "FixedSizeBinaryArray can only be created from FixedSizeList<u8> arrays, mismatched data types."
        );

        let mut builder = ArrayData::builder(DataType::FixedSizeBinary(v.value_length()))
            .len(v.len())
            .add_buffer(v.data().child_data()[0].buffers()[0].clone());
        if let Some(bitmap) = v.data().null_bitmap() {
            builder = builder
                .null_count(v.data().null_count())
                .null_bit_buffer(bitmap.bits.clone())
        }

        let data = builder.build();
        Self::from(data)
    }
}

impl fmt::Debug for BinaryArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BinaryArray\n[\n")?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

impl fmt::Debug for StringArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StringArray\n[\n")?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

impl fmt::Debug for FixedSizeBinaryArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FixedSizeBinaryArray<{}>\n[\n", self.value_length())?;
        print_long_array(self, f, |array, index, f| {
            fmt::Debug::fmt(&array.value(index), f)
        })?;
        write!(f, "]")
    }
}

impl Array for BinaryArray {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

impl Array for StringArray {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

impl Array for FixedSizeBinaryArray {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

/// A nested array type where each child (called *field*) is represented by a separate
/// array.
pub struct StructArray {
    data: ArrayDataRef,
    pub(crate) boxed_fields: Vec<ArrayRef>,
}

impl StructArray {
    /// Returns the field at `pos`.
    pub fn column(&self, pos: usize) -> &ArrayRef {
        &self.boxed_fields[pos]
    }

    /// Return the number of fields in this struct array
    pub fn num_columns(&self) -> usize {
        self.boxed_fields.len()
    }

    /// Returns the fields of the struct array
    pub fn columns(&self) -> Vec<&ArrayRef> {
        self.boxed_fields.iter().collect()
    }

    /// Returns child array refs of the struct array
    pub fn columns_ref(&self) -> Vec<ArrayRef> {
        self.boxed_fields.clone()
    }

    /// Return field names in this struct array
    pub fn column_names(&self) -> Vec<&str> {
        match self.data.data_type() {
            Struct(fields) => fields
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<&str>>(),
            _ => unreachable!("Struct array's data type is not struct!"),
        }
    }

    /// Return child array whose field name equals to column_name
    pub fn column_by_name(&self, column_name: &str) -> Option<&ArrayRef> {
        self.column_names()
            .iter()
            .position(|c| c == &column_name)
            .map(|pos| self.column(pos))
    }
}

impl From<ArrayDataRef> for StructArray {
    fn from(data: ArrayDataRef) -> Self {
        let mut boxed_fields = vec![];
        for cd in data.child_data() {
            let child_data = if data.offset != 0 || data.len != cd.len {
                slice_data(cd.clone(), data.offset, data.len)
            } else {
                cd.clone()
            };
            boxed_fields.push(make_array(child_data));
        }
        Self { data, boxed_fields }
    }
}

impl Array for StructArray {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }

    /// Returns the length (i.e., number of elements) of this array
    fn len(&self) -> usize {
        self.data().len()
    }
}

impl From<Vec<(Field, ArrayRef)>> for StructArray {
    fn from(v: Vec<(Field, ArrayRef)>) -> Self {
        let (field_types, field_values): (Vec<_>, Vec<_>) = v.into_iter().unzip();

        // Check the length of the child arrays
        let length = field_values[0].len();
        for i in 1..field_values.len() {
            assert_eq!(
                length,
                field_values[i].len(),
                "all child arrays of a StructArray must have the same length"
            );
            assert_eq!(
                field_types[i].data_type(),
                field_values[i].data().data_type(),
                "the field data types must match the array data in a StructArray"
            )
        }

        let data = ArrayData::builder(DataType::Struct(field_types))
            .child_data(field_values.into_iter().map(|a| a.data()).collect())
            .len(length)
            .build();
        Self::from(data)
    }
}

impl fmt::Debug for StructArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StructArray\n[\n")?;
        for (child_index, name) in self.column_names().iter().enumerate() {
            let column = self.column(child_index);
            writeln!(
                f,
                "-- child {}: \"{}\" ({:?})",
                child_index,
                name,
                column.data_type()
            )?;
            fmt::Debug::fmt(column, f)?;
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl From<(Vec<(Field, ArrayRef)>, Buffer, usize)> for StructArray {
    fn from(triple: (Vec<(Field, ArrayRef)>, Buffer, usize)) -> Self {
        let (field_types, field_values): (Vec<_>, Vec<_>) = triple.0.into_iter().unzip();

        // Check the length of the child arrays
        let length = field_values[0].len();
        for i in 1..field_values.len() {
            assert_eq!(
                length,
                field_values[i].len(),
                "all child arrays of a StructArray must have the same length"
            );
            assert_eq!(
                field_types[i].data_type(),
                field_values[i].data().data_type(),
                "the field data types must match the array data in a StructArray"
            )
        }

        let data = ArrayData::builder(DataType::Struct(field_types))
            .null_bit_buffer(triple.1)
            .child_data(field_values.into_iter().map(|a| a.data()).collect())
            .len(length)
            .null_count(triple.2)
            .build();
        Self::from(data)
    }
}

/// A dictionary array where each element is a single value indexed by an integer key.
/// This is mostly used to represent strings or a limited set of primitive types as integers,
/// for example when doing NLP analysis or representing chromosomes by name.
///
/// Example **with nullable** data:
///
/// ```
/// use some::array::DictionaryArray;
/// use some::datatypes::Int8Type;
/// let test = vec!["a", "a", "b", "c"];
/// let array : DictionaryArray<Int8Type> = test.iter().map(|&x| if x == "b" {None} else {Some(x)}).collect();
/// assert_eq!(array.keys().collect::<Vec<Option<i8>>>(), vec![Some(0), Some(0), None, Some(1)]);
/// ```
///
/// Example **without nullable** data:
///
/// ```
/// use some::array::DictionaryArray;
/// use some::datatypes::Int8Type;
/// let test = vec!["a", "a", "b", "c"];
/// let array : DictionaryArray<Int8Type> = test.into_iter().collect();
/// assert_eq!(array.keys().collect::<Vec<Option<i8>>>(), vec![Some(0), Some(0), Some(1), Some(2)]);
/// ```
pub struct DictionaryArray<K: ArrowPrimitiveType> {
    /// Array of keys, much like a PrimitiveArray
    data: ArrayDataRef,

    /// Pointer to the key values.
    raw_values: RawPtrBox<K::Native>,

    /// Array of any values.
    values: ArrayRef,

    /// Values are ordered.
    is_ordered: bool,
}

pub struct NullableIter<'a, T> {
    data: &'a ArrayDataRef, // TODO: Use a pointer to the null bitmap.
    ptr: *const T,
    i: usize,
    len: usize,
}

impl<'a, T> std::iter::Iterator for NullableIter<'a, T>
where
    T: Clone,
{
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i >= self.len {
            None
        } else if self.data.is_null(i) {
            self.i += 1;
            Some(None)
        } else {
            self.i += 1;
            unsafe { Some(Some((&*self.ptr.add(i)).clone())) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let i = self.i;
        if i + n >= self.len {
            self.i = self.len;
            None
        } else if self.data.is_null(i + n) {
            self.i += n + 1;
            Some(None)
        } else {
            self.i += n + 1;
            unsafe { Some(Some((&*self.ptr.add(i + n)).clone())) }
        }
    }
}

impl<'a, K: ArrowPrimitiveType> DictionaryArray<K> {
    /// Return an iterator to the keys of this dictionary.
    pub fn keys(&self) -> NullableIter<'_, K::Native> {
        NullableIter::<'_, K::Native> {
            data: &self.data,
            ptr: unsafe { self.raw_values.get().add(self.data.offset()) },
            i: 0,
            len: self.data.len(),
        }
    }

    /// Returns the lookup key by doing reverse dictionary lookup
    pub fn lookup_key(&self, value: &'static str) -> Option<K::Native> {
        let rd_buf: &StringArray =
            self.values.as_any().downcast_ref::<StringArray>().unwrap();

        (0..rd_buf.len())
            .position(|i| rd_buf.value(i) == value)
            .map(K::Native::from_usize)
            .flatten()
    }

    /// Returns an `ArrayRef` to the dictionary values.
    pub fn values(&self) -> ArrayRef {
        self.values.clone()
    }

    /// Returns a clone of the value type of this list.
    pub fn value_type(&self) -> DataType {
        self.values.data().data_type().clone()
    }

    /// The length of the dictionary is the length of the keys array.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    // Currently exists for compatibility purposes with Arrow IPC.
    pub fn is_ordered(&self) -> bool {
        self.is_ordered
    }
}

/// Constructs a `DictionaryArray` from an array data reference.
impl<T: ArrowPrimitiveType> From<ArrayDataRef> for DictionaryArray<T> {
    fn from(data: ArrayDataRef) -> Self {
        assert_eq!(
            data.buffers().len(),
            1,
            "DictionaryArray data should contain a single buffer only (keys)."
        );
        assert_eq!(
            data.child_data().len(),
            1,
            "DictionaryArray should contain a single child array (values)."
        );

        let raw_values = data.buffers()[0].raw_data();
        let dtype: &DataType = data.data_type();
        let values = make_array(data.child_data()[0].clone());
        if let DataType::Dictionary(_, _) = dtype {
            Self {
                data,
                raw_values: RawPtrBox::new(raw_values as *const T::Native),
                values,
                is_ordered: false,
            }
        } else {
            panic!("DictionaryArray must have Dictionary data type.")
        }
    }
}

/// Constructs a `DictionaryArray` from an iterator of optional strings.
impl<T: ArrowPrimitiveType + ArrowDictionaryKeyType> FromIterator<Option<&'static str>>
    for DictionaryArray<T>
{
    fn from_iter<I: IntoIterator<Item = Option<&'static str>>>(iter: I) -> Self {
        let it = iter.into_iter();
        let (lower, _) = it.size_hint();
        let key_builder = PrimitiveBuilder::<T>::new(lower);
        let value_builder = StringBuilder::new(256);
        let mut builder = StringDictionaryBuilder::new(key_builder, value_builder);
        it.for_each(|i| {
            if let Some(i) = i {
                // Note: impl ... for Result<DictionaryArray<T>> fails with
                // error[E0117]: only traits defined in the current crate can be implemented for arbitrary types
                builder
                    .append(i)
                    .expect("Unable to append a value to a dictionary array.");
            } else {
                builder
                    .append_null()
                    .expect("Unable to append a null value to a dictionary array.");
            }
        });

        builder.finish()
    }
}

/// Constructs a `DictionaryArray` from an iterator of strings.
impl<T: ArrowPrimitiveType + ArrowDictionaryKeyType> FromIterator<&'static str>
    for DictionaryArray<T>
{
    fn from_iter<I: IntoIterator<Item = &'static str>>(iter: I) -> Self {
        let it = iter.into_iter();
        let (lower, _) = it.size_hint();
        let key_builder = PrimitiveBuilder::<T>::new(lower);
        let value_builder = StringBuilder::new(256);
        let mut builder = StringDictionaryBuilder::new(key_builder, value_builder);
        it.for_each(|i| {
            builder
                .append(i)
                .expect("Unable to append a value to a dictionary array.");
        });

        builder.finish()
    }
}

impl<T: ArrowPrimitiveType> Array for DictionaryArray<T> {
    fn as_any(&self) -> &Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        self.data.clone()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        &self.data
    }
}

impl<T: ArrowPrimitiveType> fmt::Debug for DictionaryArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const MAX_LEN: usize = 10;
        let keys: Vec<_> = self.keys().take(MAX_LEN).collect();
        let elipsis = if self.keys().count() > MAX_LEN {
            "..."
        } else {
            ""
        };
        writeln!(
            f,
            "DictionaryArray {{keys: {:?}{} values: {:?}}}",
            keys, elipsis, self.values
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::thread;

    use crate::buffer::Buffer;
    use crate::datatypes::{DataType, Field};
    use crate::memory;

    #[test]
    fn test_primitive_array_from_vec() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4].to_byte_slice());
        let buf2 = buf.clone();
        let arr = Int32Array::new(5, buf, 0, 0);
        let slice = unsafe { std::slice::from_raw_parts(arr.raw_values(), 5) };
        assert_eq!(buf2, arr.values());
        assert_eq!(&[0, 1, 2, 3, 4], slice);
        assert_eq!(5, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        for i in 0..5 {
            assert!(!arr.is_null(i));
            assert!(arr.is_valid(i));
            assert_eq!(i as i32, arr.value(i));
        }
    }

    #[test]
    fn test_primitive_array_from_vec_option() {
        // Test building a primitive array with null values
        let arr = Int32Array::from(vec![Some(0), None, Some(2), None, Some(4)]);
        assert_eq!(5, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(2, arr.null_count());
        for i in 0..5 {
            if i % 2 == 0 {
                assert!(!arr.is_null(i));
                assert!(arr.is_valid(i));
                assert_eq!(i as i32, arr.value(i));
            } else {
                assert!(arr.is_null(i));
                assert!(!arr.is_valid(i));
            }
        }
    }

    #[test]
    fn test_date64_array_from_vec_option() {
        // Test building a primitive array with null values
        // we use Int32 and Int64 as a backing array, so all Int32 and Int64 conventions
        // work
        let arr: PrimitiveArray<Date64Type> =
            vec![Some(1550902545147), None, Some(1550902545147)].into();
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        for i in 0..3 {
            if i % 2 == 0 {
                assert!(!arr.is_null(i));
                assert!(arr.is_valid(i));
                assert_eq!(1550902545147, arr.value(i));
                // roundtrip to and from datetime
                assert_eq!(
                    1550902545147,
                    arr.value_as_datetime(i).unwrap().timestamp_millis()
                );
            } else {
                assert!(arr.is_null(i));
                assert!(!arr.is_valid(i));
            }
        }
    }

    #[test]
    fn test_time32_millisecond_array_from_vec() {
        // 1:        00:00:00.001
        // 37800005: 10:30:00.005
        // 86399210: 23:59:59.210
        let arr: PrimitiveArray<Time32MillisecondType> =
            vec![1, 37_800_005, 86_399_210].into();
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        let formatted = vec!["00:00:00.001", "10:30:00.005", "23:59:59.210"];
        for i in 0..3 {
            // check that we can't create dates or datetimes from time instances
            assert_eq!(None, arr.value_as_datetime(i));
            assert_eq!(None, arr.value_as_date(i));
            let time = arr.value_as_time(i).unwrap();
            assert_eq!(formatted[i], time.format("%H:%M:%S%.3f").to_string());
        }
    }

    #[test]
    fn test_time64_nanosecond_array_from_vec() {
        // Test building a primitive array with null values
        // we use Int32 and Int64 as a backing array, so all Int32 and Int64 conventions
        // work

        // 1e6:        00:00:00.001
        // 37800005e6: 10:30:00.005
        // 86399210e6: 23:59:59.210
        let arr: PrimitiveArray<Time64NanosecondType> =
            vec![1_000_000, 37_800_005_000_000, 86_399_210_000_000].into();
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        let formatted = vec!["00:00:00.001", "10:30:00.005", "23:59:59.210"];
        for i in 0..3 {
            // check that we can't create dates or datetimes from time instances
            assert_eq!(None, arr.value_as_datetime(i));
            assert_eq!(None, arr.value_as_date(i));
            let time = arr.value_as_time(i).unwrap();
            assert_eq!(formatted[i], time.format("%H:%M:%S%.3f").to_string());
        }
    }

    #[test]
    fn test_interval_array_from_vec() {
        // intervals are currently not treated specially, but are Int32 and Int64 arrays
        let arr = IntervalYearMonthArray::from(vec![Some(1), None, Some(-5)]);
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert!(arr.is_null(1));
        assert_eq!(-5, arr.value(2));

        // a day_time interval contains days and milliseconds, but we do not yet have accessors for the values
        let arr = IntervalDayTimeArray::from(vec![Some(1), None, Some(-5)]);
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert!(arr.is_null(1));
        assert_eq!(-5, arr.value(2));
    }

    #[test]
    fn test_duration_array_from_vec() {
        let arr = DurationSecondArray::from(vec![Some(1), None, Some(-5)]);
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert!(arr.is_null(1));
        assert_eq!(-5, arr.value(2));

        let arr = DurationMillisecondArray::from(vec![Some(1), None, Some(-5)]);
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert!(arr.is_null(1));
        assert_eq!(-5, arr.value(2));

        let arr = DurationMicrosecondArray::from(vec![Some(1), None, Some(-5)]);
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert!(arr.is_null(1));
        assert_eq!(-5, arr.value(2));

        let arr = DurationNanosecondArray::from(vec![Some(1), None, Some(-5)]);
        assert_eq!(3, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert!(arr.is_null(1));
        assert_eq!(-5, arr.value(2));
    }

    #[test]
    fn test_timestamp_array_from_vec() {
        let arr = TimestampSecondArray::from_vec(vec![1, -5], None);
        assert_eq!(2, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert_eq!(-5, arr.value(1));

        let arr = TimestampMillisecondArray::from_vec(vec![1, -5], None);
        assert_eq!(2, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert_eq!(-5, arr.value(1));

        let arr = TimestampMicrosecondArray::from_vec(vec![1, -5], None);
        assert_eq!(2, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert_eq!(-5, arr.value(1));

        let arr = TimestampNanosecondArray::from_vec(vec![1, -5], None);
        assert_eq!(2, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        assert_eq!(1, arr.value(0));
        assert_eq!(-5, arr.value(1));
    }

    #[test]
    fn test_primitive_array_slice() {
        let arr = Int32Array::from(vec![
            Some(0),
            None,
            Some(2),
            None,
            Some(4),
            Some(5),
            Some(6),
            None,
            None,
        ]);
        assert_eq!(9, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(4, arr.null_count());

        let arr2 = arr.slice(2, 5);
        assert_eq!(5, arr2.len());
        assert_eq!(2, arr2.offset());
        assert_eq!(1, arr2.null_count());

        for i in 0..arr2.len() {
            assert_eq!(i == 1, arr2.is_null(i));
            assert_eq!(i != 1, arr2.is_valid(i));
        }

        let arr3 = arr2.slice(2, 3);
        assert_eq!(3, arr3.len());
        assert_eq!(4, arr3.offset());
        assert_eq!(0, arr3.null_count());

        let int_arr = arr3.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(4, int_arr.value(0));
        assert_eq!(5, int_arr.value(1));
        assert_eq!(6, int_arr.value(2));
    }

    #[test]
    fn test_value_slice_no_bounds_check() {
        let arr = Int32Array::from(vec![2, 3, 4]);
        let _slice = arr.value_slice(0, 4);
    }

    #[test]
    fn test_int32_fmt_debug() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4].to_byte_slice());
        let arr = Int32Array::new(5, buf, 0, 0);
        assert_eq!(
            "PrimitiveArray<Int32>\n[\n  0,\n  1,\n  2,\n  3,\n  4,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_int32_with_null_fmt_debug() {
        let mut builder = Int32Array::builder(3);
        builder.append_slice(&[0, 1]).unwrap();
        builder.append_null().unwrap();
        builder.append_slice(&[3, 4]).unwrap();
        let arr = builder.finish();
        assert_eq!(
            "PrimitiveArray<Int32>\n[\n  0,\n  1,\n  null,\n  3,\n  4,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_boolean_fmt_debug() {
        let buf = Buffer::from(&[true, false, false].to_byte_slice());
        let arr = BooleanArray::new(3, buf, 0, 0);
        assert_eq!(
            "PrimitiveArray<Boolean>\n[\n  true,\n  false,\n  false,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_boolean_with_null_fmt_debug() {
        let mut builder = BooleanArray::builder(3);
        builder.append_value(true).unwrap();
        builder.append_null().unwrap();
        builder.append_value(false).unwrap();
        let arr = builder.finish();
        assert_eq!(
            "PrimitiveArray<Boolean>\n[\n  true,\n  null,\n  false,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_timestamp_fmt_debug() {
        let arr: PrimitiveArray<TimestampMillisecondType> =
            TimestampMillisecondArray::from_vec(vec![1546214400000, 1546214400000], None);
        assert_eq!(
            "PrimitiveArray<Timestamp(Millisecond, None)>\n[\n  2018-12-31T00:00:00,\n  2018-12-31T00:00:00,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_date32_fmt_debug() {
        let arr: PrimitiveArray<Date32Type> = vec![12356, 13548].into();
        assert_eq!(
            "PrimitiveArray<Date32(Day)>\n[\n  2003-10-31,\n  2007-02-04,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_time32second_fmt_debug() {
        let arr: PrimitiveArray<Time32SecondType> = vec![7201, 60054].into();
        assert_eq!(
            "PrimitiveArray<Time32(Second)>\n[\n  02:00:01,\n  16:40:54,\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_primitive_array_builder() {
        // Test building a primitive array with ArrayData builder and offset
        let buf = Buffer::from(&[0, 1, 2, 3, 4].to_byte_slice());
        let buf2 = buf.clone();
        let data = ArrayData::builder(DataType::Int32)
            .len(5)
            .offset(2)
            .add_buffer(buf)
            .build();
        let arr = Int32Array::from(data);
        assert_eq!(buf2, arr.values());
        assert_eq!(5, arr.len());
        assert_eq!(0, arr.null_count());
        for i in 0..3 {
            assert_eq!((i + 2) as i32, arr.value(i));
        }
    }

    #[test]
    #[should_panic(expected = "PrimitiveArray data should contain a single buffer only \
                               (values buffer)")]
    fn test_primitive_array_invalid_buffer_len() {
        let data = ArrayData::builder(DataType::Int32).len(5).build();
        Int32Array::from(data);
    }

    #[test]
    fn test_boolean_array_new() {
        // 00000010 01001000
        let buf = Buffer::from([72_u8, 2_u8]);
        let buf2 = buf.clone();
        let arr = BooleanArray::new(10, buf, 0, 0);
        assert_eq!(buf2, arr.values());
        assert_eq!(10, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        for i in 0..10 {
            assert!(!arr.is_null(i));
            assert!(arr.is_valid(i));
            assert_eq!(i == 3 || i == 6 || i == 9, arr.value(i), "failed at {}", i)
        }
    }

    #[test]
    fn test_boolean_array_from_vec() {
        let buf = Buffer::from([10_u8]);
        let arr = BooleanArray::from(vec![false, true, false, true]);
        assert_eq!(buf, arr.values());
        assert_eq!(4, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        for i in 0..4 {
            assert!(!arr.is_null(i));
            assert!(arr.is_valid(i));
            assert_eq!(i == 1 || i == 3, arr.value(i), "failed at {}", i)
        }
    }

    #[test]
    fn test_boolean_array_from_vec_option() {
        let buf = Buffer::from([10_u8]);
        let arr = BooleanArray::from(vec![Some(false), Some(true), None, Some(true)]);
        assert_eq!(buf, arr.values());
        assert_eq!(4, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(1, arr.null_count());
        for i in 0..4 {
            if i == 2 {
                assert!(arr.is_null(i));
                assert!(!arr.is_valid(i));
            } else {
                assert!(!arr.is_null(i));
                assert!(arr.is_valid(i));
                assert_eq!(i == 1 || i == 3, arr.value(i), "failed at {}", i)
            }
        }
    }

    #[test]
    fn test_boolean_array_builder() {
        // Test building a boolean array with ArrayData builder and offset
        // 000011011
        let buf = Buffer::from([27_u8]);
        let buf2 = buf.clone();
        let data = ArrayData::builder(DataType::Boolean)
            .len(5)
            .offset(2)
            .add_buffer(buf)
            .build();
        let arr = BooleanArray::from(data);
        assert_eq!(buf2, arr.values());
        assert_eq!(5, arr.len());
        assert_eq!(2, arr.offset());
        assert_eq!(0, arr.null_count());
        for i in 0..3 {
            assert_eq!(i != 0, arr.value(i), "failed at {}", i);
        }
    }

    #[test]
    #[should_panic(expected = "PrimitiveArray data should contain a single buffer only \
                               (values buffer)")]
    fn test_boolean_array_invalid_buffer_len() {
        let data = ArrayData::builder(DataType::Boolean).len(5).build();
        BooleanArray::from(data);
    }

    #[test]
    fn test_list_array() {
        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(8)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7].to_byte_slice()))
            .build();

        // Construct a buffer for value offsets, for the nested array:
        //  [[0, 1, 2], [3, 4, 5], [6, 7]]
        let value_offsets = Buffer::from(&[0, 3, 6, 8].to_byte_slice());

        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();
        let list_array = ListArray::from(list_data);

        let values = list_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int32, list_array.value_type());
        assert_eq!(3, list_array.len());
        assert_eq!(0, list_array.null_count());
        assert_eq!(6, list_array.value_offset(2));
        assert_eq!(2, list_array.value_length(2));
        assert_eq!(
            0,
            list_array
                .value(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0)
        );
        for i in 0..3 {
            assert!(list_array.is_valid(i));
            assert!(!list_array.is_null(i));
        }

        // Now test with a non-zero offset
        let list_data = ArrayData::builder(list_data_type)
            .len(3)
            .offset(1)
            .add_buffer(value_offsets)
            .add_child_data(value_data.clone())
            .build();
        let list_array = ListArray::from(list_data);

        let values = list_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int32, list_array.value_type());
        assert_eq!(3, list_array.len());
        assert_eq!(0, list_array.null_count());
        assert_eq!(6, list_array.value_offset(1));
        assert_eq!(2, list_array.value_length(1));
        assert_eq!(
            3,
            list_array
                .value(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0)
        );
    }

    #[test]
    fn test_dictionary_array() {
        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int8)
            .len(8)
            .add_buffer(Buffer::from(
                &[10_i8, 11, 12, 13, 14, 15, 16, 17].to_byte_slice(),
            ))
            .build();

        // Construct a buffer for value offsets, for the nested array:
        let keys = Buffer::from(&[2_i16, 3, 4].to_byte_slice());

        // Construct a dictionary array from the above two
        let key_type = DataType::Int16;
        let value_type = DataType::Int8;
        let dict_data_type =
            DataType::Dictionary(Box::new(key_type), Box::new(value_type));
        let dict_data = ArrayData::builder(dict_data_type.clone())
            .len(3)
            .add_buffer(keys.clone())
            .add_child_data(value_data.clone())
            .build();
        let dict_array = Int16DictionaryArray::from(dict_data);

        let values = dict_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int8, dict_array.value_type());
        assert_eq!(3, dict_array.len());

        // Null count only makes sense in terms of the component arrays.
        assert_eq!(0, dict_array.null_count());
        assert_eq!(0, dict_array.values().null_count());
        assert_eq!(Some(Some(3)), dict_array.keys().nth(1));
        assert_eq!(Some(Some(4)), dict_array.keys().nth(2));

        assert_eq!(
            dict_array.keys().collect::<Vec<Option<i16>>>(),
            vec![Some(2), Some(3), Some(4)]
        );

        // Now test with a non-zero offset
        let dict_data = ArrayData::builder(dict_data_type)
            .len(2)
            .offset(1)
            .add_buffer(keys)
            .add_child_data(value_data.clone())
            .build();
        let dict_array = Int16DictionaryArray::from(dict_data);

        let values = dict_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int8, dict_array.value_type());
        assert_eq!(2, dict_array.len());
        assert_eq!(Some(Some(3)), dict_array.keys().nth(0));
        assert_eq!(Some(Some(4)), dict_array.keys().nth(1));

        assert_eq!(
            dict_array.keys().collect::<Vec<Option<i16>>>(),
            vec![Some(3), Some(4)]
        );
    }

    #[test]
    fn test_fixed_size_list_array() {
        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(9)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7, 8].to_byte_slice()))
            .build();

        // Construct a list array from the above two
        let list_data_type = DataType::FixedSizeList(Box::new(DataType::Int32), 3);
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_child_data(value_data.clone())
            .build();
        let list_array = FixedSizeListArray::from(list_data);

        let values = list_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int32, list_array.value_type());
        assert_eq!(3, list_array.len());
        assert_eq!(0, list_array.null_count());
        assert_eq!(6, list_array.value_offset(2));
        assert_eq!(3, list_array.value_length());
        assert_eq!(
            0,
            list_array
                .value(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0)
        );
        for i in 0..3 {
            assert!(list_array.is_valid(i));
            assert!(!list_array.is_null(i));
        }

        // Now test with a non-zero offset
        let list_data = ArrayData::builder(list_data_type)
            .len(3)
            .offset(1)
            .add_child_data(value_data.clone())
            .build();
        let list_array = FixedSizeListArray::from(list_data);

        let values = list_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int32, list_array.value_type());
        assert_eq!(3, list_array.len());
        assert_eq!(0, list_array.null_count());
        assert_eq!(
            3,
            list_array
                .value(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0)
        );
        assert_eq!(6, list_array.value_offset(1));
        assert_eq!(3, list_array.value_length());
    }

    #[test]
    #[should_panic(
        expected = "FixedSizeListArray child array length should be a multiple of 3"
    )]
    fn test_fixed_size_list_array_unequal_children() {
        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(8)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7].to_byte_slice()))
            .build();

        // Construct a list array from the above two
        let list_data_type = DataType::FixedSizeList(Box::new(DataType::Int32), 3);
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_child_data(value_data.clone())
            .build();
        FixedSizeListArray::from(list_data);
    }

    #[test]
    fn test_list_array_slice() {
        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(10)
            .add_buffer(Buffer::from(
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].to_byte_slice(),
            ))
            .build();

        // Construct a buffer for value offsets, for the nested array:
        //  [[0, 1], null, null, [2, 3], [4, 5], null, [6, 7, 8], null, [9]]
        let value_offsets =
            Buffer::from(&[0, 2, 2, 2, 4, 6, 6, 9, 9, 10].to_byte_slice());
        // 01011001 00000001
        let mut null_bits: [u8; 2] = [0; 2];
        bit_util::set_bit(&mut null_bits, 0);
        bit_util::set_bit(&mut null_bits, 3);
        bit_util::set_bit(&mut null_bits, 4);
        bit_util::set_bit(&mut null_bits, 6);
        bit_util::set_bit(&mut null_bits, 8);

        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(9)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .null_bit_buffer(Buffer::from(null_bits))
            .build();
        let list_array = ListArray::from(list_data);

        let values = list_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int32, list_array.value_type());
        assert_eq!(9, list_array.len());
        assert_eq!(4, list_array.null_count());
        assert_eq!(2, list_array.value_offset(3));
        assert_eq!(2, list_array.value_length(3));

        let sliced_array = list_array.slice(1, 6);
        assert_eq!(6, sliced_array.len());
        assert_eq!(1, sliced_array.offset());
        assert_eq!(3, sliced_array.null_count());

        for i in 0..sliced_array.len() {
            if bit_util::get_bit(&null_bits, sliced_array.offset() + i) {
                assert!(sliced_array.is_valid(i));
            } else {
                assert!(sliced_array.is_null(i));
            }
        }

        // Check offset and length for each non-null value.
        let sliced_list_array =
            sliced_array.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(2, sliced_list_array.value_offset(2));
        assert_eq!(2, sliced_list_array.value_length(2));
        assert_eq!(4, sliced_list_array.value_offset(3));
        assert_eq!(2, sliced_list_array.value_length(3));
        assert_eq!(6, sliced_list_array.value_offset(5));
        assert_eq!(3, sliced_list_array.value_length(5));
    }

    #[test]
    fn test_fixed_size_list_array_slice() {
        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(10)
            .add_buffer(Buffer::from(
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].to_byte_slice(),
            ))
            .build();

        // Set null buts for the nested array:
        //  [[0, 1], null, null, [6, 7], [8, 9]]
        // 01011001 00000001
        let mut null_bits: [u8; 1] = [0; 1];
        bit_util::set_bit(&mut null_bits, 0);
        bit_util::set_bit(&mut null_bits, 3);
        bit_util::set_bit(&mut null_bits, 4);

        // Construct a fixed size list array from the above two
        let list_data_type = DataType::FixedSizeList(Box::new(DataType::Int32), 2);
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(5)
            .add_child_data(value_data.clone())
            .null_bit_buffer(Buffer::from(null_bits))
            .build();
        let list_array = FixedSizeListArray::from(list_data);

        let values = list_array.values();
        assert_eq!(value_data, values.data());
        assert_eq!(DataType::Int32, list_array.value_type());
        assert_eq!(5, list_array.len());
        assert_eq!(2, list_array.null_count());
        assert_eq!(6, list_array.value_offset(3));
        assert_eq!(2, list_array.value_length());

        let sliced_array = list_array.slice(1, 4);
        assert_eq!(4, sliced_array.len());
        assert_eq!(1, sliced_array.offset());
        assert_eq!(2, sliced_array.null_count());

        for i in 0..sliced_array.len() {
            if bit_util::get_bit(&null_bits, sliced_array.offset() + i) {
                assert!(sliced_array.is_valid(i));
            } else {
                assert!(sliced_array.is_null(i));
            }
        }

        // Check offset and length for each non-null value.
        let sliced_list_array = sliced_array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(2, sliced_list_array.value_length());
        assert_eq!(6, sliced_list_array.value_offset(2));
        assert_eq!(8, sliced_list_array.value_offset(3));
    }

    #[test]
    #[should_panic(
        expected = "ListArray data should contain a single buffer only (value offsets)"
    )]
    fn test_list_array_invalid_buffer_len() {
        let value_data = ArrayData::builder(DataType::Int32)
            .len(8)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7].to_byte_slice()))
            .build();
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type)
            .len(3)
            .add_child_data(value_data)
            .build();
        ListArray::from(list_data);
    }

    #[test]
    #[should_panic(
        expected = "ListArray should contain a single child array (values array)"
    )]
    fn test_list_array_invalid_child_array_len() {
        let value_offsets = Buffer::from(&[0, 2, 5, 7].to_byte_slice());
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type)
            .len(3)
            .add_buffer(value_offsets)
            .build();
        ListArray::from(list_data);
    }

    #[test]
    #[should_panic(expected = "offsets do not start at zero")]
    fn test_list_array_invalid_value_offset_start() {
        let value_data = ArrayData::builder(DataType::Int32)
            .len(8)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7].to_byte_slice()))
            .build();

        let value_offsets = Buffer::from(&[2, 2, 5, 7].to_byte_slice());

        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();
        ListArray::from(list_data);
    }

    #[test]
    fn test_binary_array() {
        let values: [u8; 12] = [
            b'h', b'e', b'l', b'l', b'o', b'p', b'a', b'r', b'q', b'u', b'e', b't',
        ];
        let offsets: [i32; 4] = [0, 5, 5, 12];

        // Array data: ["hello", "", "parquet"]
        let array_data = ArrayData::builder(DataType::Binary)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let binary_array = BinaryArray::from(array_data);
        assert_eq!(3, binary_array.len());
        assert_eq!(0, binary_array.null_count());
        assert_eq!([b'h', b'e', b'l', b'l', b'o'], binary_array.value(0));
        assert_eq!([] as [u8; 0], binary_array.value(1));
        assert_eq!(
            [b'p', b'a', b'r', b'q', b'u', b'e', b't'],
            binary_array.value(2)
        );
        assert_eq!(5, binary_array.value_offset(2));
        assert_eq!(7, binary_array.value_length(2));
        for i in 0..3 {
            assert!(binary_array.is_valid(i));
            assert!(!binary_array.is_null(i));
        }

        // Test binary array with offset
        let array_data = ArrayData::builder(DataType::Binary)
            .len(4)
            .offset(1)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let binary_array = BinaryArray::from(array_data);
        assert_eq!(
            [b'p', b'a', b'r', b'q', b'u', b'e', b't'],
            binary_array.value(1)
        );
        assert_eq!(5, binary_array.value_offset(0));
        assert_eq!(0, binary_array.value_length(0));
        assert_eq!(5, binary_array.value_offset(1));
        assert_eq!(7, binary_array.value_length(1));
    }

    #[test]
    fn test_binary_array_from_list_array() {
        let values: [u8; 12] = [
            b'h', b'e', b'l', b'l', b'o', b'p', b'a', b'r', b'q', b'u', b'e', b't',
        ];
        let values_data = ArrayData::builder(DataType::UInt8)
            .len(12)
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let offsets: [i32; 4] = [0, 5, 5, 12];

        // Array data: ["hello", "", "parquet"]
        let array_data1 = ArrayData::builder(DataType::Binary)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let binary_array1 = BinaryArray::from(array_data1);

        let array_data2 = ArrayData::builder(DataType::Binary)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_child_data(values_data)
            .build();
        let list_array = ListArray::from(array_data2);
        let binary_array2 = BinaryArray::from(list_array);

        assert_eq!(2, binary_array2.data().buffers().len());
        assert_eq!(0, binary_array2.data().child_data().len());

        assert_eq!(binary_array1.len(), binary_array2.len());
        assert_eq!(binary_array1.null_count(), binary_array2.null_count());
        for i in 0..binary_array1.len() {
            assert_eq!(binary_array1.value(i), binary_array2.value(i));
            assert_eq!(binary_array1.value_offset(i), binary_array2.value_offset(i));
            assert_eq!(binary_array1.value_length(i), binary_array2.value_length(i));
        }
    }

    #[test]
    fn test_string_array_from_u8_slice() {
        let values: Vec<&str> = vec!["hello", "", "parquet"];

        // Array data: ["hello", "", "parquet"]
        let string_array = StringArray::from(values);

        assert_eq!(3, string_array.len());
        assert_eq!(0, string_array.null_count());
        assert_eq!("hello", string_array.value(0));
        assert_eq!("", string_array.value(1));
        assert_eq!("parquet", string_array.value(2));
        assert_eq!(5, string_array.value_offset(2));
        assert_eq!(7, string_array.value_length(2));
        for i in 0..3 {
            assert!(string_array.is_valid(i));
            assert!(!string_array.is_null(i));
        }
    }

    #[test]
    fn test_nested_string_array() {
        let string_builder = StringBuilder::new(3);
        let mut list_of_string_builder = ListBuilder::new(string_builder);

        list_of_string_builder.values().append_value("foo").unwrap();
        list_of_string_builder.values().append_value("bar").unwrap();
        list_of_string_builder.append(true).unwrap();

        list_of_string_builder
            .values()
            .append_value("foobar")
            .unwrap();
        list_of_string_builder.append(true).unwrap();
        let list_of_strings = list_of_string_builder.finish();

        assert_eq!(list_of_strings.len(), 2);

        let first_slot = list_of_strings.value(0);
        let first_list = first_slot.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(first_list.len(), 2);
        assert_eq!(first_list.value(0), "foo");
        assert_eq!(first_list.value(1), "bar");

        let second_slot = list_of_strings.value(1);
        let second_list = second_slot.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(second_list.len(), 1);
        assert_eq!(second_list.value(0), "foobar");
    }

    #[test]
    #[should_panic(
        expected = "BinaryArray can only be created from List<u8> arrays, mismatched \
                    data types."
    )]
    fn test_binary_array_from_incorrect_list_array_type() {
        let values: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let values_data = ArrayData::builder(DataType::UInt32)
            .len(12)
            .add_buffer(Buffer::from(values[..].to_byte_slice()))
            .build();
        let offsets: [i32; 4] = [0, 5, 5, 12];

        let array_data = ArrayData::builder(DataType::Utf8)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_child_data(values_data)
            .build();
        let list_array = ListArray::from(array_data);
        BinaryArray::from(list_array);
    }

    #[test]
    #[should_panic(
        expected = "BinaryArray can only be created from list array of u8 values \
                    (i.e. List<PrimitiveArray<u8>>)."
    )]
    fn test_binary_array_from_incorrect_list_array() {
        let values: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let values_data = ArrayData::builder(DataType::UInt32)
            .len(12)
            .add_buffer(Buffer::from(values[..].to_byte_slice()))
            .add_child_data(ArrayData::builder(DataType::Boolean).build())
            .build();
        let offsets: [i32; 4] = [0, 5, 5, 12];

        let array_data = ArrayData::builder(DataType::Utf8)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_child_data(values_data)
            .build();
        let list_array = ListArray::from(array_data);
        BinaryArray::from(list_array);
    }

    #[test]
    fn test_fixed_size_binary_array() {
        let values: [u8; 15] = *b"hellotherearrow";

        let array_data = ArrayData::builder(DataType::FixedSizeBinary(5))
            .len(3)
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let fixed_size_binary_array = FixedSizeBinaryArray::from(array_data);
        assert_eq!(3, fixed_size_binary_array.len());
        assert_eq!(0, fixed_size_binary_array.null_count());
        assert_eq!(
            [b'h', b'e', b'l', b'l', b'o'],
            fixed_size_binary_array.value(0)
        );
        assert_eq!(
            [b't', b'h', b'e', b'r', b'e'],
            fixed_size_binary_array.value(1)
        );
        assert_eq!(
            [b'a', b'r', b'r', b'o', b'w'],
            fixed_size_binary_array.value(2)
        );
        assert_eq!(5, fixed_size_binary_array.value_length());
        assert_eq!(10, fixed_size_binary_array.value_offset(2));
        for i in 0..3 {
            assert!(fixed_size_binary_array.is_valid(i));
            assert!(!fixed_size_binary_array.is_null(i));
        }

        // Test binary array with offset
        let array_data = ArrayData::builder(DataType::FixedSizeBinary(5))
            .len(2)
            .offset(1)
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let fixed_size_binary_array = FixedSizeBinaryArray::from(array_data);
        assert_eq!(
            [b't', b'h', b'e', b'r', b'e'],
            fixed_size_binary_array.value(0)
        );
        assert_eq!(
            [b'a', b'r', b'r', b'o', b'w'],
            fixed_size_binary_array.value(1)
        );
        assert_eq!(2, fixed_size_binary_array.len());
        assert_eq!(5, fixed_size_binary_array.value_offset(0));
        assert_eq!(5, fixed_size_binary_array.value_length());
        assert_eq!(10, fixed_size_binary_array.value_offset(1));
    }

    #[test]
    #[should_panic(
        expected = "FixedSizeBinaryArray can only be created from list array of u8 values \
                    (i.e. FixedSizeList<PrimitiveArray<u8>>)."
    )]
    fn test_fixed_size_binary_array_from_incorrect_list_array() {
        let values: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let values_data = ArrayData::builder(DataType::UInt32)
            .len(12)
            .add_buffer(Buffer::from(values[..].to_byte_slice()))
            .add_child_data(ArrayData::builder(DataType::Boolean).build())
            .build();

        let array_data =
            ArrayData::builder(DataType::FixedSizeList(Box::new(DataType::Binary), 4))
                .len(3)
                .add_child_data(values_data)
                .build();
        let list_array = FixedSizeListArray::from(array_data);
        FixedSizeBinaryArray::from(list_array);
    }

    #[test]
    #[should_panic(expected = "BinaryArray out of bounds access")]
    fn test_binary_array_get_value_index_out_of_bound() {
        let values: [u8; 12] =
            [104, 101, 108, 108, 111, 112, 97, 114, 113, 117, 101, 116];
        let offsets: [i32; 4] = [0, 5, 5, 12];
        let array_data = ArrayData::builder(DataType::Binary)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let binary_array = BinaryArray::from(array_data);
        binary_array.value(4);
    }

    #[test]
    #[should_panic(expected = "StringArray out of bounds access")]
    fn test_string_array_get_value_index_out_of_bound() {
        let values: [u8; 12] = [
            b'h', b'e', b'l', b'l', b'o', b'p', b'a', b'r', b'q', b'u', b'e', b't',
        ];
        let offsets: [i32; 4] = [0, 5, 5, 12];
        let array_data = ArrayData::builder(DataType::Utf8)
            .len(3)
            .add_buffer(Buffer::from(offsets.to_byte_slice()))
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let string_array = StringArray::from(array_data);
        string_array.value(4);
    }

    #[test]
    fn test_binary_array_fmt_debug() {
        let values: [u8; 15] = *b"hellotherearrow";

        let array_data = ArrayData::builder(DataType::FixedSizeBinary(5))
            .len(3)
            .add_buffer(Buffer::from(&values[..]))
            .build();
        let arr = FixedSizeBinaryArray::from(array_data);
        assert_eq!(
            "FixedSizeBinaryArray<5>\n[\n  [104, 101, 108, 108, 111],\n  [116, 104, 101, 114, 101],\n  [97, 114, 114, 111, 119],\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_string_array_fmt_debug() {
        let arr: StringArray = vec!["hello", "some"].into();
        assert_eq!(
            "StringArray\n[\n  \"hello\",\n  \"some\",\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_fixed_size_binary_array_fmt_debug() {
        let arr: StringArray = vec!["hello", "some"].into();
        assert_eq!(
            "StringArray\n[\n  \"hello\",\n  \"some\",\n]",
            format!("{:?}", arr)
        );
    }

    #[test]
    fn test_struct_array_builder() {
        let boolean_data = ArrayData::builder(DataType::Boolean)
            .len(4)
            .add_buffer(Buffer::from([false, false, true, true].to_byte_slice()))
            .build();
        let int_data = ArrayData::builder(DataType::Int64)
            .len(4)
            .add_buffer(Buffer::from([42, 28, 19, 31].to_byte_slice()))
            .build();
        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, false));
        field_types.push(Field::new("b", DataType::Int64, false));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(4)
            .add_child_data(boolean_data.clone())
            .add_child_data(int_data.clone())
            .build();
        let struct_array = StructArray::from(struct_array_data);

        assert_eq!(boolean_data, struct_array.column(0).data());
        assert_eq!(int_data, struct_array.column(1).data());
    }

    #[test]
    fn test_struct_array_from() {
        let boolean_data = ArrayData::builder(DataType::Boolean)
            .len(4)
            .add_buffer(Buffer::from([12_u8]))
            .build();
        let int_data = ArrayData::builder(DataType::Int32)
            .len(4)
            .add_buffer(Buffer::from([42, 28, 19, 31].to_byte_slice()))
            .build();
        let struct_array = StructArray::from(vec![
            (
                Field::new("b", DataType::Boolean, false),
                Arc::new(BooleanArray::from(vec![false, false, true, true]))
                    as Arc<Array>,
            ),
            (
                Field::new("c", DataType::Int32, false),
                Arc::new(Int32Array::from(vec![42, 28, 19, 31])),
            ),
        ]);
        assert_eq!(boolean_data, struct_array.column(0).data());
        assert_eq!(int_data, struct_array.column(1).data());
        assert_eq!(4, struct_array.len());
        assert_eq!(0, struct_array.null_count());
        assert_eq!(0, struct_array.offset());
    }

    #[test]
    #[should_panic(
        expected = "the field data types must match the array data in a StructArray"
    )]
    fn test_struct_array_from_mismatched_types() {
        StructArray::from(vec![
            (
                Field::new("b", DataType::Int16, false),
                Arc::new(BooleanArray::from(vec![false, false, true, true]))
                    as Arc<Array>,
            ),
            (
                Field::new("c", DataType::Utf8, false),
                Arc::new(Int32Array::from(vec![42, 28, 19, 31])),
            ),
        ]);
    }

    #[test]
    fn test_struct_array_slice() {
        let boolean_data = ArrayData::builder(DataType::Boolean)
            .len(5)
            .add_buffer(Buffer::from([0b00010000]))
            .null_bit_buffer(Buffer::from([0b00010001]))
            .build();
        let int_data = ArrayData::builder(DataType::Int32)
            .len(5)
            .add_buffer(Buffer::from([0, 28, 42, 0, 0].to_byte_slice()))
            .null_bit_buffer(Buffer::from([0b00000110]))
            .build();

        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, false));
        field_types.push(Field::new("b", DataType::Int32, false));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(5)
            .add_child_data(boolean_data.clone())
            .add_child_data(int_data.clone())
            .null_bit_buffer(Buffer::from([0b00010111]))
            .build();
        let struct_array = StructArray::from(struct_array_data);

        assert_eq!(5, struct_array.len());
        assert_eq!(1, struct_array.null_count());
        assert!(struct_array.is_valid(0));
        assert!(struct_array.is_valid(1));
        assert!(struct_array.is_valid(2));
        assert!(struct_array.is_null(3));
        assert!(struct_array.is_valid(4));
        assert_eq!(boolean_data, struct_array.column(0).data());
        assert_eq!(int_data, struct_array.column(1).data());

        let c0 = struct_array.column(0);
        let c0 = c0.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(5, c0.len());
        assert_eq!(3, c0.null_count());
        assert!(c0.is_valid(0));
        assert_eq!(false, c0.value(0));
        assert!(c0.is_null(1));
        assert!(c0.is_null(2));
        assert!(c0.is_null(3));
        assert!(c0.is_valid(4));
        assert_eq!(true, c0.value(4));

        let c1 = struct_array.column(1);
        let c1 = c1.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(5, c1.len());
        assert_eq!(3, c1.null_count());
        assert!(c1.is_null(0));
        assert!(c1.is_valid(1));
        assert_eq!(28, c1.value(1));
        assert!(c1.is_valid(2));
        assert_eq!(42, c1.value(2));
        assert!(c1.is_null(3));
        assert!(c1.is_null(4));

        let sliced_array = struct_array.slice(2, 3);
        let sliced_array = sliced_array.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(3, sliced_array.len());
        assert_eq!(2, sliced_array.offset());
        assert_eq!(1, sliced_array.null_count());
        assert!(sliced_array.is_valid(0));
        assert!(sliced_array.is_null(1));
        assert!(sliced_array.is_valid(2));

        let sliced_c0 = sliced_array.column(0);
        let sliced_c0 = sliced_c0.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(3, sliced_c0.len());
        assert_eq!(2, sliced_c0.offset());
        assert!(sliced_c0.is_null(0));
        assert!(sliced_c0.is_null(1));
        assert!(sliced_c0.is_valid(2));
        assert_eq!(true, sliced_c0.value(2));

        let sliced_c1 = sliced_array.column(1);
        let sliced_c1 = sliced_c1.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(3, sliced_c1.len());
        assert_eq!(2, sliced_c1.offset());
        assert!(sliced_c1.is_valid(0));
        assert_eq!(42, sliced_c1.value(0));
        assert!(sliced_c1.is_null(1));
        assert!(sliced_c1.is_null(2));
    }

    #[test]
    #[should_panic(
        expected = "all child arrays of a StructArray must have the same length"
    )]
    fn test_invalid_struct_child_array_lengths() {
        StructArray::from(vec![
            (
                Field::new("b", DataType::Float32, false),
                Arc::new(Float32Array::from(vec![1.1])) as Arc<Array>,
            ),
            (
                Field::new("c", DataType::Float64, false),
                Arc::new(Float64Array::from(vec![2.2, 3.3])),
            ),
        ]);
    }

    #[test]
    #[should_panic(expected = "memory is not aligned")]
    fn test_primitive_array_alignment() {
        let ptr = memory::allocate_aligned(8);
        let buf = unsafe { Buffer::from_raw_parts(ptr, 8, 8) };
        let buf2 = buf.slice(1);
        let array_data = ArrayData::builder(DataType::Int32).add_buffer(buf2).build();
        Int32Array::from(array_data);
    }

    #[test]
    #[should_panic(expected = "memory is not aligned")]
    fn test_list_array_alignment() {
        let ptr = memory::allocate_aligned(8);
        let buf = unsafe { Buffer::from_raw_parts(ptr, 8, 8) };
        let buf2 = buf.slice(1);

        let values: [i32; 8] = [0; 8];
        let value_data = ArrayData::builder(DataType::Int32)
            .add_buffer(Buffer::from(values.to_byte_slice()))
            .build();

        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .add_buffer(buf2)
            .add_child_data(value_data.clone())
            .build();
        ListArray::from(list_data);
    }

    #[test]
    #[should_panic(expected = "memory is not aligned")]
    fn test_binary_array_alignment() {
        let ptr = memory::allocate_aligned(8);
        let buf = unsafe { Buffer::from_raw_parts(ptr, 8, 8) };
        let buf2 = buf.slice(1);

        let values: [u8; 12] = [0; 12];

        let array_data = ArrayData::builder(DataType::Binary)
            .add_buffer(buf2)
            .add_buffer(Buffer::from(&values[..]))
            .build();
        BinaryArray::from(array_data);
    }

    #[test]
    fn test_access_array_concurrently() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let ret = thread::spawn(move || a.value(3)).join();

        assert!(ret.is_ok());
        assert_eq!(8, ret.ok().unwrap());
    }

    #[test]
    fn test_dictionary_array_fmt_debug() {
        let key_builder = PrimitiveBuilder::<UInt8Type>::new(3);
        let value_builder = PrimitiveBuilder::<UInt32Type>::new(2);
        let mut builder = PrimitiveDictionaryBuilder::new(key_builder, value_builder);
        builder.append(12345678).unwrap();
        builder.append_null().unwrap();
        builder.append(22345678).unwrap();
        let array = builder.finish();
        assert_eq!(
            "DictionaryArray {keys: [Some(0), None, Some(1)] values: PrimitiveArray<UInt32>\n[\n  12345678,\n  22345678,\n]}\n",
            format!("{:?}", array)
        );

        let key_builder = PrimitiveBuilder::<UInt8Type>::new(20);
        let value_builder = PrimitiveBuilder::<UInt32Type>::new(2);
        let mut builder = PrimitiveDictionaryBuilder::new(key_builder, value_builder);
        for _ in 0..20 {
            builder.append(1).unwrap();
        }
        let array = builder.finish();
        assert_eq!(
            "DictionaryArray {keys: [Some(0), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)]... values: PrimitiveArray<UInt32>\n[\n  1,\n]}\n",
            format!("{:?}", array)
        );
    }

    #[test]
    fn test_dictionary_array_from_iter() {
        let test = vec!["a", "a", "b", "c"];
        let array: DictionaryArray<Int8Type> = test
            .iter()
            .map(|&x| if x == "b" { None } else { Some(x) })
            .collect();
        assert_eq!(
            "DictionaryArray {keys: [Some(0), Some(0), None, Some(1)] values: StringArray\n[\n  \"a\",\n  \"c\",\n]}\n",
            format!("{:?}", array)
        );

        let array: DictionaryArray<Int8Type> = test.into_iter().collect();
        assert_eq!(
            "DictionaryArray {keys: [Some(0), Some(0), Some(1), Some(2)] values: StringArray\n[\n  \"a\",\n  \"b\",\n  \"c\",\n]}\n",
            format!("{:?}", array)
        );
    }

    #[test]
    fn test_dictionary_array_reverse_lookup_key() {
        let test = vec!["a", "a", "b", "c"];
        let array: DictionaryArray<Int8Type> = test.into_iter().collect();

        assert_eq!(array.lookup_key("c"), Some(2));

        // Direction of building a dictionary is the iterator direction
        let test = vec!["t3", "t3", "t2", "t2", "t1", "t3", "t4", "t1", "t0"];
        let array: DictionaryArray<Int8Type> = test.into_iter().collect();

        assert_eq!(array.lookup_key("t1"), Some(2));
        assert_eq!(array.lookup_key("non-existent"), None);
    }
}
