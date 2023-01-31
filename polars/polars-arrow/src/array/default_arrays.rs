use arrow::array::{BinaryArray, BooleanArray, PrimitiveArray, Utf8Array};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;

pub trait FromData<T> {
    fn from_data_default(values: T, validity: Option<Bitmap>) -> Self;
}

impl FromData<Bitmap> for BooleanArray {
    fn from_data_default(values: Bitmap, validity: Option<Bitmap>) -> BooleanArray {
        BooleanArray::new(DataType::Boolean, values, validity)
    }
}

impl<T: NativeType> FromData<Buffer<T>> for PrimitiveArray<T> {
    fn from_data_default(values: Buffer<T>, validity: Option<Bitmap>) -> Self {
        let dt = T::PRIMITIVE;
        PrimitiveArray::new(dt.into(), values, validity)
    }
}

pub trait FromDataUtf8 {
    /// # Safety
    /// `values` buffer must contain valid utf8 between every `offset`
    unsafe fn from_data_unchecked_default(
        offsets: Buffer<i64>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self;
}

impl FromDataUtf8 for Utf8Array<i64> {
    unsafe fn from_data_unchecked_default(
        offsets: Buffer<i64>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self {
        let offsets = OffsetsBuffer::new_unchecked(offsets);
        Utf8Array::new_unchecked(DataType::LargeUtf8, offsets, values, validity)
    }
}

pub trait FromDataBinary {
    /// # Safety
    /// `values` buffer must contain valid utf8 between every `offset`
    unsafe fn from_data_unchecked_default(
        offsets: Buffer<i64>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self;
}

impl FromDataBinary for BinaryArray<i64> {
    unsafe fn from_data_unchecked_default(
        offsets: Buffer<i64>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self {
        let offsets = OffsetsBuffer::new_unchecked(offsets);
        BinaryArray::new(DataType::LargeBinary, offsets, values, validity)
    }
}
