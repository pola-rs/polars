use arrow::array::{BooleanArray, Utf8Array};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;

pub trait FromData<T> {
    fn from_data_default(values: T, validity: Option<Bitmap>) -> Self;
}

impl FromData<Bitmap> for BooleanArray {
    fn from_data_default(values: Bitmap, validity: Option<Bitmap>) -> BooleanArray {
        BooleanArray::from_data(DataType::Boolean, values, validity)
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
        Utf8Array::from_data_unchecked(DataType::LargeUtf8, offsets, values, validity)
    }
}
