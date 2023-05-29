use arrow::array::{Array, PrimitiveArray, Utf8Array};
use arrow::datatypes::DataType;
use arrow::error::Result;

use crate::prelude::{ArrayRef, LargeStringArray};

pub fn cast(array: &dyn Array, to_type: &DataType) -> Result<Box<dyn Array>> {
    match to_type {
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) if matches!(array.data_type(), DataType::LargeUtf8) => {
            let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Ok(cast_utf8_to_decimal(array, *precision, *scale))
        }
        _ => arrow::compute::cast::cast(array, to_type, Default::default()),
    }
}

#[cfg(feature = "dtype-decimal")]
use super::decimal::*;
#[cfg(feature = "dtype-decimal")]
fn cast_utf8_to_decimal(array: &Utf8Array<i64>, precision: usize, scale: usize) -> ArrayRef {
    let values: PrimitiveArray<i128> = array
        .iter()
        .map(|val| {
            val.and_then(|val| deserialize_decimal(val.as_bytes(), precision as u8, scale as u8))
        })
        .collect();
    Box::new(values.to(DataType::Decimal(precision, scale)))
}
