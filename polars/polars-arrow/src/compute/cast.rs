use arrow::array::Array;
use arrow::datatypes::DataType;
use arrow::error::Result;

pub fn cast(array: &dyn Array, to_type: &DataType) -> Result<Box<dyn Array>> {
    match to_type {
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) if matches!(array.data_type(), DataType::LargeUtf8) => {
            let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Ok(cast_utf8_to_decimal(array, Some(*precision), *scale))
        }
        _ => arrow::compute::cast::cast(array, to_type, Default::default()),
    }
}

#[cfg(feature = "dtype-decimal")]
use arrow::array::{PrimitiveArray, Utf8Array};

#[cfg(feature = "dtype-decimal")]
use super::decimal::*;
#[cfg(feature = "dtype-decimal")]
use crate::prelude::{ArrayRef, LargeStringArray};
#[cfg(feature = "dtype-decimal")]
pub fn cast_utf8_to_decimal(
    array: &Utf8Array<i64>,
    precision: Option<usize>,
    scale: usize,
) -> ArrayRef {
    let precision = precision.map(|p| p as u8);
    let values: PrimitiveArray<i128> = array
        .iter()
        .map(|val| val.and_then(|val| deserialize_decimal(val.as_bytes(), precision, scale as u8)))
        .collect();
    Box::new(values)
}
