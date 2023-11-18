use polars_error::PolarsResult;

use crate::array::Array;
use crate::datatypes::ArrowDataType;

pub fn cast(array: &dyn Array, to_type: &ArrowDataType) -> PolarsResult<Box<dyn Array>> {
    match to_type {
        #[cfg(feature = "dtype-decimal")]
        ArrowDataType::Decimal(precision, scale)
            if matches!(array.data_type(), ArrowDataType::LargeUtf8) =>
        {
            let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Ok(Box::new(cast_utf8_to_decimal(
                array,
                Some(*precision),
                *scale,
            )))
        },
        _ => crate::compute::cast::cast(array, to_type, Default::default()),
    }
}

#[cfg(feature = "dtype-decimal")]
use super::decimal::*;
#[cfg(feature = "dtype-decimal")]
use crate::array::{PrimitiveArray, Utf8Array};
#[cfg(feature = "dtype-decimal")]
use crate::legacy::prelude::LargeStringArray;
#[cfg(feature = "dtype-decimal")]
pub fn cast_utf8_to_decimal(
    array: &Utf8Array<i64>,
    precision: Option<usize>,
    scale: usize,
) -> PrimitiveArray<i128> {
    let precision = precision.map(|p| p as u8);
    array
        .iter()
        .map(|val| val.and_then(|val| deserialize_decimal(val.as_bytes(), precision, scale as u8)))
        .collect()
}
