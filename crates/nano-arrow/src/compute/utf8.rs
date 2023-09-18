//! Defines common maps to a [`Utf8Array`]

use crate::{
    array::{Array, Utf8Array},
    datatypes::DataType,
    error::{Error, Result},
    offset::Offset,
};

/// utf8_apply will apply `Fn(&str) -> String` to every value in Utf8Array.
pub fn utf8_apply<O: Offset, F: Fn(&str) -> String>(f: F, array: &Utf8Array<O>) -> Utf8Array<O> {
    let iter = array.values_iter().map(f);

    let new = Utf8Array::<O>::from_trusted_len_values_iter(iter);
    new.with_validity(array.validity().cloned())
}

/// Returns a new `Array` where each of each of the elements is upper-cased.
/// this function errors when the passed array is not a \[Large\]String array.
pub fn upper(array: &dyn Array) -> Result<Box<dyn Array>> {
    match array.data_type() {
        DataType::LargeUtf8 => Ok(Box::new(utf8_apply(
            str::to_uppercase,
            array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("A large string is expected"),
        ))),
        DataType::Utf8 => Ok(Box::new(utf8_apply(
            str::to_uppercase,
            array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("A string is expected"),
        ))),
        _ => Err(Error::InvalidArgumentError(format!(
            "upper does not support type {:?}",
            array.data_type()
        ))),
    }
}

/// Checks if an array of type `datatype` can perform upper operation
///
/// # Examples
/// ```
/// use arrow2::compute::utf8::can_upper;
/// use arrow2::datatypes::{DataType};
///
/// let data_type = DataType::Utf8;
/// assert_eq!(can_upper(&data_type), true);
///
/// let data_type = DataType::Null;
/// assert_eq!(can_upper(&data_type), false);
/// ```
pub fn can_upper(data_type: &DataType) -> bool {
    matches!(data_type, DataType::LargeUtf8 | DataType::Utf8)
}

/// Returns a new `Array` where each of each of the elements is lower-cased.
/// this function errors when the passed array is not a \[Large\]String array.
pub fn lower(array: &dyn Array) -> Result<Box<dyn Array>> {
    match array.data_type() {
        DataType::LargeUtf8 => Ok(Box::new(utf8_apply(
            str::to_lowercase,
            array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("A large string is expected"),
        ))),
        DataType::Utf8 => Ok(Box::new(utf8_apply(
            str::to_lowercase,
            array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("A string is expected"),
        ))),
        _ => Err(Error::InvalidArgumentError(format!(
            "lower does not support type {:?}",
            array.data_type()
        ))),
    }
}

/// Checks if an array of type `datatype` can perform lower operation
///
/// # Examples
/// ```
/// use arrow2::compute::utf8::can_lower;
/// use arrow2::datatypes::{DataType};
///
/// let data_type = DataType::Utf8;
/// assert_eq!(can_lower(&data_type), true);
///
/// let data_type = DataType::Null;
/// assert_eq!(can_lower(&data_type), false);
/// ```
pub fn can_lower(data_type: &DataType) -> bool {
    matches!(data_type, DataType::LargeUtf8 | DataType::Utf8)
}
