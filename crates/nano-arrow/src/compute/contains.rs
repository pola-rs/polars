//! Declares the [`contains`] operator

use crate::{
    array::{Array, BinaryArray, BooleanArray, ListArray, PrimitiveArray, Utf8Array},
    bitmap::Bitmap,
    datatypes::DataType,
    error::{Error, Result},
    offset::Offset,
    types::NativeType,
};

use super::utils::combine_validities;

/// Checks if a [`GenericListArray`] contains a value in the [`PrimitiveArray`]
/// The validity will be equal to the `And` of both arrays.
fn contains_primitive<T, O>(list: &ListArray<O>, values: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: NativeType,
    O: Offset,
{
    if list.len() != values.len() {
        return Err(Error::InvalidArgumentError(
            "Contains requires arrays of the same length".to_string(),
        ));
    }
    if list.values().data_type() != values.data_type() {
        return Err(Error::InvalidArgumentError(
            "Contains requires the inner array to be of the same logical type".to_string(),
        ));
    }

    let validity = combine_validities(list.validity(), values.validity());

    let values = list.iter().zip(values.iter()).map(|(list, values)| {
        if list.is_none() | values.is_none() {
            // validity takes care of this
            return false;
        };
        let list = list.unwrap();
        let list = list.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        let values = values.unwrap();
        list.iter().any(|x| x.map(|x| x == values).unwrap_or(false))
    });
    let values = Bitmap::from_trusted_len_iter(values);

    Ok(BooleanArray::new(DataType::Boolean, values, validity))
}

/// Checks if a [`GenericListArray`] contains a value in the [`Utf8Array`]
fn contains_utf8<O, OO>(list: &ListArray<O>, values: &Utf8Array<OO>) -> Result<BooleanArray>
where
    O: Offset,
    OO: Offset,
{
    if list.len() != values.len() {
        return Err(Error::InvalidArgumentError(
            "Contains requires arrays of the same length".to_string(),
        ));
    }
    if list.values().data_type() != values.data_type() {
        return Err(Error::InvalidArgumentError(
            "Contains requires the inner array to be of the same logical type".to_string(),
        ));
    }

    let validity = combine_validities(list.validity(), values.validity());

    let values = list.iter().zip(values.iter()).map(|(list, values)| {
        if list.is_none() | values.is_none() {
            // validity takes care of this
            return false;
        };
        let list = list.unwrap();
        let list = list.as_any().downcast_ref::<Utf8Array<OO>>().unwrap();
        let values = values.unwrap();
        list.iter().any(|x| x.map(|x| x == values).unwrap_or(false))
    });
    let values = Bitmap::from_trusted_len_iter(values);

    Ok(BooleanArray::new(DataType::Boolean, values, validity))
}

/// Checks if a [`ListArray`] contains a value in the [`BinaryArray`]
fn contains_binary<O, OO>(list: &ListArray<O>, values: &BinaryArray<OO>) -> Result<BooleanArray>
where
    O: Offset,
    OO: Offset,
{
    if list.len() != values.len() {
        return Err(Error::InvalidArgumentError(
            "Contains requires arrays of the same length".to_string(),
        ));
    }
    if list.values().data_type() != values.data_type() {
        return Err(Error::InvalidArgumentError(
            "Contains requires the inner array to be of the same logical type".to_string(),
        ));
    }

    let validity = combine_validities(list.validity(), values.validity());

    let values = list.iter().zip(values.iter()).map(|(list, values)| {
        if list.is_none() | values.is_none() {
            // validity takes care of this
            return false;
        };
        let list = list.unwrap();
        let list = list.as_any().downcast_ref::<BinaryArray<OO>>().unwrap();
        let values = values.unwrap();
        list.iter().any(|x| x.map(|x| x == values).unwrap_or(false))
    });
    let values = Bitmap::from_trusted_len_iter(values);

    Ok(BooleanArray::new(DataType::Boolean, values, validity))
}

macro_rules! primitive {
    ($list:expr, $values:expr, $l_ty:ty, $r_ty:ty) => {{
        let list = $list.as_any().downcast_ref::<ListArray<$l_ty>>().unwrap();
        let values = $values
            .as_any()
            .downcast_ref::<PrimitiveArray<$r_ty>>()
            .unwrap();
        contains_primitive(list, values)
    }};
}

/// Returns whether each element in `values` is in each element from `list`
pub fn contains(list: &dyn Array, values: &dyn Array) -> Result<BooleanArray> {
    let list_data_type = list.data_type();
    let values_data_type = values.data_type();

    match (list_data_type, values_data_type) {
        (DataType::List(_), DataType::Utf8) => {
            let list = list.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let values = values.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
            contains_utf8(list, values)
        }
        (DataType::List(_), DataType::LargeUtf8) => {
            let list = list.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let values = values.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            contains_utf8(list, values)
        }
        (DataType::LargeList(_), DataType::LargeUtf8) => {
            let list = list.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let values = values.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            contains_utf8(list, values)
        }
        (DataType::LargeList(_), DataType::Utf8) => {
            let list = list.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let values = values.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
            contains_utf8(list, values)
        }
        (DataType::List(_), DataType::Binary) => {
            let list = list.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let values = values.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            contains_binary(list, values)
        }
        (DataType::List(_), DataType::LargeBinary) => {
            let list = list.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let values = values.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            contains_binary(list, values)
        }
        (DataType::LargeList(_), DataType::LargeBinary) => {
            let list = list.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let values = values.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            contains_binary(list, values)
        }
        (DataType::LargeList(_), DataType::Binary) => {
            let list = list.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let values = values.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            contains_binary(list, values)
        }
        (DataType::List(_), DataType::Int8) => primitive!(list, values, i32, i8),
        (DataType::List(_), DataType::Int16) => primitive!(list, values, i32, i16),
        (DataType::List(_), DataType::Int32) => primitive!(list, values, i32, i32),
        (DataType::List(_), DataType::Int64) => primitive!(list, values, i32, i64),
        (DataType::List(_), DataType::UInt8) => primitive!(list, values, i32, u8),
        (DataType::List(_), DataType::UInt16) => primitive!(list, values, i32, u16),
        (DataType::List(_), DataType::UInt32) => primitive!(list, values, i32, u32),
        (DataType::List(_), DataType::UInt64) => primitive!(list, values, i32, u64),
        (DataType::List(_), DataType::Float32) => primitive!(list, values, i32, f32),
        (DataType::List(_), DataType::Float64) => primitive!(list, values, i32, f64),
        (DataType::LargeList(_), DataType::Int8) => primitive!(list, values, i64, i8),
        (DataType::LargeList(_), DataType::Int16) => primitive!(list, values, i64, i16),
        (DataType::LargeList(_), DataType::Int32) => primitive!(list, values, i64, i32),
        (DataType::LargeList(_), DataType::Int64) => primitive!(list, values, i64, i64),
        (DataType::LargeList(_), DataType::UInt8) => primitive!(list, values, i64, u8),
        (DataType::LargeList(_), DataType::UInt16) => primitive!(list, values, i64, u16),
        (DataType::LargeList(_), DataType::UInt32) => primitive!(list, values, i64, u32),
        (DataType::LargeList(_), DataType::UInt64) => primitive!(list, values, i64, u64),
        (DataType::LargeList(_), DataType::Float32) => primitive!(list, values, i64, f32),
        (DataType::LargeList(_), DataType::Float64) => primitive!(list, values, i64, f64),
        _ => Err(Error::NotYetImplemented(format!(
            "Contains is not supported between logical types \"{list_data_type:?}\" and \"{values_data_type:?}\""
        ))),
    }
}
