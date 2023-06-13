use arrow::datatypes::DataType;

use super::*;
use crate::fixed::{decode_bool, decode_primitive};
use crate::variable::decode_binary;

/// Decode `rows` into a arrow format
/// # Safety
/// This will not do any bound checks. Caller must ensure the `rows` are valid
/// encodings.
pub unsafe fn decode_rows(
    // the rows will be updated while the data is decoded
    rows: &mut [&[u8]],
    fields: &[SortField],
    data_types: &[DataType],
) -> Vec<ArrayRef> {
    assert_eq!(fields.len(), data_types.len());
    data_types
        .iter()
        .zip(fields)
        .map(|(data_type, field)| decode(rows, field, data_type))
        .collect()
}

unsafe fn decode(rows: &mut [&[u8]], field: &SortField, data_type: &DataType) -> ArrayRef {
    // not yet supported for fixed types
    assert!(!field.nulls_last, "not yet supported");
    assert!(!field.descending, "not yet supported");
    match data_type {
        DataType::Null => NullArray::new(DataType::Null, rows.len()).to_boxed(),
        DataType::Boolean => decode_bool(rows, field).to_boxed(),
        DataType::LargeBinary => decode_binary(rows, field).to_boxed(),
        DataType::LargeUtf8 => {
            let arr = decode_binary(rows, field);
            Utf8Array::<i64>::new_unchecked(
                DataType::LargeUtf8,
                arr.offsets().clone(),
                arr.values().clone(),
                arr.validity().cloned(),
            )
            .to_boxed()
        }
        DataType::Struct(fields) => {
            let values = fields
                .iter()
                .map(|struct_fld| decode(rows, field, struct_fld.data_type()))
                .collect();
            StructArray::new(data_type.clone(), values, None).to_boxed()
        }
        dt => {
            with_match_arrow_primitive_type!(dt, |$T| {
                decode_primitive::<$T>(rows, field).to_boxed()
            })
        }
    }
}
