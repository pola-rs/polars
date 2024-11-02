use arrow::datatypes::ArrowDataType;

use super::*;
use crate::fixed::{decode_bool, decode_primitive};
use crate::variable::{decode_binary, decode_binview};

/// Decode `rows` into a arrow format
/// # Safety
/// This will not do any bound checks. Caller must ensure the `rows` are valid
/// encodings.
pub unsafe fn decode_rows_from_binary<'a>(
    arr: &'a BinaryArray<i64>,
    fields: &[EncodingField],
    dtypes: &[ArrowDataType],
    rows: &mut Vec<&'a [u8]>,
) -> Vec<ArrayRef> {
    assert_eq!(arr.null_count(), 0);
    rows.clear();
    rows.extend(arr.values_iter());
    decode_rows(rows, fields, dtypes)
}

/// Decode `rows` into a arrow format
/// # Safety
/// This will not do any bound checks. Caller must ensure the `rows` are valid
/// encodings.
pub unsafe fn decode_rows(
    // the rows will be updated while the data is decoded
    rows: &mut [&[u8]],
    fields: &[EncodingField],
    dtypes: &[ArrowDataType],
) -> Vec<ArrayRef> {
    assert_eq!(fields.len(), dtypes.len());
    dtypes
        .iter()
        .zip(fields)
        .map(|(dtype, field)| decode(rows, field, dtype))
        .collect()
}

unsafe fn decode(rows: &mut [&[u8]], field: &EncodingField, dtype: &ArrowDataType) -> ArrayRef {
    match dtype {
        ArrowDataType::Null => NullArray::new(ArrowDataType::Null, rows.len()).to_boxed(),
        ArrowDataType::Boolean => decode_bool(rows, field).to_boxed(),
        ArrowDataType::BinaryView | ArrowDataType::LargeBinary => {
            decode_binview(rows, field).to_boxed()
        },
        ArrowDataType::Utf8View => {
            let arr = decode_binview(rows, field);
            arr.to_utf8view_unchecked().boxed()
        },
        ArrowDataType::LargeUtf8 => {
            let arr = decode_binary(rows, field);
            Utf8Array::<i64>::new_unchecked(
                ArrowDataType::LargeUtf8,
                arr.offsets().clone(),
                arr.values().clone(),
                arr.validity().cloned(),
            )
            .to_boxed()
        },
        ArrowDataType::Struct(fields) => {
            let values = fields
                .iter()
                .map(|struct_fld| decode(rows, field, struct_fld.dtype()))
                .collect();
            StructArray::new(dtype.clone(), rows.len(), values, None).to_boxed()
        },
        ArrowDataType::List { .. } | ArrowDataType::LargeList { .. } => {
            todo!("list decoding is not yet supported in polars' row encoding")
        },
        dt => {
            with_match_arrow_primitive_type!(dt, |$T| {
                decode_primitive::<$T>(rows, field).to_boxed()
            })
        },
    }
}
