use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;

use self::encode::fixed_size;
use self::fixed::get_null_sentinel;
use self::variable::decode_strview;
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

unsafe fn decode_validity(rows: &mut [&[u8]], field: &EncodingField) -> Option<Bitmap> {
    // 2 loop system to avoid the overhead of allocating the bitmap if all the elements are valid.

    let null_sentinel = get_null_sentinel(field);
    let first_null = (0..rows.len()).find(|&i| {
        let v;
        (v, rows[i]) = rows[i].split_at_unchecked(1);
        v[0] == null_sentinel
    });

    // No nulls just return None
    let first_null = first_null?;

    let mut bm = MutableBitmap::new();
    bm.reserve(rows.len());
    bm.extend_constant(first_null, true);
    bm.push(false);
    bm.extend_from_trusted_len_iter(rows[first_null + 1..].iter_mut().map(|row| {
        let v;
        (v, *row) = row.split_at_unchecked(1);
        v[0] != null_sentinel
    }));
    Some(bm.freeze())
}

// We inline this in an attempt to avoid the dispatch cost.
#[inline(always)]
fn dtype_and_data_to_encoded_item_len(
    dtype: &ArrowDataType,
    data: &[u8],
    field: &EncodingField,
) -> usize {
    // Fast path: if the size is fixed, we can just divide.
    if let Some(size) = fixed_size(dtype) {
        return size;
    }

    let (non_empty_sentinel, continuation_token) = if field.descending {
        (
            !variable::NON_EMPTY_SENTINEL,
            !variable::BLOCK_CONTINUATION_TOKEN,
        )
    } else {
        (
            variable::NON_EMPTY_SENTINEL,
            variable::BLOCK_CONTINUATION_TOKEN,
        )
    };

    use ArrowDataType as D;
    match dtype {
        D::Binary | D::LargeBinary | D::List(_) | D::LargeList(_) | D::BinaryView => unsafe {
            crate::variable::encoded_item_len(data, non_empty_sentinel, continuation_token)
        },
        D::Utf8 | D::LargeUtf8 | D::Utf8View => {
            let null_sentinel = get_null_sentinel(field);
            unsafe { crate::variable::encoded_str_len(data, null_sentinel, field.descending) }
        },

        D::List(list_field) | D::LargeList(list_field) => {
            let mut data = data;
            let mut item_len = 0;

            let list_continuation_token = field.list_continuation_token();

            while data[0] == list_continuation_token {
                data = &data[1..];
                let len = dtype_and_data_to_encoded_item_len(list_field.dtype(), data, field);
                data = &data[len..];
                item_len += 1 + len;
            }
            1 + item_len
        },

        D::FixedSizeBinary(_) => todo!(),
        D::FixedSizeList(fsl_field, width) => {
            let mut data = &data[1..];
            let mut item_len = 1; // validity byte

            for _ in 0..*width {
                let len = dtype_and_data_to_encoded_item_len(fsl_field.dtype(), data, field);
                data = &data[len..];
                item_len += len;
            }
            item_len
        },
        D::Struct(struct_fields) => {
            let mut data = &data[1..];
            let mut item_len = 1; // validity byte

            for struct_field in struct_fields {
                let len = dtype_and_data_to_encoded_item_len(struct_field.dtype(), data, field);
                data = &data[len..];
                item_len += len;
            }
            item_len
        },

        D::Union(_, _, _) => todo!(),
        D::Map(_, _) => todo!(),
        D::Dictionary(_, _, _) => todo!(),
        D::Decimal(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),
        D::Extension(_, _, _) => todo!(),
        D::Unknown => todo!(),

        _ => unreachable!(),
    }
}

fn rows_for_fixed_size_list<'a>(
    dtype: &ArrowDataType,
    field: &EncodingField,
    width: usize,
    rows: &mut [&'a [u8]],
    nested_rows: &mut Vec<&'a [u8]>,
) {
    nested_rows.clear();
    nested_rows.reserve(rows.len() * width);

    // Fast path: if the size is fixed, we can just divide.
    if let Some(size) = fixed_size(dtype) {
        for row in rows.iter_mut() {
            for i in 0..width {
                nested_rows.push(&row[(i * size)..][..size]);
            }
            *row = &row[size * width..];
        }
        return;
    }

    // @TODO: This is quite slow since we need to dispatch for possibly every nested type
    for row in rows.iter_mut() {
        for _ in 0..width {
            let length = dtype_and_data_to_encoded_item_len(dtype, row, field);
            let v;
            (v, *row) = row.split_at(length);
            nested_rows.push(v);
        }
    }
}

unsafe fn decode(rows: &mut [&[u8]], field: &EncodingField, dtype: &ArrowDataType) -> ArrayRef {
    match dtype {
        ArrowDataType::Null => NullArray::new(ArrowDataType::Null, rows.len()).to_boxed(),
        ArrowDataType::Boolean => decode_bool(rows, field).to_boxed(),
        ArrowDataType::BinaryView | ArrowDataType::LargeBinary => {
            decode_binview(rows, field).to_boxed()
        },
        ArrowDataType::Utf8View => {
            let arr = decode_strview(rows, field);
            arr.boxed()
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
            let validity = decode_validity(rows, field);
            let values = fields
                .iter()
                .map(|struct_fld| decode(rows, field, struct_fld.dtype()))
                .collect();
            StructArray::new(dtype.clone(), rows.len(), values, validity).to_boxed()
        },
        ArrowDataType::FixedSizeList(fsl_field, width) => {
            let validity = decode_validity(rows, field);

            // @TODO: we could consider making this into a scratchpad
            let mut nested_rows = Vec::new();
            rows_for_fixed_size_list(fsl_field.dtype(), field, *width, rows, &mut nested_rows);
            let values = decode(&mut nested_rows, field, fsl_field.dtype());

            FixedSizeListArray::new(dtype.clone(), rows.len(), values, validity).to_boxed()
        },
        ArrowDataType::List(list_field) | ArrowDataType::LargeList(list_field) => {
            let mut validity = MutableBitmap::new();

            // @TODO: we could consider making this into a scratchpad
            let num_rows = rows.len();
            let mut nested_rows = Vec::new();
            let mut offsets = Vec::with_capacity(rows.len() + 1);
            offsets.push(0);

            let list_null_sentinel = field.list_null_sentinel();
            let list_continuation_token = field.list_continuation_token();
            let list_termination_token = field.list_termination_token();

            // @TODO: make a specialized loop for fixed size list_field.dtype()
            for (i, row) in rows.iter_mut().enumerate() {
                while row[0] == list_continuation_token {
                    *row = &row[1..];
                    let len = dtype_and_data_to_encoded_item_len(list_field.dtype(), row, field);
                    nested_rows.push(&row[..len]);
                    *row = &row[len..];
                }

                offsets.push(nested_rows.len() as i64);

                // @TODO: Might be better to make this a 2-loop system.
                if row[0] == list_null_sentinel {
                    *row = &row[1..];
                    validity.reserve(num_rows);
                    validity.extend_constant(i - validity.len(), true);
                    validity.push(false);
                    continue;
                }

                assert_eq!(row[0], list_termination_token);
                *row = &row[1..];
            }

            let validity = if validity.is_empty() {
                None
            } else {
                validity.extend_constant(num_rows - validity.len(), true);
                Some(validity.freeze())
            };
            assert_eq!(offsets.len(), rows.len() + 1);

            let values = decode(&mut nested_rows, field, list_field.dtype());

            ListArray::<i64>::new(
                dtype.clone(),
                unsafe { OffsetsBuffer::new_unchecked(Buffer::from(offsets)) },
                values,
                validity,
            )
            .to_boxed()
        },
        dt => {
            with_match_arrow_primitive_type!(dt, |$T| {
                decode_primitive::<$T>(rows, field).to_boxed()
            })
        },
    }
}
