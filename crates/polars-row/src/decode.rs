use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;

use self::encode::fixed_size;
use self::fixed::get_null_sentinel;
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

unsafe fn decode_validity(
    rows: &mut [&[u8]],
    field: &EncodingField,
) -> Option<Bitmap> {
    let mut i = 0;
    let mut bm = MutableBitmap::new();
    let null_sentinel = get_null_sentinel(field);
    while i < rows.len() {
        let v;
        (v, rows[i]) = rows[i].split_at_unchecked(1);
        if v[0] == null_sentinel {
            bm.reserve(rows.len());
            bm.extend_constant(i, true);
            bm.push(false);
            i += 1;
            break;
        }
        i += 1;
    }

    if i == rows.len() {
        return None;
    }

    bm.extend_from_trusted_len_iter(rows[i..].iter_mut().map(|row| {
        let v;
        (v, *row) = row.split_at_unchecked(1);
        v[0] != null_sentinel
    }));

    Some(bm.freeze())
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
        for r in 0..rows.len() {
            for i in 0..width {
                nested_rows.push(&rows[r][(i * size)..][..size]);
            }
            rows[r] = &rows[r][size * width..];
        }
        return;
    }

    use ArrowDataType as D;
    match dtype {
        D::FixedSizeBinary(_) => todo!(),
        D::BinaryView
        | D::Utf8View
        | D::Binary
        | D::LargeBinary
        | D::Utf8
        | D::LargeUtf8
        | D::List(_)
        | D::LargeList(_) => {
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

            for r in 0..rows.len() {
                for i in 0..width {
                    let length = unsafe {
                        crate::variable::encoded_item_len(
                            rows[r],
                            non_empty_sentinel,
                            continuation_token,
                        )
                    };
                    let v;
                    (v, rows[r]) = rows[r].split_at(length);
                    nested_rows.push(v);
                }
            }
        },
        D::FixedSizeList(field, _) => todo!(),
        D::Struct(vec) => todo!(),
        D::Dictionary(integer_type, arrow_data_type, _) => todo!(),
        D::Extension(pl_small_str, arrow_data_type, pl_small_str1) => todo!(),
        D::Unknown => todo!(),

        D::Union(_, _, _) => todo!(),
        D::Map(_, _) => todo!(),
        D::Decimal(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),

        _ => unreachable!(),
    }
}

fn offsets_from_dtype_and_data<'a>(
    dtype: &ArrowDataType,
    field: &EncodingField,
    data: &'a [u8],
    offsets: &mut Vec<usize>,
) {
    offsets.clear();

    // Fast path: if the size is fixed, we can just divide.
    if let Some(size) = fixed_size(dtype) {
        assert!(size == 0 || data.len() % size == 0);
        offsets.extend((0..data.len() / size).map(|i| i * size));
        return;
    }

    use ArrowDataType as D;
    match dtype {
        D::FixedSizeBinary(_) => todo!(),
        D::BinaryView
        | D::Utf8View
        | D::Binary
        | D::LargeBinary
        | D::Utf8
        | D::LargeUtf8
        | D::List(_)
        | D::LargeList(_) => {
            let mut data = data;
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
            let mut offset = 0;
            while !data.is_empty() {
                let length = unsafe {
                    crate::variable::encoded_item_len(
                        data,
                        non_empty_sentinel,
                        continuation_token,
                    )
                };
                offsets.push(offset);
                data = &data[length..];
                offset += length;
            }
        },
        D::FixedSizeList(_, _) => todo!(),
        D::Struct(_) => todo!(),
        D::Dictionary(_, _, _) => todo!(),
        D::Extension(_, _, _) => todo!(),
        D::Unknown => todo!(),

        D::Union(_, _, _) => todo!(),
        D::Map(_, _) => todo!(),
        D::Decimal(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),

        _ => unreachable!(),
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
            let validity = decode_validity(rows, field);
            let values = fields
                .iter()
                .map(|struct_fld| decode(rows, field, struct_fld.dtype()))
                .collect();
            StructArray::new(dtype.clone(), rows.len(), values, validity).to_boxed()
        },
        ArrowDataType::FixedSizeList(fsl_field, width) => {
            let validity = decode_validity(rows, field);

            // @TODO: scratchpad
            let mut nested_rows = Vec::new();
            rows_for_fixed_size_list(fsl_field.dtype(), field, *width, rows, &mut nested_rows);
            let values = decode(&mut nested_rows, field, fsl_field.dtype());

            FixedSizeListArray::new(dtype.clone(), rows.len(), values, validity).to_boxed()
        },
        ArrowDataType::List(list_field) | ArrowDataType::LargeList(list_field) => {
            let arr = decode_binary(rows, field);

            let mut offsets = Vec::with_capacity(rows.len());
            // @TODO: Make into scratchpad
            let mut nested_offsets = Vec::new();
            offsets_from_dtype_and_data(
                list_field.dtype(),
                field,
                arr.values().as_ref(),
                &mut nested_offsets,
            );
            // @TODO: This might cause realloc, fix.
            nested_offsets.push(arr.values().len());
            let mut nested_rows = nested_offsets
                .windows(2)
                .map(|vs| &arr.values()[vs[0]..vs[1]])
                .collect::<Vec<_>>();

            let mut i = 0;
            for offset in arr.offsets().iter() {
                while nested_offsets[i] != offset.as_usize() {
                    i += 1;
                }

                offsets.push(i as i64);
            }
            assert_eq!(offsets.len(), rows.len() + 1);

            let values = decode(&mut nested_rows, field, list_field.dtype());
            let (_, _, _, validity) = arr.into_inner();

            // @TODO: Handle validity
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
