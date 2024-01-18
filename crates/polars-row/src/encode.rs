use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, PrimitiveArray,
    StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_utils::vec::PushUnchecked;

use crate::fixed::FixedLengthEncoding;
use crate::row::{RowsEncoded, SortField};
use crate::{with_match_arrow_primitive_type, ArrayRef};

pub fn convert_columns(columns: &[ArrayRef], fields: &[SortField]) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized(columns, fields, &mut rows);
    rows
}

pub fn convert_columns_no_order(columns: &[ArrayRef]) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized_no_order(columns, &mut rows);
    rows
}

pub fn convert_columns_amortized_no_order(columns: &[ArrayRef], rows: &mut RowsEncoded) {
    convert_columns_amortized(
        columns,
        std::iter::repeat(&SortField::default()).take(columns.len()),
        rows,
    );
}

pub fn convert_columns_amortized<'a, I: IntoIterator<Item = &'a SortField>>(
    columns: &'a [ArrayRef],
    fields: I,
    rows: &mut RowsEncoded,
) {
    let fields = fields.into_iter();
    assert_eq!(fields.size_hint().0, columns.len());
    if columns.iter().any(|arr| {
        matches!(
            arr.data_type(),
            ArrowDataType::Struct(_) | ArrowDataType::Utf8View
        )
    }) {
        let mut flattened_columns = Vec::with_capacity(columns.len() * 5);
        let mut flattened_fields = Vec::with_capacity(columns.len() * 5);

        for (arr, field) in columns.iter().zip(fields) {
            match arr.data_type() {
                ArrowDataType::Struct(_) => {
                    let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                    for arr in arr.values() {
                        flattened_columns.push(arr.clone() as ArrayRef);
                        flattened_fields.push(field.clone())
                    }
                },
                ArrowDataType::Utf8View => {
                    let arr = arr.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                    flattened_columns.push(arr.to_binview().boxed());
                    flattened_fields.push(field.clone());
                },
                _ => {
                    flattened_columns.push(arr.clone());
                    flattened_fields.push(field.clone());
                },
            }
        }
        let values_size =
            allocate_rows_buf(&flattened_columns, &mut rows.values, &mut rows.offsets);
        for (arr, field) in flattened_columns.iter().zip(flattened_fields.iter()) {
            // Safety:
            // we allocated rows with enough bytes.
            unsafe { encode_array(&**arr, field, rows) }
        }
        // safety: values are initialized
        unsafe { rows.values.set_len(values_size) }
    } else {
        let values_size = allocate_rows_buf(columns, &mut rows.values, &mut rows.offsets);
        for (arr, field) in columns.iter().zip(fields) {
            // Safety:
            // we allocated rows with enough bytes.
            unsafe { encode_array(&**arr, field, rows) }
        }
        // safety: values are initialized
        unsafe { rows.values.set_len(values_size) }
    }
}

fn encode_primitive<T: NativeType + FixedLengthEncoding>(
    arr: &PrimitiveArray<T>,
    field: &SortField,
    out: &mut RowsEncoded,
) {
    if arr.null_count() == 0 {
        unsafe { crate::fixed::encode_slice(arr.values().as_slice(), out, field) };
    } else {
        unsafe {
            crate::fixed::encode_iter(arr.into_iter().map(|v| v.copied()), out, field);
        }
    }
}

/// Ecnodes an array into `out`
///
/// # Safety
/// `out` must have enough bytes allocated otherwise it will be out of bounds.
unsafe fn encode_array(array: &dyn Array, field: &SortField, out: &mut RowsEncoded) {
    match array.data_type() {
        ArrowDataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            crate::fixed::encode_iter(array.into_iter(), out, field);
        },
        ArrowDataType::LargeBinary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            crate::variable::encode_iter(array.into_iter(), out, field)
        },
        ArrowDataType::BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            crate::variable::encode_iter(array.into_iter(), out, field)
        },
        ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            panic!("should be cast to binary")
        },
        ArrowDataType::Dictionary(_, _, _) => {
            let array = array
                .as_any()
                .downcast_ref::<DictionaryArray<u32>>()
                .unwrap();
            let iter = array
                .iter_typed::<Utf8Array<i64>>()
                .unwrap()
                .map(|opt_s| opt_s.map(|s| s.as_bytes()));
            crate::variable::encode_iter(iter, out, field)
        },
        dt => {
            with_match_arrow_primitive_type!(dt, |$T| {
                let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                encode_primitive(array, field, out);
            })
        },
    };
}

pub fn encoded_size(data_type: &ArrowDataType) -> usize {
    use ArrowDataType::*;
    match data_type {
        UInt8 => u8::ENCODED_LEN,
        UInt16 => u16::ENCODED_LEN,
        UInt32 => u32::ENCODED_LEN,
        UInt64 => u64::ENCODED_LEN,
        Int8 => i8::ENCODED_LEN,
        Int16 => i16::ENCODED_LEN,
        Int32 => i32::ENCODED_LEN,
        Int64 => i64::ENCODED_LEN,
        Float32 => f32::ENCODED_LEN,
        Float64 => f64::ENCODED_LEN,
        Boolean => bool::ENCODED_LEN,
        dt => unimplemented!("{dt:?}"),
    }
}

// Returns the length that the caller must set on the `values` buf  once the bytes
// are initialized.
pub fn allocate_rows_buf(
    columns: &[ArrayRef],
    values: &mut Vec<u8>,
    offsets: &mut Vec<usize>,
) -> usize {
    let has_variable = columns.iter().any(|arr| {
        matches!(
            arr.data_type(),
            ArrowDataType::BinaryView | ArrowDataType::Dictionary(_, _, _)
        )
    });

    let num_rows = columns[0].len();
    if has_variable {
        // row size of the fixed-length columns
        // those can be determined without looping over the arrays
        let row_size_fixed: usize = columns
            .iter()
            .map(|arr| {
                if matches!(
                    arr.data_type(),
                    ArrowDataType::BinaryView | ArrowDataType::Dictionary(_, _, _)
                ) {
                    0
                } else {
                    encoded_size(arr.data_type())
                }
            })
            .sum();

        offsets.clear();
        offsets.reserve(num_rows + 1);

        // first write lengths to this buffer
        let lengths = offsets;

        // for the variable length columns we must iterate to determine the length per row location
        let mut processed_count = 0;
        for array in columns.iter() {
            match array.data_type() {
                ArrowDataType::BinaryView => {
                    let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                    if processed_count == 0 {
                        for opt_val in array.into_iter() {
                            unsafe {
                                lengths.push_unchecked(
                                    row_size_fixed + crate::variable::encoded_len(opt_val),
                                );
                            }
                        }
                    } else {
                        for (opt_val, row_length) in array.into_iter().zip(lengths.iter_mut()) {
                            *row_length += crate::variable::encoded_len(opt_val)
                        }
                    }
                    processed_count += 1;
                },
                ArrowDataType::Dictionary(_, _, _) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<DictionaryArray<u32>>()
                        .unwrap();
                    let iter = array
                        .iter_typed::<Utf8Array<i64>>()
                        .unwrap()
                        .map(|opt_s| opt_s.map(|s| s.as_bytes()));
                    if processed_count == 0 {
                        for opt_val in iter {
                            unsafe {
                                lengths.push_unchecked(
                                    row_size_fixed + crate::variable::encoded_len(opt_val),
                                )
                            }
                        }
                    } else {
                        for (opt_val, row_length) in iter.zip(lengths.iter_mut()) {
                            *row_length += crate::variable::encoded_len(opt_val)
                        }
                    }
                    processed_count += 1;
                },
                _ => {
                    // the rest is fixed
                },
            }
        }
        // now we use the lengths and the same buffer to determine the offsets
        let offsets = lengths;
        // we write lagged because the offsets will be written by the encoding column
        let mut current_offset = 0_usize;
        let mut lagged_offset = 0_usize;

        for length in offsets.iter_mut() {
            let to_write = lagged_offset;
            lagged_offset = current_offset;
            current_offset += *length;

            *length = to_write;
        }
        // ensure we have len + 1 offsets
        offsets.push(lagged_offset);

        // Only reserve. The init will be done later
        values.reserve(current_offset);
        current_offset
    } else {
        let row_size: usize = columns
            .iter()
            .map(|arr| encoded_size(arr.data_type()))
            .sum();
        let n_bytes = num_rows * row_size;
        values.clear();
        values.reserve(n_bytes);

        // note that offsets are shifted to the left
        // assume 2 fields with a len of 1
        // e.g. in arrow we would have 0, 2, 4, 6

        // now we write 0, 0, 2, 4

        // and when we encode field 1, we update the offset
        // so that becomes: 0, 1, 3, 5

        // and when the final field, field 2 is written
        // the offsets are correct:
        // 0, 2, 4, 6
        offsets.clear();
        offsets.reserve(num_rows + 1);
        let mut current_offset = 0;
        offsets.push(current_offset);
        for _ in 0..num_rows {
            offsets.push(current_offset);
            current_offset += row_size;
        }
        n_bytes
    }
}

#[cfg(test)]
mod test {
    use arrow::array::Int32Array;

    use super::*;
    use crate::decode::decode_rows_from_binary;
    use crate::variable::{decode_binview, BLOCK_SIZE, EMPTY_SENTINEL, NON_EMPTY_SENTINEL};

    #[test]
    fn test_fixed_and_variable_encode() {
        let a = Int32Array::from_vec(vec![1, 2, 3]);
        let b = Int32Array::from_vec(vec![213, 12, 12]);
        let c = Utf8ViewArray::from_slice([Some("a"), Some(""), Some("meep")]);

        let encoded = convert_columns_no_order(&[Box::new(a), Box::new(b), Box::new(c)]);
        assert_eq!(encoded.offsets, &[0, 44, 55, 99,]);
        assert_eq!(encoded.values.len(), 99);
        assert!(encoded.values.ends_with(&[0, 0, 0, 4]));
        assert!(encoded.values.starts_with(&[1, 128, 0, 0, 1, 1, 128]));
    }

    #[test]
    fn test_str_encode() {
        let sentence = "The black cat walked under a ladder but forget it's milk so it ...";
        let arr =
            BinaryViewArray::from_slice([Some("a"), Some(""), Some("meep"), Some(sentence), None]);

        let field = SortField {
            descending: false,
            nulls_last: false,
        };
        let arr = arrow::compute::cast::cast(&arr, &ArrowDataType::BinaryView, Default::default())
            .unwrap();
        let rows_encoded = convert_columns(&[arr], &[field]);
        let row1 = rows_encoded.get(0);

        // + 2 for the start valid byte and for the continuation token
        assert_eq!(row1.len(), BLOCK_SIZE + 2);
        let mut expected = [0u8; BLOCK_SIZE + 2];
        expected[0] = NON_EMPTY_SENTINEL;
        expected[1] = b'a';
        *expected.last_mut().unwrap() = 1;
        assert_eq!(row1, expected);

        let row2 = rows_encoded.get(1);
        let expected = &[EMPTY_SENTINEL];
        assert_eq!(row2, expected);

        let row3 = rows_encoded.get(2);
        let mut expected = [0u8; BLOCK_SIZE + 2];
        expected[0] = NON_EMPTY_SENTINEL;
        *expected.last_mut().unwrap() = 4;
        expected[1..5].copy_from_slice(b"meep");
        assert_eq!(row3, expected);

        let row4 = rows_encoded.get(3);
        let expected = [
            2, 84, 104, 101, 32, 98, 108, 97, 99, 107, 32, 99, 97, 116, 32, 119, 97, 108, 107, 101,
            100, 32, 117, 110, 100, 101, 114, 32, 97, 32, 108, 97, 100, 255, 100, 101, 114, 32, 98,
            117, 116, 32, 102, 111, 114, 103, 101, 116, 32, 105, 116, 39, 115, 32, 109, 105, 108,
            107, 32, 115, 111, 32, 105, 116, 32, 46, 255, 46, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
        ];
        assert_eq!(row4, expected);
        let row5 = rows_encoded.get(4);
        let expected = &[0u8];
        assert_eq!(row5, expected);
    }

    #[test]
    fn test_str_encode_block_size() {
        // create a key of exactly block size
        // and check the round trip
        let mut val = String::new();
        for i in 0..BLOCK_SIZE {
            val.push(char::from_u32(i as u32).unwrap())
        }

        let a = [val.as_str(), val.as_str(), val.as_str()];

        let field = SortField {
            descending: false,
            nulls_last: false,
        };
        let arr = BinaryViewArray::from_slice_values(a);
        let rows_encoded = convert_columns_no_order(&[arr.clone().boxed()]);

        let mut rows = rows_encoded.iter().collect::<Vec<_>>();
        let decoded = unsafe { decode_binview(&mut rows, &field) };
        assert_eq!(decoded, arr);
    }

    #[test]
    fn test_reverse_variable() {
        let a = Utf8ViewArray::from_slice_values(["one", "two", "three", "four", "five", "six"]);

        let fields = &[SortField {
            descending: true,
            nulls_last: false,
        }];

        let dtypes = [ArrowDataType::Utf8View];

        unsafe {
            let encoded = convert_columns(&[Box::new(a.clone())], fields);
            let out = decode_rows_from_binary(&encoded.into_array(), fields, &dtypes, &mut vec![]);

            let arr = &out[0];
            let decoded = arr.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            assert_eq!(decoded, &a);
        }
    }
}
