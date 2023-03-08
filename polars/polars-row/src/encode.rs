use arrow::array::{Array, BinaryArray, BooleanArray, DictionaryArray, PrimitiveArray, Utf8Array};
use arrow::datatypes::{DataType as ArrowDataType, DataType};
use arrow::types::NativeType;

use crate::encodings::fixed::FixedLengthEncoding;
use crate::row::{RowsEncoded, SortField};
use crate::{with_match_arrow_primitive_type, ArrayRef};

pub fn convert_columns(columns: &[ArrayRef], fields: &[SortField]) -> RowsEncoded {
    assert_eq!(fields.len(), columns.len());

    let mut rows = allocate_rows_buf(columns);
    for (arr, field) in columns.iter().zip(fields.iter()) {
        // Safety:
        // we allocated rows with enough bytes.
        unsafe { encode_array(&**arr, field, &mut rows) }
    }
    rows
}

fn encode_primitive<T: NativeType + FixedLengthEncoding>(
    arr: &PrimitiveArray<T>,
    field: &SortField,
    out: &mut RowsEncoded,
) {
    if arr.null_count() == 0 {
        crate::encodings::fixed::encode_slice(arr.values().as_slice(), out, field);
    } else {
        crate::encodings::fixed::encode_iter(arr.into_iter().map(|v| v.copied()), out, field);
    }
}

/// Ecnodes an array into `out`
///
/// # Safety
/// `out` must have enough bytes allocated otherwise it will be out of bounds.
unsafe fn encode_array(array: &dyn Array, field: &SortField, out: &mut RowsEncoded) {
    match array.data_type() {
        DataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            crate::encodings::fixed::encode_iter(array.into_iter(), out, field);
        }
        DataType::LargeBinary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            crate::encodings::variable::encode_iter(array.into_iter(), out, field)
        }
        DataType::LargeUtf8 => {
            panic!("should be cast to binary")
        }
        DataType::Dictionary(_, _, _) => {
            let array = array
                .as_any()
                .downcast_ref::<DictionaryArray<u32>>()
                .unwrap();
            let iter = array
                .iter_typed::<Utf8Array<i64>>()
                .unwrap()
                .map(|opt_s| opt_s.map(|s| s.as_bytes()));
            crate::encodings::variable::encode_iter(iter, out, field)
        }
        dt => {
            with_match_arrow_primitive_type!(dt, |$T| {
                let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                encode_primitive(array, field, out);
            })
        }
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
        _ => unimplemented!(),
    }
}

pub fn allocate_rows_buf(columns: &[ArrayRef]) -> RowsEncoded {
    let has_variable = columns.iter().any(|arr| {
        matches!(
            arr.data_type(),
            ArrowDataType::LargeBinary | ArrowDataType::Dictionary(_, _, _)
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
                    ArrowDataType::LargeBinary | ArrowDataType::Dictionary(_, _, _)
                ) {
                    0
                } else {
                    encoded_size(arr.data_type())
                }
            })
            .sum();

        let mut lengths = vec![row_size_fixed; num_rows];

        // for the variable length columns we must iterate to determine the length per row location
        for array in columns.iter() {
            match array.data_type() {
                ArrowDataType::LargeBinary => {
                    let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
                    for (opt_val, row_length) in array.into_iter().zip(lengths.iter_mut()) {
                        *row_length += crate::encodings::variable::encoded_len(opt_val)
                    }
                }
                ArrowDataType::Dictionary(_, _, _) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<DictionaryArray<u32>>()
                        .unwrap();
                    let iter = array
                        .iter_typed::<Utf8Array<i64>>()
                        .unwrap()
                        .map(|opt_s| opt_s.map(|s| s.as_bytes()));
                    for (opt_val, row_length) in iter.zip(lengths.iter_mut()) {
                        *row_length += crate::encodings::variable::encoded_len(opt_val)
                    }
                }
                _ => {
                    // the rest is fixed
                }
            }
        }
        let mut offsets = Vec::with_capacity(num_rows + 1);
        let mut current_offset = 0_usize;
        offsets.push(current_offset);

        for length in lengths {
            offsets.push(current_offset);
            #[cfg(target_pointer_width = "64")]
            {
                // don't do overflow check, counting exabytes here.
                current_offset += length;
            }
            #[cfg(not(target_pointer_width = "64"))]
            {
                current_offset = current_offset.checked_add(length).expect("overflow");
            }
        }

        // todo! allocate uninit
        let buf = vec![0u8; current_offset];
        RowsEncoded::new(buf, offsets)
    } else {
        let row_size: usize = columns
            .iter()
            .map(|arr| encoded_size(arr.data_type()))
            .sum();
        let n_bytes = num_rows * row_size;
        // todo! allocate uninit
        let buf = vec![0u8; n_bytes];

        // note that offsets are shifted to the left
        // assume 2 fields with a len of 1
        // e.g. in arrow we would have 0, 2, 4, 6

        // now we write 0, 0, 2, 4

        // and when we encode field 1, we update the offset
        // so that becomes: 0, 1, 3, 5

        // and when the final field, field 2 is written
        // the offsets are correct:
        // 0, 2, 4, 6
        let mut offsets = Vec::with_capacity(num_rows + 1);
        let mut current_offset = 0;
        offsets.push(current_offset);
        for _ in 0..num_rows {
            offsets.push(current_offset);
            current_offset += row_size;
        }
        RowsEncoded::new(buf, offsets)
    }
}

#[cfg(test)]
mod test {
    use arrow::array::Utf8Array;

    use super::*;
    use crate::encodings::variable::{BLOCK_SIZE, EMPTY_SENTINEL, NON_EMPTY_SENTINEL};

    #[test]
    fn test_str_encode() {
        let sentence = "The black cat walked under a ladder but forget it's milk so it ...";
        let arr =
            Utf8Array::<i64>::from_iter([Some("a"), Some(""), Some("meep"), Some(sentence), None]);

        let field = SortField {
            descending: false,
            nulls_last: false,
        };
        let arr = arrow::compute::cast::cast(&arr, &ArrowDataType::LargeBinary, Default::default())
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
        (&mut expected[1..5]).copy_from_slice(b"meep");
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
}
