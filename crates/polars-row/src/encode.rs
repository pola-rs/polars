use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, PrimitiveArray,
    StructArray, Utf8ViewArray,
};
use arrow::bitmap::utils::ZipValidity;
use arrow::compute::utils::combine_validities_and;
use arrow::datatypes::ArrowDataType;
use arrow::legacy::prelude::{LargeBinaryArray, LargeListArray};
use arrow::types::NativeType;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::vec::PushUnchecked;

use crate::fixed::FixedLengthEncoding;
use crate::row::{EncodingField, RowsEncoded};
use crate::{with_match_arrow_primitive_type, ArrayRef};

pub fn convert_columns(columns: &[ArrayRef], fields: &[EncodingField]) -> RowsEncoded {
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
        std::iter::repeat(&EncodingField::default()).take(columns.len()),
        rows,
    );
}

enum Encoder {
    // For list encoding we recursively call encode on the inner until we
    // have a leaf we can encode.
    // On allocation we already encode the leaves and set those to `rows`.
    List {
        enc: Vec<Encoder>,
        rows: Option<LargeBinaryArray>,
        original: LargeListArray,
        field: EncodingField,
    },
    Leaf(ArrayRef),
}

impl Encoder {
    fn list_iter(&self) -> impl Iterator<Item = Option<&[u8]>> {
        match self {
            Encoder::Leaf(_) => unreachable!(),
            Encoder::List { original, rows, .. } => {
                let rows = rows.as_ref().unwrap();
                // This should be 0 due to rows encoding;
                assert_eq!(rows.null_count(), 0);

                let offsets = original.offsets().windows(2);
                let zipped = ZipValidity::new_with_validity(offsets, original.validity());

                let binary_offsets = rows.offsets();
                let row_values = rows.values().as_slice();

                zipped.map(|opt_window| {
                    opt_window.map(|window| {
                        unsafe {
                            // Offsets of the list
                            let start = *window.get_unchecked_release(0);
                            let end = *window.get_unchecked_release(1);

                            // Offsets in the binary values.
                            let start = *binary_offsets.get_unchecked_release(start as usize);
                            let end = *binary_offsets.get_unchecked_release(end as usize);

                            let start = start as usize;
                            let end = end as usize;

                            row_values.get_unchecked_release(start..end)
                        }
                    })
                })
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            Encoder::List { original, .. } => original.len(),
            Encoder::Leaf(arr) => arr.len(),
        }
    }

    fn data_type(&self) -> &ArrowDataType {
        match self {
            Encoder::List { original, .. } => original.data_type(),
            Encoder::Leaf(arr) => arr.data_type(),
        }
    }

    fn is_variable(&self) -> bool {
        match self {
            Encoder::Leaf(arr) => {
                matches!(
                    arr.data_type(),
                    ArrowDataType::BinaryView
                        | ArrowDataType::Dictionary(_, _, _)
                        | ArrowDataType::LargeBinary
                )
            },
            Encoder::List { .. } => true,
        }
    }
}

fn get_encoders(arr: &dyn Array, encoders: &mut Vec<Encoder>, field: &EncodingField) -> usize {
    let mut added = 0;
    match arr.data_type() {
        ArrowDataType::Struct(_) => {
            let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
            for value_arr in arr.values() {
                // A hack to make outer validity work.
                // TODO! improve
                if arr.null_count() > 0 {
                    let new_validity = combine_validities_and(arr.validity(), value_arr.validity());
                    value_arr.with_validity(new_validity);
                    added += get_encoders(value_arr.as_ref(), encoders, field);
                } else {
                    added += get_encoders(value_arr.as_ref(), encoders, field);
                }
            }
        },
        ArrowDataType::Utf8View => {
            let arr = arr.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            encoders.push(Encoder::Leaf(arr.to_binview().boxed()));
            added += 1
        },
        ArrowDataType::LargeList(_) => {
            let arr = arr.as_any().downcast_ref::<LargeListArray>().unwrap();
            let mut inner = vec![];
            get_encoders(arr.values().as_ref(), &mut inner, field);
            encoders.push(Encoder::List {
                enc: inner,
                original: arr.clone(),
                rows: None,
                field: *field,
            });
            added += 1;
        },
        _ => {
            encoders.push(Encoder::Leaf(arr.to_boxed()));
            added += 1;
        },
    }
    added
}

pub fn convert_columns_amortized<'a, I: IntoIterator<Item = &'a EncodingField>>(
    columns: &'a [ArrayRef],
    fields: I,
    rows: &mut RowsEncoded,
) {
    let fields = fields.into_iter();
    assert_eq!(fields.size_hint().0, columns.len());
    if columns.iter().any(|arr| {
        matches!(
            arr.data_type(),
            ArrowDataType::Struct(_) | ArrowDataType::Utf8View | ArrowDataType::LargeList(_)
        )
    }) {
        let mut flattened_columns = Vec::with_capacity(columns.len() * 5);
        let mut flattened_fields = Vec::with_capacity(columns.len() * 5);

        for (arr, field) in columns.iter().zip(fields) {
            let added = get_encoders(arr.as_ref(), &mut flattened_columns, field);
            for _ in 0..added {
                flattened_fields.push(*field);
            }
        }
        let values_size = allocate_rows_buf(
            &mut flattened_columns,
            &flattened_fields,
            &mut rows.values,
            &mut rows.offsets,
        );
        for (arr, field) in flattened_columns.iter().zip(flattened_fields.iter()) {
            // SAFETY:
            // we allocated rows with enough bytes.
            unsafe { encode_array(arr, field, rows) }
        }
        // SAFETY: values are initialized
        unsafe { rows.values.set_len(values_size) }
    } else {
        let mut encoders = columns
            .iter()
            .map(|arr| Encoder::Leaf(arr.clone()))
            .collect::<Vec<_>>();
        let fields = fields.cloned().collect::<Vec<_>>();
        let values_size =
            allocate_rows_buf(&mut encoders, &fields, &mut rows.values, &mut rows.offsets);
        for (enc, field) in encoders.iter().zip(fields) {
            // SAFETY:
            // we allocated rows with enough bytes.
            unsafe { encode_array(enc, &field, rows) }
        }
        // SAFETY: values are initialized
        unsafe { rows.values.set_len(values_size) }
    }
}

fn encode_primitive<T: NativeType + FixedLengthEncoding>(
    arr: &PrimitiveArray<T>,
    field: &EncodingField,
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
unsafe fn encode_array(encoder: &Encoder, field: &EncodingField, out: &mut RowsEncoded) {
    match encoder {
        Encoder::List { .. } => {
            let iter = encoder.list_iter();
            crate::variable::encode_iter(iter, out, &EncodingField::new_unsorted())
        },
        Encoder::Leaf(array) => {
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
                ArrowDataType::Utf8View => {
                    panic!("should be binview")
                },
                ArrowDataType::Dictionary(_, _, _) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<DictionaryArray<u32>>()
                        .unwrap();
                    let iter = array
                        .iter_typed::<Utf8ViewArray>()
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
        },
    }
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
        Decimal(_, _) => i128::ENCODED_LEN,
        Float32 => f32::ENCODED_LEN,
        Float64 => f64::ENCODED_LEN,
        Boolean => bool::ENCODED_LEN,
        dt => unimplemented!("{dt:?}"),
    }
}

// Returns the length that the caller must set on the `values` buf  once the bytes
// are initialized.
fn allocate_rows_buf(
    columns: &mut [Encoder],
    fields: &[EncodingField],
    values: &mut Vec<u8>,
    offsets: &mut Vec<usize>,
) -> usize {
    let has_variable = columns.iter().any(|enc| enc.is_variable());

    let num_rows = columns[0].len();
    if has_variable {
        // row size of the fixed-length columns
        // those can be determined without looping over the arrays
        let row_size_fixed: usize = columns
            .iter()
            .map(|enc| {
                if enc.is_variable() {
                    0
                } else {
                    encoded_size(enc.data_type())
                }
            })
            .sum();

        offsets.clear();
        offsets.reserve(num_rows + 1);

        // first write lengths to this buffer
        let lengths = offsets;

        // for the variable length columns we must iterate to determine the length per row location
        let mut processed_count = 0;
        for (enc, enc_field) in columns.iter_mut().zip(fields) {
            match enc {
                Encoder::List {
                    enc: inner_enc,
                    rows,
                    field,
                    original,
                } => {
                    let field = *field;
                    let fields = inner_enc.iter().map(|_| field).collect::<Vec<_>>();
                    // Nested lists don't yet work as that requires the leaves not only allocating, but also
                    // encoding. To make that work we must add a flag `in_list` that tell the leaves to immediately
                    // encode the rows instead of only setting the length.
                    // This needs a bit refactoring, might require allocation and encoding to be in
                    // the same function.
                    if let ArrowDataType::LargeList(inner) = original.data_type() {
                        assert!(
                            !matches!(inner.data_type, ArrowDataType::LargeList(_)),
                            "should not be nested"
                        )
                    }
                    // Create the row encoding for the inner type.
                    let mut values_rows = RowsEncoded::default();

                    // Allocate and immediately row-encode the inner types recursively.
                    let values_size = allocate_rows_buf(
                        inner_enc,
                        &fields,
                        &mut values_rows.values,
                        &mut values_rows.offsets,
                    );

                    // For single nested it does work as we encode here.
                    unsafe {
                        for enc in inner_enc {
                            encode_array(enc, &field, &mut values_rows)
                        }
                        values_rows.values.set_len(values_size)
                    };
                    let values_rows = values_rows.into_array();
                    *rows = Some(values_rows);

                    let iter = enc.list_iter();

                    if processed_count == 0 {
                        for opt_val in iter {
                            unsafe {
                                lengths.push_unchecked(
                                    row_size_fixed
                                        + crate::variable::encoded_len(
                                            opt_val,
                                            &EncodingField::new_unsorted(),
                                        ),
                                );
                            }
                        }
                    } else {
                        for (opt_val, row_length) in iter.zip(lengths.iter_mut()) {
                            *row_length += crate::variable::encoded_len(
                                opt_val,
                                &EncodingField::new_unsorted(),
                            )
                        }
                    }
                    processed_count += 1;
                },
                Encoder::Leaf(array) => {
                    match array.data_type() {
                        ArrowDataType::BinaryView => {
                            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                            if processed_count == 0 {
                                for opt_val in array.into_iter() {
                                    unsafe {
                                        lengths.push_unchecked(
                                            row_size_fixed
                                                + crate::variable::encoded_len(opt_val, enc_field),
                                        );
                                    }
                                }
                            } else {
                                for (opt_val, row_length) in
                                    array.into_iter().zip(lengths.iter_mut())
                                {
                                    *row_length += crate::variable::encoded_len(opt_val, enc_field)
                                }
                            }
                            processed_count += 1;
                        },
                        ArrowDataType::LargeBinary => {
                            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
                            if processed_count == 0 {
                                for opt_val in array.into_iter() {
                                    unsafe {
                                        lengths.push_unchecked(
                                            row_size_fixed
                                                + crate::variable::encoded_len(opt_val, enc_field),
                                        );
                                    }
                                }
                            } else {
                                for (opt_val, row_length) in
                                    array.into_iter().zip(lengths.iter_mut())
                                {
                                    *row_length += crate::variable::encoded_len(opt_val, enc_field)
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
                                .iter_typed::<Utf8ViewArray>()
                                .unwrap()
                                .map(|opt_s| opt_s.map(|s| s.as_bytes()));
                            if processed_count == 0 {
                                for opt_val in iter {
                                    unsafe {
                                        lengths.push_unchecked(
                                            row_size_fixed
                                                + crate::variable::encoded_len(opt_val, enc_field),
                                        )
                                    }
                                }
                            } else {
                                for (opt_val, row_length) in iter.zip(lengths.iter_mut()) {
                                    *row_length += crate::variable::encoded_len(opt_val, enc_field)
                                }
                            }
                            processed_count += 1;
                        },
                        _ => {
                            // the rest is fixed
                        },
                    }
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
    use arrow::offset::Offsets;

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

        let field = EncodingField::new_sorted(false, false);
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

        let field = EncodingField::new_sorted(false, false);
        let arr = BinaryViewArray::from_slice_values(a);
        let rows_encoded = convert_columns_no_order(&[arr.clone().boxed()]);

        let mut rows = rows_encoded.iter().collect::<Vec<_>>();
        let decoded = unsafe { decode_binview(&mut rows, &field) };
        assert_eq!(decoded, arr);
    }

    #[test]
    fn test_reverse_variable() {
        let a = Utf8ViewArray::from_slice_values(["one", "two", "three", "four", "five", "six"]);

        let fields = &[EncodingField::new_sorted(true, false)];

        let dtypes = [ArrowDataType::Utf8View];

        unsafe {
            let encoded = convert_columns(&[Box::new(a.clone())], fields);
            let out = decode_rows_from_binary(&encoded.into_array(), fields, &dtypes, &mut vec![]);

            let arr = &out[0];
            let decoded = arr.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            assert_eq!(decoded, &a);
        }
    }

    #[test]
    fn test_list_encode() {
        let values = Utf8ViewArray::from_slice_values([
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ]);
        let dtype = LargeListArray::default_datatype(values.data_type().clone());
        let array = LargeListArray::new(
            dtype,
            Offsets::<i64>::try_from(vec![0i64, 1, 4, 7, 7, 9, 10])
                .unwrap()
                .into(),
            values.boxed(),
            None,
        );
        let fields = &[EncodingField::new_sorted(true, false)];

        let out = convert_columns(&[array.boxed()], fields);
        let out = out.into_array();
        assert_eq!(
            out.values().iter().map(|v| *v as usize).sum::<usize>(),
            82411
        );
    }
}
