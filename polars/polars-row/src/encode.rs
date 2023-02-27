use arrow::array::{Array, BinaryArray, BooleanArray, PrimitiveArray};
use arrow::datatypes::{DataType as ArrowDataType, DataType};
use arrow::types::NativeType;

use crate::encodings::fixed::FixedLengthEncoding;
use crate::row::RowsEncoded;
use crate::sort_field::SortField;
use crate::{with_match_arrow_primitive_type, ArrayRef};

pub fn convert_columns(columns: &[ArrayRef], fields: Vec<SortField>) -> RowsEncoded {
    assert_eq!(fields.len(), columns.len());

    let mut rows = allocate_rows_buf(columns, &fields);
    for (arr, field) in columns.iter().zip(fields.iter()) {
        // Safety:
        // we allocated rows with enough bytes.
        unsafe { encode_array(&**arr, field, &mut rows) }
    }

    // we set fields later so we don't have aliasing borrows.
    rows.fields = fields;
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
        dt => {
            with_match_arrow_primitive_type!(dt, |$T| {
                let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                encode_primitive(array, field, out);
            })
        }
    };
}

pub fn allocate_rows_buf(columns: &[ArrayRef], fields: &[SortField]) -> RowsEncoded {
    let has_variable = fields
        .iter()
        .any(|f| matches!(f.data_type, ArrowDataType::LargeBinary));

    let num_rows = columns[0].len();
    if has_variable {
        // row size of the fixed-length columns
        // those can be determined without looping over the arrays
        let row_size_fixed: usize = fields
            .iter()
            .map(|f| {
                if matches!(f.data_type, ArrowDataType::LargeBinary) {
                    0
                } else {
                    f.encoded_size()
                }
            })
            .sum();

        let mut lengths = vec![row_size_fixed; num_rows];

        // for the variable length columns we must iterate to determine the length per row location
        for array in columns.iter() {
            if matches!(array.data_type(), ArrowDataType::LargeBinary) {
                let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
                for (opt_val, row_length) in array.into_iter().zip(lengths.iter_mut()) {
                    *row_length += crate::encodings::variable::encoded_len(opt_val)
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
        RowsEncoded::new(buf, offsets, None)
    } else {
        let row_size: usize = fields.iter().map(|f| f.encoded_size()).sum();
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
        RowsEncoded::new(buf, offsets, None)
    }
}
