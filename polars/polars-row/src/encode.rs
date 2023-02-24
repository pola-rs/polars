use arrow::array::{Array, PrimitiveArray};
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
        encode_array(&**arr, field, &mut rows)
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

fn encode_array(array: &dyn Array, field: &SortField, out: &mut RowsEncoded) {
    match array.data_type() {
        DataType::Boolean => todo!(),
        DataType::LargeUtf8 => todo!(),
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
        todo!()
    } else {
        let row_size: usize = fields.iter().map(|f| f.encoded_size()).sum();
        let n_bytes = num_rows * row_size;
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
