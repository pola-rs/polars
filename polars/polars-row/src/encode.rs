use crate::row::RowsEncoded;
use crate::sort_field::SortField;
use arrow::datatypes::DataType as ArrowDataType;
use crate::ArrayRef;

fn encode_column() {
    todo!()
}

pub fn allocate_rows_buf(columns: &[ArrayRef], fields: Box<[SortField]>) -> RowsEncoded {
    let has_variable = fields.iter().any(|f| matches!(f.data_type, ArrowDataType::LargeBinary));

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
        };
        RowsEncoded::new(buf.into(), offsets.into(), fields)
    }
}