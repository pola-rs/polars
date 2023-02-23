use crate::sort_field::SortField;

pub struct RowsEncoded {
    pub(crate) buf: Vec<u8>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) fields: Vec<SortField>,
}

impl RowsEncoded {
    pub(crate) fn new(buf: Vec<u8>, offsets: Vec<usize>, fields: Option<Vec<SortField>>) -> Self {
        RowsEncoded {
            buf,
            offsets,
            fields: fields.unwrap_or_default(),
        }
    }
}
