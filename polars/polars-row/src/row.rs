use crate::sort_field::SortField;

pub struct RowsEncoded {
    pub(crate) buf: Box<[u8]>,
    pub(crate) offsets: Box<[usize]>,
    pub(crate) fields: Box<[SortField]>
}

impl RowsEncoded {
    pub(crate) fn new(buf: Box<[u8]>, offsets: Box<[usize]>, fields: Box<[SortField]>) -> Self {
        RowsEncoded {
            buf,
            offsets,
            fields
        }
    }
}