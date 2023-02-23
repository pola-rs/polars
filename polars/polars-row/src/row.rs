use crate::sort_field::SortField;

pub struct RowsEncoded {
    buf: Box<[u8]>,
    offsets: Box<[usize]>,
    fields: Box<[SortField]>
}

impl RowsEncoded {
    pub(crate) fn new(buf: Box<[u8]>, offsets: Box<[usize]>, fields: Box<[SortField]>) -> Self {
        RowsEncoded {
            buf,
            offsets,
            fields
        }
    }

    pub(crate) fn offsets(&self) -> &[usize] {
        &*self.offsets
    }

    pub(crate) fn offsets_mut(&mut self) -> &mut [usize] {
        &mut *self.offsets
    }
}