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

    pub fn iter(&self) -> RowsEncodedIter {
        let iter = self.offsets[1..].iter();
        let offset = self.offsets[0];
        RowsEncodedIter {
            offset,
            end: iter,
            buf: &self.buf,
        }
    }
}

pub struct RowsEncodedIter<'a> {
    offset: usize,
    end: std::slice::Iter<'a, usize>,
    buf: &'a [u8],
}

impl<'a> Iterator for RowsEncodedIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let new_offset = *self.end.next()?;
        let payload = unsafe { self.buf.get_unchecked(self.offset..new_offset) };
        self.offset = new_offset;
        Some(payload)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.end.size_hint()
    }
}
