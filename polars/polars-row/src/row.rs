use arrow::array::BinaryArray;
use arrow::datatypes::DataType;
use arrow::offset::Offsets;

#[derive(Clone)]
pub struct SortField {
    /// Whether to sort in descending order
    pub descending: bool,
    /// Whether to sort nulls first
    pub nulls_last: bool,
}

pub struct RowsEncoded {
    pub(crate) buf: Vec<u8>,
    pub(crate) offsets: Vec<usize>,
}

impl RowsEncoded {
    pub(crate) fn new(buf: Vec<u8>, offsets: Vec<usize>) -> Self {
        RowsEncoded { buf, offsets }
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

    pub fn into_array(self) -> BinaryArray<i64> {
        assert_eq!(
            std::mem::size_of::<usize>(),
            std::mem::size_of::<i64>(),
            "only supported on 64bit arch"
        );
        assert!(
            (*self.offsets.last().unwrap() as u64) < i64::MAX as u64,
            "overflow"
        );

        // Safety: we checked overflow
        let offsets = unsafe { std::mem::transmute::<Vec<usize>, Vec<i64>>(self.offsets) };

        // Safety: monotonically increasing
        let offsets = unsafe { Offsets::new_unchecked(offsets) };

        BinaryArray::new(DataType::LargeBinary, offsets.into(), self.buf.into(), None)
    }

    #[cfg(test)]
    pub fn get(&self, i: usize) -> &[u8] {
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        &self.buf[start..end]
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
