/// Decodes according to [Plain strings](https://github.com/apache/parquet-format/blob/master/Encodings.md#plain-plain--0),
/// prefixes, lengths and values
/// # Implementation
/// This struct does not allocate on the heap.
use crate::error::Error;

#[derive(Debug)]
pub struct BinaryIter<'a> {
    values: &'a [u8],
    length: Option<usize>,
}

impl<'a> BinaryIter<'a> {
    pub fn new(values: &'a [u8], length: Option<usize>) -> Self {
        Self { values, length }
    }
}

impl<'a> Iterator for BinaryIter<'a> {
    type Item = Result<&'a [u8], Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.len() < 4 {
            return None;
        }
        if let Some(x) = self.length.as_mut() {
            *x = x.saturating_sub(1)
        }
        let length = u32::from_le_bytes(self.values[0..4].try_into().unwrap()) as usize;
        self.values = &self.values[4..];
        if length > self.values.len() {
            return Some(Err(Error::oos(
                "A string in plain encoding declares a length that is out of range",
            )));
        }
        let (result, remaining) = self.values.split_at(length);
        self.values = remaining;
        Some(Ok(result))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length.unwrap_or_default(), self.length)
    }
}
