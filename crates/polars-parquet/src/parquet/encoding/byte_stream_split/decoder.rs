use crate::parquet::error::ParquetError;

const MAX_ELEMENT_SIZE: usize = 8;

/// Decodes using the [Byte Stream Split](https://github.com/apache/parquet-format/blob/master/Encodings.md#byte-stream-split-byte_stream_split--9) encoding.
/// # Implementation
/// A fixed size buffer is stored inline to support reading types of up to 8 bytes in size.
#[derive(Debug)]
pub struct Decoder<'a> {
    values: &'a [u8],
    buffer: [u8; MAX_ELEMENT_SIZE],
    num_elements: usize,
    position: usize,
    element_size: usize,
}

impl<'a> Decoder<'a> {
    pub fn try_new(values: &'a [u8], element_size: usize) -> Result<Self, ParquetError> {
        if element_size > MAX_ELEMENT_SIZE {
            // Since Parquet format version 2.11 it's valid to use byte stream split for fixed-length byte array data,
            // which could be larger than 8 bytes, but Polars doesn't yet support reading byte stream split encoded FLBA data.
            return Err(ParquetError::oos(format!(
                "Byte stream split decoding only supports up to {} byte element sizes",
                MAX_ELEMENT_SIZE
            )));
        }

        let values_size = values.len();
        if values_size % element_size != 0 {
            return Err(ParquetError::oos(format!(
                "Values array length ({}) is not a multiple of the element size ({})",
                values_size, element_size
            )));
        }
        let num_elements = values.len() / element_size;

        Ok(Self {
            values,
            buffer: [0; MAX_ELEMENT_SIZE],
            num_elements,
            position: 0,
            element_size,
        })
    }

    pub fn move_next(&mut self) -> bool {
        if self.position >= self.num_elements {
            return false;
        }

        debug_assert!(self.element_size <= MAX_ELEMENT_SIZE);
        debug_assert!(self.values.len() >= self.num_elements * self.element_size);
        for n in 0..self.element_size {
            unsafe {
                // SAFETY:
                // We have the invariants that element_size <= MAX_ELEMENT_SIZE,
                // buffer.len() == MAX_ELEMENT_SIZE,
                // position < num_elements and
                // values.len() >= num_elements * element_size.
                *self.buffer.get_unchecked_mut(n) = *self
                    .values
                    .get_unchecked((self.num_elements * n) + self.position)
            }
        }

        self.position += 1;
        true
    }

    /// The number of remaining values
    pub fn len(&self) -> usize {
        self.num_elements - self.position
    }

    pub fn current_value(&self) -> &[u8] {
        &self.buffer[0..self.element_size]
    }

    pub fn iter_converted<'b, T, F>(&'b mut self, converter: F) -> DecoderIterator<'a, 'b, T, F>
    where
        F: Copy + Fn(&[u8]) -> T,
    {
        DecoderIterator {
            decoder: self,
            converter,
        }
    }
}

#[derive(Debug)]
pub struct DecoderIterator<'a, 'b, T, F>
where
    F: Copy + Fn(&[u8]) -> T,
{
    decoder: &'b mut Decoder<'a>,
    converter: F,
}

impl<'a, 'b, T, F> Iterator for DecoderIterator<'a, 'b, T, F>
where
    F: Copy + Fn(&[u8]) -> T,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.decoder.move_next() {
            Some((self.converter)(self.decoder.current_value()))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.decoder.len(), Some(self.decoder.len()))
    }
}
