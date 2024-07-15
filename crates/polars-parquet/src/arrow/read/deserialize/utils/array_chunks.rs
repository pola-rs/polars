use crate::parquet::types::NativeType as ParquetNativeType;

/// A slice of chunks that fit the `P` type.
///
/// This is essentially the equivalent of [`ChunksExact`][std::slice::ChunksExact], but with a size
/// and type known at compile-time. This makes the compiler able to reason much more about the
/// code. Especially, since the chunk-sizes for this type are almost always powers of 2 and
/// bitshifts or special instructions would be much better to use.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArrayChunks<'a, P: ParquetNativeType> {
    pub(crate) bytes: &'a [P::Bytes],
}

impl<'a, P: ParquetNativeType> ArrayChunks<'a, P> {
    /// Create a new [`ArrayChunks`]
    ///
    /// This returns null if the `bytes` slice's length is not a multiple of the size of `P::Bytes`.
    pub(crate) fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() % std::mem::size_of::<P::Bytes>() != 0 {
            return None;
        }

        // SAFETY:
        // We know that that the alignment, size and provenance are the same.
        let bytes = unsafe { std::mem::transmute::<&[u8], &[P::Bytes]>(bytes) };

        Some(Self { bytes })
    }
}

impl<'a, P: ParquetNativeType> Iterator for ArrayChunks<'a, P> {
    type Item = &'a P::Bytes;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.bytes.first()?;
        self.bytes = &self.bytes[1..];
        Some(item)
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let item = self.bytes.get(n)?;
        self.bytes = &self.bytes[n + 1..];
        Some(item)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.bytes.len(), Some(self.bytes.len()))
    }
}

impl<'a, P: ParquetNativeType> ExactSizeIterator for ArrayChunks<'a, P> {}
