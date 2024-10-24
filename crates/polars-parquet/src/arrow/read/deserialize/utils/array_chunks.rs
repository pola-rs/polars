use arrow::types::AlignedBytes;

/// A slice of chunks that fit an [`AlignedBytes`] type.
///
/// This is essentially the equivalent of [`ChunksExact`][std::slice::ChunksExact], but with a size
/// and type known at compile-time. This makes the compiler able to reason much more about the
/// code. Especially, since the chunk-sizes for this type are almost always powers of 2 and
/// bitshifts or special instructions would be much better to use.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArrayChunks<'a, B: AlignedBytes> {
    pub(crate) bytes: &'a [B::Unaligned],
}

impl<'a, B: AlignedBytes> ArrayChunks<'a, B> {
    /// Create a new [`ArrayChunks`]
    ///
    /// This returns null if the `bytes` slice's length is not a multiple of the size of `P::Bytes`.
    pub(crate) fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() % B::SIZE != 0 {
            return None;
        }

        let bytes = bytemuck::cast_slice(bytes);

        Some(Self { bytes })
    }

    pub(crate) unsafe fn get_unchecked(&self, at: usize) -> B {
        B::from_unaligned(*unsafe { self.bytes.get_unchecked(at) })
    }

    pub fn truncate(&self, length: usize) -> ArrayChunks<'a, B> {
        let length = length.min(self.bytes.len());

        Self {
            bytes: unsafe { self.bytes.get_unchecked(..length) },
        }
    }

    pub unsafe fn slice_unchecked(&self, start: usize, end: usize) -> ArrayChunks<'a, B> {
        debug_assert!(start <= self.bytes.len());
        debug_assert!(end <= self.bytes.len());

        Self {
            bytes: unsafe { self.bytes.get_unchecked(start..end) },
        }
    }

    pub fn as_ptr(&self) -> *const B::Unaligned {
        self.bytes.as_ptr()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, B: AlignedBytes> Iterator for ArrayChunks<'a, B> {
    type Item = &'a B::Unaligned;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.bytes.first()?;
        self.bytes = &self.bytes[1..];
        Some(item)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.bytes.len(), Some(self.bytes.len()))
    }
}

impl<'a, B: AlignedBytes> ExactSizeIterator for ArrayChunks<'a, B> {}
