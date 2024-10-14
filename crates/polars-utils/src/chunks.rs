/// A copy of the [`std::slice::Chunks`] that exposes the inner `slice` and `chunk_size`.
#[derive(Clone, Debug)]
pub struct Chunks<'a, T> {
    slice: &'a [T],
    chunk_size: usize,
}

impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        let item;
        (item, self.slice) = self.slice.split_at(self.chunk_size.min(self.slice.len()));

        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.slice.len().div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        let rem = self.slice.len() % self.chunk_size;
        let offset = if rem == 0 {
            self.chunk_size
        } else {
            rem
        };

        let item;
        (self.slice, item) = self.slice.split_at(self.slice.len() - offset);

        Some(item)
    }
}

impl<'a, T> ExactSizeIterator for Chunks<'a, T> {}

impl<'a, T> Chunks<'a, T> {
    pub const fn new(slice: &'a [T], chunk_size: usize) -> Self {
        Self { slice, chunk_size }
    }

    pub const fn as_slice(&self) -> &'a [T] {
        self.slice
    }

    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn skip_in_place(&mut self, n: usize) {
        let n = n * self.chunk_size;
        self.slice = &self.slice[n.min(self.slice.len())..];
    }
}
