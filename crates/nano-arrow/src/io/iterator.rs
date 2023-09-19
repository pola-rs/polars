pub use streaming_iterator::StreamingIterator;

/// A [`StreamingIterator`] with an internal buffer of [`Vec<u8>`] used to efficiently
/// present items of type `T` as `&[u8]`.
/// It is generic over the type `T` and the transformation `F: T -> &[u8]`.
pub struct BufStreamingIterator<I, F, T>
where
    I: Iterator<Item = T>,
    F: FnMut(T, &mut Vec<u8>),
{
    iterator: I,
    f: F,
    buffer: Vec<u8>,
    is_valid: bool,
}

impl<I, F, T> BufStreamingIterator<I, F, T>
where
    I: Iterator<Item = T>,
    F: FnMut(T, &mut Vec<u8>),
{
    #[inline]
    pub fn new(iterator: I, f: F, buffer: Vec<u8>) -> Self {
        Self {
            iterator,
            f,
            buffer,
            is_valid: false,
        }
    }
}

impl<I, F, T> StreamingIterator for BufStreamingIterator<I, F, T>
where
    I: Iterator<Item = T>,
    F: FnMut(T, &mut Vec<u8>),
{
    type Item = [u8];

    #[inline]
    fn advance(&mut self) {
        let a = self.iterator.next();
        if let Some(a) = a {
            self.is_valid = true;
            self.buffer.clear();
            (self.f)(a, &mut self.buffer);
        } else {
            self.is_valid = false;
        }
    }

    #[inline]
    fn get(&self) -> Option<&Self::Item> {
        if self.is_valid {
            Some(&self.buffer)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}
