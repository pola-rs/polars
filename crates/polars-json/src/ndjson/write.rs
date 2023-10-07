//! APIs to serialize and write to [NDJSON](http://ndjson.org/).
use std::io::Write;

use arrow::array::Array;
use arrow::error::Error;
pub use fallible_streaming_iterator::FallibleStreamingIterator;

use super::super::json::write::new_serializer;

fn serialize(array: &dyn Array, buffer: &mut Vec<u8>) {
    let mut serializer = new_serializer(array, 0, usize::MAX);
    (0..array.len()).for_each(|_| {
        buffer.extend_from_slice(serializer.next().unwrap());
        buffer.push(b'\n');
    });
}

/// [`FallibleStreamingIterator`] that serializes an [`Array`] to bytes of valid NDJSON
/// where every line is an element of the array.
/// # Implementation
/// Advancing this iterator CPU-bounded
#[derive(Debug, Clone)]
pub struct Serializer<A, I>
where
    A: AsRef<dyn Array>,
    I: Iterator<Item = Result<A, Error>>,
{
    arrays: I,
    buffer: Vec<u8>,
}

impl<A, I> Serializer<A, I>
where
    A: AsRef<dyn Array>,
    I: Iterator<Item = Result<A, Error>>,
{
    /// Creates a new [`Serializer`].
    pub fn new(arrays: I, buffer: Vec<u8>) -> Self {
        Self { arrays, buffer }
    }
}

impl<A, I> FallibleStreamingIterator for Serializer<A, I>
where
    A: AsRef<dyn Array>,
    I: Iterator<Item = Result<A, Error>>,
{
    type Item = [u8];

    type Error = Error;

    fn advance(&mut self) -> Result<(), Error> {
        self.buffer.clear();
        self.arrays
            .next()
            .map(|maybe_array| maybe_array.map(|array| serialize(array.as_ref(), &mut self.buffer)))
            .transpose()?;
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        if !self.buffer.is_empty() {
            Some(&self.buffer)
        } else {
            None
        }
    }
}

/// An iterator adapter that receives an implementer of [`Write`] and
/// an implementer of [`FallibleStreamingIterator`] (such as [`Serializer`])
/// and writes a valid NDJSON
/// # Implementation
/// Advancing this iterator mixes CPU-bounded (serializing arrays) tasks and IO-bounded (write to the writer).
pub struct FileWriter<W, I>
where
    W: Write,
    I: FallibleStreamingIterator<Item = [u8], Error = Error>,
{
    writer: W,
    iterator: I,
}

impl<W, I> FileWriter<W, I>
where
    W: Write,
    I: FallibleStreamingIterator<Item = [u8], Error = Error>,
{
    /// Creates a new [`FileWriter`].
    pub fn new(writer: W, iterator: I) -> Self {
        Self { writer, iterator }
    }

    /// Returns the inner content of this iterator
    ///
    /// There are two use-cases for this function:
    /// * to continue writing to its writer
    /// * to re-use an internal buffer of its iterator
    pub fn into_inner(self) -> (W, I) {
        (self.writer, self.iterator)
    }
}

impl<W, I> Iterator for FileWriter<W, I>
where
    W: Write,
    I: FallibleStreamingIterator<Item = [u8], Error = Error>,
{
    type Item = Result<(), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iterator.next().transpose()?;
        Some(item.and_then(|x| {
            self.writer.write_all(x)?;
            Ok(())
        }))
    }
}
