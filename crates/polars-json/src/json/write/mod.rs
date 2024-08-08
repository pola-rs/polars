//! APIs to write to JSON
mod serialize;
mod utf8;

use std::io::Write;

use arrow::array::Array;
use arrow::datatypes::ArrowSchema;
use arrow::io::iterator::StreamingIterator;
use arrow::record_batch::RecordBatchT;
pub use fallible_streaming_iterator::*;
use polars_error::{PolarsError, PolarsResult};
pub(crate) use serialize::new_serializer;
use serialize::serialize;
pub use utf8::serialize_to_utf8;

/// [`FallibleStreamingIterator`] that serializes an [`Array`] to bytes of valid JSON
/// # Implementation
/// Advancing this iterator CPU-bounded
#[derive(Debug, Clone)]
pub struct Serializer<A, I>
where
    A: AsRef<dyn Array>,
    I: Iterator<Item = PolarsResult<A>>,
{
    arrays: I,
    buffer: Vec<u8>,
}

impl<A, I> Serializer<A, I>
where
    A: AsRef<dyn Array>,
    I: Iterator<Item = PolarsResult<A>>,
{
    /// Creates a new [`Serializer`].
    pub fn new(arrays: I, buffer: Vec<u8>) -> Self {
        Self { arrays, buffer }
    }
}

impl<A, I> FallibleStreamingIterator for Serializer<A, I>
where
    A: AsRef<dyn Array>,
    I: Iterator<Item = PolarsResult<A>>,
{
    type Item = [u8];

    type Error = PolarsError;

    fn advance(&mut self) -> PolarsResult<()> {
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

/// [`FallibleStreamingIterator`] that serializes a [`RecordBatchT`] into bytes of JSON
/// in a (pandas-compatible) record-oriented format.
///
/// # Implementation
/// Advancing this iterator is CPU-bounded.
pub struct RecordSerializer<'a> {
    schema: ArrowSchema,
    index: usize,
    end: usize,
    iterators: Vec<Box<dyn StreamingIterator<Item = [u8]> + Send + Sync + 'a>>,
    buffer: Vec<u8>,
}

impl<'a> RecordSerializer<'a> {
    /// Creates a new [`RecordSerializer`].
    pub fn new<A>(schema: ArrowSchema, chunk: &'a RecordBatchT<A>, buffer: Vec<u8>) -> Self
    where
        A: AsRef<dyn Array>,
    {
        let end = chunk.len();
        let iterators = chunk
            .arrays()
            .iter()
            .map(|arr| new_serializer(arr.as_ref(), 0, usize::MAX))
            .collect();

        Self {
            schema,
            index: 0,
            end,
            iterators,
            buffer,
        }
    }
}

impl<'a> FallibleStreamingIterator for RecordSerializer<'a> {
    type Item = [u8];

    type Error = PolarsError;

    fn advance(&mut self) -> PolarsResult<()> {
        self.buffer.clear();
        if self.index == self.end {
            return Ok(());
        }

        let mut is_first_row = true;
        write!(&mut self.buffer, "{{")?;
        for (f, ref mut it) in self.schema.fields.iter().zip(self.iterators.iter_mut()) {
            if !is_first_row {
                write!(&mut self.buffer, ",")?;
            }
            write!(&mut self.buffer, "\"{}\":", f.name)?;

            self.buffer.extend_from_slice(it.next().unwrap());
            is_first_row = false;
        }
        write!(&mut self.buffer, "}}")?;

        self.index += 1;
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

/// Writes valid JSON from an iterator of (assumed JSON-encoded) bytes to `writer`
pub fn write<W, I>(writer: &mut W, mut blocks: I) -> PolarsResult<()>
where
    W: std::io::Write,
    I: FallibleStreamingIterator<Item = [u8], Error = PolarsError>,
{
    writer.write_all(b"[")?;
    let mut is_first_row = true;
    while let Some(block) = blocks.next()? {
        if !is_first_row {
            writer.write_all(b",")?;
        }
        is_first_row = false;
        writer.write_all(block)?;
    }
    writer.write_all(b"]")?;
    Ok(())
}
