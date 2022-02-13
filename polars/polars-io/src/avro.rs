use std::io::{Read, Seek, Write};

use super::{finish_reader, ArrowChunk, ArrowReader, ArrowResult};
use crate::prelude::*;
use polars_core::prelude::*;
use std::ops::Deref;

use arrow::io::avro::{read, write};

/// Read Appache Avro format into a DataFrame
///
/// # Example
/// ```
/// use std::fs::File;
/// use polars_core::prelude::*;
/// use polars_io::avro::AvroReader;
/// use polars_io::SerReader;
///
/// fn example() -> Result<DataFrame> {
///     let file = File::open("file.avro").expect("file not found");
///     
///     AvroReader::new(file)
///             .finish()
/// }
/// ```
#[must_use]
pub struct AvroReader<R> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
}

impl<R: Read + Seek> AvroReader<R> {
    /// Get schema of the Avro File
    pub fn schema(&mut self) -> Result<Schema> {
        let (_, schema, _, _) = read::read_metadata(&mut self.reader)?;
        Ok(schema.into())
    }

    /// Get arrow schema of the avro File, this is faster than a polars schema.
    pub fn arrow_schema(&mut self) -> Result<ArrowSchema> {
        let (_, schema, _, _) = read::read_metadata(&mut self.reader)?;
        Ok(schema)
    }

    /// Stop reading when `n` rows are read.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }
}

impl<R> ArrowReader for read::Reader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }
}

impl<R> SerReader<R> for AvroReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        AvroReader {
            reader,
            rechunk: true,
            n_rows: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let (avro_schema, schema, codec, file_marker) = read::read_metadata(&mut self.reader)?;

        let avro_reader = read::Reader::new(
            read::Decompressor::new(
                read::BlockStreamIterator::new(&mut self.reader, file_marker),
                codec,
            ),
            avro_schema,
            schema.clone().fields,
            None,
        );

        finish_reader(avro_reader, rechunk, None, None, None, &schema, None)
    }
}

pub use write::Compression as AvroCompression;

/// Write a DataFrame to Appache Avro format
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::avro::AvroWriter;
/// use std::fs::File;
/// use polars_io::SerWriter;
///
/// fn example(df: &mut DataFrame) -> Result<()> {
///     let mut file = File::create("file.avro").expect("could not create file");
///
///     AvroWriter::new(&mut file)
///         .finish(df)
/// }
///
/// ```
#[must_use]
pub struct AvroWriter<W> {
    writer: W,
    compression: Option<write::Compression>,
}

impl<W> AvroWriter<W>
where
    W: Write,
{
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<write::Compression>) -> Self {
        self.compression = compression;
        self
    }
}

impl<W> SerWriter<W> for AvroWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self {
        Self {
            writer,
            compression: None,
        }
    }

    fn finish(mut self, df: &mut DataFrame) -> Result<()> {
        let schema = df.schema().to_arrow();
        let avro_fields = write::to_avro_schema(&schema)?;

        for chunk in df.iter_chunks() {
            let mut serializers = chunk
                .iter()
                .zip(avro_fields.iter())
                .map(|(array, field)| write::new_serializer(array.deref(), &field.schema))
                .collect::<Vec<_>>();

            let mut block = write::Block::new(chunk.len(), vec![]);
            let mut compressed_block = write::CompressedBlock::default();

            write::serialize(&mut serializers, &mut block);
            let _was_compressed =
                write::compress(&mut block, &mut compressed_block, self.compression)?;

            write::write_metadata(&mut self.writer, avro_fields.clone(), self.compression)?;

            write::write_block(&mut self.writer, &compressed_block)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{write, AvroReader, AvroWriter};
    use crate::prelude::*;
    use polars_core::df;
    use polars_core::prelude::*;
    use std::io::Cursor;
    #[test]
    fn write_and_read_avro() -> Result<()> {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut write_df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
        )?;

        AvroWriter::new(&mut buf).finish(&mut write_df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf).finish()?;
        assert!(write_df.frame_equal(&read_df));
        Ok(())
    }

    #[test]
    fn test_write_and_read_with_compression() -> Result<()> {
        let mut write_df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
        )?;

        let compressions = vec![
            None,
            Some(write::Compression::Deflate),
            Some(write::Compression::Snappy),
        ];

        for compression in compressions.into_iter() {
            let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

            AvroWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut write_df)?;
            buf.set_position(0);

            let read_df = AvroReader::new(buf).finish()?;
            assert!(write_df.frame_equal(&read_df));
        }

        Ok(())
    }
}
