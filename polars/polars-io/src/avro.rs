use std::io::{Read, Seek, Write};

use super::{finish_reader, ArrowChunk, ArrowReader, ArrowResult};
use crate::prelude::*;
use polars_core::prelude::*;
use std::ops::Deref;

use arrow::io::avro::{read, write};

/// Read Apache Avro format into a DataFrame
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
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
}

impl<R: Read + Seek> AvroReader<R> {
    /// Get schema of the Avro File
    pub fn schema(&mut self) -> Result<Schema> {
        let (_, schema, _, _) = read::read_metadata(&mut self.reader)?;
        Ok((&schema.fields).into())
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

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
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
            columns: None,
            projection: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let (avro_schema, schema, codec, file_marker) = read::read_metadata(&mut self.reader)?;

        if let Some(columns) = self.columns {
            self.projection = Some(columns_to_projection(columns, &schema)?);
        }

        let (prj, arrow_schema) = if let Some(projection) = self.projection {
            let mut prj = vec![false; avro_schema.len()];
            for &index in projection.iter() {
                prj[index] = true;
            }

            (Some(prj), apply_projection(&schema, &projection))
        } else {
            (None, schema.clone())
        };

        let avro_reader = read::Reader::new(
            read::Decompressor::new(
                read::BlockStreamIterator::new(&mut self.reader, file_marker),
                codec,
            ),
            avro_schema,
            schema.fields,
            prj,
        );

        finish_reader(
            avro_reader,
            rechunk,
            self.n_rows,
            None,
            None,
            &arrow_schema,
            None,
        )
    }
}

pub use write::Compression as AvroCompression;

/// Write a DataFrame to Apache Avro format
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

    #[test]
    fn test_with_projection() -> Result<()> {
        let mut df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
        )?;

        let expected_df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2]
        )?;

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        AvroWriter::new(&mut buf).finish(&mut df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf)
            .with_projection(Some(vec![0, 1]))
            .finish()?;

        assert!(expected_df.frame_equal(&read_df));

        Ok(())
    }

    #[test]
    fn test_with_columns() -> Result<()> {
        let mut df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
        )?;

        let expected_df = df!(
            "i64" => &[1, 2],
            "utf8" => &["a", "b"]
        )?;

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        AvroWriter::new(&mut buf).finish(&mut df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf)
            .with_columns(Some(vec!["i64".to_string(), "utf8".to_string()]))
            .finish()?;

        assert!(expected_df.frame_equal(&read_df));

        Ok(())
    }
}
